// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LoopifyInsertSliceChain: detect unrolled insert_slice chains and convert
// them to scf.for loops with iter_args.
//
// Background
// ----------
// When JAX traces a Python loop like:
//
//   for k in range(90):
//       q_k  = q[k]          # tensor.extract_slice at constant offset k
//       out  = compute(q_k, carry)
//       carry = out           # inter-iteration dependency
//       outs.append(out)
//   result = jnp.stack(outs)  # stablehlo.concatenate
//
// IREE's StableHLO input lowering converts stablehlo.concatenate to a chain:
//
//   %a0 = tensor.insert_slice %out0 into %empty [0,0][1,N][1,1]
//   %a1 = tensor.insert_slice %out1 into %a0    [1,0][1,N][1,1]
//   ...
//   %result = tensor.insert_slice %out89 into %a88 [89,0][1,N][1,1]
//
// Where each %outK is derived from:
//   tensor.extract_slice %src [K,0][1,N][1,1]   (constant offset K)
//   ... (computation, possibly using output from iteration K-1 as carry)
//
// This pass detects such chains (length >= min_chain_length, default 8) and
// replaces them with an scf.for loop:
//
//   %result = scf.for %k = 0 to 90 step 1
//       iter_args(%acc = %empty, %carry0 = %init0, ...) {
//     %slice = tensor.extract_slice %src[%k, 0][1, N][1, 1]
//     %out_k = <body using %slice and carry>
//     %new_acc = tensor.insert_slice %out_k into %acc [%k, 0][1, N][1, 1]
//     scf.yield %new_acc, %out_k, ...
//   }
//
// This allows IREE's DispatchCreation to form ONE dispatch for the loop body,
// generating a single GPU kernel that loops over all levels internally —
// eliminating 90x kernel launch overhead.
//
// Multiple concat outputs that share the same loop body (same source tensors,
// same N) are grouped into ONE scf.for with multiple accumulators.

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Verifier.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_LOOPIFYINSERTSLICECHAINPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the constant integer value of an index-type Value, or -1 if unknown.
static int64_t getConstantIndex(Value v) {
  if (!v) return -1;
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val))) return val.getSExtValue();
  IntegerAttr attr;
  if (matchPattern(v, m_Constant(&attr))) return attr.getInt();
  return -1;
}

/// Given a tensor.insert_slice op, return the static offset along `dim`,
/// or -1 if the offset is not a compile-time constant.
static int64_t getInsertSliceOffset(tensor::InsertSliceOp op, int64_t dim) {
  auto offsets = op.getMixedOffsets();
  if (dim >= (int64_t)offsets.size()) return -1;
  auto &ofr = offsets[dim];
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr)) return ia.getInt();
    return -1;
  }
  return getConstantIndex(ofr.dyn_cast<Value>());
}

/// Given a tensor.extract_slice op, return the static offset along `dim`.
static int64_t getExtractSliceOffset(tensor::ExtractSliceOp op, int64_t dim) {
  auto offsets = op.getMixedOffsets();
  if (dim >= (int64_t)offsets.size()) return -1;
  auto &ofr = offsets[dim];
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr)) return ia.getInt();
    return -1;
  }
  return getConstantIndex(ofr.dyn_cast<Value>());
}

/// Return the static size of tensor type along `dim`, or -1.
static int64_t getStaticDimSize(Value v, int64_t dim) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || dim >= ty.getRank()) return -1;
  return ty.isDynamicDim(dim) ? -1 : ty.getDimSize(dim);
}

//===----------------------------------------------------------------------===//
// Chain analysis
//===----------------------------------------------------------------------===//

/// A single element of an unrolled chain.
struct ChainElement {
  tensor::InsertSliceOp insertOp; // The insert_slice at this iteration
  int64_t iterIndex;              // The loop iteration index (offset along dim)
  Value sourceValue;              // What gets inserted (%outK)
};

/// A detected unrolled chain: all insert_slice ops that form one loop.
struct UnrolledChain {
  SmallVector<ChainElement> elements; // Ordered by iterIndex
  int64_t sliceDim;                   // The dimension being looped over
  int64_t numIters;                   // Total number of iterations
  Value initialDest;                  // The initial "empty" tensor
  RankedTensorType accType;           // Type of the accumulator
};

/// Try to build an unrolled chain starting from `first`.
/// `first` must be an insert_slice whose dest is a tensor.empty (or zeros).
/// Returns true and fills `chain` if successful.
static bool buildChain(tensor::InsertSliceOp first, int64_t sliceDim,
                       int64_t minLen, UnrolledChain &chain) {
  // `first` must have a constant offset along sliceDim.
  int64_t off0 = getInsertSliceOffset(first, sliceDim);
  if (off0 < 0) return false;

  // Check that the slice at this dim has size 1.
  auto sliceSizes = first.getMixedSizes();
  if (sliceDim >= (int64_t)sliceSizes.size()) return false;
  auto &sz = sliceSizes[sliceDim];
  int64_t dimSz = -1;
  if (auto attr = sz.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr)) dimSz = ia.getInt();
  } else {
    dimSz = getConstantIndex(sz.dyn_cast<Value>());
  }
  if (dimSz != 1) return false; // Only handle unit-slice-size per iteration

  // Walk the chain: each link is an insert_slice whose dest is the result
  // of the previous insert_slice.
  SmallVector<ChainElement> elems;
  elems.push_back({first, off0, first.getSource()});

  tensor::InsertSliceOp cur = first;
  while (true) {
    // Does cur have exactly one user that is another insert_slice?
    tensor::InsertSliceOp next;
    int numInsertUses = 0;
    for (auto *user : cur->getUsers()) {
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user)) {
        // Must use cur's result as its dest operand (not source).
        if (ins.getDest() == cur.getResult()) {
          next = ins;
          ++numInsertUses;
        }
      }
    }
    if (numInsertUses != 1 || !next) break;

    int64_t off = getInsertSliceOffset(next, sliceDim);
    if (off < 0) break;
    // Must be exactly one step forward.
    int64_t prevOff = elems.back().iterIndex;
    if (off != prevOff + dimSz) break;
    // Slice sizes must match.
    auto nextSizes = next.getMixedSizes();
    if (sliceDim < (int64_t)nextSizes.size()) {
      auto &nsz = nextSizes[sliceDim];
      int64_t nDimSz = -1;
      if (auto attr = nsz.dyn_cast<Attribute>()) {
        if (auto ia = dyn_cast<IntegerAttr>(attr)) nDimSz = ia.getInt();
      } else {
        nDimSz = getConstantIndex(nsz.dyn_cast<Value>());
      }
      if (nDimSz != 1) break;
    }

    elems.push_back({next, off, next.getSource()});
    cur = next;
  }

  if ((int64_t)elems.size() < minLen) return false;

  // Verify offsets are consecutive starting from 0.
  if (elems[0].iterIndex != 0) return false;
  for (size_t i = 1; i < elems.size(); ++i) {
    if (elems[i].iterIndex != (int64_t)i) return false;
  }

  auto resultType =
      dyn_cast<RankedTensorType>(cur.getResult().getType());
  if (!resultType) return false;

  chain.elements = std::move(elems);
  chain.sliceDim = sliceDim;
  chain.numIters = (int64_t)chain.elements.size();
  chain.initialDest = first.getDest();
  chain.accType = resultType;
  return true;
}

//===----------------------------------------------------------------------===//
// Carry and body analysis
//===----------------------------------------------------------------------===//

/// Collect all ops (transitively) that produce values used by the computation
/// of `sourceValue`, stopping at `boundary` values (which are treated as
/// available inputs). Does NOT cross insert_slice dest operands.
static void collectBodyOps(Value sourceValue, const DenseSet<Value> &boundary,
                            SetVector<Operation *> &bodyOps,
                            SetVector<Value> &bodyValues) {
  SmallVector<Value> worklist = {sourceValue};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (bodyValues.contains(v)) continue;
    if (boundary.count(v)) continue;
    // Block arguments are invariant.
    if (isa<BlockArgument>(v)) continue;

    Operation *defOp = v.getDefiningOp();
    if (!defOp) continue;

    bodyValues.insert(v);
    if (!bodyOps.contains(defOp)) {
      bodyOps.insert(defOp);
      // Add ALL results of this op so bodyValues represents the complete set of
      // values produced by ops in bodyOps, not just the backward-traced path.
      // This is critical for multi-result ops (e.g., linalg.generic producing
      // both q_v_out and pflx): the trace may reach the op via pflx but
      // q_v_out must also be in bodyValues for grouping to detect it.
      for (Value result : defOp->getResults())
        bodyValues.insert(result);
      for (Value operand : defOp->getOperands()) {
        worklist.push_back(operand);
      }
    }
  }
}

/// For each iteration k in [0, numIters), find which values produced by
/// iteration k-1's body are used as inputs to iteration k's body.
/// These are the "carry" values that need to be passed as iter_args.
///
/// Strategy:
///   - iter0Body = ops/values reachable from elements[0].sourceValue, bounded
///     by function args and constants.
///   - For iter1, find all operands of its body ops that are in iter0Body.
///     Those are carry values (produced by iter0, consumed by iter1).
///
static void detectCarry(const UnrolledChain &chain,
                         const DenseSet<Value> &boundary,
                         SmallVector<Value> &carryOutputs,
                         SmallVector<Value> &carryInitialValues) {
  if (chain.numIters < 2) return;

  // Collect iter0 body (all ops/values reachable from elements[0].sourceValue).
  SetVector<Operation *> iter0Ops;
  SetVector<Value> iter0Values;
  collectBodyOps(chain.elements[0].sourceValue, boundary, iter0Ops,
                 iter0Values);

  // Collect iter1 body ops EXCLUSIVE to iter1 by stopping at all iter0 output
  // values. Without this, iter1's backward trace would walk through iter0's
  // entire computation (since iter1 depends on iter0's carry output), causing
  // false carry detection of every intermediate value in iter0.
  //
  // We stop at: (a) iter0's direct output, and (b) all results of its defining
  // op (e.g., if the defining op produces both q_out and pflx, we stop at both
  // so neither is traversed through).
  DenseSet<Value> iter1Boundary = boundary;
  iter1Boundary.insert(chain.elements[0].sourceValue);
  if (Operation *defOp = chain.elements[0].sourceValue.getDefiningOp()) {
    for (Value res : defOp->getResults())
      iter1Boundary.insert(res);
  }
  // Also stop at all other values in iter0 that might be used as carries
  // but aren't the direct output (e.g., intermediate outputs of a multi-result
  // op shared across chains). Add the full iter0Values as boundary so that
  // iter1OpsOnly only captures ops that directly bridge iter0→iter1.
  for (Value v : iter0Values)
    iter1Boundary.insert(v);

  SetVector<Operation *> iter1OpsOnly;
  SetVector<Value> iter1ValuesOnly;
  collectBodyOps(chain.elements[1].sourceValue, iter1Boundary, iter1OpsOnly,
                 iter1ValuesOnly);

  // Find values produced in iter0 that the iter1-exclusive ops use directly.
  // These are the actual carry values passed between iterations.
  DenseSet<Value> carrySet;
  for (Operation *op : iter1OpsOnly) {
    for (Value operand : op->getOperands()) {
      if (iter0Values.contains(operand) && !carrySet.count(operand)) {
        carrySet.insert(operand);
        carryOutputs.push_back(operand);
      }
    }
  }

  // For each carry output from iter0, find the corresponding initial value:
  // what does iter0 use in place of the "previous iteration" output?
  // Since iter0 has no predecessor, it must use some constant/argument.
  // We find this by looking at iter1's body: it uses carryOutput (iter0's
  // result). The analogous value in iter0 is whatever iter0 used in the
  // same "carry input" position.
  //
  // Heuristic: for each carry value V from iter0, find the corresponding
  // "initial" value for iter0 by matching structural similarity with iter1:
  //   - Find ops in iter1 that USE V.
  //   - Find the corresponding op in iter0 (same opcode, same non-carry args).
  //   - The "carry input" to iter0's version is what it uses instead of V.
  //
  // Simpler fallback: look for the value that iter0 uses in the same
  // structural position. Since the unrolled code uses the same compute
  // pattern for each iter, we can find iter0's "carry input" by tracing
  // back through iter0's ops to find an operand that is NOT in iter0Values
  // (i.e., comes from outside iter0 — the initial carry).
  for (Value carryOut : carryOutputs) {
    // Find ops in iter1 (exclusive to iter1) that use carryOut.
    for (Operation *iter1Op : iter1OpsOnly) {
      bool usesCarry = false;
      unsigned carryOperandIdx = 0;
      for (auto [idx, operand] : llvm::enumerate(iter1Op->getOperands())) {
        if (operand == carryOut) {
          usesCarry = true;
          carryOperandIdx = idx;
          break;
        }
      }
      if (!usesCarry) continue;

      // Find the corresponding op in iter0 (same opcode).
      // The analogous operand in iter0 is the initial carry.
      for (Operation *iter0Op : iter0Ops) {
        if (iter0Op->getName() != iter1Op->getName()) continue;
        if (iter0Op->getNumOperands() != iter1Op->getNumOperands()) continue;
        if (carryOperandIdx < iter0Op->getNumOperands()) {
          Value initCarry = iter0Op->getOperand(carryOperandIdx);
          // initCarry must NOT be in iter0Values (it comes from outside),
          // and must have the same type as carryOut (avoid type mismatches).
          if (!iter0Values.contains(initCarry) &&
              initCarry.getType() == carryOut.getType()) {
            carryInitialValues.push_back(initCarry);
            goto found_init;
          }
        }
      }
      // Fallback: use a zero constant of the carry type.
      {
        // We don't have a builder here; leave null and handle in codegen.
        carryInitialValues.push_back(nullptr);
        goto found_init;
      }
    found_init:
      break;
    }
  }

  // Ensure carryInitialValues has same size as carryOutputs.
  while (carryInitialValues.size() < carryOutputs.size())
    carryInitialValues.push_back(nullptr);
}

//===----------------------------------------------------------------------===//
// scf.for construction
//===----------------------------------------------------------------------===//

/// Convert an unrolled chain to an scf.for loop.
/// Returns the final value (result of the scf.for's acc iter_arg).
static LogicalResult convertChainToLoop(UnrolledChain &chain,
                                         IRRewriter &rewriter) {
  Location loc = chain.elements[0].insertOp.getLoc();

  // Boundary values: all values NOT inside the loop body.
  // Start with all block arguments of the enclosing function.
  DenseSet<Value> boundary;
  Block *block = chain.elements[0].insertOp->getBlock();
  for (Value arg : block->getArguments())
    boundary.insert(arg);
  // Also treat the source tensors of extract_slice ops (the full-rank inputs)
  // as boundary values, since they're the loop inputs.
  // We'll discover these as we collect body ops.

  // Detect carry.
  SmallVector<Value> carryOutputs, carryInitials;
  detectCarry(chain, boundary, carryOutputs, carryInitials);

  // Gather all "source tensors": full-rank tensors that are sliced at each
  // iteration by tensor.extract_slice with a constant offset equal to k.
  // These become the loop-invariant inputs, sliced dynamically by %k.
  // We find them by looking at iter0's extract_slice ops.
  SetVector<Operation *> iter0Ops;
  SetVector<Value> iter0Values;
  collectBodyOps(chain.elements[0].sourceValue, boundary, iter0Ops,
                 iter0Values);

  // Find extract_slice ops in iter0 that slice along `chain.sliceDim` at
  // offset 0 (i.e., they become dynamic slices in the loop body).
  SmallVector<std::pair<tensor::ExtractSliceOp, Value>> dynamicSlices;

  for (Operation *op : iter0Ops) {
    auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extSlice) continue;
    int64_t off = getExtractSliceOffset(extSlice, chain.sliceDim);
    if (off != 0) continue; // Only those that start at iteration 0
    // Check that this dimension has size 1.
    auto sizes = extSlice.getMixedSizes();
    if (chain.sliceDim >= (int64_t)sizes.size()) continue;
    auto &dimSz = sizes[chain.sliceDim];
    int64_t sz = -1;
    if (auto attr = dimSz.dyn_cast<Attribute>()) {
      if (auto ia = dyn_cast<IntegerAttr>(attr)) sz = ia.getInt();
    } else {
      sz = getConstantIndex(dimSz.dyn_cast<Value>());
    }
    if (sz != 1) continue;
    dynamicSlices.push_back({extSlice, extSlice.getSource()});
  }

  if (dynamicSlices.empty()) {
    // No loop-varying inputs found; bail out.
    return failure();
  }

  // Verify all iterations have matching extract_slice sources.
  // For iter k, find all extract_slices at offset k with the same source.
  // (Quick sanity check for correctness.)
  for (int64_t k = 1; k < chain.numIters; ++k) {
    SetVector<Operation *> iterKOps;
    SetVector<Value> iterKValues;
    collectBodyOps(chain.elements[k].sourceValue, boundary, iterKOps,
                   iterKValues);
    for (auto &[extSlice0, src] : dynamicSlices) {
      bool found = false;
      for (Operation *op : iterKOps) {
        auto extK = dyn_cast<tensor::ExtractSliceOp>(op);
        if (!extK) continue;
        if (extK.getSource() != src) continue;
        int64_t off = getExtractSliceOffset(extK, chain.sliceDim);
        if (off != k) continue;
        found = true;
        break;
      }
      if (!found) {
        // Pattern doesn't match; bail.
        return failure();
      }
    }
  }

  // Build the scf.for loop.
  ImplicitLocOpBuilder b(loc, rewriter);
  b.setInsertionPoint(chain.elements[0].insertOp);

  Value lb = b.create<arith::ConstantIndexOp>(0);
  Value ub = b.create<arith::ConstantIndexOp>(chain.numIters);
  Value step = b.create<arith::ConstantIndexOp>(1);

  // iter_args: [acc, carry0, carry1, ...]
  SmallVector<Value> initArgs;
  initArgs.push_back(chain.initialDest); // acc
  for (size_t i = 0; i < carryOutputs.size(); ++i) {
    Value init = carryInitials[i];
    if (!init) {
      // Create a zero tensor of the carry output's type.
      auto ty = dyn_cast<RankedTensorType>(carryOutputs[i].getType());
      if (!ty) return failure();
      init = b.create<arith::ConstantOp>(
          b.getZeroAttr(ty));
    }
    initArgs.push_back(init);
  }

  SmallVector<Type> iterArgTypes;
  for (Value v : initArgs)
    iterArgTypes.push_back(v.getType());

  auto forOp = b.create<scf::ForOp>(
      lb, ub, step, initArgs,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value inductionVar,
          ValueRange iterArgs) {
        ImplicitLocOpBuilder bb(bodyLoc, bodyBuilder);

        // Build IRMapping: map iter0 ops to the loop body.
        IRMapping mapping;

        // Map each iter0 extract_slice (at offset 0) to a dynamic extract.
        for (auto &[extSlice0, src] : dynamicSlices) {
          // Build dynamic offsets: replace the sliceDim offset with inductionVar.
          SmallVector<OpFoldResult> offsets;
          auto staticOffsets = extSlice0.getMixedOffsets();
          for (size_t d = 0; d < staticOffsets.size(); ++d) {
            if ((int64_t)d == chain.sliceDim)
              offsets.push_back(inductionVar);
            else
              offsets.push_back(staticOffsets[d]);
          }
          auto newExt = bb.create<tensor::ExtractSliceOp>(
              extSlice0.getType(), src, offsets, extSlice0.getMixedSizes(),
              extSlice0.getMixedStrides());
          mapping.map(extSlice0.getResult(), newExt.getResult());
        }

        // Map carry INPUTS (not outputs) to the loop's carry iter_args.
        // When we clone iter0's body ops, they use `carryInitials[i]` as their
        // carry-slot input (e.g., pflx_init for iteration 0). We remap that
        // to `carry_iter_arg` so the cloned op uses the loop-carried value
        // from the previous iteration instead of the fixed initial constant.
        for (auto [carryInit, iterArg] :
             llvm::zip(carryInitials, iterArgs.drop_front(1))) {
          if (carryInit)
            mapping.map(carryInit, iterArg);
        }

        // Clone iter0 body ops in topological order.
        SetVector<Operation *> bodyOps(iter0Ops.begin(), iter0Ops.end());
        for (auto &[ext, src] : dynamicSlices)
          bodyOps.remove(ext.getOperation());
        mlir::topologicalSort(bodyOps);

        for (Operation *op : bodyOps) {
          Operation *cloned = bb.clone(*op, mapping);
          for (auto [orig, clonedRes] :
               llvm::zip(op->getResults(), cloned->getResults()))
            mapping.map(orig, clonedRes);
        }

        // The loop body's output is the mapped version of iter0's source.
        Value outK = mapping.lookupOrDefault(chain.elements[0].sourceValue);

        // Insert into accumulator at dynamic index.
        Value acc = iterArgs[0];
        auto insertSizes = chain.elements[0].insertOp.getMixedSizes();
        auto insertStrides = chain.elements[0].insertOp.getMixedStrides();
        SmallVector<OpFoldResult> dynOffsets;
        for (size_t d = 0; d < chain.elements[0].insertOp.getMixedOffsets().size(); ++d) {
          if ((int64_t)d == chain.sliceDim)
            dynOffsets.push_back(inductionVar);
          else
            dynOffsets.push_back(chain.elements[0].insertOp.getMixedOffsets()[d]);
        }
        Value newAcc = bb.create<tensor::InsertSliceOp>(outK, acc, dynOffsets,
                                                         insertSizes, insertStrides);

        // Yield: [new_acc, updated_carry0, updated_carry1, ...]
        SmallVector<Value> yieldVals = {newAcc};
        for (Value carryOut : carryOutputs) {
          yieldVals.push_back(mapping.lookupOrDefault(carryOut));
        }
        bb.create<scf::YieldOp>(yieldVals);
      });

  // Replace the final insert_slice result with the scf.for acc result.
  Value finalResult = forOp.getResult(0);
  rewriter.replaceOp(chain.elements.back().insertOp, finalResult);

  // Erase intermediate chain ops (all but the last, which we just replaced).
  for (int64_t i = (int64_t)chain.elements.size() - 2; i >= 0; --i) {
    rewriter.eraseOp(chain.elements[i].insertOp);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Grouped conversion: multiple chains → one scf.for with N accumulators
//===----------------------------------------------------------------------===//

/// Convert a group of chains sharing the same (sliceDim, numIters) into ONE
/// scf.for loop with one accumulator iter_arg per chain.
///
/// Without grouping, processing chains one-at-a-time creates N separate
/// scf.for loops where chain k's body includes ALL ops needed by chain k,
/// which for sequential computations (like graupel) means an O(N²) blowup
/// in total work. With grouping, ONE loop body computes everything once and
/// fills all N accumulators per iteration.
static LogicalResult convertChainGroupToLoop(SmallVectorImpl<UnrolledChain> &chains,
                                              IRRewriter &rewriter) {
  assert(!chains.empty());
  int64_t sliceDim = chains[0].sliceDim;
  int64_t numIters = chains[0].numIters;
  Location loc = chains[0].elements[0].insertOp.getLoc();
  llvm::errs() << "[LoopifyGroup] chains=" << chains.size()
               << " sliceDim=" << sliceDim << " numIters=" << numIters << "\n";

  // Boundary: block arguments of the enclosing function.
  DenseSet<Value> boundary;
  Block *block = chains[0].elements[0].insertOp->getBlock();
  for (Value arg : block->getArguments())
    boundary.insert(arg);

  // Compute the UNION of all chains' iter0 bodies.
  // Using union avoids a membership check — every chain's iter0 source is
  // guaranteed to be in the union by construction, even when chain bodies
  // are not nested (e.g., independent outputs from the same multi-result op).
  SetVector<Operation *> unionOps;
  SetVector<Value> unionValues;
  for (auto &chain : chains)
    collectBodyOps(chain.elements[0].sourceValue, boundary, unionOps,
                   unionValues);

  // For carry detection, use the chain with the largest individual body
  // (most ops → most inter-iteration dependencies → best proxy for carry).
  size_t carryChainIdx = 0;
  {
    size_t maxSize = 0;
    for (size_t i = 0; i < chains.size(); ++i) {
      SetVector<Operation *> ops;
      SetVector<Value> vals;
      collectBodyOps(chains[i].elements[0].sourceValue, boundary, ops, vals);
      if (ops.size() > maxSize) {
        maxSize = ops.size();
        carryChainIdx = i;
      }
    }
  }

  // Detect carry from the chain with the most deps.
  SmallVector<Value> carryOutputs, carryInitials;
  detectCarry(chains[carryChainIdx], boundary, carryOutputs, carryInitials);

  // Find extract_slice ops in the union body that slice at offset 0 along
  // sliceDim with size 1 — these become dynamic slices inside the loop body.
  SmallVector<std::pair<tensor::ExtractSliceOp, Value>> dynamicSlices;
  for (Operation *op : unionOps) {
    auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extSlice) continue;
    if (getExtractSliceOffset(extSlice, sliceDim) != 0) continue;
    auto sizes = extSlice.getMixedSizes();
    if (sliceDim >= (int64_t)sizes.size()) continue;
    int64_t sz = -1;
    auto &dimSzOFR = sizes[sliceDim];
    if (auto attr = dimSzOFR.dyn_cast<Attribute>()) {
      if (auto ia = dyn_cast<IntegerAttr>(attr)) sz = ia.getInt();
    } else {
      sz = getConstantIndex(dimSzOFR.dyn_cast<Value>());
    }
    if (sz != 1) continue;
    dynamicSlices.push_back({extSlice, extSlice.getSource()});
  }
  llvm::errs() << "[LoopifyGroup] unionOps=" << unionOps.size()
               << " dynamicSlices=" << dynamicSlices.size()
               << " carryOutputs=" << carryOutputs.size() << "\n";
  if (dynamicSlices.empty()) {
    llvm::errs() << "[LoopifyGroup] FAIL: dynamicSlices empty\n";
    return failure();
  }

  // Build the combined scf.for.
  // iter_args layout: [acc_0, ..., acc_{N-1}, carry_0, ..., carry_M]
  ImplicitLocOpBuilder b(loc, rewriter);
  b.setInsertionPoint(chains[0].elements[0].insertOp);

  Value lb = b.create<arith::ConstantIndexOp>(0);
  Value ub = b.create<arith::ConstantIndexOp>(numIters);
  Value step = b.create<arith::ConstantIndexOp>(1);

  size_t nAccs = chains.size();
  SmallVector<Value> initArgs;
  for (auto &chain : chains)
    initArgs.push_back(chain.initialDest);
  for (size_t i = 0; i < carryOutputs.size(); ++i) {
    Value init = carryInitials[i];
    if (!init) {
      auto ty = dyn_cast<RankedTensorType>(carryOutputs[i].getType());
      if (!ty) {
        llvm::errs() << "[LoopifyGroup] FAIL: carry type not RankedTensor\n";
        return failure();
      }
      init = b.create<arith::ConstantOp>(b.getZeroAttr(ty));
    }
    initArgs.push_back(init);
  }

  auto forOp = b.create<scf::ForOp>(
      lb, ub, step, initArgs,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value inductionVar,
          ValueRange iterArgs) {
        ImplicitLocOpBuilder bb(bodyLoc, bodyBuilder);
        IRMapping mapping;

        // Map iter0 extract_slices to dynamic versions.
        for (auto &[extSlice0, src] : dynamicSlices) {
          SmallVector<OpFoldResult> offsets;
          auto staticOffsets = extSlice0.getMixedOffsets();
          for (size_t d = 0; d < staticOffsets.size(); ++d) {
            offsets.push_back((int64_t)d == sliceDim
                                  ? OpFoldResult(inductionVar)
                                  : staticOffsets[d]);
          }
          auto newExt = bb.create<tensor::ExtractSliceOp>(
              extSlice0.getType(), src, offsets, extSlice0.getMixedSizes(),
              extSlice0.getMixedStrides());
          mapping.map(extSlice0.getResult(), newExt.getResult());
        }

        // Map carry initials → carry iter_args so cloned body ops use the
        // loop-carried value from the previous iteration.
        for (auto [carryInit, iterArg] :
             llvm::zip(carryInitials, iterArgs.drop_front(nAccs))) {
          if (carryInit)
            mapping.map(carryInit, iterArg);
        }

        // Clone union body in topological order using mlir::topologicalSort.
        // This correctly handles multi-hop deps (e.g., chain N's body uses
        // outputs from chain N-1 as intermediate values) that the manual
        // topo sort misses when mapping is only pre-populated with initial
        // values (extract_slices + carry initials).
        SetVector<Operation *> bodyOps(unionOps.begin(), unionOps.end());
        for (auto &[ext, src] : dynamicSlices)
          bodyOps.remove(ext.getOperation());
        mlir::topologicalSort(bodyOps);

        for (Operation *op : bodyOps) {
          Operation *cloned = bb.clone(*op, mapping);
          for (auto [orig, clonedRes] :
               llvm::zip(op->getResults(), cloned->getResults()))
            mapping.map(orig, clonedRes);
        }

        // For each chain: insert its output into the corresponding accumulator.
        SmallVector<Value> yieldVals;
        for (size_t i = 0; i < chains.size(); ++i) {
          Value outK =
              mapping.lookupOrDefault(chains[i].elements[0].sourceValue);
          Value acc = iterArgs[i];
          auto &elem0 = chains[i].elements[0];
          SmallVector<OpFoldResult> dynOffsets;
          for (size_t d = 0; d < elem0.insertOp.getMixedOffsets().size(); ++d) {
            dynOffsets.push_back((int64_t)d == sliceDim
                                     ? OpFoldResult(inductionVar)
                                     : elem0.insertOp.getMixedOffsets()[d]);
          }
          yieldVals.push_back(bb.create<tensor::InsertSliceOp>(
              outK, acc, dynOffsets, elem0.insertOp.getMixedSizes(),
              elem0.insertOp.getMixedStrides()));
        }
        for (Value carryOut : carryOutputs)
          yieldVals.push_back(mapping.lookupOrDefault(carryOut));

        bb.create<scf::YieldOp>(yieldVals);
      });

  llvm::errs() << "[LoopifyGroup] SUCCESS: built 1 scf.for for "
               << chains.size() << " chains\n";
  // Replace each chain's final insert_slice with the scf.for acc result.
  for (size_t i = 0; i < chains.size(); ++i) {
    rewriter.replaceOp(chains[i].elements.back().insertOp, forOp.getResult(i));
    for (int64_t j = (int64_t)chains[i].elements.size() - 2; j >= 0; --j)
      rewriter.eraseOp(chains[i].elements[j].insertOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Forall + For construction: cells parallel, levels sequential
//===----------------------------------------------------------------------===//

/// Scalarize a purely element-wise linalg.generic op.
/// Given scalar inputs already mapped in `mapping`, inline the linalg body
/// as scalar ops and map each result to the corresponding scalar output.
/// Ops with non-parallel (reduction) iterators are skipped silently.
static void scalarizeLinalgGenericOp(linalg::GenericOp genOp, IRMapping &mapping,
                                     ImplicitLocOpBuilder &b) {
  // Only handle all-parallel iterator types.
  for (auto it : genOp.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel)
      return;

  Block &body = genOp.getRegion().front();
  int numInputs = genOp.getNumDpsInputs();
  int numOutputs = genOp.getNumDpsInits();

  // Build body mapping: block arg → scalar value from outer mapping.
  IRMapping bodyMapping;
  for (int i = 0; i < numInputs; ++i) {
    Value inp = genOp.getDpsInputs()[i];
    bodyMapping.map(body.getArgument(i), mapping.lookupOrDefault(inp));
  }
  // Output block args: map to zero using each arg's own type (may be i1, f64, etc.).
  for (int i = 0; i < numOutputs; ++i) {
    Type outElemTy = body.getArgument(numInputs + i).getType();
    bodyMapping.map(body.getArgument(numInputs + i),
                    b.create<arith::ConstantOp>(b.getZeroAttr(outElemTy)));
  }

  // Clone body ops (except linalg.yield) and collect yield values.
  SmallVector<Value> yieldVals;
  for (auto &bodyOp : body) {
    if (isa<linalg::YieldOp>(&bodyOp)) {
      for (Value v : bodyOp.getOperands())
        yieldVals.push_back(bodyMapping.lookupOrDefault(v));
      break;
    }
    Operation *cloned = b.clone(bodyOp, bodyMapping);
    for (auto [orig, res] :
         llvm::zip(bodyOp.getResults(), cloned->getResults()))
      bodyMapping.map(orig, res);
  }

  // Map genOp results → scalar yield values.
  for (int i = 0; i < numOutputs && i < (int)yieldVals.size(); ++i)
    mapping.map(genOp.getResult(i), yieldVals[i]);
}

/// Convert a group of chains into a GPU-optimal structure:
///   flow.dispatch.region {
///     scf.forall (%c in [0, numCells)) shared_outs(acc0 = empty0, ...) {
///       scf.for %k = 0 to numIters iter_args(col0=init, carry=0.0) {
///         %val = tensor.extract %src[%k, %c]   // scalar
///         <scalarized body>
///         %new_col = tensor.insert %out into %col[%k, 0]
///         scf.yield %new_col, ..., %scalar_carry, ...
///       }
///       scf.forall.in_parallel {
///         tensor.parallel_insert_slice %col into %acc[0, %c][numIters, 1]
///       }
///     }
///     flow.return %forall_result0, ...
///   }
///
/// The scf.forall has #iree_codegen.workgroup_mapping<x> so IREE's codegen
/// recognizes it as workgroup distribution. ALL writes to global memory go
/// through tensor.parallel_insert_slice inside the workgroup-mapped forall,
/// satisfying IREE's workgroup distribution verification.
///
/// The flow.dispatch.region wrapper ensures IREE keeps everything in a single
/// dispatch, preventing the body from being split into many small dispatches.
static LogicalResult convertChainGroupToForallDispatch(
    SmallVectorImpl<UnrolledChain> &chains, IRRewriter &rewriter) {
  assert(!chains.empty());
  int64_t sliceDim = chains[0].sliceDim;
  int64_t numIters = chains[0].numIters;
  Location loc = chains[0].elements[0].insertOp.getLoc();
  llvm::errs() << "[ForallDispatch] chains=" << chains.size()
               << " sliceDim=" << sliceDim << " numIters=" << numIters << "\n";

  // --- Collect boundary and union body ops ---
  DenseSet<Value> boundary;
  Block *block = chains[0].elements[0].insertOp->getBlock();
  for (Value arg : block->getArguments())
    boundary.insert(arg);

  SetVector<Operation *> unionOps;
  SetVector<Value> unionValues;
  for (auto &chain : chains)
    collectBodyOps(chain.elements[0].sourceValue, boundary, unionOps,
                   unionValues);

  // --- Find level-slices: extract_slice at offset 0 along sliceDim, size 1 ---
  SmallVector<std::pair<tensor::ExtractSliceOp, Value>> levelSlices;
  for (Operation *op : unionOps) {
    auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extSlice) continue;
    if (getExtractSliceOffset(extSlice, sliceDim) != 0) continue;
    auto sizes = extSlice.getMixedSizes();
    if (sliceDim >= (int64_t)sizes.size()) continue;
    int64_t sz = -1;
    auto &dimSzOFR = sizes[sliceDim];
    if (auto attr = dimSzOFR.dyn_cast<Attribute>()) {
      if (auto ia = dyn_cast<IntegerAttr>(attr)) sz = ia.getInt();
    } else {
      sz = getConstantIndex(dimSzOFR.dyn_cast<Value>());
    }
    if (sz != 1) continue;
    levelSlices.push_back({extSlice, extSlice.getSource()});
  }
  if (levelSlices.empty()) {
    llvm::errs() << "[ForallDispatch] FAIL: no level slices\n";
    return failure();
  }

  // --- Determine cell dimension and numCells ---
  auto srcType =
      dyn_cast<RankedTensorType>(levelSlices[0].second.getType());
  if (!srcType) return failure();

  int64_t cellDim = -1, numCells = -1;
  for (int64_t d = 0; d < srcType.getRank(); ++d) {
    if (d == sliceDim) continue;
    if (!srcType.isDynamicDim(d) && srcType.getDimSize(d) > 0) {
      cellDim = d;
      numCells = srcType.getDimSize(d);
      break;
    }
  }
  if (numCells <= 0 || cellDim < 0) {
    llvm::errs() << "[ForallDispatch] FAIL: cannot determine cell dimension\n";
    return failure();
  }
  llvm::errs() << "[ForallDispatch] numCells=" << numCells
               << " cellDim=" << cellDim << "\n";

  // --- Detect carry ---
  size_t carryChainIdx = 0;
  {
    size_t maxSize = 0;
    for (size_t i = 0; i < chains.size(); ++i) {
      SetVector<Operation *> ops;
      SetVector<Value> vals;
      collectBodyOps(chains[i].elements[0].sourceValue, boundary, ops, vals);
      if (ops.size() > maxSize) {
        maxSize = ops.size();
        carryChainIdx = i;
      }
    }
  }
  SmallVector<Value> carryOutputs, carryInitials;
  detectCarry(chains[carryChainIdx], boundary, carryOutputs, carryInitials);
  size_t nAccs = chains.size();
  size_t nCarry = carryOutputs.size();
  llvm::errs() << "[ForallDispatch] carryOutputs=" << nCarry << "\n";

  // --- Column tensor type: [numIters, 1] for sliceDim=0, cellDim=1 ---
  SmallVector<int64_t> colShape(srcType.getRank());
  for (int64_t d = 0; d < srcType.getRank(); ++d)
    colShape[d] = (d == sliceDim) ? numIters : 1;

  // --- Create flow.dispatch.region wrapping the entire forall ---
  // The dispatch.region forces IREE to keep everything in one kernel.
  // We do NOT set mapping attributes — IREE's codegen adds those later.
  rewriter.setInsertionPoint(chains[0].elements[0].insertOp);

  SmallVector<Type> dispatchResultTypes;
  for (auto &chain : chains)
    dispatchResultTypes.push_back(chain.accType);

  auto regionOp = IREE::Flow::DispatchRegionOp::create(
      rewriter, loc, dispatchResultTypes,
      /*result_dims=*/ValueRange{},
      /*workload=*/ValueRange{});
  Block &regionBody = regionOp.getBody().emplaceBlock();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&regionBody);
    ImplicitLocOpBuilder b(loc, rewriter);

    // Shared outputs: initial empty accumulator tensors (one per chain).
    SmallVector<Value> sharedInits;
    for (auto &chain : chains)
      sharedInits.push_back(chain.initialDest);

    // Create scf.forall WITH WorkgroupMappingAttr so IREE's codegen
    // knows how to distribute it to GPU workgroups (one thread per cell).
    auto forallOp = scf::ForallOp::create(
        b, loc,
        ArrayRef<OpFoldResult>{b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(numCells)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
        ValueRange(sharedInits),
        /*mapping=*/rewriter.getArrayAttr(
            {IREE::Codegen::WorkgroupMappingAttr::get(
                rewriter.getContext(),
                IREE::Codegen::WorkgroupId::IdX)}));

    {

    Value cellId = forallOp.getInductionVars()[0];
    auto sharedOuts = forallOp.getRegionIterArgs();

    // Set insertion point before the in_parallel terminator.
    scf::InParallelOp terminator = forallOp.getTerminator();
    rewriter.setInsertionPoint(terminator);
    ImplicitLocOpBuilder fb(loc, rewriter);
    Value zero = fb.create<arith::ConstantIndexOp>(0);

    // Initial column accumulators (empty tensors per chain).
    SmallVector<Value> forInitArgs;
    {
      SmallVector<OpFoldResult> colMixedSizes;
      for (int64_t d : colShape)
        colMixedSizes.push_back(rewriter.getIndexAttr(d));
      for (size_t i = 0; i < nAccs; ++i) {
        Type chainElemType = chains[i].accType.getElementType();
        forInitArgs.push_back(
            tensor::EmptyOp::create(rewriter, loc, colMixedSizes,
                                    chainElemType));
      }
    }

    // Scalar carry initials: extract from carry initial tensor at cellId.
    for (size_t i = 0; i < nCarry; ++i) {
      Value carryInit = carryInitials[i];
      Value scalarInit;
      if (!carryInit) {
        Type carryElemTy;
        if (auto carryTy =
                dyn_cast<RankedTensorType>(carryOutputs[i].getType()))
          carryElemTy = carryTy.getElementType();
        else
          carryElemTy = carryOutputs[i].getType();
        scalarInit = fb.create<arith::ConstantOp>(fb.getZeroAttr(carryElemTy));
      } else {
        auto carryTy = dyn_cast<RankedTensorType>(carryInit.getType());
        if (carryTy) {
          SmallVector<Value> extractIdx;
          for (int64_t d = 0; d < carryTy.getRank(); ++d)
            extractIdx.push_back(d == cellDim ? cellId : zero);
          scalarInit = fb.create<tensor::ExtractOp>(carryInit, extractIdx);
        } else {
          scalarInit = carryInit;
        }
      }
      forInitArgs.push_back(scalarInit);
    }

    // --- scf.for over levels [0, numIters) ---
    Value lb = fb.create<arith::ConstantIndexOp>(0);
    Value ub = fb.create<arith::ConstantIndexOp>(numIters);
    Value step = fb.create<arith::ConstantIndexOp>(1);

    auto forOp = fb.create<scf::ForOp>(
        lb, ub, step, forInitArgs,
        [&](OpBuilder &forBodyBuilder, Location forBodyLoc, Value k,
            ValueRange forIterArgs) {
          ImplicitLocOpBuilder kb(forBodyLoc, forBodyBuilder);
          IRMapping mapping;

          // Map level-slices → scalar extracts: tensor.extract %src[k, cellId].
          for (auto &[extSlice0, src] : levelSlices) {
            auto srcTy = cast<RankedTensorType>(src.getType());
            SmallVector<Value> extractIdx;
            for (int64_t d = 0; d < srcTy.getRank(); ++d) {
              if (d == sliceDim) extractIdx.push_back(k);
              else if (d == cellDim) extractIdx.push_back(cellId);
              else extractIdx.push_back(zero);
            }
            Value scalar = kb.create<tensor::ExtractOp>(src, extractIdx);
            mapping.map(extSlice0.getResult(), scalar);
          }

          // Map carry initials → scalar carry iter_args.
          for (auto [carryInit, iterArg] :
               llvm::zip(carryInitials, forIterArgs.drop_front(nAccs))) {
            if (carryInit)
              mapping.map(carryInit, iterArg);
          }

          // Scalarize and clone body ops in topological order.
          SetVector<Operation *> bodyOps(unionOps.begin(), unionOps.end());
          for (auto &[ext, src] : levelSlices)
            bodyOps.remove(ext.getOperation());
          mlir::topologicalSort(bodyOps);

          auto extractScalar = [&](Value tensorVal) -> Value {
            auto tensorTy = cast<RankedTensorType>(tensorVal.getType());
            SmallVector<Value> idx;
            for (int64_t d = 0; d < tensorTy.getRank(); ++d) {
              int64_t dimSz = tensorTy.getDimSize(d);
              if (d == sliceDim)
                idx.push_back(dimSz == 1 ? zero : k);
              else if (d == cellDim)
                idx.push_back(dimSz == 1 ? zero : cellId);
              else
                idx.push_back(zero);
            }
            return kb.create<tensor::ExtractOp>(tensorVal, idx);
          };

          for (Operation *op : bodyOps) {
            if (llvm::all_of(op->getResults(), [&](Value v) {
                  return mapping.lookupOrDefault(v) != v;
                })) {
              continue;
            }

            if (!isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
                     tensor::ExtractOp, linalg::LinalgOp, tensor::EmptyOp,
                     arith::ConstantOp>(op)) {
              for (Value operand : op->getOperands()) {
                Value mapped = mapping.lookupOrDefault(operand);
                if (!isa<RankedTensorType>(mapped.getType())) continue;
                mapping.map(operand, extractScalar(mapped));
              }
            }

            if (auto genOp = dyn_cast<linalg::GenericOp>(op)) {
              for (int i = 0; i < genOp.getNumDpsInputs(); ++i) {
                Value inp = genOp.getDpsInputs()[i];
                Value mapped = mapping.lookupOrDefault(inp);
                if (isa<RankedTensorType>(mapped.getType()))
                  mapping.map(inp, extractScalar(mapped));
              }
              scalarizeLinalgGenericOp(genOp, mapping, kb);
            } else if (isa<tensor::EmptyOp>(op)) {
              for (Value result : op->getResults()) {
                if (mapping.lookupOrDefault(result) == result) {
                  auto resTy = cast<RankedTensorType>(result.getType());
                  mapping.map(result, kb.create<arith::ConstantOp>(
                                          kb.getZeroAttr(resTy.getElementType())));
                }
              }
            } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
              auto resultTy =
                  dyn_cast<RankedTensorType>(constOp.getResult().getType());
              if (resultTy) {
                if (auto denseAttr =
                        dyn_cast<DenseElementsAttr>(constOp.getValue())) {
                  if (denseAttr.isSplat()) {
                    Type scalarTy = resultTy.getElementType();
                    TypedAttr scalarAttr;
                    if (isa<FloatType>(scalarTy)) {
                      APFloat splatF = *denseAttr.getValues<APFloat>().begin();
                      scalarAttr = FloatAttr::get(scalarTy, splatF);
                    } else if (isa<IntegerType>(scalarTy)) {
                      APInt splatI = *denseAttr.getValues<APInt>().begin();
                      scalarAttr = IntegerAttr::get(scalarTy, splatI);
                    }
                    if (scalarAttr) {
                      auto sc = kb.create<arith::ConstantOp>(scalarAttr);
                      mapping.map(constOp.getResult(), sc.getResult());
                      continue;
                    }
                  }
                }
              }
              Operation *cloned = kb.clone(*op, mapping);
              for (auto [orig, res] :
                   llvm::zip(op->getResults(), cloned->getResults()))
                mapping.map(orig, res);
            } else if (auto extSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
              Value mappedSrc =
                  mapping.lookupOrDefault(extSliceOp.getSource());
              Value scalar;
              if (!isa<RankedTensorType>(mappedSrc.getType()))
                scalar = mappedSrc;
              else
                scalar = extractScalar(mappedSrc);
              mapping.map(extSliceOp.getResult(), scalar);
            } else if (auto extOp = dyn_cast<tensor::ExtractOp>(op)) {
              Value mappedSrc = mapping.lookupOrDefault(extOp.getTensor());
              if (!isa<RankedTensorType>(mappedSrc.getType())) {
                mapping.map(extOp.getResult(), mappedSrc);
              } else {
                Operation *cloned = kb.clone(*op, mapping);
                mapping.map(extOp.getResult(), cloned->getResult(0));
              }
            } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
              Value fillVal = mapping.lookupOrDefault(fillOp.getInputs()[0]);
              if (isa<RankedTensorType>(fillVal.getType()))
                fillVal = extractScalar(fillVal);
              for (Value result : fillOp.getResults())
                mapping.map(result, fillVal);
            } else if (isa<linalg::LinalgOp>(op)) {
              for (Value result : op->getResults()) {
                if (auto resTy = dyn_cast<RankedTensorType>(result.getType()))
                  mapping.map(result, kb.create<arith::ConstantOp>(
                                          kb.getZeroAttr(resTy.getElementType())));
              }
            } else {
              Operation *cloned = kb.clone(*op, mapping);
              for (auto [orig, res] :
                   llvm::zip(op->getResults(), cloned->getResults()))
                mapping.map(orig, res);
            }
          } // end body ops loop

          // Yield: [new_col0, ..., new_colN, carry0_out, ..., carryM_out]
          SmallVector<Value> yieldVals;
          for (size_t i = 0; i < nAccs; ++i) {
            Value scalarOut =
                mapping.lookupOrDefault(chains[i].elements[0].sourceValue);
            if (isa<RankedTensorType>(scalarOut.getType()))
              scalarOut = extractScalar(scalarOut);
            Value col = forIterArgs[i];
            SmallVector<Value> insertIdx;
            for (int64_t d = 0; d < (int64_t)colShape.size(); ++d)
              insertIdx.push_back(d == sliceDim ? k : zero);
            Value newCol =
                kb.create<tensor::InsertOp>(scalarOut, col, insertIdx);
            yieldVals.push_back(newCol);
          }
          for (Value carryOut : carryOutputs) {
            Value mapped = mapping.lookupOrDefault(carryOut);
            if (isa<RankedTensorType>(mapped.getType()))
              mapped = extractScalar(mapped);
            yieldVals.push_back(mapped);
          }
          kb.create<scf::YieldOp>(yieldVals);
        }); // end scf.for

    // --- Populate scf.forall.in_parallel terminator ---
    rewriter.setInsertionPointToStart(terminator.getBody());
    for (size_t i = 0; i < nAccs; ++i) {
      Value col = forOp.getResult(i);
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (int64_t d = 0; d < (int64_t)colShape.size(); ++d) {
        if (d == cellDim) {
          offsets.push_back(cellId);
          sizes.push_back(rewriter.getIndexAttr(1));
        } else {
          offsets.push_back(rewriter.getIndexAttr(0));
          sizes.push_back(rewriter.getIndexAttr(colShape[d]));
        }
        strides.push_back(rewriter.getIndexAttr(1));
      }
      tensor::ParallelInsertSliceOp::create(rewriter, loc, col, sharedOuts[i],
                                            offsets, sizes, strides);
    }

    }

    // flow.return the forall results from the dispatch region.
    rewriter.setInsertionPointAfter(forallOp);
    SmallVector<Value> regionResults;
    for (size_t i = 0; i < nAccs; ++i)
      regionResults.push_back(forallOp.getResult(i));
    IREE::Flow::ReturnOp::create(rewriter, loc, regionResults);
  }

  llvm::errs() << "[ForallDispatch] SUCCESS: created dispatch.region with "
                  "forall(%c in [0," << numCells << "), %k in [0," << numIters
               << ")) for " << chains.size() << " chains\n";

  // Replace each chain's final insert_slice with dispatch region result.
  for (size_t i = 0; i < chains.size(); ++i) {
    rewriter.replaceOp(chains[i].elements.back().insertOp,
                       regionOp->getResult(i));
    for (int64_t j = (int64_t)chains[i].elements.size() - 2; j >= 0; --j)
      rewriter.eraseOp(chains[i].elements[j].insertOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct LoopifyInsertSliceChainPass
    : public iree_compiler::Preprocessing::impl::LoopifyInsertSliceChainPassBase<
          LoopifyInsertSliceChainPass> {
  using iree_compiler::Preprocessing::impl::LoopifyInsertSliceChainPassBase<
      LoopifyInsertSliceChainPass>::LoopifyInsertSliceChainPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Step 1: collect ALL chains from tensor.empty heads in one pass.
    SmallVector<UnrolledChain> allChains;
    funcOp.walk([&](tensor::InsertSliceOp op) {
      if (!isa_and_nonnull<tensor::EmptyOp>(op.getDest().getDefiningOp()))
        return;
      for (int64_t dim = 0; dim < 4; ++dim) {
        UnrolledChain chain;
        if (buildChain(op, dim, minChainLength, chain)) {
          allChains.push_back(std::move(chain));
          break;
        }
      }
    });
    if (allChains.empty()) {
      llvm::errs() << "[Loopify] no chains found in function\n";
      return;
    }

    for (size_t i = 0; i < allChains.size(); ++i)
      llvm::errs() << "  chain[" << i << "]: sliceDim=" << allChains[i].sliceDim
                   << " numIters=" << allChains[i].numIters << "\n";
    // Step 2: group chains by (sliceDim, numIters).
    // Chains in the same group share the same loop trip count and can be
    // combined into ONE scf.for to avoid redundant body recomputation.
    SmallVector<SmallVector<size_t>> groups;
    SmallVector<std::pair<int64_t, int64_t>> groupKeys;
    for (size_t i = 0; i < allChains.size(); ++i) {
      auto key =
          std::make_pair(allChains[i].sliceDim, allChains[i].numIters);
      bool found = false;
      for (size_t g = 0; g < groupKeys.size(); ++g) {
        if (groupKeys[g] == key) {
          groups[g].push_back(i);
          found = true;
          break;
        }
      }
      if (!found) {
        groupKeys.push_back(key);
        groups.push_back({i});
      }
    }

    llvm::errs() << "[Loopify] totalChains=" << allChains.size()
                 << " groups=" << groups.size() << "\n";
    for (size_t g = 0; g < groups.size(); ++g)
      llvm::errs() << "  group[" << g << "]: sliceDim=" << groupKeys[g].first
                   << " numIters=" << groupKeys[g].second
                   << " chains=" << groups[g].size() << "\n";
    // Step 3: convert each group.
    for (auto &indices : groups) {
      SmallVector<UnrolledChain> groupChains;
      for (size_t idx : indices) {
        if (allChains[idx].elements[0].insertOp->getBlock())
          groupChains.push_back(allChains[idx]);
      }
      if (groupChains.empty()) continue;

      rewriter.setInsertionPoint(groupChains[0].elements[0].insertOp);
      if (useForall) {
        // GPU-optimal mode: scf.forall(cells) { scf.for(levels) { scalar } }
        // wrapped in flow.dispatch.region with workgroup mapping.
        if (failed(convertChainGroupToForallDispatch(groupChains, rewriter))) {
          llvm::errs() << "[Loopify] forall dispatch failed, falling back "
                          "to scf.for\n";
          if (failed(convertChainGroupToLoop(groupChains, rewriter))) {
            for (auto &chain : groupChains) {
              if (!chain.elements[0].insertOp->getBlock()) continue;
              rewriter.setInsertionPoint(chain.elements[0].insertOp);
              (void)convertChainToLoop(chain, rewriter);
            }
          }
        }
      } else {
        // Original mode: scf.for(levels) { tensor<1xN> compute }
        if (failed(convertChainGroupToLoop(groupChains, rewriter))) {
          // Fall back: convert each chain individually.
          for (auto &chain : groupChains) {
            if (!chain.elements[0].insertOp->getBlock()) continue;
            rewriter.setInsertionPoint(chain.elements[0].insertOp);
            (void)convertChainToLoop(chain, rewriter);
          }
        }
      }
    }
    if (useForall) {
      llvm::errs() << "[Loopify] verifying IR after dispatch region conversion...\n";
      if (failed(mlir::verify(funcOp))) {
        llvm::errs() << "[Loopify] IR INVALID after dispatch region conversion!\n";
        signalPassFailure();
        return;
      }
      llvm::errs() << "[Loopify] IR valid\n";
    }
    llvm::errs() << "[Loopify] runOnOperation complete\n";
  }
};

} // namespace

} // namespace mlir::iree_compiler::Preprocessing
