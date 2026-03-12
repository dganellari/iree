// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GPULoopifyUnrolledSliceChain: detect unrolled insert_slice chains inside
// GPU dispatch functions and convert them to scf.forall + scf.for loops
// suitable for GPU codegen.
//
// This pass runs at the *codegen* level — inside a dispatch function —
// unlike the preprocessing-level LoopifyInsertSliceChain pass. Operating at
// codegen avoids fighting with IREE's dispatch creation, tiling, and
// distribution passes that can split or restructure loops created too early.
//
// Pattern detected:
//   %a0 = tensor.insert_slice %out0 into %empty [0,0][1,N][1,1]
//   %a1 = tensor.insert_slice %out1 into %a0    [1,0][1,N][1,1]
//   ...
//   %result = tensor.insert_slice %out89 into %a88 [89,0][1,N][1,1]
//
// Where each %outK comes from:
//   tensor.extract_slice %src [K,0][1,N][1,1]
//   ... (computation, possibly carrying values from iteration K-1)
//
// Generated structure:
//   scf.forall (%cell) in (numCells) shared_outs(%acc = %empty) {
//     scf.for %k = 0 to numLevels step 1
//         iter_args(%col = %col_init, %carry0 = init0, ...) {
//       %val = tensor.extract %src[%k, %cell]
//       <scalarized body>
//       %new_col = tensor.insert %out into %col[%k, 0]
//       scf.yield %new_col, %carry_out, ...
//     }
//     scf.forall.in_parallel {
//       tensor.parallel_insert_slice %col into %acc [0, %cell][L, 1]
//     }
//   }

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOOPIFYUNROLLEDSLICECHAINPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the constant integer value of an index-type Value, or -1.
static int64_t getConstantIndex(Value v) {
  if (!v)
    return -1;
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  IntegerAttr attr;
  if (matchPattern(v, m_Constant(&attr)))
    return attr.getInt();
  return -1;
}

/// Return the static offset of an insert_slice along `dim`, or -1.
static int64_t getInsertSliceOffset(tensor::InsertSliceOp op, int64_t dim) {
  auto offsets = op.getMixedOffsets();
  if (dim >= (int64_t)offsets.size())
    return -1;
  auto &ofr = offsets[dim];
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt();
    return -1;
  }
  return getConstantIndex(ofr.dyn_cast<Value>());
}

/// Return the static offset of an extract_slice along `dim`, or -1.
static int64_t getExtractSliceOffset(tensor::ExtractSliceOp op, int64_t dim) {
  auto offsets = op.getMixedOffsets();
  if (dim >= (int64_t)offsets.size())
    return -1;
  auto &ofr = offsets[dim];
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt();
    return -1;
  }
  return getConstantIndex(ofr.dyn_cast<Value>());
}

/// Return the static size of a mixed-offset-size entry, or -1.
static int64_t getMixedSize(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt();
    return -1;
  }
  return getConstantIndex(ofr.dyn_cast<Value>());
}

//===----------------------------------------------------------------------===//
// Chain analysis
//===----------------------------------------------------------------------===//

struct ChainElement {
  tensor::InsertSliceOp insertOp;
  int64_t iterIndex;
  Value sourceValue; // What gets inserted (%outK)
};

struct UnrolledChain {
  SmallVector<ChainElement> elements;
  int64_t sliceDim;
  int64_t numIters;
  Value initialDest;
  RankedTensorType accType;
};

/// Build an unrolled chain starting from `first` (whose dest is tensor.empty).
static bool buildChain(tensor::InsertSliceOp first, int64_t sliceDim,
                       int64_t minLen, UnrolledChain &chain) {
  int64_t off0 = getInsertSliceOffset(first, sliceDim);
  if (off0 < 0)
    return false;

  auto sliceSizes = first.getMixedSizes();
  if (sliceDim >= (int64_t)sliceSizes.size())
    return false;
  if (getMixedSize(sliceSizes[sliceDim]) != 1)
    return false;

  SmallVector<ChainElement> elems;
  elems.push_back({first, off0, first.getSource()});

  tensor::InsertSliceOp cur = first;
  while (true) {
    tensor::InsertSliceOp next;
    int numInsertUses = 0;
    for (auto *user : cur->getUsers()) {
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user)) {
        if (ins.getDest() == cur.getResult()) {
          next = ins;
          ++numInsertUses;
        }
      }
    }
    if (numInsertUses != 1 || !next)
      break;

    int64_t off = getInsertSliceOffset(next, sliceDim);
    if (off < 0)
      break;
    if (off != elems.back().iterIndex + 1)
      break;

    auto nextSizes = next.getMixedSizes();
    if (sliceDim < (int64_t)nextSizes.size()) {
      if (getMixedSize(nextSizes[sliceDim]) != 1)
        break;
    }

    elems.push_back({next, off, next.getSource()});
    cur = next;
  }

  if ((int64_t)elems.size() < minLen)
    return false;
  if (elems[0].iterIndex != 0)
    return false;
  for (size_t i = 1; i < elems.size(); ++i) {
    if (elems[i].iterIndex != (int64_t)i)
      return false;
  }

  auto resultType = dyn_cast<RankedTensorType>(cur.getResult().getType());
  if (!resultType)
    return false;

  chain.elements = std::move(elems);
  chain.sliceDim = sliceDim;
  chain.numIters = (int64_t)chain.elements.size();
  chain.initialDest = first.getDest();
  chain.accType = resultType;
  return true;
}

//===----------------------------------------------------------------------===//
// Body & carry analysis
//===----------------------------------------------------------------------===//

/// Collect all ops transitively producing `sourceValue`, stopping at boundary.
static void collectBodyOps(Value sourceValue, const DenseSet<Value> &boundary,
                           SetVector<Operation *> &bodyOps,
                           SetVector<Value> &bodyValues) {
  SmallVector<Value> worklist = {sourceValue};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (bodyValues.contains(v))
      continue;
    if (boundary.count(v))
      continue;
    if (isa<BlockArgument>(v))
      continue;

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;

    bodyValues.insert(v);
    if (!bodyOps.contains(defOp)) {
      bodyOps.insert(defOp);
      for (Value result : defOp->getResults())
        bodyValues.insert(result);
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
  }
}

/// Detect inter-iteration carry values by comparing iter0 and iter1 bodies.
static void detectCarry(const UnrolledChain &chain,
                        const DenseSet<Value> &boundary,
                        SmallVector<Value> &carryOutputs,
                        SmallVector<Value> &carryInitialValues) {
  if (chain.numIters < 2)
    return;

  // Collect iter0 body.
  SetVector<Operation *> iter0Ops;
  SetVector<Value> iter0Values;
  collectBodyOps(chain.elements[0].sourceValue, boundary, iter0Ops,
                 iter0Values);

  // Collect iter1 body, stopping at iter0's values.
  DenseSet<Value> iter1Boundary = boundary;
  iter1Boundary.insert(chain.elements[0].sourceValue);
  if (Operation *defOp = chain.elements[0].sourceValue.getDefiningOp()) {
    for (Value res : defOp->getResults())
      iter1Boundary.insert(res);
  }
  for (Value v : iter0Values)
    iter1Boundary.insert(v);

  SetVector<Operation *> iter1OpsOnly;
  SetVector<Value> iter1ValuesOnly;
  collectBodyOps(chain.elements[1].sourceValue, iter1Boundary, iter1OpsOnly,
                 iter1ValuesOnly);

  // Values from iter0 used by iter1-exclusive ops are carry values.
  DenseSet<Value> carrySet;
  for (Operation *op : iter1OpsOnly) {
    for (Value operand : op->getOperands()) {
      if (iter0Values.contains(operand) && !carrySet.count(operand)) {
        carrySet.insert(operand);
        carryOutputs.push_back(operand);
      }
    }
  }

  // Find initial values for each carry (what iter0 uses in place of carry).
  for (Value carryOut : carryOutputs) {
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
      if (!usesCarry)
        continue;

      for (Operation *iter0Op : iter0Ops) {
        if (iter0Op->getName() != iter1Op->getName())
          continue;
        if (iter0Op->getNumOperands() != iter1Op->getNumOperands())
          continue;
        if (carryOperandIdx < iter0Op->getNumOperands()) {
          Value initCarry = iter0Op->getOperand(carryOperandIdx);
          if (!iter0Values.contains(initCarry) &&
              initCarry.getType() == carryOut.getType()) {
            carryInitialValues.push_back(initCarry);
            goto found_init;
          }
        }
      }
      {
        carryInitialValues.push_back(nullptr);
        goto found_init;
      }
    found_init:
      break;
    }
  }

  while (carryInitialValues.size() < carryOutputs.size())
    carryInitialValues.push_back(nullptr);
}

//===----------------------------------------------------------------------===//
// Scalarize linalg.generic helper
//===----------------------------------------------------------------------===//

static void scalarizeLinalgGenericOp(linalg::GenericOp genOp,
                                     IRMapping &mapping,
                                     ImplicitLocOpBuilder &b) {
  for (auto it : genOp.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel)
      return;

  Block &body = genOp.getRegion().front();
  int numInputs = genOp.getNumDpsInputs();
  int numOutputs = genOp.getNumDpsInits();

  IRMapping bodyMapping;
  for (int i = 0; i < numInputs; ++i) {
    Value inp = genOp.getDpsInputs()[i];
    bodyMapping.map(body.getArgument(i), mapping.lookupOrDefault(inp));
  }
  for (int i = 0; i < numOutputs; ++i) {
    Type outElemTy = body.getArgument(numInputs + i).getType();
    bodyMapping.map(body.getArgument(numInputs + i),
                    b.create<arith::ConstantOp>(b.getZeroAttr(outElemTy)));
  }

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

  for (int i = 0; i < numOutputs && i < (int)yieldVals.size(); ++i)
    mapping.map(genOp.getResult(i), yieldVals[i]);
}

//===----------------------------------------------------------------------===//
// Core transformation: chain group → scf.forall + scf.for
//===----------------------------------------------------------------------===//

static LogicalResult
convertChainGroupToForallFor(SmallVectorImpl<UnrolledChain> &chains,
                             IRRewriter &rewriter) {
  assert(!chains.empty());
  int64_t sliceDim = chains[0].sliceDim;
  int64_t numIters = chains[0].numIters;
  Location loc = chains[0].elements[0].insertOp.getLoc();
  llvm::errs() << "[GPULoopify] chains=" << chains.size()
               << " sliceDim=" << sliceDim << " numIters=" << numIters << "\n";

  // Boundary values: block arguments of the enclosing function.
  DenseSet<Value> boundary;
  Block *block = chains[0].elements[0].insertOp->getBlock();
  for (Value arg : block->getArguments())
    boundary.insert(arg);

  // Union of all chains' iter0 bodies.
  SetVector<Operation *> unionOps;
  SetVector<Value> unionValues;
  for (auto &chain : chains)
    collectBodyOps(chain.elements[0].sourceValue, boundary, unionOps,
                   unionValues);

  // Find level-slices: extract_slice at offset 0 along sliceDim, size 1.
  SmallVector<std::pair<tensor::ExtractSliceOp, Value>> levelSlices;
  for (Operation *op : unionOps) {
    auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extSlice)
      continue;
    if (getExtractSliceOffset(extSlice, sliceDim) != 0)
      continue;
    auto sizes = extSlice.getMixedSizes();
    if (sliceDim >= (int64_t)sizes.size())
      continue;
    if (getMixedSize(sizes[sliceDim]) != 1)
      continue;
    levelSlices.push_back({extSlice, extSlice.getSource()});
  }
  if (levelSlices.empty()) {
    llvm::errs() << "[GPULoopify] FAIL: no level slices\n";
    return failure();
  }

  // Determine cell dimension and numCells.
  auto srcType = dyn_cast<RankedTensorType>(levelSlices[0].second.getType());
  if (!srcType)
    return failure();

  int64_t cellDim = -1, numCells = -1;
  for (int64_t d = 0; d < srcType.getRank(); ++d) {
    if (d == sliceDim)
      continue;
    if (!srcType.isDynamicDim(d) && srcType.getDimSize(d) > 0) {
      cellDim = d;
      numCells = srcType.getDimSize(d);
      break;
    }
  }
  if (numCells <= 0 || cellDim < 0) {
    llvm::errs() << "[GPULoopify] FAIL: cannot determine cell dimension\n";
    return failure();
  }
  llvm::errs() << "[GPULoopify] numCells=" << numCells
               << " cellDim=" << cellDim << "\n";

  // Detect carry using chain with largest body.
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
  llvm::errs() << "[GPULoopify] carryOutputs=" << nCarry << "\n";

  // Column tensor shape: [numIters, 1] for sliceDim=0, cellDim=1.
  SmallVector<int64_t> colShape(srcType.getRank());
  for (int64_t d = 0; d < srcType.getRank(); ++d)
    colShape[d] = (d == sliceDim) ? numIters : 1;

  // Build scf.forall over cells.
  rewriter.setInsertionPoint(chains[0].elements[0].insertOp);
  ImplicitLocOpBuilder b(loc, rewriter);

  SmallVector<Value> sharedInits;
  for (auto &chain : chains)
    sharedInits.push_back(chain.initialDest);

  auto forallOp = scf::ForallOp::create(
      b, loc, ArrayRef<OpFoldResult>{b.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(numCells)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)}, ValueRange(sharedInits),
      /*mapping=*/std::nullopt);

  Value cellId = forallOp.getInductionVars()[0];
  auto sharedOuts = forallOp.getRegionIterArgs();

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
          tensor::EmptyOp::create(rewriter, loc, colMixedSizes, chainElemType));
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

  // scf.for over levels [0, numIters).
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
            if (d == sliceDim)
              extractIdx.push_back(k);
            else if (d == cellDim)
              extractIdx.push_back(cellId);
            else
              extractIdx.push_back(zero);
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
          // Skip if all results already mapped.
          if (llvm::all_of(op->getResults(), [&](Value v) {
                return mapping.lookupOrDefault(v) != v;
              })) {
            continue;
          }

          // For non-special ops with tensor operands, extract scalars.
          if (!isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
                   tensor::ExtractOp, linalg::LinalgOp, tensor::EmptyOp,
                   arith::ConstantOp>(op)) {
            for (Value operand : op->getOperands()) {
              Value mapped = mapping.lookupOrDefault(operand);
              if (!isa<RankedTensorType>(mapped.getType()))
                continue;
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

  // Populate scf.forall.in_parallel terminator.
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

  llvm::errs() << "[GPULoopify] SUCCESS: created forall(%c in [0,"
               << numCells << "), %k in [0," << numIters << ")) for "
               << chains.size() << " chains\n";

  // Replace each chain's final insert_slice with forall result.
  for (size_t i = 0; i < chains.size(); ++i) {
    rewriter.replaceOp(chains[i].elements.back().insertOp,
                       forallOp.getResult(i));
    for (int64_t j = (int64_t)chains[i].elements.size() - 2; j >= 0; --j)
      rewriter.eraseOp(chains[i].elements[j].insertOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Fallback: chain group → plain scf.for (no forall)
//===----------------------------------------------------------------------===//

static LogicalResult
convertChainGroupToLoop(SmallVectorImpl<UnrolledChain> &chains,
                        IRRewriter &rewriter) {
  assert(!chains.empty());
  int64_t sliceDim = chains[0].sliceDim;
  int64_t numIters = chains[0].numIters;
  Location loc = chains[0].elements[0].insertOp.getLoc();

  DenseSet<Value> boundary;
  Block *block = chains[0].elements[0].insertOp->getBlock();
  for (Value arg : block->getArguments())
    boundary.insert(arg);

  SetVector<Operation *> unionOps;
  SetVector<Value> unionValues;
  for (auto &chain : chains)
    collectBodyOps(chain.elements[0].sourceValue, boundary, unionOps,
                   unionValues);

  // Detect carry from largest-body chain.
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

  // Find dynamic slices.
  SmallVector<std::pair<tensor::ExtractSliceOp, Value>> dynamicSlices;
  for (Operation *op : unionOps) {
    auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extSlice)
      continue;
    if (getExtractSliceOffset(extSlice, sliceDim) != 0)
      continue;
    auto sizes = extSlice.getMixedSizes();
    if (sliceDim >= (int64_t)sizes.size())
      continue;
    if (getMixedSize(sizes[sliceDim]) != 1)
      continue;
    dynamicSlices.push_back({extSlice, extSlice.getSource()});
  }
  if (dynamicSlices.empty())
    return failure();

  size_t nAccs = chains.size();
  ImplicitLocOpBuilder b(loc, rewriter);
  b.setInsertionPoint(chains[0].elements[0].insertOp);

  Value lb = b.create<arith::ConstantIndexOp>(0);
  Value ub = b.create<arith::ConstantIndexOp>(numIters);
  Value step = b.create<arith::ConstantIndexOp>(1);

  SmallVector<Value> initArgs;
  for (auto &chain : chains)
    initArgs.push_back(chain.initialDest);
  for (size_t i = 0; i < carryOutputs.size(); ++i) {
    Value init = carryInitials[i];
    if (!init) {
      auto ty = dyn_cast<RankedTensorType>(carryOutputs[i].getType());
      if (!ty)
        return failure();
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

        for (auto [carryInit, iterArg] :
             llvm::zip(carryInitials, iterArgs.drop_front(nAccs))) {
          if (carryInit)
            mapping.map(carryInit, iterArg);
        }

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

        SmallVector<Value> yieldVals;
        for (size_t i = 0; i < chains.size(); ++i) {
          Value outK =
              mapping.lookupOrDefault(chains[i].elements[0].sourceValue);
          Value acc = iterArgs[i];
          auto &elem0 = chains[i].elements[0];
          SmallVector<OpFoldResult> dynOffsets;
          for (size_t d = 0;
               d < elem0.insertOp.getMixedOffsets().size(); ++d) {
            dynOffsets.push_back(
                (int64_t)d == sliceDim
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

  for (size_t i = 0; i < chains.size(); ++i) {
    rewriter.replaceOp(chains[i].elements.back().insertOp,
                       forOp.getResult(i));
    for (int64_t j = (int64_t)chains[i].elements.size() - 2; j >= 0; --j)
      rewriter.eraseOp(chains[i].elements[j].insertOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct GPULoopifyUnrolledSliceChainPass
    : public impl::GPULoopifyUnrolledSliceChainPassBase<
          GPULoopifyUnrolledSliceChainPass> {
  using impl::GPULoopifyUnrolledSliceChainPassBase<
      GPULoopifyUnrolledSliceChainPass>::
      GPULoopifyUnrolledSliceChainPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Step 1: collect all chains from tensor.empty heads.
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
      llvm::errs() << "[GPULoopify] no chains found\n";
      return;
    }

    for (size_t i = 0; i < allChains.size(); ++i)
      llvm::errs() << "  chain[" << i
                   << "]: sliceDim=" << allChains[i].sliceDim
                   << " numIters=" << allChains[i].numIters << "\n";

    // Step 2: group chains by (sliceDim, numIters).
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

    llvm::errs() << "[GPULoopify] totalChains=" << allChains.size()
                 << " groups=" << groups.size() << "\n";

    // Step 3: convert each group.
    for (auto &indices : groups) {
      SmallVector<UnrolledChain> groupChains;
      for (size_t idx : indices) {
        if (allChains[idx].elements[0].insertOp->getBlock())
          groupChains.push_back(allChains[idx]);
      }
      if (groupChains.empty())
        continue;

      rewriter.setInsertionPoint(groupChains[0].elements[0].insertOp);
      if (useForall) {
        // GPU-optimal: scf.forall(cells) { scf.for(levels) { scalar } }
        if (failed(convertChainGroupToForallFor(groupChains, rewriter))) {
          llvm::errs()
              << "[GPULoopify] forall failed, falling back to scf.for\n";
          if (failed(convertChainGroupToLoop(groupChains, rewriter))) {
            for (auto &chain : groupChains) {
              if (!chain.elements[0].insertOp->getBlock())
                continue;
              // Individual chain conversion not implemented at codegen;
              // skip.
            }
          }
        }
      } else {
        if (failed(convertChainGroupToLoop(groupChains, rewriter))) {
          llvm::errs() << "[GPULoopify] loop conversion failed\n";
        }
      }
    }

    if (failed(mlir::verify(funcOp))) {
      llvm::errs() << "[GPULoopify] IR INVALID after conversion!\n";
      signalPassFailure();
      return;
    }
    llvm::errs() << "[GPULoopify] runOnOperation complete\n";
  }
};

} // namespace

} // namespace mlir::iree_compiler
