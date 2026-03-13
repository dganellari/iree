// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PREPROCESSING_COMMON_PASSES
#define IREE_PREPROCESSING_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def ApplyPDLPatternsPass : Pass<"iree-preprocessing-apply-pdl-patterns", "ModuleOp"> {
  let summary = "Parse an input file containing PDL patterns and apply them as patterns";
  let options = [
    Option<"patternsFile", "patterns-file", "std::string",
      /*default=*/"", "File path to file containing PDL patterns to use.">,
  ];
  let dependentDialects = [
    "iree_compiler::IREE::Codegen::IREECodegenDialect",
    "iree_compiler::IREE::Flow::FlowDialect",
    "iree_compiler::IREE::Stream::StreamDialect",
    "iree_compiler::IREE::Util::UtilDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect, pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def AttrBasedPipelinePass : Pass<"iree-preprocessing-attr-based-pipeline", ""> {
  let summary = "Run a pass pipeline specified as an attribute on any function-like operation";
  let description = [{
    The textual representation of the pass pipeline specified in the attribute
    `preprocessing_pipeline=#util.preprocessing_pipeline<...>` is to be run on the
    function-like operation.
  }];
  let dependentDialects = [
    "iree_compiler::IREE::Flow::FlowDialect",
    "iree_compiler::IREE::Util::UtilDialect",
  ];
}

def Convert1X1FilterConv2DToMatmulPass:
    Pass<"iree-preprocessing-convert-1x1-filter-conv2d-to-matmul", ""> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def ConvertConvFilterToChannelsLastPass:
    Pass<"iree-preprocessing-convert-conv-filter-to-channels-last", ""> {
  let summary = "Convert linalg convolutions filter from hwcf to channel last layout.";
  let options = [
    Option<"filterLayout", "filter-layout", "std::string",
      /*default=*/"\"fhwc\"", "Filter layout of convolution.">,
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "IREE::TensorExt::IREETensorExtDialect",
  ];
}

def ConvertConvToChannelsLastPass :
    Pass<"iree-preprocessing-convert-conv-to-channels-last", ""> {
  let summary = "Convert linalg convolutions to channels last.";
  let options = [
    Option<"tilingFactor", "tiling-factor", "int",
           /*default=*/"0",
           "Tiling factor for the channel dimension of NCHW-like convolutions. "
           "Defaults to fully transposing all channel-like dimensions">,
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def FoldAttentionWithTransposePass :
    Pass<"iree-preprocessing-fold-attention-with-transpose", ""> {
  let summary = "Fold attention operation with transpose";
}

def InterpreterPass : Pass<"iree-preprocessing-transform-interpreter"> {
  let summary = "transform dialect interpreter";
  let description = [{
    This pass runs the transform dialect interpreter and applies the named
    sequence transformation named `__transform_main`.

    TODO: Drop this pass in favor of the one upstream. The one upstream requires
    separate loading of the module and thus isn't suited for single-use
    transform scripts.
  }];
  let dependentDialects = ["::mlir::transform::TransformDialect"];
  let options = [
    Option<"disableExpensiveChecks", "disable-expensive-checks", "bool",
           "false",
           "Disable expensive checks in the interpreter for a faster run.">,
    Option<"transformSpecPath", "transform-spec-path", "std::string",
           /*default=*/"", "File path to the transform spec to use.">,
  ];
}

def MakeSingleDispatchForFunctionPass :
    InterfacePass<"iree-preprocessing-make-single-dispatch-for-function", "mlir::FunctionOpInterface"> {
  let summary = "Convert entire function into a single dispatch";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
  ];
}

def PadToIntrinsicsPass :
    Pass<"iree-preprocessing-pad-to-intrinsics", "ModuleOp"> {
  let summary = "Pad linalg ops such that we can use target's intrinsics.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
  let options = [
      Option<"padTargetType", "pad-target-type", "Preprocessing::PadTargetType",
             /*default=*/"Preprocessing::PadTargetType::All",
             "Pad target-op type that is being specified.",
             [{::llvm::cl::values(
               clEnumValN(Preprocessing::PadTargetType::ConvOp, "conv",
                "Pad convolution ops to intrinsics."),
               clEnumValN(Preprocessing::PadTargetType::ContractionOp, "contraction",
                "Pad contraction ops to intrinsics."),
               clEnumValN(Preprocessing::PadTargetType::All, "all",
                "Pad all ops to intrinsics.")
              )}]>,
  ];
}

def PadLinalgOpsPass :
    Pass<"iree-preprocessing-pad-linalg-ops", ""> {
  let summary = "Pad linalg ops to the next integer multiple of paddingSize.";
  let options = [
    Option<"paddingSize", "pad-size", "int",
           /*default=*/"4",
           "Specify the padding size">,
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def TransposeMatmulPass : Pass<"iree-preprocessing-transpose-matmul-pass"> {
  let summary = "Convert Linalg matmul ops to transposed variants";
  let options = [
    Option<"input", "input", "Preprocessing::TransposeMatmulInput",
           /*default=*/"Preprocessing::TransposeMatmulInput::Lhs",
           "Input to transpose, one of: 'lhs' (default), 'rhs'",
           [{llvm::cl::values(
              clEnumValN(Preprocessing::TransposeMatmulInput::Lhs, "lhs",
                         "Transpose LHS input matrix."),
              clEnumValN(Preprocessing::TransposeMatmulInput::Rhs, "rhs",
                         "Transpose RHS input matrix.")
           )}]>
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def GeneralizeLinalgMatMulPass :
    InterfacePass<"iree-preprocessing-generalize-linalg-matmul-experimental", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg matmul ops to linalg.generics.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def LoopifyInsertSliceChainPass :
    InterfacePass<"iree-preprocessing-loopify-insert-slice-chain", "mlir::FunctionOpInterface"> {
  let summary = "Convert unrolled tensor.insert_slice chains to scf.for or scf.forall loops";
  let description = [{
    Detects chains of tensor.insert_slice ops with consecutive constant offsets
    (produced when IREE lowers stablehlo.concatenate of N operands), and
    converts them to loops.

    Default mode (use-forall=false): creates scf.for over the slice dimension
    (levels), keeping tensor<1xN> slice computations per iteration.

    Forall mode (use-forall=true): creates scf.forall over the cell dimension
    (N cells in parallel) with scf.for over levels inside. Each cell-thread
    carries scalar values in registers and writes a column (numLevels x 1) into
    the shared output via tensor.parallel_insert_slice. This matches the
    GPU-optimal dispatch(N_cells){ for(level) { scalar_compute } } structure.

    Example (forall mode):
      %result = scf.forall (%c = 0 to N) shared_outs(%acc = %empty) {
        %col = scf.for %k = 0 to 90 iter_args(%col_acc=%empty_col, %carry=%init) {
          %val = tensor.extract %src[%k, %c]
          %out = <scalar body using %val and %carry>
          %new_col = tensor.insert %out into %col_acc[%k, 0]
          scf.yield %new_col, %out
        }
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %col into %acc[0, %c][90, 1][1, 1]
        }
      }
  }];
  let options = [
    Option<"minChainLength", "min-chain-length", "int64_t",
           /*default=*/"8",
           "Minimum chain length to trigger loopification">,
    Option<"useForall", "use-forall", "bool",
           /*default=*/"false",
           "If true, generate scf.forall over cells with scf.for over levels "
           "(GPU-optimal structure) instead of scf.for over levels">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::iree_compiler::IREE::Flow::FlowDialect",
    "mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "IREE::TensorExt::IREETensorExtDialect",
  ];
}

#endif  // IREE_PREPROCESSING_COMMON_PASSES
