"""Static RV1106 ONNX compatibility checks.

This is a preflight checker, not a replacement for RKNN-Toolkit2.  The
operator names and broad support levels follow Rockchip's published RKNN
Toolkit2 operator table; target-specific shape/attribute checks cover the
restrictions that are useful to catch before invoking the proprietary
compiler.  RKNN remains authoritative for edge cases and fusion decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import onnx


# RKNN Toolkit2 2.3.2's public ONNX table marks these as unsupported.  Keep
# this explicit so a new ONNX operator is reported as unknown instead of being
# silently assumed to be executable.
UNSUPPORTED_OPS = frozenset({
    "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Bernoulli",
    "BitShift", "BitwiseAnd", "BitwiseNot", "BitwiseOr", "BitwiseXor",
    "BlackmanWindow", "CastLike", "Ceil", "Celu", "CenterCropPad", "Col2Im",
    "Compress", "ConcatFromSequence", "ConvInteger", "Cosh", "CumSum",
    "DeformConv", "Det", "DFT", "DynamicQuantizeLinear", "Einsum", "GlobalLpPool",
    "GridSample", "GroupNormalization", "HammingWindow", "HannWindow", "Hardmax",
    "IsInf", "IsNaN", "Loop", "LpPool", "MatMulInteger", "Mean", "MelWeightMatrix",
    "Multinomial", "Neg", "NegativeLogLikelihoodLoss", "NonMaxSuppression",
    "NonZero", "Not", "OneHot", "Optional", "OptionalGetElement", "OptionalHasElement",
    "Or", "QLinearConv", "QLinearMatMul", "RandomNormal", "RandomNormalLike",
    "RandomUniform", "RandomUniformLike", "Range", "ReduceL1", "ReduceL2",
    "ReduceLogSum", "ReduceLogSumExp", "ReduceProd", "ReduceSumSquare", "RNN",
    "Round", "Scan", "ScatterElements", "Selu", "SequenceAt", "SequenceConstruct",
    "SequenceEmpty", "SequenceErase", "SequenceInsert", "SequenceLength",
    "SequenceMap", "Shrink", "SoftmaxCrossEntropyLoss", "Softsign", "SplitToSequence",
    "STFT", "StringNormalizer", "Tan", "TfIdfVectorizer", "ThresholdedRelu", "Tile",
    "TopK", "Trilu", "Unique", "Xor",
})


# Operators with a public entry but restrictions that matter on RV1106.
PARTIAL_OPS = frozenset({
    "Div", "EyeLike", "GRU", "If", "LSTM", "LogSoftmax", "Resize", "RoiAlign",
    "Slice", "Softmax", "Concat",
})


@dataclass
class Finding:
    level: str
    node: str
    op_type: str
    message: str

    def json(self) -> Dict[str, str]:
        return asdict(self)


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def _node_name(node: onnx.NodeProto, index: int) -> str:
    return node.name or f"{node.op_type}[{index}]"


def _shape_map(model: onnx.ModelProto) -> Dict[str, List[Optional[int]]]:
    result: Dict[str, List[Optional[int]]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dims: List[Optional[int]] = []
        for dim in tensor.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else None)
        result[value.name] = dims
    return result


def check_rv1106(model: onnx.ModelProto) -> List[Finding]:
    """Return compatibility findings ordered by graph order.

    ``error`` means RKNN's published table says the op is unsupported or the
    graph violates a high-confidence RV1106 restriction. ``warning`` means a
    supported/partial op has target-specific constraints that this static
    checker cannot prove completely. ``unknown`` means the op is absent from
    the published table and needs a real RKNN build probe.
    """
    shapes = _shape_map(model)
    findings: List[Finding] = []
    known = UNSUPPORTED_OPS | PARTIAL_OPS | {
        "Add", "ArgMax", "ArgMin", "AveragePool", "BatchNormalization", "Cast", "Clip",
        "Concat", "Constant", "ConstantOfShape", "Conv", "ConvTranspose", "Cos",
        "DepthToSpace", "DequantizeLinear", "Div", "Dropout", "Elu", "Equal", "Erf",
        "Exp", "Expand", "Flatten", "Floor", "Gather", "GatherElements", "Gemm",
        "GlobalAveragePool", "GlobalMaxPool", "Greater", "GreaterOrEqual", "HardSigmoid",
        "HardSwish", "Identity", "InstanceNormalization", "LayerNormalization",
        "LeakyRelu", "Less", "LessOrEqual", "Log", "LpNormalization", "LRN", "MatMul",
        "Max", "MaxPool", "MaxRoiPool", "MaxUnpool", "Mish", "Min", "Mod", "Mul",
        "Pad", "Pow", "PRelu", "QuantizeLinear", "ReduceMax", "ReduceMean", "ReduceMin",
        "ReduceSum", "Relu", "Reshape", "ReverseSequence", "ScatterND", "Shape",
        "Sigmoid", "Sin", "Size", "Slice", "Softmax", "Softplus", "SpaceToDepth",
        "Split", "Sqrt", "Squeeze", "Sub", "Sum", "Tanh", "Transpose", "Unsqueeze",
        "Where",
    }
    for index, node in enumerate(model.graph.node):
        name = _node_name(node, index)
        op = node.op_type
        if op in UNSUPPORTED_OPS:
            findings.append(Finding("error", name, op, "listed as unsupported by RKNN Toolkit2"))
            continue
        if op not in known:
            findings.append(Finding("unknown", name, op, "not present in the published support table"))
            continue
        if op == "Conv":
            groups = int(_attr(node, "group", 1))
            if groups < 1:
                findings.append(Finding("error", name, op, "group must be positive"))
            if groups > 1:
                findings.append(Finding("warning", name, op,
                                        "grouped/depthwise Conv needs an RKNN build probe"))
        elif op in {"AveragePool", "MaxPool"}:
            strides = _attr(node, "strides", [1, 1])
            if any(int(x) > 8 for x in strides):
                findings.append(Finding("error", name, op, "RV1106 pool stride exceeds published limit 8"))
        elif op == "Resize":
            mode = _attr(node, "mode", "nearest")
            if mode not in {"nearest", "linear"}:
                findings.append(Finding("error", name, op,
                                        f"RV1106 supports nearest/bilinear Resize, got {mode!r}"))
        elif op in {"GRU", "LogSoftmax", "RoiAlign", "Slice", "Softmax"}:
            findings.append(Finding("warning", name, op,
                                    "supported only with target-specific restrictions; verify with RKNN"))
        elif op == "Concat":
            findings.append(Finding("warning", name, op,
                                    "channel alignment/layout constraints require shape-aware RKNN verification"))
        if op in {"Conv", "AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool",
                  "BatchNormalization", "Relu", "LeakyRelu", "PRelu"}:
            for input_name in node.input[:1]:
                shape = shapes.get(input_name)
                if shape and shape[0] not in (None, 1):
                    findings.append(Finding("warning", name, op,
                                            f"RV1106 NPU path commonly requires batch=1; got {shape[0]}"))
    return findings


def legalize_rv1106(model: onnx.ModelProto) -> tuple[onnx.ModelProto, List[Finding]]:
    """Simplify then run the RV1106 compatibility gate.

    Actual decompositions should be added only with numerical tests.  This
    intentionally does not rewrite unsupported operators into guessed forms.
    """
    from onnxsim import simplify

    simplified, ok = simplify(model, check_n=0)
    if not ok:
        raise RuntimeError("onnxsim simplification failed")
    return simplified, check_rv1106(simplified)
