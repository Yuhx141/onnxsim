"""A Quark-ONNX-API-shaped quantization shim backed by onnxsim's own
quantizers -- no ``amd-quark`` install required.

It reproduces the public calling convention of ``quark.onnx`` (names and
signatures read from the ``amd-quark`` 0.13 wheel's ``quark/onnx/__init__.py``
and ``quark/onnx/quantization/config``) so a Quark script can switch to
onnxsim by changing its import:

    # before
    from quark.onnx import ModelQuantizer, QConfig
    # after
    from onnxsim.quark_compat import ModelQuantizer, QConfig

    config = QConfig.get_default_config("A8W8")
    ModelQuantizer(config).quantize_model(
        "model.onnx", "model.quant.onnx", calibration_data_reader
    )

The implementation is independent: Quark's source was read for its public
names and preset *meanings*, not copied.

**Scope, deliberately narrower than Quark:**

- Presets with a real backend here: ``A8W8``, ``A16W8`` (and the
  ``S8S8_AAWS`` / ``U8S8_AAWS`` / ``U8U8_AAWA`` / ``U16S8_AAWS`` /
  ``S16S8_ASWS`` / ``XINT8`` spellings, see ``_PRESETS``), ``FP16``, ``BF16``.
  Integer presets use :func:`onnxsim.full_qdq.quantize_full_qdq`: signed /
  unsigned 8- and 16-bit activations, symmetric (``A8W8``, ``A16W8``,
  ``XINT8``: centred zero point) or asymmetric with percentile calibration
  (``*_AAWS``), power-of-2 scales for ``XINT8``, and symmetric int8 weights
  **per tensor** like Quark (``extra_options={"PerChannel": True}`` for per
  channel; the weight-rounding algorithms need it and switch it on). Scales and
  zero points match Quark's for the probed models
  (``tests/test_quark_parity.py``). Calibration follows the preset: MinMax
  (``A8W8``, ``A16W8``), Percentile (99.999; ``S8S8_AAWS`` 99.9999; the
  ``Int8Spec`` family's default, as in Quark; agrees with Quark to histogram
  binning, ~5e-4) and, for ``XINT8``, Quark's power-of-two MinMSE
  (``method="minmse_pof2"`` in :func:`onnxsim.calibration.calibrate`): the same
  2048-bin histogram and five candidate scales, so activation scales are
  identical to Quark's, and weights and biases get the same MinMSE search --
  biases are **int8** with a per-tensor power-of-two scale like Quark's
  (``extra_options={"Int32Bias": True}`` keeps int32). Like Quark, every
  non-weight constant of a quantized node (LayerNorm scale, Mul operand, ...)
  is quantized as an int8 weight (activation dtype for Add / Sub / Mul / Div /
  Min / Max constants under ``A16W8``'s ``AlignEltwiseQuantType``), and
  Softmax outputs are calibrated to the fixed range (0, 1) except under
  ``XINT8``. Not matched: Quark's ``Entropy`` (a different algorithm than
  :func:`onnxsim.calibration.calibrate`'s ``"entropy"``; inner scales differ by
  up to ~30% on the probed MLP), ``Distribution`` / ``LayerwisePercentile``
  (``CalibMethod`` members that raise), and its non-power-of-two ``MinMSE``
  (Quark's ``CalibMethod.MinMSE`` is the power-of-two search, which is what
  :class:`CalibMethod` maps it to). Quark's NPU graph rewrites for ``XINT8``
  (shift/cut adjustment, ...) did not change any probed scale.
- Per-layer overrides: ``layer_type_config`` then ``specific_layer_config``
  (which wins) retarget the *activation* dtype / symmetry of a layer's inputs
  (``input_tensors``, or the deprecated ``activation``) and outputs
  (``output_tensors``) among int8/uint8/int16/uint16; a ``None`` key in
  ``layer_type_config`` and ``exclude`` keep nodes float. Node names may be
  Quark's ``^...*`` regular expressions (subgraph tuples raise). Weight dtypes
  other than int8 raise; ``bias`` specs are ignored (biases stay int32).
  Scales / zero points match Quark's (``tests/test_quark_parity.py``).
- ``UINT8_DYNAMIC_QUANT``: :mod:`onnxsim.quark_dynamic` emits Quark's /
  ONNX Runtime's dynamic pattern (``DynamicQuantizeLinear`` +
  ``MatMulInteger`` / ``ConvInteger``); no calibration data is needed.
- Block formats (``BFP16``, ``MX4/6/9``, ``MXFP4/6/8``, ``MXINT8``):
  :mod:`onnxsim.quark_fakequant_graph` inserts the same ``com.amd.quark``
  ``BFPQuantizeDequantize`` / ``MXQuantizeDequantize`` nodes Quark's quantizer
  does -- same tensors, names, attributes and block axes for the ops it lists
  (activations, outputs, weights *and* biases) -- so the model runs wherever
  Quark's ONNX custom-op library is registered
  (``quark.onnx.operators.custom_ops.get_library_path()``); **onnxsim cannot
  execute those nodes itself**. ``tests/test_quark_parity.py`` checks this
  against the installed ``amd-quark`` in CI: identical placement on the probed
  ops and bit-identical outputs under ONNX Runtime. Not replicated: Quark's
  model pre-processing (BatchNormalization folding, ``ReduceMean`` ->
  ``GlobalAveragePool``, the implicit CLE below). Options (``QConfig(..., extra_options=...)``):
  ``BlockFormatActivations=False`` quantizes only the constants, offline, so the
  model runs anywhere; ``BlockFormatFoldWeights=True`` folds the constants
  offline (via :mod:`onnxsim.quark_block_formats`) instead of leaving a node on
  them. ``algo_config`` is not applied to block formats.
  Dynamic quantization raises ``NotImplementedError``.
- ``algo_config``: SmoothQuant (``alpha``) and CLE (Conv chains plus Gemm /
  MatMul chains, :mod:`onnxsim.quark_cle`) run on the float model before
  quantization. Note Quark enables CLE implicitly in *every* preset
  (``include_cle=True``) while onnxsim only runs it when ``CLEConfig`` is
  listed, so a preset's weights differ from Quark's wherever a CLE pattern
  exists; AdaQuant (``num_iterations``, ``learning_rate``,
  ``reg_param``) and BiasCorrection run after it, against the float model.
  AdaQuant only reoptimizes MatMul/Gemm layers whose output is not folded
  with a following Relu, and leaves the rest as calibrated.
  AutoMixprecision replaces the plain quantization step with
  :func:`onnxsim.quark_auto_mixprecision.auto_mixprecision`: a single
  ``target_layer_config`` whose *activation* is the other of ``uint8`` /
  ``uint16`` (weights stay int8); the dict/list multi-config forms,
  ``subgraph_json`` and ``sensitivity_cache_file`` raise, and ``dual_quant_nodes``
  / ``no_input_qdq_shared`` / ``shared_param_mode`` / ``worker_num`` have no
  effect (every tensor already has its own Q/DQ pair; analysis is serial).
  AdaRound and GPTQ refine the weight codes layer by layer
  (:mod:`onnxsim.quark_weight_rounding`; Conv / Gemm / MatMul, guarded so a
  layer's reconstruction error never gets worse). AdaRound honours
  ``drop_ratio`` (QDrop-style mixing of quantized and float layer inputs),
  ``selective_update``, ``lr_adjust`` and ``data_size``; ``update_bias`` is
  accepted and ignored, as in Quark (only AdaQuant reads it); ``early_stop`` /
  ``output_qdq`` / ``batch_size`` / ``num_batches`` have no effect. GPTQ with
  ``bits`` / ``group_size`` / ``per_channel`` / ``mse`` / ``weight_symmetric``
  set re-grids the weights the way Quark's GPTQ does (``bits``-bit codes,
  per-tensor / per-channel / per-group scales, scales and zero points written
  back into the QDQ weights; ``group_size`` needs a blocked
  ``DequantizeLinear``, opset >= 21, which ONNX Runtime only runs next to
  activation Q/DQ with ``session.disable_quant_qdq=1``); with none of them set
  the model's own scales are kept. ``act_order`` together with ``group_size``
  raises. Quark 0.13's GPTQ error-propagation step is a no-op (it indexes a
  triangular factor by column), so Quark's result is round-to-nearest on its
  grid; onnxsim really propagates the error (and is never worse). Quarot folds the R1 residual-stream rotation into the float
  weights before quantization (:mod:`onnxsim.quark_quarot`; needs
  ``r_config_path``; R2-R4 do not exist, as in Quark's ONNX flow). An
  ``algo_config`` that cannot run for a preset (block formats, FP16 / BF16)
  raises ``NotImplementedError`` unless ``ignore_unsupported_algos=True``.
- ``extra_options`` are stored, not interpreted -- except ``PerChannel``,
  ``Int32Bias``, ``AlignEltwiseQuantType`` and the block-format options above.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import onnx

# -- data-type specs ---------------------------------------------------------


@dataclass(eq=True, unsafe_hash=True)  # hashable: usable as a dict key, as in Quark
class QSpec:
    """Base of the per-tensor spec classes (Quark's ``Int8Spec`` etc.).

    ``dtype`` is the onnx-style name; ``symmetric`` / ``pof2`` mirror the
    fields Quark's specs expose that change which backend path is valid.
    """

    dtype: str = "int8"
    symmetric: bool = True
    pof2: bool = False
    #: ``"minmax"``, ``"percentile[:p]"``, ``"entropy"``, ``"mse"``,
    #: ``"minmse_pof2"`` (Quark's MinMSE) or a :class:`CalibMethod`
    calibration_method: Any = "minmax"
    is_dynamic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.calibration_method, CalibMethod):
            self.calibration_method = _CALIB_NAMES[self.calibration_method]


class CalibMethod(Enum):
    """Quark's ``CalibMethod`` (``quark.onnx.CalibMethod``). ``MinMSE`` is its
    power-of-two MinMSE search (:mod:`onnxsim.calibration` ``"minmse_pof2"``);
    ``Distribution`` / ``LayerwisePercentile`` are not implemented."""

    MinMax = 0
    MinMSE = 1
    Percentile = 2
    Entropy = 3
    LayerwisePercentile = 4
    Distribution = 5


_CALIB_NAMES = {
    CalibMethod.MinMax: "minmax",
    CalibMethod.MinMSE: "minmse_pof2",
    CalibMethod.Percentile: "percentile:99.999",
    CalibMethod.Entropy: "entropy",
    CalibMethod.LayerwisePercentile: "layerwise_percentile",
    CalibMethod.Distribution: "distribution",
}


def _spec(
    name: str,
    dtype: str,
    symmetric: bool,
    pof2: bool = False,
    calibration_method: str = "minmax",
):
    def __init__(self, **kwargs: Any) -> None:
        fields: Dict[str, Any] = dict(
            dtype=dtype,
            symmetric=symmetric,
            pof2=pof2,
            calibration_method=calibration_method,
        )
        fields.update(kwargs)  # a caller may override e.g. ``symmetric``
        QSpec.__init__(self, **fields)

    return type(name, (QSpec,), {"__init__": __init__, "__doc__": f"{name}."})


# Quark's defaults: the integer specs calibrate with a 99.999 percentile, the
# power-of-2 ones with MinMSE, everything else with MinMax.
_PCT_DEFAULT = "percentile:99.999"
Int8Spec = _spec("Int8Spec", "int8", True, calibration_method=_PCT_DEFAULT)
UInt8Spec = _spec("UInt8Spec", "uint8", False, calibration_method=_PCT_DEFAULT)
Int16Spec = _spec("Int16Spec", "int16", True, calibration_method=_PCT_DEFAULT)
UInt16Spec = _spec("UInt16Spec", "uint16", False, calibration_method=_PCT_DEFAULT)
XInt8Spec = _spec("XInt8Spec", "int8", True, True, "minmse_pof2")
XUInt8Spec = _spec("XUInt8Spec", "uint8", True, True, "minmse_pof2")
Float16Spec = _spec("Float16Spec", "float16", False)
BFloat16Spec = _spec("BFloat16Spec", "bfloat16", False)
BFP16Spec = _spec("BFP16Spec", "bfp16", False)
MX4Spec = _spec("MX4Spec", "mx4", False)
MX6Spec = _spec("MX6Spec", "mx6", False)
MX9Spec = _spec("MX9Spec", "mx9", False)

MXFP4E2M1Spec = _spec("MXFP4E2M1Spec", "mxfp4_e2m1", False)
MXFP6E3M2Spec = _spec("MXFP6E3M2Spec", "mxfp6_e3m2", False)
MXFP6E2M3Spec = _spec("MXFP6E2M3Spec", "mxfp6_e2m3", False)
MXFP8E5M2Spec = _spec("MXFP8E5M2Spec", "mxfp8_e5m2", False)
MXFP8E4M3Spec = _spec("MXFP8E4M3Spec", "mxfp8_e4m3", False)
MXInt8Spec = _spec("MXInt8Spec", "mxint8", False)


def _block_fn(dtype: str) -> Optional[Callable[[np.ndarray, int], np.ndarray]]:
    """Weight fake-quantizer ``f(array, axis)`` for a block-format / half dtype."""
    from onnxsim import quark_block_formats as bf

    if dtype == "bfp16":
        return lambda a, ax: bf.bfp16(a, axis=ax)
    if dtype in ("mx4", "mx6", "mx9"):
        bw = {"mx4": 11, "mx6": 13, "mx9": 16}[dtype]
        return lambda a, ax: bf.bfp_prime(a, bit_width=bw, axis=ax)
    if dtype == "mxint8":
        return lambda a, ax: bf.mx(a, element_dtype="int8", axis=ax)
    if dtype.startswith("mxfp"):
        elem = dtype.replace("mxfp", "fp", 1)  # mxfp8_e4m3 -> fp8_e4m3
        return lambda a, ax: bf.mx(a, element_dtype=elem, axis=ax)
    if dtype == "float16":
        return lambda a, ax: bf.fp16_round(a)
    if dtype == "bfloat16":
        return lambda a, ax: bf.bf16_round(a)
    return None


PROMOTABLE_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")
_BLOCK_DTYPES = {
    "bfp16",
    "mx4",
    "mx6",
    "mx9",
    "mxint8",
    "mxfp4_e2m1",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "mxfp8_e5m2",
    "mxfp8_e4m3",
}
_FAKEQUANT_DTYPES = _BLOCK_DTYPES | {"float16", "bfloat16"}


@dataclass(eq=True, unsafe_hash=True)
class QLayerConfig:
    """Spec of one layer group (Quark's ``QLayerConfig``).

    ``activation`` is Quark's deprecated spelling of ``input_tensors`` (giving
    both raises, as in Quark). A spec left ``None`` means "inherit": the global
    config's, for a global config that is Int8. Positional order is
    ``(activation, weight)`` for backward compatibility.
    """

    activation: Optional[QSpec] = None
    weight: Optional[QSpec] = None
    input_tensors: Optional[QSpec] = None
    bias: Optional[QSpec] = None
    output_tensors: Optional[QSpec] = None

    def __post_init__(self) -> None:
        if self.input_tensors is not None and self.activation is not None:
            raise ValueError(
                "Both `activation` and `input_tensors` are provided. Please just "
                "use `input_tensors`."
            )
        if self.activation is None:
            self.activation = self.input_tensors
        self.input_tensors = self.activation

    def resolved(self) -> "QLayerConfig":
        """The global form: a missing activation / weight spec is Int8."""
        return QLayerConfig(
            activation=self.activation or Int8Spec(),
            weight=self.weight or Int8Spec(),
            bias=self.bias,
            output_tensors=self.output_tensors,
        )


def _activation_inputs(node: onnx.NodeProto, inits: "set[str]") -> List[str]:
    """Inputs before the first constant, which Quark treats as activations."""
    out: List[str] = []
    for x in node.input:
        if not x or x in inits:
            break
        out.append(x)
    return out


def _match_nodes(model: onnx.ModelProto, patterns: List[Any]) -> List[str]:
    """Node names selected by ``patterns``: plain names, and Quark's
    ``^...*``-style regular expressions (must contain ``.*``)."""
    names = [n.name for n in model.graph.node]
    out: List[str] = []
    for p in patterns:
        if isinstance(p, tuple):
            raise NotImplementedError("subgraph patterns are not supported")
        if not isinstance(p, str):
            raise TypeError(f"expected a node name or pattern, got {type(p).__name__}")
        if p.startswith("^"):
            if ".*" not in p:
                raise ValueError(
                    f"invalid pattern {p!r}: patterns start with ^ and contain .*"
                )
            hit = [n for n in names if n and re.search(p, n)]
            if not hit:
                raise ValueError(f"pattern {p!r} matches no node")
            out += hit
        else:
            out.append(p)
    return out


# -- algorithm configs (stored only, see module docstring) --------------------


@dataclass(eq=True)
class AlgoConfig:
    """Base class of the algorithm configs; ``name`` is the Quark algo name."""

    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


def _algo(name: str):
    def __init__(self, **params: Any) -> None:
        AlgoConfig.__init__(self, name=name, params=params)

    return type(
        name.title().replace("_", "") + "Config", (AlgoConfig,), {"__init__": __init__}
    )


SmoothQuantConfig = _algo("smooth_quant")
CLEConfig = _algo("cle")
BiasCorrectionConfig = _algo("bias_correction")
GPTQConfig = _algo("gptq")
AdaRoundConfig = _algo("adaround")
AdaQuantConfig = _algo("adaquant")
QuarotConfig = _algo("quarot")
AutoMixprecisionConfig = _algo("auto_mixprecision")


# -- QConfig and presets -------------------------------------------------------


class QConfig:
    """Mirror of ``quark.onnx.QConfig`` (global spec, per-layer overrides,
    excluded nodes, algorithms, extra options)."""

    def __init__(
        self,
        global_config: QLayerConfig,
        specific_layer_config: Optional[Dict[Any, List[str]]] = None,
        layer_type_config: Optional[Dict[Any, List[str]]] = None,
        exclude: Optional[List[Any]] = None,
        algo_config: Optional[List[AlgoConfig]] = None,
        use_external_data_format: bool = False,
        **extra_options: Any,
    ) -> None:
        self.global_config = global_config
        self.specific_layer_config = specific_layer_config or {}
        self.layer_type_config = layer_type_config or {}
        self.exclude = exclude or []
        self.algo_config = algo_config or []
        self.use_external_data_format = use_external_data_format
        # Quark passes these as ``extra_options={...}``; accept both spellings.
        self.extra_options = dict(
            extra_options.pop("extra_options", {}), **extra_options
        )

    @staticmethod
    def get_default_config(config_name: str) -> "QConfig":
        """Preset by Quark name (``"A8W8"``, ``"XINT8"``, ``"BF16"``, ...)."""
        try:
            return _PRESETS[config_name]()
        except KeyError:
            raise ValueError(
                f"unknown preset {config_name!r}; known: {sorted(_PRESETS)}"
            ) from None


def _layer(act: type, wt: type, **act_kwargs: Any) -> QLayerConfig:
    return QLayerConfig(activation=act(**act_kwargs), weight=wt())


# Quark calibrates the asymmetric ("AA") presets with percentiles (99.999;
# S8S8_AAWS 99.9999), S16S8_ASWS with a symmetric 99.999 percentile.
_PCT = dict(symmetric=False, calibration_method="percentile:99.999")
_PCT4 = dict(symmetric=False, calibration_method="percentile:99.9999")


_PRESETS: Dict[str, Callable[[], QConfig]] = {
    "XINT8": lambda: QConfig(_layer(XUInt8Spec, XInt8Spec)),
    "UINT8_DYNAMIC_QUANT": lambda: QConfig(
        _layer(Int8Spec, UInt8Spec, is_dynamic=True)
    ),
    "A8W8": lambda: QConfig(_layer(Int8Spec, Int8Spec, calibration_method="minmax")),
    "S8S8_AAWS": lambda: QConfig(_layer(Int8Spec, Int8Spec, **_PCT4)),
    "U8S8_AAWS": lambda: QConfig(_layer(UInt8Spec, Int8Spec, **_PCT)),
    "U8U8_AAWA": lambda: QConfig(_layer(UInt8Spec, UInt8Spec, **_PCT)),
    "A16W8": lambda: QConfig(
        _layer(Int16Spec, Int8Spec, calibration_method="minmax"),
        AlignEltwiseQuantType=True,
    ),
    "S16S8_ASWS": lambda: QConfig(_layer(Int16Spec, Int8Spec)),
    "U16S8_AAWS": lambda: QConfig(_layer(UInt16Spec, Int8Spec, **_PCT)),
    "FP16": lambda: QConfig(_layer(Float16Spec, Float16Spec)),
    "BF16": lambda: QConfig(_layer(BFloat16Spec, BFloat16Spec)),
    "BFP16": lambda: QConfig(_layer(BFP16Spec, BFP16Spec)),
    "MX4": lambda: QConfig(_layer(MX4Spec, MX4Spec)),
    "MX6": lambda: QConfig(_layer(MX6Spec, MX6Spec)),
    "MX9": lambda: QConfig(_layer(MX9Spec, MX9Spec)),
    "MXFP4E2M1": lambda: QConfig(_layer(MXFP4E2M1Spec, MXFP4E2M1Spec)),
    "MXFP6E3M2": lambda: QConfig(_layer(MXFP6E3M2Spec, MXFP6E3M2Spec)),
    "MXFP6E2M3": lambda: QConfig(_layer(MXFP6E2M3Spec, MXFP6E2M3Spec)),
    "MXFP8E5M2": lambda: QConfig(_layer(MXFP8E5M2Spec, MXFP8E5M2Spec)),
    "MXFP8E4M3": lambda: QConfig(_layer(MXFP8E4M3Spec, MXFP8E4M3Spec)),
    "MXINT8": lambda: QConfig(_layer(MXInt8Spec, MXInt8Spec)),
}


def _preset_finetune_params(cls: type) -> Dict[str, Any]:
    """The ``FastFinetune`` options Quark's ``*_ADAROUND`` / ``*_ADAQUANT``
    presets carry. Their dict has no ``UpdateBias`` key, so Quark's training
    default (on) applies to AdaQuant, unlike ``AdaQuantConfig()`` (off)."""
    if cls is AdaQuantConfig:
        return dict(_FASTFT_PRESET, learning_rate=1e-5, update_bias=True)
    if cls is AdaRoundConfig:
        return dict(_FASTFT_PRESET, learning_rate=0.1)
    return {}


def _algo_variant(base: str, cls: type) -> Callable[[], QConfig]:
    def make() -> QConfig:
        return _with_algo(_PRESETS[base](), cls(**_preset_finetune_params(cls)))

    return make


for _n in list(_PRESETS):
    _block = _n.startswith(("BFP", "MX"))
    if _n in ("BF16", "UINT8_DYNAMIC_QUANT") or _n.startswith("FP16"):
        continue
    if not _block:
        _PRESETS[f"{_n}_ADAROUND"] = _algo_variant(_n, AdaRoundConfig)
    _PRESETS[f"{_n}_ADAQUANT"] = _algo_variant(_n, AdaQuantConfig)


# Quark's mixed-format presets (no ADAROUND variants -- and, apart from the
# BF16_MIXED_* ones registered below, no ADAQUANT ones -- exist in Quark):
# bfloat16 activations over block-format constants, block-format activations
# over int8 constants.
def _cnn_accurate(act: type, wt: type) -> QConfig:
    return QConfig(
        _layer(act, wt, symmetric=False, calibration_method="percentile:99.9999"),
        algo_config=[AdaRoundConfig(**_preset_finetune_params(AdaRoundConfig))],
    )


def _s16s16_mixed_s8s8() -> QConfig:
    """int16 activations / weights (asymmetric, percentile 99.9999), every
    Conv / Gemm / MatMul promoted to int8 (Quark: AutoMixprecision, threshold
    disabled)."""
    return QConfig(
        _layer(
            Int16Spec,
            Int16Spec,
            symmetric=False,
            calibration_method="percentile:99.9999",
        ),
        algo_config=[
            AutoMixprecisionConfig(
                target_layer_config=QLayerConfig(
                    activation=Int8Spec(symmetric=False), weight=Int8Spec()
                ),
                metric_threshold=0,
            )
        ],
    )


def _mixed_block(block: Callable[[], QSpec], *algos: AlgoConfig) -> QConfig:
    """bfloat16 everywhere, every Conv / Gemm / MatMul promoted to ``block``
    (Quark: AutoMixprecision with the metric threshold disabled, dual nodes at
    the boundaries, biases left unquantized)."""
    target = QLayerConfig(activation=block(), weight=block())
    return QConfig(
        _layer(BFloat16Spec, BFloat16Spec),
        algo_config=[
            AutoMixprecisionConfig(
                target_layer_config=target, dual_quant_nodes=True, metric_threshold=0
            ),
            *algos,
        ],
        QuantizeBias=False,
    )


_PRESETS.update(
    {
        "BF16_MIXED_BFP16": lambda: _mixed_block(BFP16Spec),
        "BF16_MIXED_MXINT8": lambda: _mixed_block(MXInt8Spec),
        "BF16_MIXED_BFP16_ADAQUANT": lambda: _mixed_block(BFP16Spec, AdaQuantConfig()),
        "BF16_MIXED_MXINT8_ADAQUANT": lambda: _mixed_block(
            MXInt8Spec, AdaQuantConfig()
        ),
        "BF16_BFP16": lambda: QConfig(_layer(BFloat16Spec, BFP16Spec)),
        "BF16_MXINT8": lambda: QConfig(_layer(BFloat16Spec, MXInt8Spec)),
        "MX9_INT8": lambda: QConfig(_layer(MX9Spec, Int8Spec)),
        # The "amateur" CNN presets: asymmetric uint8 / uint16 activations,
        # per-tensor symmetric int8 / int16 weights; ACCURATE adds percentile
        # 99.9999 calibration and AdaRound (Quark's FastFinetune defaults).
        # VINT8: signed power-of-2 int8 everywhere, every op type quantized, no
        # Relu folding, int8 biases, one Q/DQ pair per consumer (the VAIML
        # deployment flavour of XINT8; Quark has no ADAROUND/ADAQUANT variant).
        "VINT8": lambda: QConfig(
            _layer(XInt8Spec, XInt8Spec),
            RemoveQDQConvRelu=False,
            Int32Bias=False,
            DedicatedQDQPair=True,
            QuantizeAllOpTypes=True,
        ),
        "S16S16_MIXED_S8S8": lambda: _s16s16_mixed_s8s8(),
        "INT8_CNN_DEFAULT": lambda: QConfig(
            _layer(UInt8Spec, Int8Spec, symmetric=False, calibration_method="minmax")
        ),
        "INT16_CNN_DEFAULT": lambda: QConfig(
            _layer(UInt16Spec, Int16Spec, symmetric=False, calibration_method="minmax")
        ),
        "INT8_CNN_ACCURATE": lambda: _cnn_accurate(UInt8Spec, Int8Spec),
        "INT16_CNN_ACCURATE": lambda: _cnn_accurate(UInt16Spec, Int16Spec),
    }
)


# FP16 / BF16 AdaQuant: Quark emits the FP16 / BF16 graph with tuned
# weights / biases. The graph half is reproduced; the tuning is not (like
# BFP16_ADAQUANT, quantize_model raises unless ignore_unsupported_algos=True).
_PRESETS["FP16_ADAQUANT"] = _algo_variant("FP16", AdaQuantConfig)
_PRESETS["BF16_ADAQUANT"] = _algo_variant("BF16", AdaQuantConfig)


def _with_algo(cfg: QConfig, algo: AlgoConfig) -> QConfig:
    cfg.algo_config = [algo]
    return cfg


class Config:
    """Mirror of ``quark.onnx.Config`` (wraps a global config)."""

    def __init__(self, global_quant_config: QConfig) -> None:
        self.global_quant_config = global_quant_config


# -- calibration-reader adapter ------------------------------------------------


def _drain_reader(
    reader: Any, limit: Optional[int] = None
) -> List[Dict[str, np.ndarray]]:
    """Materialize an onnxruntime-style ``CalibrationDataReader`` (anything
    with ``get_next() -> dict | None``), or pass through a list of dicts."""
    if reader is None:
        return []
    if isinstance(reader, (list, tuple)):
        return [dict(b) for b in reader]
    batches: List[Dict[str, np.ndarray]] = []
    while limit is None or len(batches) < limit:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append({k: np.asarray(v) for k, v in batch.items()})
    return batches


_RUNNABLE_ALGOS = {
    "quarot",
    "smooth_quant",
    "cle",
    "adaquant",
    "adaround",
    "gptq",
    "bias_correction",
    "auto_mixprecision",
}
# Quark AdaQuant param -> onnxsim.apply_adaquant kwarg (``legacy_engine=True``).
_ADAQUANT_PARAMS = {
    "num_iterations": "num_iterations",
    "learning_rate": "weight_learning_rate",
    "reg_param": "reg_param",
}
# ``extra_options["FastFinetune"]`` keys (they win over the algo config's
# values, as in Quark) -> AdaRoundConfig / AdaQuantConfig params.
_FASTFT_KEYS = {
    "DataSize": "data_size",
    "FixedSeed": "fixed_seed",
    "BatchSize": "batch_size",
    "NumBatches": "num_batches",
    "NumIterations": "num_iterations",
    "LearningRate": "learning_rate",
    "EarlyStop": "early_stop",
    "OutputIndex": "output_index",
    "LRAdjust": "lr_adjust",
    "SelectiveUpdate": "selective_update",
    "UpdateBias": "update_bias",
    "OutputQDQ": "output_qdq",
    "DropRatio": "drop_ratio",
    "MemOptLevel": "mem_opt_level",
    "Parallel": "parallel",
    "RegParam": "reg_param",
    "BetaRange": "beta_range",
    "WarmStart": "warm_start",
    "SelectMaxMemLayer": "select_max_mem_layer",
    "TargetOpType": "target_op_type",
    "RefModelPath": "ref_model_path",
}
# AdaRoundConfig / AdaQuantConfig fields that Quark 0.13's ``_get_config`` never
# copies into ``extra_options["FastFinetune"]`` (it stores them on the config and
# stops there), so there they only take effect through extra_options.
_FASTFT_NOT_FORWARDED = (
    "output_index",
    "reg_param",
    "beta_range",
    "warm_start",
    "parallel",
    "dynamic_batch",
    "ref_model_path",
)
# What Quark's ``*_ADAROUND`` / ``*_ADAQUANT`` presets put in
# ``extra_options["FastFinetune"]`` (read from the amd-quark 0.13 wheel).
_FASTFT_PRESET = {
    "data_size": 1000,
    "fixed_seed": 1705472343,
    "batch_size": 2,
    "num_iterations": 1000,
    "early_stop": True,
}


# -- quantizer -----------------------------------------------------------------


class ModelQuantizer:
    """Mirror of ``quark.onnx.ModelQuantizer``.

    ``last_approximations`` lists every place the requested spec was mapped
    to the nearest thing onnxsim can actually do (empty when exact).
    """

    def __init__(self, config: Union[QConfig, Config]) -> None:
        if isinstance(config, Config):
            config = config.global_quant_config
        if not isinstance(config, QConfig):
            raise TypeError(f"expected QConfig or Config, got {type(config).__name__}")
        self.config = config
        self.last_approximations: List[str] = []
        self._overrides_applied = False
        #: the :class:`~onnxsim.quark_auto_mixprecision.AutoMixprecisionResult`
        #: (sensitivity ranking, moved layers, scores) of the last run, if any
        self.last_auto_mixprecision: Any = None
        #: ``{"adaround" | "gptq": [LayerReport, ...]}`` of the last run
        self.last_weight_rounding: Dict[str, Any] = {}

    def quantize_model(
        self,
        model_input: Union[str, onnx.ModelProto],
        model_output: Optional[str] = None,
        calibration_data_reader: Any = None,
        ignore_unsupported_algos: bool = False,
    ) -> onnx.ModelProto:
        """Quantize and (if ``model_output`` is given) save. Returns the model."""
        cfg = self.config
        self.last_approximations = []
        self._overrides_applied = False
        self.last_auto_mixprecision = None
        self.last_weight_rounding = {}
        cfg.global_config = cfg.global_config.resolved()
        act, wt = cfg.global_config.activation, cfg.global_config.weight
        assert act is not None and wt is not None  # resolved() fills both

        if wt.is_dynamic:
            raise NotImplementedError("dynamic weight quantization is not supported")
        # the weight-rounding algorithms work on the int8 weight codes
        can_run = _RUNNABLE_ALGOS - (
            {"adaquant", "adaround", "gptq"} if wt.dtype == "int16" else set()
        )
        unsupported = [a.name for a in cfg.algo_config if a.name not in can_run]
        if unsupported and not ignore_unsupported_algos:
            raise NotImplementedError(
                f"algo_config [{', '.join(unsupported)}] is not executed by "
                "onnxsim.quark_compat; pass ignore_unsupported_algos=True to "
                "quantize without them"
            )
        runnable = [a for a in cfg.algo_config if a.name in can_run]

        if isinstance(model_input, str):
            model_input = onnx.load(model_input)

        half = act.dtype in ("float16", "bfloat16")
        if half and cfg.extra_options.get("ConvertToHalf"):
            if runnable and not ignore_unsupported_algos:
                raise NotImplementedError(
                    "algo_config is not applied to float presets "
                    f"({act.dtype}); pass ignore_unsupported_algos=True to "
                    "convert without it"
                )
            from onnxsim.onnx_simplifier import quantize_bf16, quantize_fp16

            fn = quantize_fp16 if act.dtype == "float16" else quantize_bf16
            result = fn(model_input)
        elif self._mixed_block_target(act) is not None:
            result = self._quantize_mixed_block(
                model_input, act, ignore_unsupported_algos
            )
        elif act.is_dynamic:
            result = self._quantize_dynamic(model_input, act, wt)
        elif wt.dtype in _FAKEQUANT_DTYPES or act.dtype in _FAKEQUANT_DTYPES:
            if cfg.algo_config and not ignore_unsupported_algos:
                what = "float presets" if half else "block formats"
                raise NotImplementedError(
                    f"algo_config is not applied to {what} "
                    f"({act.dtype}/{wt.dtype}); pass ignore_unsupported_algos=True "
                    "to quantize without it"
                )
            result = self._quantize_block(model_input, act, wt)
        else:
            result = self._quantize_int(
                model_input, act, wt, calibration_data_reader, runnable
            )

        if (
            cfg.specific_layer_config or cfg.layer_type_config
        ) and not self._overrides_applied:
            self._approx(
                "per-layer / per-type overrides ignored for "
                f"{act.dtype}/{wt.dtype} (global spec used)"
            )
        for msg in self.last_approximations:
            warnings.warn(f"onnxsim.quark_compat: {msg}", UserWarning, stacklevel=2)
        if model_output:
            onnx.save(result, model_output)
        return result

    def _mixed_block_target(self, act: QSpec) -> Optional[AlgoConfig]:
        """The ``AutoMixprecisionConfig`` of a bfloat16 model whose target
        layers use a block format (``BF16_MIXED_BFP16`` / ``_MXINT8``), if any."""
        if act.dtype != "bfloat16":
            return None
        for a in self.config.algo_config:
            t = a.params.get("target_layer_config")
            if (
                a.name == "auto_mixprecision"
                and isinstance(t, QLayerConfig)
                and t.weight is not None
                and t.weight.dtype in _BLOCK_DTYPES
            ):
                return a
        return None

    def _quantize_mixed_block(
        self, model: onnx.ModelProto, act: QSpec, ignore_unsupported: bool
    ) -> onnx.ModelProto:
        from onnxsim.quark_preset_graphs import apply_mixed_block_format

        algo = self._mixed_block_target(act)
        assert algo is not None
        p = algo.params
        target = p["target_layer_config"]
        others = [a.name for a in self.config.algo_config if a is not algo]
        if others and not ignore_unsupported:
            raise NotImplementedError(
                f"algo_config [{', '.join(others)}] is not applied to block formats; "
                "pass ignore_unsupported_algos=True to quantize without it"
            )
        if p.get("metric_threshold", 0):
            raise NotImplementedError(
                "AutoMixprecisionConfig.metric_threshold must be 0 (promote every "
                "candidate): onnxsim cannot evaluate com.amd.quark custom ops"
            )
        for key in ("subgraph_json", "sensitivity_cache_file"):
            if p.get(key) is not None:
                raise NotImplementedError(
                    f"AutoMixprecisionConfig.{key} is not supported"
                )
        self._approx(
            f"{target.weight.dtype} layers: every candidate is promoted (Quark's "
            "metric_threshold=0), no sensitivity ranking; the model uses "
            "com.amd.quark custom ops that onnxsim cannot execute"
        )
        return apply_mixed_block_format(
            model,
            target.weight.dtype,
            exclude=[e for e in self.config.exclude if isinstance(e, str)],
            target_ops=tuple(p.get("target_op_type") or PROMOTABLE_OPS),
            include_layers=p.get("include_layers") or (),
            exclude_layers=p.get("exclude_layers") or (),
        )

    def _quantize_dynamic(
        self, model: onnx.ModelProto, act: QSpec, wt: QSpec
    ) -> onnx.ModelProto:
        from onnxsim.quark_dynamic import quantize_dynamic_integer

        cfg = self.config
        if cfg.algo_config:
            raise NotImplementedError(
                "algo_config is not applied to dynamic quantization"
            )
        if wt.dtype not in ("int8", "uint8"):
            raise NotImplementedError(f"weight dtype {wt.dtype} unsupported")
        if act.dtype not in ("int8", "uint8"):
            raise NotImplementedError(
                f"dynamic activation dtype {act.dtype} unsupported"
            )
        if act.dtype != "uint8":
            self._approx(
                "dynamic activations are quantized uint8 asymmetric "
                "(DynamicQuantizeLinear), whatever the activation spec's dtype"
            )
        exclude = _match_nodes(
            model, [e for e in cfg.exclude if isinstance(e, (str, tuple))]
        )
        _, _, type_excluded = self._layer_overrides(model, allow_dtypes=False)
        self._overrides_applied = True
        return quantize_dynamic_integer(
            model, weight_dtype=wt.dtype, exclude_nodes=exclude + type_excluded
        )

    def _quantize_block(
        self, model: onnx.ModelProto, act: QSpec, wt: QSpec
    ) -> onnx.ModelProto:
        from onnxsim.quark_fakequant_graph import apply_fake_quant_format

        opts = self.config.extra_options
        exclude = [e for e in self.config.exclude if isinstance(e, str)]
        if act.dtype in _BLOCK_DTYPES and wt.dtype == "int8":  # MX9_INT8
            from onnxsim.quark_preset_graphs import (
                apply_block_activations_int8_constants,
            )

            self._approx(
                f"{act.dtype} uses com.amd.quark custom ops: the model needs Quark's "
                "ONNX custom-op library to run (onnxsim cannot execute it)"
            )
            return apply_block_activations_int8_constants(model, act.dtype, exclude)
        fn = _block_fn(wt.dtype)
        # bfloat16 activations over block-format constants (BF16_BFP16 / BF16_MXINT8)
        mixed = act.dtype == "bfloat16" and wt.dtype in _BLOCK_DTYPES
        if fn is None or (
            act.dtype in _FAKEQUANT_DTYPES and act.dtype != wt.dtype and not mixed
        ):
            raise NotImplementedError(
                f"weight dtype {wt.dtype} with activation dtype {act.dtype}: "
                "block / half formats must match on both sides"
            )
        quantize_acts = act.dtype in _FAKEQUANT_DTYPES and opts.get(
            "BlockFormatActivations", True
        )
        if act.dtype in _FAKEQUANT_DTYPES and not quantize_acts:
            self._approx(
                f"{act.dtype} activations are not quantized "
                "(extra_options BlockFormatActivations=False) -- weights only"
            )
        elif quantize_acts:
            self._approx(
                f"{act.dtype} uses com.amd.quark custom ops: the model needs Quark's "
                "ONNX custom-op library to run (onnxsim cannot execute it)"
            )
        fold = bool(opts.get("BlockFormatFoldWeights", False))
        return apply_fake_quant_format(
            model,
            act.dtype if mixed else wt.dtype,
            activations=bool(quantize_acts),
            fold_weights=fold,
            fold_fn=fn,
            exclude=exclude,
            const_dtype=wt.dtype if mixed and quantize_acts and not fold else None,
        )

    def _approx(self, msg: str) -> None:
        self.last_approximations.append(msg)

    def _quarot(self, model: onnx.ModelProto, algo: AlgoConfig) -> onnx.ModelProto:
        from onnxsim.quark_quarot import rotate_model

        p = algo.params
        if not p.get("r_config_path"):
            raise ValueError(
                "QuarotConfig.r_config_path is required (a JSON file with "
                '"R1_pairs": [{"prev_nodes", "next_nodes", "norm_node"}, ...])'
            )
        self._approx(
            "Quarot: only the R1 (residual-stream) rotation is applied; the matrix "
            "is a (random) Hadamard for power-of-two sizes, else random orthogonal"
        )
        return rotate_model(
            model,
            p["r_config_path"],
            r_matrix_dim=p.get("r_matrix_dim", 4096),
            use_random_had=bool(p.get("use_random_had", False)),
        )

    def _finetune(
        self,
        name: str,
        float_model: onnx.ModelProto,
        quantized: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        algo: AlgoConfig,
    ) -> onnx.ModelProto:
        """Quark's FastFinetune (``AdaRoundConfig`` / ``AdaQuantConfig``) via
        :mod:`onnxsim.quark_finetune`; ``extra_options["FastFinetune"]`` keys
        override the config's params, and ``QuantizationPreference="accuracy"``
        applies Quark's own overrides (``EarlyStop`` off, ``UpdateBias`` and
        ``OutputQDQ`` on)."""
        from onnxsim.quark_finetune import TARGET_OPS, FinetuneOptions, finetune

        p = dict(algo.params)
        dropped = [k for k in _FASTFT_NOT_FORWARDED if k in p]
        for k in dropped:
            del p[k]
        if dropped:
            self._approx(
                f"{name}: Quark's config does not forward {', '.join(dropped)} "
                "(only extra_options['FastFinetune'] does), so they are ignored here too"
            )
        ff = self.config.extra_options.get("FastFinetune")
        if isinstance(ff, dict):
            p.update({_FASTFT_KEYS[k]: v for k, v in ff.items() if k in _FASTFT_KEYS})
        if self.config.extra_options.get("QuantizationPreference") == "accuracy":
            p.update(early_stop=False, update_bias=True, output_qdq=True)
        if "data_size" in p:
            calibration = calibration[: int(p["data_size"])]
        ref = p.get("ref_model_path")
        if isinstance(ref, str) and os.path.exists(ref):
            float_model = onnx.load(ref)
        adaquant = name == "adaquant"
        targets = tuple(p.get("target_op_type") or TARGET_OPS)
        opt = FinetuneOptions(
            algorithm=name,
            num_iterations=int(p.get("num_iterations", 3000 if adaquant else 1000)),
            learning_rate=p.get("learning_rate"),
            batch_size=int(p.get("batch_size", 1)),
            num_batches=int(p.get("num_batches", 1)),
            early_stop=bool(p.get("early_stop", False)),
            reg_param=float(p.get("reg_param", 0.01)),
            beta_range=tuple(p.get("beta_range", (20.0, 2.0))),  # type: ignore[arg-type]
            warm_start=float(p.get("warm_start", 0.2)),
            drop_ratio=float(p.get("drop_ratio", 1.0)),
            lr_adjust=tuple(p["lr_adjust"]) if p.get("lr_adjust") else None,  # type: ignore[arg-type]
            selective_update=bool(p.get("selective_update", False)),
            update_bias=bool(p.get("update_bias", False)) and adaquant,
            output_qdq=bool(p.get("output_qdq", False)),
            parallel=bool(p.get("parallel", False)),
            mem_opt_level=int(p.get("mem_opt_level", 1)),
            output_index=p.get("output_index"),
            select_max_mem_layer=bool(p.get("select_max_mem_layer", False)),
            target_ops=targets,
            seed=int(p.get("fixed_seed", 1705472343)),
        )
        self._approx(
            f"{name} is a numpy port of Quark's FastFinetune loop: mini-batches "
            "come from numpy's generator instead of torch.randperm and the "
            "arithmetic is float64, so results match Quark statistically "
            "(bit for bit given the same mini-batch indices)"
        )
        # update_bias: only AdaQuant reads it, as in Quark.
        out, self.last_weight_rounding[name] = finetune(
            float_model, quantized, calibration, opt
        )
        return out

    def _gptq(
        self,
        float_model: onnx.ModelProto,
        quantized: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        algo: AlgoConfig,
    ) -> onnx.ModelProto:
        from onnxsim.quark_weight_rounding import gptq_int8

        p = algo.params
        bits = int(p.get("bits", 8))
        group_size = int(p.get("group_size", -1))
        sym = bool(p.get("weight_symmetric", True))
        mse = bool(p.get("mse", False))
        requantize = (
            bits != 8 or group_size != -1 or not sym or mse or "per_channel" in p
        )
        if requantize:
            self._approx(
                "GPTQ re-grids the weights like Quark's GPTQ (GPTQConfig.bits / "
                "group_size / per_channel / mse / weight_symmetric) from all "
                "calibration batches, and does propagate rounding error (Quark "
                "0.13's update is a no-op)"
            )
        else:
            self._approx(
                "GPTQ keeps quantize_full_qdq's per-channel scales "
                "(GPTQConfig.per_channel / mse are not used)"
            )
        kwargs: Dict[str, Any] = {
            "bits": bits,
            "group_size": group_size,
            "weight_symmetric": sym,
            "mse": mse,
            "per_channel": bool(p.get("per_channel", False)),
            "requantize": requantize,
        }
        if "perc_damp" in p:
            kwargs["perc_damp"] = p["perc_damp"]
        if "block_size" in p:
            kwargs["block_size"] = p["block_size"]
        if "act_order" in p:
            kwargs["act_order"] = bool(p["act_order"])
        out, self.last_weight_rounding["gptq"] = gptq_int8(
            float_model, quantized, calibration, **kwargs
        )
        return out

    @staticmethod
    def _promotes_int16_to_int8(act: QSpec, algo: AlgoConfig) -> bool:
        """A signed 16-bit model whose AutoMixprecision target is signed int8
        (Quark's ``S16S16_MIXED_S8S8``)."""
        target = algo.params.get("target_layer_config")
        return (
            act.dtype == "int16"
            and isinstance(target, QLayerConfig)
            and target.activation is not None
            and target.activation.dtype == "int8"
        )

    def _promote_all_int16_to_int8(
        self,
        model: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        act: QSpec,
        wt: QSpec,
        exclude: List[str],
        algo: AlgoConfig,
        per_channel: bool,
    ) -> onnx.ModelProto:
        """``S16S16_MIXED_S8S8``: int16 activations and weights, with every
        Conv / ConvTranspose / Gemm / MatMul promoted to int8 inputs, weights
        and biases (Quark runs AutoMixprecision with the metric threshold
        disabled, so every candidate is promoted). The outputs of the
        promoted layers stay int16 and no convert pairs are inserted."""
        from onnxsim.full_qdq import quantize_full_qdq
        from onnxsim.quark_preset_graphs import (
            promoted_activations,
            requantize_biases_int8,
        )

        p = algo.params
        target = p["target_layer_config"]
        if p.get("metric_threshold", 0):
            raise NotImplementedError(
                "AutoMixprecisionConfig.metric_threshold must be 0 for an "
                "int16 -> int8 mix (promote every candidate)"
            )
        for key in ("subgraph_json", "sensitivity_cache_file"):
            if p.get(key) is not None:
                raise NotImplementedError(
                    f"AutoMixprecisionConfig.{key} is not supported"
                )
        if target.weight.dtype != "int8":
            raise NotImplementedError("target_layer_config weight must be int8")
        ops = tuple(p.get("target_op_type") or PROMOTABLE_OPS)
        include, drop = p.get("include_layers") or (), p.get("exclude_layers") or ()
        self._approx(
            "int16 -> int8 mix: every candidate is promoted (Quark's "
            "metric_threshold=0), no sensitivity ranking; the weights of "
            "Conv / Gemm / MatMul layers are int8 (promoted), and no layer "
            "keeps int16 weights"
        )
        tensor_dtypes = {
            t: "int8" for t in promoted_activations(model, ops, include, drop)
        }
        quantized = quantize_full_qdq(
            model,
            calibration_data=calibration,
            activation_dtype="int16",
            exclude_nodes=exclude,
            method=act.calibration_method,
            symmetric_activations=act.symmetric,
            per_channel=per_channel,
            weight_dtype="int8",
            tensor_dtypes=tensor_dtypes,
            convert_inputs=False,
        )
        return requantize_biases_int8(quantized, model, ops, include, drop)

    def _layer_overrides(
        self, model: onnx.ModelProto, allow_dtypes: bool = True
    ) -> "tuple[Dict[str, str], Dict[str, bool], List[str]]":
        """``(tensor_dtypes, tensor_symmetric, excluded nodes)`` from
        ``layer_type_config`` then ``specific_layer_config`` (the latter wins,
        as in Quark). A layer's ``input_tensors`` spec applies to its
        activation inputs (those before the first constant), ``output_tensors``
        to its outputs; weight / bias overrides are not supported."""
        cfg = self.config
        inits = {i.name for i in model.graph.initializer}
        dtypes: Dict[str, str] = {}
        symmetric: Dict[str, bool] = {}
        excluded: List[str] = []
        by_name = {n.name: n for n in model.graph.node if n.name}

        def apply(node: onnx.NodeProto, layer: QLayerConfig) -> None:
            if layer.weight is not None and layer.weight.dtype not in (
                "int8",
                "uint8",
            ):
                raise NotImplementedError(
                    f"per-layer weight dtype {layer.weight.dtype} is not supported "
                    "(weights are int8)"
                )
            if layer.bias is not None:
                self._approx("per-layer bias specs ignored (biases stay int32)")
            for spec, tensors in (
                (layer.activation, _activation_inputs(node, inits)),
                (layer.output_tensors, list(node.output)),
            ):
                if spec is None:
                    continue
                dt = self._int_act_dtype(spec)
                for t in tensors:
                    dtypes[t] = dt
                    symmetric[t] = spec.symmetric

        for layer, op_types in cfg.layer_type_config.items():
            if layer is None:
                excluded += [
                    n.name for n in model.graph.node if n.op_type in op_types and n.name
                ]
                continue
            for n in model.graph.node:
                if n.op_type in op_types:
                    apply(n, layer)
        for layer, names in cfg.specific_layer_config.items():
            for name in _match_nodes(model, names):
                if name not in by_name:
                    raise ValueError(f"specific_layer_config: no node named {name!r}")
                apply(by_name[name], layer)
        if dtypes and not allow_dtypes:
            self._approx("per-layer activation dtypes ignored (dynamic quantization)")
            dtypes, symmetric = {}, {}
        return dtypes, symmetric, excluded

    def _int_act_dtype(self, spec: QSpec) -> str:
        """The ``quantize_full_qdq`` activation dtype for an int spec."""
        if spec.dtype in ("int8", "uint8", "int16", "uint16"):
            return spec.dtype
        raise NotImplementedError(f"activation dtype {spec.dtype} unsupported")

    def _auto_mixprecision(
        self,
        model: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        base_dtype: str,
        act: QSpec,
        exclude: List[str],
        algo: AlgoConfig,
    ) -> onnx.ModelProto:
        from onnxsim.quark_auto_mixprecision import auto_mixprecision

        p = algo.params
        target = p.get("target_layer_config")
        if not isinstance(target, QLayerConfig):
            raise NotImplementedError(
                "AutoMixprecisionConfig needs a single QLayerConfig as "
                "target_layer_config (the dict / list multi-config forms are "
                "not supported)"
            )
        for key in ("subgraph_json", "sensitivity_cache_file"):
            if p.get(key) is not None:
                raise NotImplementedError(
                    f"AutoMixprecisionConfig.{key} is not supported"
                )
        if (target.weight or Int8Spec()).dtype not in ("int8", "uint8"):
            raise NotImplementedError("target_layer_config weight must be int8")
        if target.activation is None:
            raise ValueError("target_layer_config needs an activation spec")
        target_dtype = self._int_act_dtype(target.activation)
        if target_dtype == base_dtype:
            raise ValueError(
                "AutoMixprecision target activation precision equals the base "
                f"precision ({base_dtype}); nothing to mix"
            )
        self._approx(
            "AutoMixprecision mixes activation precision only (weights stay int8)"
        )
        optimize = p.get("metric_optimize_object", "speed")
        res = auto_mixprecision(
            model,
            calibration,
            base_dtype=base_dtype,
            target_dtype=target_dtype,
            target_op_types=tuple(
                p.get("target_op_type") or ("Conv", "Gemm", "MatMul")
            ),
            include_layers=p.get("include_layers") or (),
            exclude_layers=p.get("exclude_layers") or (),
            exclude_nodes=exclude,
            metric=p.get("metric_default", "l2"),
            metric_distance_fn=p.get("metric_distance_fn"),
            metric_evaluate_fn=p.get("metric_evaluate_fn"),
            metric_threshold=p.get("metric_threshold", 0),
            optimize=optimize,
            metric_output_index=p.get("metric_output_index", 0),
            data_size=p.get("data_size", 0),
            method=act.calibration_method,
        )
        self.last_auto_mixprecision = res
        return res.model

    def _quantize_int(
        self,
        model: onnx.ModelProto,
        act: QSpec,
        wt: QSpec,
        reader: Any,
        algos: List[AlgoConfig],
    ) -> onnx.ModelProto:
        from onnxsim.full_qdq import quantize_full_qdq

        if wt.dtype not in ("int8", "uint8", "int16"):
            raise NotImplementedError(f"weight dtype {wt.dtype} unsupported")
        if wt.dtype == "uint8":
            self._approx("weights quantized int8-symmetric instead of uint8")
        act_dtype = self._int_act_dtype(act)

        calibration = _drain_reader(reader)
        if not calibration:
            raise ValueError("calibration_data_reader is required for integer presets")
        exclude = _match_nodes(
            model, [e for e in self.config.exclude if isinstance(e, (str, tuple))]
        )
        t_dtypes, t_sym, type_excluded = self._layer_overrides(model)
        self._overrides_applied = True
        exclude += type_excluded
        opts = self.config.extra_options
        by_name = {a.name: a for a in algos}

        # Quark's presets quantize weights per tensor; the weight-rounding
        # algorithms below work per output channel.
        per_channel = bool(self.config.extra_options.get("PerChannel", False))
        legacy_adaquant = "adaquant" in by_name and by_name["adaquant"].params.get(
            "legacy_engine"
        )
        if not per_channel and ("gptq" in by_name or legacy_adaquant):
            per_channel = True
            self._approx("weights quantized per channel (needed by the algorithm)")

        # Float -> float pre-quantization passes (quantize_full_qdq is fed
        # the transformed model; the untouched one stays the reference).
        float_model = model
        work = model
        if "quarot" in by_name:
            work = self._quarot(work, by_name["quarot"])
        if "smooth_quant" in by_name:
            from onnxsim.smoothquant import apply_smoothquant

            alpha = by_name["smooth_quant"].params.get("alpha", 0.5)
            work = apply_smoothquant(work, calibration_data=calibration, alpha=alpha)
        if "cle" in by_name:
            from onnxsim.onnx_simplifier import cross_layer_equalize
            from onnxsim.quark_cle import equalize_linear_layers

            work = equalize_linear_layers(cross_layer_equalize(work))
        if work is not model:
            float_model = work

        mixed_algo = by_name.get("auto_mixprecision")
        if mixed_algo is not None and self._promotes_int16_to_int8(act, mixed_algo):
            quantized = self._promote_all_int16_to_int8(
                work, calibration, act, wt, exclude, mixed_algo, per_channel
            )
        elif mixed_algo is not None:
            quantized = self._auto_mixprecision(
                work, calibration, act_dtype, act, exclude, by_name["auto_mixprecision"]
            )
        else:
            quantized = quantize_full_qdq(
                work,
                calibration_data=calibration,
                activation_dtype=act_dtype,
                exclude_nodes=exclude,
                method=act.calibration_method,
                symmetric_activations=act.symmetric,
                power_of_two=act.pof2 or wt.pof2,
                per_channel=per_channel,
                weight_dtype="int16" if wt.dtype == "int16" else "int8",
                fold_relu=bool(opts.get("RemoveQDQConvRelu", True)),
                # Quark's XINT8 (power-of-2 weights): MinMSE scale search on
                # weights and int8 biases (``Int32Bias=True`` keeps int32)
                pof2_mode="minmse" if wt.pof2 else "ceil",
                int8_bias=wt.pof2 and not self.config.extra_options.get("Int32Bias"),
                int8_constants=True,
                align_eltwise_dtype=bool(
                    self.config.extra_options.get("AlignEltwiseQuantType")
                ),
                softmax_unit_range=not act.pof2,
                tensor_dtypes=t_dtypes or None,
                tensor_symmetric=t_sym or None,
            )
            if opts.get("Int32Bias", True) is False:
                from onnxsim.quark_preset_graphs import requantize_biases_int8

                self._approx(
                    "int8 bias (Int32Bias=False): symmetric per tensor"
                    + (", power-of-2 scale" if act.pof2 or wt.pof2 else "")
                )
                quantized = requantize_biases_int8(
                    quantized,
                    work,
                    ("Conv", "ConvTranspose", "Gemm"),
                    power_of_two=act.pof2 or wt.pof2,
                )
            if opts.get("DedicatedQDQPair", False):
                from onnxsim.quark_preset_graphs import dedicate_qdq_pairs

                quantized = dedicate_qdq_pairs(quantized)

        # Post-quantization passes, which compare against the float model.
        if "adaquant" in by_name and by_name["adaquant"].params.get("legacy_engine"):
            from onnxsim.adaquant import apply_adaquant

            params = by_name["adaquant"].params
            quantized = apply_adaquant(
                float_model,
                quantized,
                calibration_data=calibration,
                **{
                    kwarg: params[key]
                    for key, kwarg in _ADAQUANT_PARAMS.items()
                    if key in params
                },
            )
        elif "adaquant" in by_name:
            quantized = self._finetune(
                "adaquant", float_model, quantized, calibration, by_name["adaquant"]
            )
        if "adaround" in by_name:
            quantized = self._finetune(
                "adaround", float_model, quantized, calibration, by_name["adaround"]
            )
        if "gptq" in by_name:
            quantized = self._gptq(float_model, quantized, calibration, by_name["gptq"])
        if "bias_correction" in by_name:
            from onnxsim.bias_correction import correct_bias

            quantized = correct_bias(
                float_model, quantized, calibration_data=calibration
            )
        return quantized


__all__ = [
    "AdaQuantConfig",
    "AdaRoundConfig",
    "AlgoConfig",
    "AutoMixprecisionConfig",
    "BFP16Spec",
    "BFloat16Spec",
    "BiasCorrectionConfig",
    "CalibMethod",
    "CLEConfig",
    "Config",
    "Float16Spec",
    "GPTQConfig",
    "Int16Spec",
    "Int8Spec",
    "MX4Spec",
    "MX6Spec",
    "MX9Spec",
    "MXFP4E2M1Spec",
    "MXFP6E2M3Spec",
    "MXFP6E3M2Spec",
    "MXFP8E4M3Spec",
    "MXFP8E5M2Spec",
    "MXInt8Spec",
    "ModelQuantizer",
    "QConfig",
    "QLayerConfig",
    "QSpec",
    "QuarotConfig",
    "SmoothQuantConfig",
    "UInt16Spec",
    "UInt8Spec",
    "XInt8Spec",
    "XUInt8Spec",
]
