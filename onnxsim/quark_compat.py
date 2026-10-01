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
  Integer presets use :func:`onnxsim.full_qdq.quantize_full_qdq`, whose
  activations are **uint8/uint16 only** (asymmetric) and whose weights are
  symmetric int8. A preset that asks for signed or power-of-2 activations
  (``A8W8`` int8, ``XINT8``) is therefore *approximated* by uint8 / uint16
  activations; the approximation is recorded in
  ``ModelQuantizer.last_approximations`` and emitted as a ``UserWarning``.
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
  ``GlobalAveragePool``, CLE). Options (``QConfig(..., extra_options=...)``):
  ``BlockFormatActivations=False`` quantizes only the constants, offline, so the
  model runs anywhere; ``BlockFormatFoldWeights=True`` folds the constants
  offline (via :mod:`onnxsim.quark_block_formats`) instead of leaving a node on
  them. ``algo_config`` is not applied to block formats.
  Dynamic quantization raises ``NotImplementedError``.
- ``algo_config``: SmoothQuant (``alpha``) and CLE run on the float model
  before quantization; AdaQuant (``num_iterations``, ``learning_rate``,
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
  AdaRound and GPTQ refine the int8 weight codes layer by layer
  (:mod:`onnxsim.quark_weight_rounding`; Conv / Gemm / MatMul, guarded so a
  layer's reconstruction error never gets worse); ``update_bias``, ``drop_ratio``
  (AdaRound) and ``bits != 8`` / ``group_size`` / asymmetric weights (GPTQ)
  raise. Quarot folds the R1 residual-stream rotation into the float
  weights before quantization (:mod:`onnxsim.quark_quarot`; needs
  ``r_config_path``; R2-R4 do not exist, as in Quark's ONNX flow). An
  ``algo_config`` that cannot run for a preset (block formats, FP16 / BF16)
  raises ``NotImplementedError`` unless ``ignore_unsupported_algos=True``.
- ``extra_options`` are stored, not interpreted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
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
    calibration_method: str = "minmax"
    is_dynamic: bool = False


def _spec(name: str, dtype: str, symmetric: bool, pof2: bool = False):
    def __init__(self, **kwargs: Any) -> None:
        QSpec.__init__(self, dtype=dtype, symmetric=symmetric, pof2=pof2, **kwargs)

    return type(name, (QSpec,), {"__init__": __init__, "__doc__": f"{name}."})


Int8Spec = _spec("Int8Spec", "int8", True)
UInt8Spec = _spec("UInt8Spec", "uint8", False)
Int16Spec = _spec("Int16Spec", "int16", True)
UInt16Spec = _spec("UInt16Spec", "uint16", False)
XInt8Spec = _spec("XInt8Spec", "int8", True, pof2=True)
XUInt8Spec = _spec("XUInt8Spec", "uint8", False, pof2=True)
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
    """Activation + weight spec of one layer group (Quark's ``QLayerConfig``)."""

    activation: QSpec = field(default_factory=Int8Spec)
    weight: QSpec = field(default_factory=Int8Spec)


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


def _layer(act: type, wt: type) -> QLayerConfig:
    return QLayerConfig(activation=act(), weight=wt())


_PRESETS: Dict[str, Callable[[], QConfig]] = {
    "XINT8": lambda: QConfig(_layer(XUInt8Spec, XInt8Spec)),
    "A8W8": lambda: QConfig(_layer(Int8Spec, Int8Spec)),
    "S8S8_AAWS": lambda: QConfig(_layer(Int8Spec, Int8Spec)),
    "U8S8_AAWS": lambda: QConfig(_layer(UInt8Spec, Int8Spec)),
    "U8U8_AAWA": lambda: QConfig(_layer(UInt8Spec, UInt8Spec)),
    "A16W8": lambda: QConfig(_layer(Int16Spec, Int8Spec)),
    "S16S8_ASWS": lambda: QConfig(_layer(Int16Spec, Int8Spec)),
    "U16S8_AAWS": lambda: QConfig(_layer(UInt16Spec, Int8Spec)),
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


def _algo_variant(base: str, cls: type) -> Callable[[], QConfig]:
    def make() -> QConfig:
        return _with_algo(_PRESETS[base](), cls())

    return make


for _n in list(_PRESETS):
    _block = _n.startswith(("BFP", "MX"))
    if _n == "BF16" or _n.startswith("FP16"):
        continue
    if not _block:
        _PRESETS[f"{_n}_ADAROUND"] = _algo_variant(_n, AdaRoundConfig)
    _PRESETS[f"{_n}_ADAQUANT"] = _algo_variant(_n, AdaQuantConfig)


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
# Quark AdaQuant param -> onnxsim.apply_adaquant kwarg.
_ADAQUANT_PARAMS = {
    "num_iterations": "num_iterations",
    "learning_rate": "weight_learning_rate",
    "reg_param": "reg_param",
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
        self.last_auto_mixprecision = None
        self.last_weight_rounding = {}
        act, wt = cfg.global_config.activation, cfg.global_config.weight

        for spec in (act, wt):
            if spec.is_dynamic:
                raise NotImplementedError("dynamic quantization is not supported")
        unsupported = [a.name for a in cfg.algo_config if a.name not in _RUNNABLE_ALGOS]
        if unsupported and not ignore_unsupported_algos:
            raise NotImplementedError(
                f"algo_config [{', '.join(unsupported)}] is not executed by "
                "onnxsim.quark_compat; pass ignore_unsupported_algos=True to "
                "quantize without them"
            )
        runnable = [a for a in cfg.algo_config if a.name in _RUNNABLE_ALGOS]

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

        if cfg.specific_layer_config or cfg.layer_type_config:
            self._approx("per-layer / per-type overrides ignored (global spec used)")
        for msg in self.last_approximations:
            warnings.warn(f"onnxsim.quark_compat: {msg}", UserWarning, stacklevel=2)
        if model_output:
            onnx.save(result, model_output)
        return result

    def _quantize_block(
        self, model: onnx.ModelProto, act: QSpec, wt: QSpec
    ) -> onnx.ModelProto:
        from onnxsim.quark_fakequant_graph import apply_fake_quant_format

        fn = _block_fn(wt.dtype)
        if fn is None or (act.dtype in _FAKEQUANT_DTYPES and act.dtype != wt.dtype):
            raise NotImplementedError(
                f"weight dtype {wt.dtype} with activation dtype {act.dtype}: "
                "block / half formats must match on both sides"
            )
        opts = self.config.extra_options
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
        return apply_fake_quant_format(
            model,
            wt.dtype,
            activations=bool(quantize_acts),
            fold_weights=bool(opts.get("BlockFormatFoldWeights", False)),
            fold_fn=fn,
            exclude=[e for e in self.config.exclude if isinstance(e, str)],
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

    def _adaround(
        self,
        float_model: onnx.ModelProto,
        quantized: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        algo: AlgoConfig,
    ) -> onnx.ModelProto:
        from onnxsim.quark_weight_rounding import adaround_int8

        p = algo.params
        if p.get("update_bias"):
            raise NotImplementedError("AdaRoundConfig.update_bias is not supported")
        if p.get("drop_ratio", 1.0) != 1.0:
            raise NotImplementedError(
                "AdaRoundConfig.drop_ratio (QDrop) is not supported"
            )
        self._approx(
            "AdaRound is layer-wise against the float model's activations "
            "(Quark optimizes subgraph blocks)"
        )
        kwargs: Dict[str, Any] = {
            k: p[k]
            for k in ("num_iterations", "learning_rate", "reg_param", "warm_start")
            if k in p
        }
        if "beta_range" in p:
            kwargs["beta_range"] = tuple(p["beta_range"])
        if "fixed_seed" in p:
            kwargs["seed"] = int(p["fixed_seed"]) % (2**32)
        if p.get("target_op_type"):
            kwargs["target_ops"] = tuple(
                o for o in p["target_op_type"] if o in ("Conv", "Gemm", "MatMul")
            )
        out, self.last_weight_rounding["adaround"] = adaround_int8(
            float_model, quantized, calibration, **kwargs
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
        if p.get("bits", 8) != 8:
            raise NotImplementedError("GPTQConfig.bits must be 8 (int8 QDQ weights)")
        if p.get("group_size", -1) != -1:
            raise NotImplementedError("GPTQConfig.group_size must be -1 (ungrouped)")
        if not p.get("weight_symmetric", True):
            raise NotImplementedError("GPTQConfig.weight_symmetric must be True")
        self._approx(
            "GPTQ keeps quantize_full_qdq's per-channel scales "
            "(GPTQConfig.per_channel / mse are not used)"
        )
        kwargs: Dict[str, Any] = {}
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

    def _int_act_dtype(self, spec: QSpec) -> str:
        """The ``quantize_full_qdq`` activation dtype for an int spec (uint8 or
        uint16), recording an approximation for the signed ones."""
        if spec.dtype in ("int16", "uint16"):
            if spec.dtype == "int16":
                self._approx("int16 activations mapped to uint16 (asymmetric)")
            return "uint16"
        if spec.dtype in ("int8", "uint8"):
            if spec.dtype == "int8":
                self._approx("int8 activations mapped to uint8 (asymmetric)")
            return "uint8"
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
        if target.weight.dtype not in ("int8", "uint8"):
            raise NotImplementedError("target_layer_config weight must be int8")
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

        if wt.dtype not in ("int8", "uint8"):
            raise NotImplementedError(f"weight dtype {wt.dtype} unsupported")
        if wt.dtype == "uint8":
            self._approx("weights quantized int8-symmetric instead of uint8")
        act_dtype = self._int_act_dtype(act)
        if act.pof2 or wt.pof2:
            self._approx("power-of-2 scales not enforced (float scales used)")

        calibration = _drain_reader(reader)
        if not calibration:
            raise ValueError("calibration_data_reader is required for integer presets")
        exclude = [e for e in self.config.exclude if isinstance(e, str)]
        by_name = {a.name: a for a in algos}

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

            work = cross_layer_equalize(work)
        if work is not model:
            float_model = work

        if "auto_mixprecision" in by_name:
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
            )

        # Post-quantization passes, which compare against the float model.
        if "adaquant" in by_name:
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
        if "adaround" in by_name:
            quantized = self._adaround(
                float_model, quantized, calibration, by_name["adaround"]
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
