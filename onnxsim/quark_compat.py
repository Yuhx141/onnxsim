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
- Block formats (``BFP16``, ``MX4/6/9``, ``MXFP4/6/8``, ``MXINT8``): the
  *weights* of MatMul/Gemm/Conv are fake-quantized offline with
  :mod:`onnxsim.quark_block_formats` (bit-exact against Quark's CPU kernels)
  and stored back as float32, blocks along the reduction axis. Quark also
  inserts runtime custom ops (domain ``com.vai.quantize``) that quantize
  *activations*; onnxsim has no such op, so activations stay float -- recorded
  in ``last_approximations``. ``algo_config`` is not applied to block formats.
  Dynamic quantization raises ``NotImplementedError``.
- ``algo_config``: SmoothQuant (``alpha``) and CLE run on the float model
  before quantization; AdaQuant (``num_iterations``, ``learning_rate``,
  ``reg_param``) and BiasCorrection run after it, against the float model.
  AdaQuant only reoptimizes MatMul/Gemm layers whose output is not folded
  with a following Relu, and leaves the rest as calibrated. AdaRound, GPTQ,
  Quarot and AutoMixprecision are accepted and stored so configs round-trip,
  but **not executed** (onnxsim's AdaRound/GPTQ target its int4 weight-only
  scheme): ``quantize_model`` raises ``NotImplementedError`` naming them
  unless ``ignore_unsupported_algos=True``.
- ``extra_options`` are stored, not interpreted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import onnx

# -- data-type specs ---------------------------------------------------------


@dataclass(eq=True)
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
    """Weight fake-quantizer ``f(array, axis)`` for a block-format dtype."""
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


@dataclass(eq=True)
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
    "XINT8": lambda: QConfig(_layer(XInt8Spec, XInt8Spec)),
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


_RUNNABLE_ALGOS = {"smooth_quant", "cle", "adaquant", "bias_correction"}
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

        if wt.dtype in _BLOCK_DTYPES or act.dtype in _BLOCK_DTYPES:
            if cfg.algo_config and not ignore_unsupported_algos:
                raise NotImplementedError(
                    "algo_config is not applied to block formats "
                    f"({act.dtype}/{wt.dtype}); pass ignore_unsupported_algos=True "
                    "to quantize without it"
                )
            result = self._quantize_block(model_input, act, wt)
        elif act.dtype in ("float16", "bfloat16"):
            from onnxsim.onnx_simplifier import quantize_bf16, quantize_fp16

            fn = quantize_fp16 if act.dtype == "float16" else quantize_bf16
            result = fn(model_input)
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
        fn = _block_fn(wt.dtype)
        if fn is None:
            raise NotImplementedError(
                f"weight dtype {wt.dtype} with block-format activation {act.dtype} "
                "has no onnxsim backend"
            )
        if act.dtype in _BLOCK_DTYPES:
            self._approx(
                f"{act.dtype} activations are not quantized (Quark inserts runtime "
                "ops for them; onnxsim has none) -- weights only"
            )
        m = onnx.ModelProto()
        m.CopyFrom(model)
        inits = {t.name: t for t in m.graph.initializer}
        excluded = {e for e in self.config.exclude if isinstance(e, str)}
        done = set()
        for node in m.graph.node:
            if node.op_type not in ("MatMul", "Gemm", "Conv") or len(node.input) < 2:
                continue
            if node.name in excluded or node.output[0] in excluded:
                continue
            w = inits.get(node.input[1])
            if w is None or w.data_type != onnx.TensorProto.FLOAT or w.name in done:
                continue
            arr = onnx.numpy_helper.to_array(w)
            if arr.ndim < 2:
                continue
            if node.op_type == "Conv":
                axis = 1  # [O, I/g, k...]: input channels
            elif node.op_type == "Gemm":
                trans_b = any(a.name == "transB" and a.i for a in node.attribute)
                axis = 1 if trans_b else 0  # [N, K] vs [K, N]: the K axis
            else:
                axis = arr.ndim - 2  # MatMul B [..., K, N]: the K axis
            w.CopyFrom(onnx.numpy_helper.from_array(fn(arr, axis), w.name))
            done.add(w.name)
        return m

    def _approx(self, msg: str) -> None:
        self.last_approximations.append(msg)

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
        if act.dtype in ("int16", "uint16"):
            act_dtype = "uint16"
            if act.dtype == "int16":
                self._approx("int16 activations mapped to uint16 (asymmetric)")
        else:
            act_dtype = "uint8"
            if act.dtype == "int8":
                self._approx("int8 activations mapped to uint8 (asymmetric)")
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
        if "smooth_quant" in by_name:
            from onnxsim.smoothquant import apply_smoothquant

            alpha = by_name["smooth_quant"].params.get("alpha", 0.5)
            work = apply_smoothquant(work, calibration_data=calibration, alpha=alpha)
        if "cle" in by_name:
            from onnxsim.onnx_simplifier import cross_layer_equalize

            work = cross_layer_equalize(work)
        if work is not model:
            float_model = work

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
]
