"""Automatic mixed precision for QDQ models, shaped after AMD Quark's
``AutoMixprecisionConfig`` flow (``quark.onnx.algorithm.mprecision``).
Independent implementation: Quark's source was read for the contract and the
metric definitions, not copied.

Given a float model, :func:`auto_mixprecision` quantizes it with
:func:`onnxsim.full_qdq.quantize_full_qdq` at a *base* activation precision,
then moves layers ("candidates") to a *target* precision one at a time:

1. **Baseline score**: ``metric(float_out, quantized_out)``; lower is better.
2. **Sensitivity**: each candidate is moved to the target precision *on its
   own*; its score is the metric of that model. Candidates are ranked
   ascending (closest to float first), for either objective.
3. **Greedy mixing**, walking the ranking and keeping candidates moved
   cumulatively. ``metric_threshold``:

   - ``None``: sensitivity analysis only, the baseline model is returned;
   - ``0`` (Quark's default): the threshold is disabled, every candidate moves;
   - ``optimize="speed"``: the baseline is the *higher* precision and the
     target the lower one -- keep moving candidates while the score stays
     ``<= threshold``, and undo the first one that pushes it above (stop);
   - ``optimize="quality"``: the baseline is the *lower* precision and the
     target the higher one -- keep moving candidates until the score drops
     to ``<= threshold`` (stop).

   With a non-zero threshold, "speed" returns the baseline unchanged if its
   score already exceeds the threshold, and "quality" if it already meets it.

A candidate is a node of one of ``target_op_types``; moving it to the target
precision sets the dtype of its float activation inputs and its output (and of
a directly-following Relu's output, which :func:`quantize_full_qdq` folds into
the output quantizer). Tensors at precision boundaries simply keep their own
Q/DQ pair, so no extra boundary nodes are needed.

**Scope.** Only the *activation* precision is mixed (``uint8`` <-> ``uint16``,
the two activation types :func:`quantize_full_qdq` supports); weights stay
symmetric int8. Layer-wise only (no subgraph partitioning), a single target
precision, serial analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import onnx

_EPS = 1e-10
_DTYPES = ("uint8", "uint16")

#: ``outputs[sample][output_index]`` arrays; what the metric functions score.
Outputs = List[List[np.ndarray]]
MetricFn = Callable[[Outputs, Outputs], float]


# -- metrics (lower is better) ---------------------------------------------------


def _pairs(float_out: Outputs, quant_out: Outputs):
    if len(float_out) != len(quant_out):
        raise ValueError(
            "float_out and quant_out must have the same number of samples, "
            f"got {len(float_out)} vs {len(quant_out)}"
        )
    for f_sample, q_sample in zip(float_out, quant_out):
        for f, q in zip(f_sample, q_sample):
            yield f, q


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def l2_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean L2 norm of the difference over all (sample, output) pairs."""
    return _mean(
        [
            float(np.linalg.norm(np.asarray(f, np.float32) - np.asarray(q, np.float32)))
            for f, q in _pairs(float_out, quant_out)
        ]
    )


def kl_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean KL(P_float || P_quant); each output is shifted to be non-negative
    and normalized to a distribution."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float64).ravel()
        q = np.asarray(q, np.float64).ravel()
        f = f - min(f.min(), 0.0)
        q = q - min(q.min(), 0.0)
        p = f / f.sum() if f.sum() > 0 else np.ones_like(f) / f.size
        r = q / q.sum() if q.sum() > 0 else np.ones_like(q) / q.size
        vals.append(float(np.sum(p * np.log((p + _EPS) / (r + _EPS)))))
    return _mean(vals)


def cosine_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean cosine distance ``1 - cos_sim`` (0 when either norm is ~0)."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float32).ravel()
        q = np.asarray(q, np.float32).ravel()
        nf, nq = float(np.linalg.norm(f)), float(np.linalg.norm(q))
        sim = 1.0 if nf < _EPS or nq < _EPS else float(np.dot(f, q) / (nf * nq))
        vals.append(1.0 - sim)
    return _mean(vals)


def sqnr_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean *negative* SQNR in dB (use a negative threshold, e.g. ``-30``)."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float32)
        q = np.asarray(q, np.float32)
        signal = max(float(np.mean(f**2)), _EPS)
        noise = max(float(np.mean((f - q) ** 2)), _EPS)
        vals.append(-10.0 * float(np.log10(signal / noise)))
    return _mean(vals)


def psnr_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean *negative* PSNR in dB (peak = max |float output|)."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float32)
        q = np.asarray(q, np.float32)
        mse = float(np.mean((f - q) ** 2)) or _EPS
        peak = float(np.max(np.abs(f))) or _EPS
        vals.append(-(20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(mse))))
    return _mean(vals)


BUILTIN_METRICS: Dict[str, MetricFn] = {
    "l2": l2_metric,
    "kl": kl_metric,
    "cosine": cosine_metric,
    "sqnr": sqnr_metric,
    "psnr": psnr_metric,
}


def resolve_metric(
    metric: str = "l2",
    distance_fn: Optional[MetricFn] = None,
    evaluate_fn: Optional[Callable[[Outputs], float]] = None,
) -> MetricFn:
    """``distance_fn`` (lower is better) wins over ``evaluate_fn`` (higher is
    better, adapted to ``evaluate(float) - evaluate(quant)``) over the named
    built-in ``metric``. Giving both callables is an error."""
    if distance_fn is not None and evaluate_fn is not None:
        raise ValueError("distance_fn and evaluate_fn are mutually exclusive")
    if distance_fn is not None:
        return distance_fn
    if evaluate_fn is not None:
        return lambda f, q: float(evaluate_fn(f)) - float(evaluate_fn(q))
    if metric not in BUILTIN_METRICS:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(BUILTIN_METRICS)}")
    return BUILTIN_METRICS[metric]


# -- the algorithm -----------------------------------------------------------------


@dataclass
class SensitivityResult:
    """One candidate's score when moved to the target precision on its own."""

    name: str
    nodes: List[str]
    tensors: List[str]
    score: float
    enabled: bool = True


@dataclass
class AutoMixprecisionResult:
    model: onnx.ModelProto
    baseline_score: float
    final_score: float
    ranked: List[SensitivityResult] = field(default_factory=list)
    moved: List[str] = field(default_factory=list)  # candidate names, in order
    threshold_reached: bool = False


def _candidates(
    model: onnx.ModelProto,
    target_op_types: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
) -> List[SensitivityResult]:
    g = model.graph
    inits = {t.name for t in g.initializer}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    outputs = {o.name for o in g.output}
    out: List[SensitivityResult] = []
    for n in g.node:
        if n.op_type not in target_op_types or not n.output:
            continue
        ids = {n.name, n.output[0]} - {""}
        if include and not (ids & set(include)):
            continue
        if ids & set(exclude):
            continue
        tensors = [x for x in n.input if x and x not in inits]
        tensors.append(n.output[0])
        users = consumers.get(n.output[0], [])
        if (
            len(users) == 1
            and users[0].op_type == "Relu"
            and n.output[0] not in outputs
        ):
            tensors.append(users[0].output[0])  # full_qdq folds the Relu in
        name = n.name or n.output[0]
        out.append(SensitivityResult(name, [name], tensors, float("nan")))
    return out


def auto_mixprecision(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    base_dtype: str,
    target_dtype: str,
    target_op_types: Sequence[str] = ("Conv", "Gemm", "MatMul"),
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
    exclude_nodes: Sequence[str] = (),
    metric: str = "l2",
    metric_distance_fn: Optional[MetricFn] = None,
    metric_evaluate_fn: Optional[Callable[[Outputs], float]] = None,
    metric_threshold: Optional[float] = 0.0,
    optimize: str = "speed",
    metric_output_index: Optional[int] = 0,
    data_size: int = 0,
    per_channel: bool = True,
    method: str = "minmax",
    providers: Optional[Sequence[str]] = None,
) -> AutoMixprecisionResult:
    """Mixed-precision quantization of ``model`` (see the module docstring).

    :param exclude_nodes: nodes kept in float by the quantizer itself (they are
            not quantized at either precision, unlike ``exclude_layers``, which
            only stops a layer from being a mixing candidate)
    :param base_dtype: activation precision of the starting model
            (``"uint8"`` or ``"uint16"``)
    :param target_dtype: precision candidates move to (the other one)
    :param metric_output_index: model output scored by the metric; ``None``
            scores every output
    :param data_size: use only the first ``data_size`` calibration batches for
            scoring (``0`` = all)
    """
    if base_dtype not in _DTYPES or target_dtype not in _DTYPES:
        raise ValueError(f"dtypes must be in {_DTYPES}")
    if base_dtype == target_dtype:
        raise ValueError("base_dtype and target_dtype must differ")
    if optimize not in ("speed", "quality"):
        raise ValueError("optimize must be 'speed' or 'quality'")
    if not calibration_data:
        raise ValueError("calibration_data is required")
    metric_fn = resolve_metric(metric, metric_distance_fn, metric_evaluate_fn)

    import onnxruntime as ort

    from onnxsim.calibration import calibrate
    from onnxsim.full_qdq import quantize_full_qdq

    prov = list(providers) if providers else ["CPUExecutionProvider"]
    eval_data = list(calibration_data)[: data_size or None]

    def run(m: onnx.ModelProto) -> Outputs:
        sess = ort.InferenceSession(m.SerializeToString(), providers=prov)
        res: Outputs = []
        for batch in eval_data:
            outs = sess.run(None, batch)
            res.append(
                [outs[metric_output_index]] if metric_output_index is not None else outs
            )
        return res

    float_out = run(model)

    # Calibrate once: every float activation tensor, reused by every trial.
    inits = {t.name for t in model.graph.initializer}
    acts = [i.name for i in model.graph.input if i.name not in inits]
    acts += [o for n in model.graph.node for o in n.output]
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=acts,
    )

    def quantize(moved_tensors: Set[str]) -> onnx.ModelProto:
        return quantize_full_qdq(
            model,
            calibration_data=calibration_data,
            activation_dtype=base_dtype,
            per_channel=per_channel,
            exclude_nodes=exclude_nodes,
            method=method,
            providers=providers,
            ranges=ranges,
            tensor_dtypes={t: target_dtype for t in moved_tensors} or None,
        )

    def score_of(moved_tensors: Set[str]):
        q = quantize(moved_tensors)
        return q, metric_fn(float_out, run(q))

    baseline, baseline_score = score_of(set())
    result = AutoMixprecisionResult(baseline, baseline_score, baseline_score)

    cands = _candidates(model, target_op_types, include_layers, exclude_layers)
    for c in cands:
        c.score = score_of(set(c.tensors))[1]
    result.ranked = sorted(cands, key=lambda c: c.score)

    if metric_threshold is None or not result.ranked:
        return result
    if metric_threshold != 0:
        if optimize == "speed" and baseline_score > metric_threshold:
            return result  # already past the threshold: no room to optimize
        if optimize == "quality" and baseline_score <= metric_threshold:
            return result  # already good enough

    moved: Set[str] = set()
    cur_model, cur_score = baseline, baseline_score
    for c in result.ranked:
        if not c.enabled:
            continue
        trial_model, score = score_of(moved | set(c.tensors))
        if metric_threshold == 0:
            keep, stop = True, False
        elif optimize == "speed":
            keep, stop = score <= metric_threshold, score > metric_threshold
        else:
            keep, stop = True, score <= metric_threshold
        if keep:
            moved |= set(c.tensors)
            result.moved.append(c.name)
            cur_model, cur_score = trial_model, score
        if stop:
            result.threshold_reached = optimize == "quality"
            break
    result.model, result.final_score = cur_model, cur_score
    return result


__all__ = [
    "AutoMixprecisionResult",
    "BUILTIN_METRICS",
    "SensitivityResult",
    "auto_mixprecision",
    "cosine_metric",
    "kl_metric",
    "l2_metric",
    "psnr_metric",
    "resolve_metric",
    "sqnr_metric",
]
