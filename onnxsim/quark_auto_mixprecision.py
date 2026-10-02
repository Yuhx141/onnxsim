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

**Forms of the target.** ``targets`` is a list of ``(dtype, symmetric)``
precisions; with one entry every candidate moves to it, with several each
candidate is scored under every one and moves to the best-scoring (Quark's
list-of-``QLayerConfig`` mode); ``candidate_targets`` pins named candidates to
an entry (Quark's ``{QLayerConfig: [names]}`` form -- candidates it does not
name use entry 0). ``subgraphs`` (Quark's ``subgraph_json``, see
:func:`parse_subgraph_json`) makes each group of nodes one candidate that is
scored and moved together. ``cache_file`` stores the sensitivity ranking in
Quark's JSON schema (a candidate's ``"enabled": false`` pins it) and reuses it
while the model / configuration fingerprint matches; ``worker_num`` scores
candidates on that many threads; ``no_input_qdq_shared`` keeps nodes whose
input activation is read by several nodes out of the mixing step.

**Scope.** Only the *activation* precision is mixed (the integer activation
types :func:`quantize_full_qdq` supports); weights stay symmetric int8.
``dual_quant_nodes`` adds a converting Q/DQ pair in front of every consumer
whose precision differs from the tensor's (Quark's boundary insertion, but with
onnxsim's own scale / zero point for the pair); ``shared_param_mode`` has no
meaning here (every tensor has its own scale / zero point initializers).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx

_EPS = 1e-10
_DTYPES = ("int8", "uint8", "int16", "uint16")

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


#: ``(activation dtype, symmetric or None to keep the model's setting)``
Target = Tuple[str, Optional[bool]]


@dataclass
class SensitivityResult:
    """One candidate's score when moved to a target precision on its own."""

    name: str
    nodes: List[str]
    tensors: List[str]
    score: float
    enabled: bool = True
    #: score under each entry of ``targets`` (``score`` is their minimum)
    all_config_scores: List[float] = field(default_factory=list)
    best_config_index: int = 0


@dataclass
class AutoMixprecisionResult:
    model: onnx.ModelProto
    baseline_score: float
    final_score: float
    ranked: List[SensitivityResult] = field(default_factory=list)
    moved: List[str] = field(default_factory=list)  # candidate names, in order
    threshold_reached: bool = False


# -- subgraph partitions (Quark's ``subgraph_json``) ---------------------------------


@dataclass
class SubgraphSpec:
    name: str
    start_nodes: List[str]
    end_nodes: List[str]
    resolved_nodes: List[str] = field(default_factory=list)


def _reach(
    model: onnx.ModelProto, starts: Sequence[str], ends: Sequence[str]
) -> List[str]:
    """Nodes reachable from ``starts``, not walking past an end node."""
    nodes = {n.name: n for n in model.graph.node}
    readers: Dict[str, List[str]] = {}
    for n in model.graph.node:
        for x in n.input:
            readers.setdefault(x, []).append(n.name)
    stop = set(ends)
    seen: List[str] = []
    todo = list(starts)
    while todo:
        name = todo.pop(0)
        if name in seen or name not in nodes:
            continue
        seen.append(name)
        if name in stop:
            continue
        for o in nodes[name].output:
            todo += [r for r in readers.get(o, []) if r not in seen]
    return seen


def parse_subgraph_json(
    path: Union[str, Path], float_model: onnx.ModelProto, quant_model: onnx.ModelProto
) -> List[SubgraphSpec]:
    """Quark's subgraph partition file::

        {"quantized": false, "num_subgraphs": 2,
         "subgraphs": [{"name": "a", "start_nodes": [...], "end_nodes": [...]}]}

    Each subgraph is every node reachable from its start nodes up to its end
    nodes (on ``float_model``, or on ``quant_model`` when ``"quantized"`` is
    true); nodes of a later subgraph already claimed by an earlier one are
    dropped from it; every remaining node forms a final ``__ungrouped__``
    entry. Unknown boundary nodes and a wrong ``num_subgraphs`` raise."""
    data = json.loads(Path(path).read_text())
    entries = data.get("subgraphs", [])
    declared = data.get("num_subgraphs")
    if declared is not None and declared != len(entries):
        raise ValueError(
            f"num_subgraphs={declared} does not match the number of subgraph "
            f"entries ({len(entries)})"
        )
    quantized = bool(data.get("quantized", False))
    f_names = {n.name for n in float_model.graph.node}
    q_names = {n.name for n in quant_model.graph.node}
    specs: List[SubgraphSpec] = []
    assigned: Set[str] = set()
    for e in entries:
        starts, ends = list(e["start_nodes"]), list(e["end_nodes"])
        known = q_names if quantized else f_names
        for n in starts + ends:
            if n not in known:
                raise ValueError(
                    f"subgraph {e['name']!r}: node {n!r} not found in the "
                    f"{'quantized' if quantized else 'float'} model"
                )
        resolved = _reach(quant_model if quantized else float_model, starts, ends)
        resolved = [n for n in resolved if n in q_names or quantized]
        resolved = [n for n in resolved if n not in assigned]
        assigned.update(resolved)
        specs.append(SubgraphSpec(e["name"], starts, ends, resolved))
    rest = [n.name for n in float_model.graph.node if n.name not in assigned]
    if rest:
        specs.append(SubgraphSpec("__ungrouped__", [], [], rest))
    return specs


# -- candidates ----------------------------------------------------------------------


def _node_key(n: onnx.NodeProto) -> str:
    return n.name or (n.output[0] if n.output else "")


def _node_tensors(model: onnx.ModelProto) -> Dict[str, List[str]]:
    """Per node: the float activation tensors whose precision moves with it
    (its activation inputs and output, plus a directly-following Relu's output,
    which :func:`quantize_full_qdq` folds into the output quantizer)."""
    g = model.graph
    inits = {t.name for t in g.initializer}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    outputs = {o.name for o in g.output}
    out: Dict[str, List[str]] = {}
    for n in g.node:
        if not n.output:
            continue
        tensors = [x for x in n.input if x and x not in inits]
        tensors.append(n.output[0])
        users = consumers.get(n.output[0], [])
        if (
            len(users) == 1
            and users[0].op_type == "Relu"
            and n.output[0] not in outputs
        ):
            tensors.append(users[0].output[0])
        out[_node_key(n)] = tensors
    return out


def _candidate_nodes(
    model: onnx.ModelProto,
    target_op_types: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
    restrict: Optional[Sequence[str]] = None,
) -> List[str]:
    out: List[str] = []
    for n in model.graph.node:
        if n.op_type not in target_op_types or not n.output:
            continue
        ids = {n.name, n.output[0]} - {""}
        if restrict is not None and _node_key(n) not in restrict:
            continue
        if include and not (ids & set(include)):
            continue
        if ids & set(exclude):
            continue
        out.append(_node_key(n))
    return out


def _fingerprint(
    model: onnx.ModelProto,
    base_dtype: str,
    targets: Sequence[Target],
    candidate_targets: Dict[str, Target],
    target_op_types: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]],
) -> str:
    """Hash of everything a ranking depends on structurally (graph topology,
    precisions, op filter, partition) -- not the weight values."""
    parts = [
        f"{n.name}|{n.op_type}|{','.join(n.input)}|{','.join(n.output)}"
        for n in sorted(model.graph.node, key=lambda n: (n.name, n.op_type))
    ]
    parts.append(f"base:{base_dtype}")
    parts.append(f"targets:{json.dumps(list(targets))}")
    parts.append(f"pinned:{json.dumps(candidate_targets, sort_keys=True)}")
    parts.append("ops:" + ",".join(sorted(target_op_types)))
    parts.append("include:" + ",".join(sorted(include)))
    parts.append("exclude:" + ",".join(sorted(exclude)))
    if subgraphs is not None:
        parts.append(json.dumps([[a, list(b)] for a, b in subgraphs]))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def save_sensitivity(
    ranked: Sequence[SensitivityResult], path: Union[str, Path], key: str
) -> None:
    """Write ``ranked`` in Quark's sensitivity-cache JSON schema."""
    payload = {
        "version": "onnxsim",
        "cache_key": key,
        "results": [
            {
                "name": r.name,
                "candidate_nodes": r.nodes,
                "score": r.score,
                "all_config_scores": r.all_config_scores,
                "best_config_index": r.best_config_index,
                "enabled": r.enabled,
            }
            for r in ranked
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_sensitivity(
    path: Union[str, Path],
    key: str,
    node_tensors: Dict[str, List[str]],
) -> Optional[List[SensitivityResult]]:
    """Read a cache written by :func:`save_sensitivity` (or by Quark); ``None``
    -- with a warning -- when its fingerprint is not ``key``. A Quark-written
    file therefore never matches (the fingerprint covers the quantized graph
    layout) and is recomputed."""
    payload = json.loads(Path(path).read_text())
    if payload.get("cache_key") != key:
        warnings.warn(
            f"sensitivity cache {path} is stale (model or configuration "
            "changed); recomputing",
            UserWarning,
            stacklevel=3,
        )
        return None
    out = []
    for e in payload["results"]:
        tensors: List[str] = []
        for n in e["candidate_nodes"]:
            tensors += [t for t in node_tensors.get(n, []) if t not in tensors]
        out.append(
            SensitivityResult(
                e["name"],
                list(e["candidate_nodes"]),
                tensors,
                float(e["score"]),
                bool(e.get("enabled", True)),
                [float(x) for x in e.get("all_config_scores", [])],
                int(e.get("best_config_index", 0)),
            )
        )
    return out


def auto_mixprecision(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    base_dtype: str,
    target_dtype: Optional[str] = None,
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
    targets: Optional[Sequence[Target]] = None,
    candidate_targets: Optional[Dict[str, Target]] = None,
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    cache_file: Optional[Union[str, Path]] = None,
    worker_num: int = 1,
    no_input_qdq_shared: bool = False,
    dual_quant_nodes: bool = False,
    quantize_kwargs: Optional[Dict[str, Any]] = None,
) -> AutoMixprecisionResult:
    """Mixed-precision quantization of ``model`` (see the module docstring).

    :param exclude_nodes: nodes kept in float by the quantizer itself (they are
            not quantized at either precision, unlike ``exclude_layers``, which
            only stops a layer from being a mixing candidate)
    :param base_dtype: activation precision of the starting model
    :param target_dtype: precision candidates move to (a single target; give
            ``targets`` instead for several)
    :param targets: ``(dtype, symmetric)`` entries (``symmetric=None`` keeps
            the model's own setting); several entries -> each candidate takes
            the best-scoring one
    :param candidate_targets: ``{candidate node name: (dtype, symmetric)}``
            pinned targets, never scored; other nodes use their candidate's
            best entry of ``targets``
    :param subgraphs: ``[(name, [node names])]``: each group is one candidate
    :param cache_file: JSON file for the sensitivity ranking: reused when its
            fingerprint matches, written otherwise
    :param worker_num: threads used to score candidates
    :param dual_quant_nodes: re-quantize (an extra Q/DQ pair) a tensor for a
            consumer whose precision differs; off, such a node consumes one
            precision and produces the other, as Quark does
    :param quantize_kwargs: keyword arguments of the :func:`quantize_full_qdq`
            call that makes the baseline (``calibration_data`` / ``activation_dtype``
            / ``ranges`` / ``convert_inputs`` are taken from here); its own
            ``tensor_dtypes`` / ``tensor_symmetric`` apply to every trial. Default:
            ``per_channel`` / ``exclude_nodes`` / ``method`` only
    :param no_input_qdq_shared: skip nodes whose first activation input is
            read by more than one node in the mixing step
    :param metric_output_index: model output scored by the metric; ``None``
            scores every output
    :param data_size: use only the first ``data_size`` calibration batches for
            scoring (``0`` = all)
    """
    if targets is None:
        if target_dtype is None:
            raise ValueError("target_dtype or targets is required")
        targets = [(target_dtype, None)]
        if base_dtype == target_dtype:
            raise ValueError("base_dtype and target_dtype must differ")
    targets = list(targets)
    if not targets:
        raise ValueError("targets must not be empty")
    for dt, _ in [(base_dtype, None)] + targets:
        if dt not in _DTYPES:
            raise ValueError(f"dtypes must be in {_DTYPES}")
    pinned: Dict[str, Target] = dict(candidate_targets or {})
    for dt, _ in pinned.values():
        if dt not in _DTYPES:
            raise ValueError(f"dtypes must be in {_DTYPES}")
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
        method=(quantize_kwargs or {}).get("method", method),
        extra_tensor_names=acts,
    )

    node_tensors = _node_tensors(model)

    def assign(nodes: Sequence[str], default: int) -> Dict[str, Target]:
        out: Dict[str, Target] = {}
        for n in nodes:
            tgt = pinned[n] if n in pinned else targets[default]  # type: ignore[index]
            for t in node_tensors.get(n, []):
                out[t] = tgt
        return out

    base_kw: Dict[str, Any] = dict(
        quantize_kwargs
        if quantize_kwargs is not None
        else dict(per_channel=per_channel, exclude_nodes=exclude_nodes, method=method)
    )
    for k in ("calibration_data", "activation_dtype", "ranges", "convert_inputs"):
        base_kw.pop(k, None)
    base_td = dict(base_kw.pop("tensor_dtypes", None) or {})
    base_ts = dict(base_kw.pop("tensor_symmetric", None) or {})

    def quantize(moved: Dict[str, Target]) -> onnx.ModelProto:
        sym = {**base_ts, **{t: s for t, (_, s) in moved.items() if s is not None}}
        dts = {**base_td, **{t: d for t, (d, _) in moved.items()}}
        return quantize_full_qdq(
            model,
            calibration_data=calibration_data,
            activation_dtype=base_dtype,
            providers=providers,
            ranges=ranges,
            tensor_dtypes=dts or None,
            tensor_symmetric=sym or None,
            convert_inputs=dual_quant_nodes,
            **base_kw,
        )

    def score_of(moved: Dict[str, Target]):
        q = quantize(moved)
        return q, metric_fn(float_out, run(q))

    baseline, baseline_score = score_of({})
    result = AutoMixprecisionResult(baseline, baseline_score, baseline_score)

    # -- sensitivity: cached, or each candidate under each target
    ops = tuple(target_op_types)
    if subgraphs is not None:
        groups = [
            (name, _candidate_nodes(model, ops, include_layers, exclude_layers, nodes))
            for name, nodes in subgraphs
        ]
    else:
        groups = [
            (n, [n])
            for n in _candidate_nodes(model, ops, include_layers, exclude_layers)
        ]
    groups = [(name, nodes) for name, nodes in groups if nodes]
    key = _fingerprint(
        model,
        base_dtype,
        targets,
        pinned,
        ops,
        include_layers,
        exclude_layers,
        subgraphs,
    )
    ranked: Optional[List[SensitivityResult]] = None
    if cache_file is not None and Path(cache_file).exists():
        ranked = load_sensitivity(cache_file, key, node_tensors)
    if ranked is None:

        def score_group(item: Tuple[str, List[str]]) -> SensitivityResult:
            name, nodes = item
            scores = [score_of(assign(nodes, i))[1] for i in range(len(targets))]
            best = min(range(len(scores)), key=lambda i: scores[i])
            tensors = list(
                dict.fromkeys(t for n in nodes for t in node_tensors.get(n, []))
            )
            return SensitivityResult(
                name, nodes, tensors, scores[best], True, scores, best
            )

        workers = max(int(worker_num), 1)
        if workers > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                scored = list(ex.map(score_group, groups))
        else:
            scored = [score_group(g) for g in groups]
        ranked = sorted(scored, key=lambda c: c.score)
        if cache_file is not None:
            save_sensitivity(ranked, cache_file, key)
    result.ranked = ranked

    if metric_threshold is None or not result.ranked:
        return result
    if metric_threshold != 0:
        if optimize == "speed" and baseline_score > metric_threshold:
            return result  # already past the threshold: no room to optimize
        if optimize == "quality" and baseline_score <= metric_threshold:
            return result  # already good enough

    shared = _shared_inputs(model) if no_input_qdq_shared else set()
    moved: Dict[str, Target] = {}
    cur_model, cur_score = baseline, baseline_score
    for c in result.ranked:
        if not c.enabled:
            continue
        nodes = [n for n in c.nodes if n not in shared]
        if not nodes:
            continue
        trial_model, score = score_of({**moved, **assign(nodes, c.best_config_index)})
        if metric_threshold == 0:
            keep, stop = True, False
        elif optimize == "speed":
            keep, stop = score <= metric_threshold, score > metric_threshold
        else:
            keep, stop = True, score <= metric_threshold
        if keep:
            moved.update(assign(nodes, c.best_config_index))
            result.moved.append(c.name)
            cur_model, cur_score = trial_model, score
        if stop:
            result.threshold_reached = optimize == "quality"
            break
    result.model, result.final_score = cur_model, cur_score
    return result


def _shared_inputs(model: onnx.ModelProto) -> Set[str]:
    """Nodes whose first activation input is read by more than one node."""
    inits = {t.name for t in model.graph.initializer}
    readers: Dict[str, int] = {}
    for n in model.graph.node:
        for x in set(n.input):
            readers[x] = readers.get(x, 0) + 1
    out: Set[str] = set()
    for n in model.graph.node:
        first = next((x for x in n.input if x and x not in inits), None)
        if first is not None and readers.get(first, 0) > 1:
            out.add(_node_key(n))
    return out


__all__: Any = [
    "AutoMixprecisionResult",
    "BUILTIN_METRICS",
    "SensitivityResult",
    "SubgraphSpec",
    "auto_mixprecision",
    "cosine_metric",
    "kl_metric",
    "l2_metric",
    "load_sensitivity",
    "parse_subgraph_json",
    "psnr_metric",
    "resolve_metric",
    "save_sensitivity",
    "sqnr_metric",
]
