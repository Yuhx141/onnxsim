"""QuaRot-style residual-stream rotation ("R1") of an ONNX model, shaped after
Quark's ``QuaRotConfig`` / ``rotation_transforms`` flow. Independent
implementation: Quark's source was read for the config schema and the
transformation, not copied.

This is **not** :func:`onnxsim.quarot.apply_quarot`, which inserts a runtime
activation rotation and quantizes to INT4. Here the rotation is folded
*offline into the weights*, so the float model computes exactly the same
function; what changes is the basis of the residual stream, whose outlier
channels get spread out -- a later quantization step then sees weights and
activations with far fewer outliers.

With an orthogonal ``R`` ([d, d]) and the stream as a row vector, rotating the
stream means ``x -> x R``. Per pair in the config's ``R1_pairs``:

- ``norm_node`` (optional): an RMSNorm-style node holding a per-channel scale
  ``gamma`` (and optionally a bias ``beta``). Since ``RMSNorm(x R) =
  RMSNorm(x) R`` only holds for ``gamma = 1``, ``gamma`` is folded into the
  ``next_nodes`` weights (input channels scaled by ``gamma``), ``beta`` into
  their bias (``b += beta @ W``), and the norm's scale is set to ones / its
  bias removed.
- ``prev_nodes`` *write* into the stream: their output channels are rotated
  (``W -> W R``, bias ``b -> b R``); a ``Gather`` embedding table ``[vocab, d]``
  is rotated on its last axis.
- ``next_nodes`` *read* from the stream: their input channels are rotated
  inversely (``W -> R^T W``).

A node must appear in exactly one pair as a writer (otherwise it would be
rotated twice); the config decides which pair owns it. Config schema (JSON, the
same keys as Quark's)::

    {"R1_pairs": [{"prev_nodes": ["embed", "mm_out"],
                   "next_nodes": ["mm_in"],
                   "norm_node": "norm1_scale"}, ...]}

Weight layouts: ``MatMul`` / ``Gemm(transB=0)`` weights are ``[in, out]``,
``Gemm(transB=1)`` is ``[out, in]``. The weight of a node is its constant float
input of rank >= 2 (``Gather``: the table); a norm's ``gamma`` / ``beta`` are
its constant rank-1 float inputs, in order. Only rotation "R1" exists here
(Quark's ONNX implementation has R2-R4 as TODO as well). The model must be
float32; the transformation is exact up to float rounding.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper


def make_rotation(
    dim: int,
    random_hadamard: bool = False,
    seed: int = 0,
    signs: Optional[np.ndarray] = None,
    orthogonal_fallback: bool = False,
) -> np.ndarray:
    """An orthogonal ``[dim, dim]`` float64 matrix, Quark's R1 matrix.

    Exactly Quark's: the normalized Hadamard matrix of :mod:`onnxsim.quark_hadamard`
    (Sylvester for a power of two, else ``kron(H_K, H_{dim/K})`` with the
    tabulated ``H_K`` for ``K`` in 12, 20, 28, 36, 40, 52, 60, 108, 140, 156,
    172). With ``random_hadamard`` its rows are multiplied by a random +-1
    vector (a "random Hadamard"): ``signs`` when given (the ``dim`` signs
    Quark draws with ``torch.randint``, which numpy cannot reproduce), else a
    seeded one. A ``dim`` without a Hadamard matrix raises ``ValueError`` as in
    Quark, unless ``orthogonal_fallback``: then the Q of a seeded Gaussian's QR
    decomposition (Haar-random orthogonal, not a Quark matrix).
    """
    from onnxsim.quark_hadamard import hadamard_rotation, supports

    if dim < 1:
        raise ValueError("dim must be >= 1")
    rng = np.random.default_rng(seed)
    if supports(dim):
        if random_hadamard and signs is None:
            signs = rng.choice([-1.0, 1.0], size=dim)
        return hadamard_rotation(dim, signs if random_hadamard else None)
    if not orthogonal_fallback:
        # same message as Quark's `Could not find an Hadamard matrix ...`
        hadamard_rotation(dim)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))[None, :]


def _check_rotation(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"rotation must be a square matrix, got shape {r.shape}")
    if not np.allclose(r.T @ r, np.eye(r.shape[0]), atol=1e-6):
        raise ValueError("rotation matrix is not orthogonal")
    return r


class _Rotator:
    def __init__(self, model: onnx.ModelProto, r: np.ndarray) -> None:
        self.m = onnx.ModelProto()
        self.m.CopyFrom(model)
        self.r = r
        self.inits = {t.name: t for t in self.m.graph.initializer}
        self.nodes = {n.name: n for n in self.m.graph.node if n.name}
        self.d = r.shape[0]
        self.writers_done: set = set()

    # -- lookup ---------------------------------------------------------------

    def node(self, name: str) -> onnx.NodeProto:
        if name not in self.nodes:
            raise ValueError(f"node {name!r} not found in the model")
        return self.nodes[name]

    def _float_inits(self, node: onnx.NodeProto, rank: Optional[int]) -> List[str]:
        out = []
        for x in node.input:
            t = self.inits.get(x)
            if t is None or t.data_type != onnx.TensorProto.FLOAT:
                continue
            if rank is None:
                ok = True
            elif rank == 1:
                ok = len(t.dims) == 1
            else:
                ok = len(t.dims) >= rank
            if ok:
                out.append(x)
        return out

    def weight_name(self, node: onnx.NodeProto) -> str:
        if node.op_type == "Gather":
            if node.input[0] not in self.inits:
                raise ValueError(f"Gather {node.name!r} has no constant data table")
            return node.input[0]
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
            if node.input[1] in self.inits:
                return node.input[1]
            raise ValueError(
                f"{node.op_type} {node.name!r}: weight (input 1) is not constant"
            )
        cands = self._float_inits(node, 2)
        if not cands:
            raise ValueError(f"node {node.name!r} has no constant weight")
        return cands[0]

    def axes(self, node: onnx.NodeProto) -> Tuple[int, int]:
        """``(in_axis, out_axis)`` into the weight array."""
        if node.op_type == "Gather":
            return -2, -1
        if node.op_type == "Gemm" and any(
            a.name == "transB" and a.i for a in node.attribute
        ):
            return -1, -2
        return -2, -1

    def bias_name(self, node: onnx.NodeProto) -> Optional[str]:
        if (
            node.op_type == "Gemm"
            and len(node.input) >= 3
            and node.input[2] in self.inits
        ):
            return node.input[2]
        return None

    def get(self, name: str) -> np.ndarray:
        return numpy_helper.to_array(self.inits[name]).astype(np.float64)

    def put(self, name: str, arr: np.ndarray) -> None:
        self.inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    # -- the three transformations -------------------------------------------

    def rotate_reader(self, node: onnx.NodeProto) -> None:
        w = self.get(self.weight_name(node))
        in_axis, _ = self.axes(node)
        if w.shape[in_axis] != self.d:
            raise ValueError(
                f"{node.name!r}: input dimension {w.shape[in_axis]} != rotation size {self.d}"
            )
        # in_axis == -2: W [.., in, out] -> R^T W ;  in_axis == -1: W [out, in] -> W R.
        # The products are evaluated on the same operand layouts as Quark's (the
        # [in, out] case through a transposed view), so the float64 sums run in
        # the same BLAS order and round to the same float32 weights.
        if in_axis == -2:
            w2 = np.swapaxes(np.matmul(np.swapaxes(w, -1, -2), self.r), -1, -2)
        else:
            w2 = np.matmul(w, self.r)
        self.put(self.weight_name(node), w2)

    def rotate_writer(self, node: onnx.NodeProto, direct: bool = False) -> None:
        if node.name in self.writers_done:
            raise ValueError(
                f"{node.name!r} is a prev_node of more than one pair (it would be rotated twice)"
            )
        self.writers_done.add(node.name)
        wname = self.weight_name(node)
        w = self.get(wname)
        _, out_axis = self.axes(node)
        if w.shape[out_axis] != self.d:
            raise ValueError(
                f"{node.name!r}: output dimension {w.shape[out_axis]} != rotation size {self.d}"
            )
        # out_axis == -1: W [.., in, out] -> W R ;  out_axis == -2: W [out, in] -> R^T W
        if out_axis == -1 and direct:
            # Quark rotates the first pair's embedding table as W @ R, untransposed
            w2 = np.matmul(w, self.r)
        elif out_axis == -1:
            w2 = np.swapaxes(np.matmul(self.r.T, np.swapaxes(w, -1, -2)), -1, -2)
        else:
            w2 = np.matmul(self.r.T, w)
        self.put(wname, w2)
        b = self.bias_name(node)
        if b is not None:
            bias = self.get(b)
            if bias.shape[-1] == self.d:
                self.put(b, np.matmul(self.r.T, bias))

    def fold_norm(self, norm: onnx.NodeProto, nexts: List[onnx.NodeProto]) -> None:
        vecs = self._float_inits(norm, 1)
        if not vecs:
            raise ValueError(f"norm node {norm.name!r} has no constant scale")
        gamma_name = vecs[0]
        beta_name = vecs[1] if len(vecs) > 1 else None
        gamma = self.get(gamma_name)
        beta = self.get(beta_name) if beta_name else None
        if gamma.shape != (self.d,):
            raise ValueError(
                f"norm node {norm.name!r}: scale has shape {gamma.shape}, expected ({self.d},)"
            )
        for nxt in nexts:
            wname = self.weight_name(nxt)
            w = self.get(wname)
            in_axis, _ = self.axes(nxt)
            if beta is not None:
                # x_n W = (x_hat * gamma) W + beta W : the beta term needs the
                # *unscaled* weight.
                contrib = beta @ w if in_axis == -2 else w @ beta
                b = self.bias_name(nxt)
                if b is None:
                    raise ValueError(
                        f"norm node {norm.name!r} has a bias but {nxt.name!r} has no "
                        "constant bias to fold it into (use a Gemm with a bias)"
                    )
                self.put(b, self.get(b) + contrib)
            scale = gamma[:, None] if in_axis == -2 else gamma[None, :]
            self.put(wname, w * scale)
        self.put(gamma_name, np.ones_like(gamma))
        if beta_name is not None:
            self.put(beta_name, np.zeros_like(beta))  # type: ignore[arg-type]

    def run(self, pairs: List[Dict[str, Any]]) -> onnx.ModelProto:
        readers_seen: set = set()
        for pair_idx, pair in enumerate(pairs):
            prevs = [self.node(n) for n in pair.get("prev_nodes", [])]
            nexts = [self.node(n) for n in pair.get("next_nodes", [])]
            for n in nexts:
                if n.name in readers_seen:
                    raise ValueError(f"{n.name!r} is a next_node of more than one pair")
                readers_seen.add(n.name)
            if pair.get("norm_node"):
                self.fold_norm(self.node(pair["norm_node"]), nexts)
            for node_idx, n in enumerate(prevs):
                first_gather = pair_idx == 0 and node_idx == 0 and n.op_type == "Gather"
                self.rotate_writer(n, direct=first_gather)
            for n in nexts:
                self.rotate_reader(n)
        return self.m


def rotate_model(
    model: onnx.ModelProto,
    rotation_config: Union[str, Dict[str, Any]],
    r1: Optional[np.ndarray] = None,
    r_matrix_dim: Optional[int] = None,
    use_random_had: bool = False,
    seed: int = 0,
) -> onnx.ModelProto:
    """Fold an R1 rotation into ``model`` (see the module docstring).

    :param rotation_config: the config dict, or the path of a JSON file
    :param r1: the orthogonal rotation matrix; when omitted it is built with
            :func:`make_rotation` from ``r_matrix_dim`` / ``use_random_had`` /
            ``seed``
    """
    if isinstance(rotation_config, str):
        with open(rotation_config) as f:
            rotation_config = json.load(f)
    pairs = rotation_config.get("R1_pairs", [])  # type: ignore[union-attr]
    if r1 is None:
        if r_matrix_dim is None:
            raise ValueError("pass either r1 or r_matrix_dim")
        r1 = make_rotation(r_matrix_dim, use_random_had, seed)
    r = _check_rotation(r1)
    if r_matrix_dim is not None and r.shape[0] != r_matrix_dim:
        raise ValueError(
            f"r1 is {r.shape[0]}x{r.shape[0]}, r_matrix_dim={r_matrix_dim}"
        )
    return _Rotator(model, r).run(pairs)


__all__ = ["make_rotation", "rotate_model"]
