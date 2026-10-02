"""Hadamard rotation matrices, matching AMD Quark's R1 rotation matrix.

Quark builds the Quarot R1 matrix from ``scipy.linalg.hadamard`` (the Sylvester
construction) when the hidden size is a power of two, and otherwise from a
table of known Hadamard matrices of order 12, 20, 28, 36, 40, 52, 60, 108, 140,
156 and 172 (the "Sloane" library): for a size ``n = K * 2**j`` it takes the
*largest* tabulated ``K`` that divides ``n`` with a power-of-two quotient, and
the rotation is ``kron(H_K, H_{2**j}) / sqrt(n)``. Any other size has no
Hadamard matrix there, and Quark raises ``Could not find an Hadamard matrix for
the size n=...``.

The tabulated matrices are specific +-1 matrices (Paley / Williamson / Sloane
constructions, each with its own row/column ordering and signs), so matching
Quark bit-for-bit needs those exact matrices. They are stored here as
bit-packed, zlib-compressed, base64 tables (one bit per entry, ``1`` = ``+1``)
that were checked against Quark's own matrices; every one is verified to
satisfy ``H @ H.T == n * I`` when decoded.
"""

from __future__ import annotations

import base64
import sys
import zlib
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

# fmt: off
_TABLES: Dict[int, str] = {
    12: (
        "eNpr4JV98dr9lt3Fb91b5q54ezzLGgBe0AqB"
    ),
    20: (
        "eNoBMgDN/4AADYV57CvLYV6bCvzYV+bCvzYV+bCrzYXebCrzYdebCrzYlebIrzaFebwrzeFeawrzp18b"
        "rg=="
    ),
    28: (
        "eNoBYgCd///9//7Daw32G1hrsOrD3YdWHuw6sLdi1Ym7JqyN2jVobuGrw3cNXhu4arDew12G9hp//AAK"
        "w0Ty1hgnmrDRPNWEie6sBE+1YSJ5qxkTjVnImGreRMNU8i4ah5Gw1TyNhonkso80+Q=="
    ),
    36: (
        "eNoBogBd////3///Ri7Ri/oxdoxf0Yu0Yr6MbaMd9GNtGK+jK2jJfRpbRovo4to8X0cW0eL6OLaLF9LF"
        "tJi+pi2oxfYxbcYvsYtqMX6MW9GL9GLejF+jFn//wAALRiwudNoxQXOu0YgLnbaMUFzttGCC562jFBc5"
        "bRmguYto3QXMW0ToLuLaB0F7FtE6C5i2mdBYxbXOgsYtjnQaMW1zoNGLS50OjFhc6OBjU+s="
    ),
    40: (
        "eNoByAA3/4AACAAA2FedhXnsK87CvLYV62Femwr5sK/NhXzYV+bCvmwr82FfNhX5sK+bCrzYW82F3mwt"
        "5sKvNhrzYdebDXmwq82KvNiV5slebIrzaK82hXm4V5vCvNwrzeFebhXmsK87CvOAAAf//9hXknqG7CvB"
        "PUO2FeSeoZsK9k9QzYVzJ6jmwrGT1PNhUMnq+bCgZPW82FQyet5sIhk9rzYVDJ7XmwKGT6vNhUMnlebG"
        "oZOK82dQyYV5t6hkwrzT1DLhXmHqGbCvNPUMxD5tnQ=="
    ),
    52: (
        "eNoBUgGt/qeUfjTLCQU8o/OmSEip4R+dMkJFTwj+6YISKnxH10wQmVPiPLpwhMqfEeXShC5U+I8uhCHy"
        "p8RZdSEHlX4gy6kIPKvxJlxISeUfiTLiQk8o/GmWEhcD08nt+mW4Gp5vb9MtwNTze26Z7gKnu9t0x3AV"
        "P97LpjuEqd72XTHcZU73sukO5yp3vZdAd3lRvey6A7vKre5l2B2eV29zLsDs8rt6mX4HJ5fb1Mss1CQp"
        "5OBxZqEhTzcDizEJKnm4FFmISVPdwKLIQkqe7g0WQhJU53BoshCypjuDRZCHlSHcmiyEPKgO7NEkIeVA"
        "d2aJIQ8rA7s0CQp5WB1ZoEhTy8Dnt5ZqPxTyvbizUfinne3FmI/VPO9qLMR+qe97UWYj5U973osxHyp7"
        "3rRbiPlT3vGi/EfKlvfNF+I+VLe+aL8Q8q29s0X4h5Xt7ZoPxTyvbyzQfinlGjWqww=="
    ),
    60: (
        "eNot0T1LQgEUh/E2hwpdBCfFQQhd+gLaEC1+gBxCEKeLODW4KCIIers0hHC3UBCEQIqEEES8INXgFC4l"
        "vjQJKqaDQiKIlOfx7D84PP/UgdyxtXLk8LaC40c1tJ77pxfRc9PHdaDrCdt88ZzRvrI0l339V+tkzPlh"
        "+eXy7TSxcg2KVWd6VHPPwF8KuA7WwfdP4FJM8MQDbgUFG37wIiB4mgP3wckheOUSXKiBG+DJHPz/5w5v"
        "4+Al+CwP3oAPR+CCV7C6BmfBig/cBNvN4OeE4Js0+NYh+CQEfjUJ1mzgO7CSAUuhmPbtBJO32FDBP5JX"
        "KYfBPcnbdXfAn3tcBevgB/LO3tkmwjb1P48t6DY="
    ),
    108: (
        "eNpV1M0rw3EcwHGJ8tBKDitpagdF0dSiVstBu1GSHVZqU8rDxEEpRWNCm1FymWmbcvAUoTyU2Dzl8hsS"
        "LivhYLY5WCTUks3D9v59/4FXfT+fz3sgBU+izu6uyVl3CLflGq8kdNew1Dg8dqb/XCr8aK9NvalPK2nu"
        "ej+tUxz5e6ThKXmT0u4xWk0Fl/vb2oPsPOfeq7Y6suVO35nL0PkenqdnDdaNSOl1WdVJi6LvOjc6Pm/u"
        "XByUKlW9gQV7sIPYURuwr3xiEWL9I8SiFcTuiXklwFwfxFaJFZuICcRGDcQmiS3YiR0Ti/0VsAsFMSux"
        "q2pis8Rif5XE3CJsUwMsJMLig0lgX0Zi8cEkMO80sSixpwCxny34x0yfxN6JGTzE9oi5nomt9gEL9RIT"
        "iFXqib0QE+zEnMTUD8QeFcAEFTEHMfUZsUAXMJuSmINYq4/Y337/YnolsXVisjFiy83ArE3EbHnAinTE"
        "DollSYlNELMMExsqAdYqJ5a43DjmziC2VgUscbk/mC8HWDITcewyjdgUsSsRNkcsmYkY5hFhbzXA2kUY"
        "mtR2pw8TQ5P8PbIdYivEMjuJ2RDAeUsDsUFiQSkxBtBvSifGAFbMmIntsraLrG3HOWtrZG3zvwGi9+S8"
    ),
    140: (
        "eNpl1s9L03Ecx/EuE/rODoqTBAkKdvA7xOmh0zAQD23egrWDsGUIYrMuHmL4jWKxmNBBpOUgSQo95JBF"
        "oiK6jaWHySaDwJa/ttlhelgifEE2WtKUtvw+v5+/4AGfN6/X68UVvGue3uVXtR79iS36eKTT5fW58uJ+"
        "wtjf0uYOa53bcqI7EO+7f9z38qZZ0p302pPvH+rt74Sl4domoScbdO+cSnKdWPphXfE/iN1zaKTG1Pyh"
        "f9SW+zq3t1bTNV4wZBdDU6Z1S+S27knuekfDRNShWcgXc7esxc+Bp8bo2HRYKLQGB5Lt9YFNYtJDwMzK"
        "xKRUmFVitlSYIDE+FzDNKsyMHRizSMyoDZhmCzEfrcBs1BPjJcbUTcxPPTDxOmL8xJTWiSn/gRJzp52Y"
        "IjHxBDHnB6HAlGRiDol5biKmqMIkiQm7gBmUiUkSk5aImSfmzxQxeWIiA8T87gQmsE1MmphHp8SkiEmH"
        "iFkgJhQk5tcIMAYnMbt2YIQdYt4S410kxkPMUSsxlRCpYs60xFRCpIqJuIkpNQITyBKT0QAjFohJqTBh"
        "YnzEeIPEeCRgjgzEVBOtgjkTiPkfr/8wETcxJR0whiwxGQ0wlgIxE8RMhomZIeZTGzEbEjBiDzHfHMSM"
        "E/OGmLvTxFwungvMZAsxH8zADArEXC6eC8yNLmK+NABzdYyY18TE+olRtOA5xtRETJYYbQ0x/g5gYlFi"
        "lJVcxpiMxBwQE68lRlnJZYy0RoyyksuYjJGYZWKeJYjBPhgSM8PErBAT2iPmmBjsgzJml5jZfWK+qzBL"
        "xGypMHPE+DBWcjEVRrWcVrmcNg+4nJwcK/JfkcvXhA=="
    ),
    156: (
        "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"
    ),
    172: (
        "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"
        "VDR0hIqep6JgVWShomua2/8BSkp+Pg=="
    ),
}
# fmt: on

# Tried largest first, like Quark (which picks the first match in this order).
KNOWN_SIZES: Tuple[int, ...] = tuple(sorted(_TABLES, reverse=True))


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def sylvester_hadamard(n: int) -> np.ndarray:
    """The Sylvester Hadamard matrix of power-of-two order ``n`` (+-1, float64)."""
    if not is_pow2(n):
        raise ValueError(f"sylvester_hadamard needs a power of two, got {n}")
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


@lru_cache(maxsize=None)
def _tabulated(k: int) -> np.ndarray:
    bits = np.unpackbits(
        np.frombuffer(zlib.decompress(base64.b64decode(_TABLES[k])), dtype=np.uint8)
    )[: k * k]
    h = bits.astype(np.float64).reshape(k, k) * 2.0 - 1.0
    if not np.array_equal(h @ h.T, k * np.eye(k)):  # pragma: no cover - table guard
        raise RuntimeError(f"corrupt Hadamard table of order {k}")
    h.setflags(write=False)
    return h


def hadamard_factor(n: int) -> Tuple[np.ndarray, int]:
    """``(H_K, K)`` for size ``n``: ``K == 1`` and the Sylvester matrix for a
    power of two, else the tabulated matrix of the largest ``K`` dividing ``n``
    with a power-of-two quotient. Raises like Quark when there is none."""
    if is_pow2(n):
        return sylvester_hadamard(n), 1
    for k in KNOWN_SIZES:
        if n > 0 and n % k == 0 and is_pow2(n // k):
            return _tabulated(k), k
    raise ValueError(f"Could not find an Hadamard matrix for the size n={n}.")


def supports(n: int) -> bool:
    """Whether Quark (and so :func:`hadamard_matrix`) supports size ``n``."""
    try:
        hadamard_factor(n)
    except ValueError:
        return False
    return True


def hadamard_matrix(n: int) -> np.ndarray:
    """The un-normalized +-1 Hadamard matrix ``kron(H_K, H_{n/K})`` of Quark."""
    h_k, k = hadamard_factor(n)
    if k == 1:
        return h_k
    return np.kron(h_k, sylvester_hadamard(n // k))


def _sqrt64(n: int) -> float:
    """``sqrt(n)`` in float64 as Quark computes it, ``torch.tensor(n).sqrt()``.

    torch's float64 ``sqrt`` is not always correctly rounded (it is 1 ulp off
    for e.g. n = 2, 8, 19, 32, 76, 128, 312, 512, ...), so bit-identical
    matrices need torch's value: it is used when torch is already imported (it
    is whenever Quark is), else numpy's correctly rounded one (which differs by
    at most 1 ulp of float64 for those sizes).
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return float(torch.tensor(n, dtype=torch.float64).sqrt())
        except Exception:  # pragma: no cover - broken torch install
            pass
    return float(np.sqrt(float(n)))


def _sqrt32(n: int) -> float:
    """``torch.tensor(n).sqrt()`` in torch's default float32, as a float."""
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return float(torch.tensor(n).sqrt())
        except Exception:  # pragma: no cover - broken torch install
            pass
    return float(np.sqrt(np.float32(n)))


def hadamard_rotation(n: int, signs: Optional[np.ndarray] = None) -> np.ndarray:
    """Quark's R1 matrix as float64: the normalized Hadamard matrix, optionally
    with its rows multiplied by a +-1 vector ``signs`` (Quark's "random
    Hadamard", ``diag(signs) @ H``).

    The two cases normalize the way Quark does, so the results are
    bit-identical to Quark's: multiplying by ``1 / sqrt(n)`` in float64 without
    ``signs``, and dividing by ``sqrt(n)`` *in float32* with them (Quark's
    random Hadamard divides by a float32 tensor, so that matrix is orthogonal
    only to ~1e-7). See :func:`_sqrt64` for the role torch plays.
    """
    h = hadamard_matrix(n)
    if signs is None:
        return h * (1.0 / _sqrt64(n))
    s = np.asarray(signs, dtype=np.float64).reshape(-1)
    if s.shape[0] != n or not np.all(np.abs(s) == 1.0):
        raise ValueError(f"signs must be {n} values of +-1")
    return (s[:, None] * h) / _sqrt32(n)


__all__ = [
    "KNOWN_SIZES",
    "hadamard_factor",
    "hadamard_matrix",
    "hadamard_rotation",
    "is_pow2",
    "supports",
    "sylvester_hadamard",
]
