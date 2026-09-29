import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

tinygrad = pytest.importorskip("tinygrad")

from tinygrad_bridge import (  # noqa: E402
    XDNAUnavailable,
    numpy_to_iron,
    numpy_to_tinygrad,
    plan_tinygrad_matmul,
    run_tinygrad_matmul,
    tinygrad_dtype_to_numpy,
    tinygrad_to_iron,
    tinygrad_to_numpy,
    xdna_dtype_to_numpy,
)
from xdna_backend import describe_tensor  # noqa: E402

iron = pytest.importorskip("aie.iron")


def test_describe_tensor_handles_tinygrad_and_iron_dtypes():
    from tinygrad import Tensor

    assert describe_tensor(Tensor(np.zeros((2, 2), dtype=np.int8))).dtype == "i8"
    assert describe_tensor(SimpleNamespace(shape=(2, 2), dtype=np.int8)).dtype == "i8"
    assert describe_tensor(np.zeros((2, 2), dtype=np.int16)).dtype == "i16"


def test_tinygrad_dtype_mapping_rejects_unknown():
    assert tinygrad_dtype_to_numpy("dtypes.char") == np.dtype("int8")
    assert tinygrad_dtype_to_numpy("dtypes.float") == np.dtype("float32")
    assert xdna_dtype_to_numpy("i8") == np.dtype("int8")
    with pytest.raises(ValueError, match="tinygrad dtype"):
        tinygrad_dtype_to_numpy("dtypes.whatever")
    with pytest.raises(ValueError, match="NumPy"):
        xdna_dtype_to_numpy("f8")


def test_tinygrad_to_numpy_realizes_lazy_graph():
    from tinygrad import Tensor

    lazy = Tensor.randn(4, 8) @ Tensor.randn(8, 4)
    array = tinygrad_to_numpy(lazy)
    assert isinstance(array, np.ndarray)
    assert array.shape == (4, 4) and array.flags.c_contiguous


def test_tinygrad_iron_round_trip_on_cpu_tensors():
    from tinygrad import Tensor

    staged = tinygrad_to_iron(Tensor(np.arange(6, dtype=np.int8).reshape(2, 3)))
    assert np.asarray(staged.numpy()).reshape(2, 3).tolist() == [[0, 1, 2], [3, 4, 5]]
    empty = numpy_to_iron(np.empty((2, 2), dtype=np.int32))
    assert np.asarray(empty.numpy()).shape == (2, 2)


def test_numpy_to_tinygrad_round_trip_and_device_kwarg():
    from tinygrad import Tensor

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert isinstance(numpy_to_tinygrad(array), Tensor)
    assert numpy_to_tinygrad(array, device="CPU").numpy().tolist() == array.tolist()


def test_plan_tinygrad_matmul_matches_backend_plan():
    from tinygrad import Tensor

    plan = plan_tinygrad_matmul(
        Tensor(np.zeros((512, 512), dtype=np.int8)),
        Tensor(np.zeros((512, 512), dtype=np.int8)),
        dtype="i8",
        output_dtype="i32",
        columns=8,
    )
    assert plan.tile == (64, 64, 64)
    assert plan.kernel == "matmul_i8_oi32_m64k64n64_c8"


def test_run_reports_npu_unavailable_without_matching_artifact():
    from tinygrad import Tensor

    with pytest.raises(XDNAUnavailable, match="no XDNA artifact"):
        run_tinygrad_matmul(
            {"kernels": {}},
            Tensor(np.zeros((4, 4), dtype=np.int8)),
            Tensor(np.zeros((4, 4), dtype=np.int8)),
            dtype="i8",
            columns=1,
        )


def test_run_uses_planned_kernel_key(tmp_path):
    from tinygrad import Tensor

    manifest = {"kernels": {"other": {"xclbin": "x", "insts": "y"}}}
    with pytest.raises(XDNAUnavailable, match="no XDNA artifact"):
        run_tinygrad_matmul(
            manifest,
            Tensor(np.zeros((64, 64), dtype=np.int8)),
            Tensor(np.zeros((64, 64), dtype=np.int8)),
            dtype="i8",
            columns=1,
            base_dir=tmp_path,
        )
