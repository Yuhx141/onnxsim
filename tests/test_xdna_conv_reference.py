import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from conv_lowering import plan_conv_gemm  # noqa: E402
from conv_reference import execute_conv_reference, im2col_nchw  # noqa: E402


def _value(name, shape):
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(
                shape=SimpleNamespace(dim=[SimpleNamespace(dim_value=v) for v in shape])
            )
        ),
    )


def test_grouped_conv_reference_matches_direct_convolution():
    conv = SimpleNamespace(
        op_type="Conv",
        input=["x", "w", "b"],
        output=["y"],
        name="grouped",
        attribute=[
            SimpleNamespace(name="group", ints=(2,), i=0, s=b""),
            SimpleNamespace(name="pads", ints=(1, 1, 1, 1), i=0, s=b""),
        ],
    )
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[conv],
            input=[_value("x", (1, 4, 3, 3))],
            value_info=[],
            output=[_value("y", (1, 6, 3, 3))],
            initializer=[
                SimpleNamespace(name="w", dims=(6, 2, 3, 3)),
                SimpleNamespace(name="b", dims=(6,)),
            ],
        )
    )
    plan = plan_conv_gemm(model, 0, columns=1)
    x = np.arange(36, dtype=np.int8).reshape(1, 4, 3, 3)
    w = np.ones((6, 2, 3, 3), dtype=np.int8)
    b = np.arange(6, dtype=np.int8)
    result = execute_conv_reference(x, w, b, plan)
    expected = np.zeros((1, 6, 3, 3), dtype=np.int32)
    for oc in range(6):
        group = oc // 3
        for oh in range(3):
            for ow in range(3):
                for ic in range(2):
                    c = group * 2 + ic
                    for iy in range(3):
                        for ix in range(3):
                            yy, xx = oh + iy - 1, ow + ix - 1
                            if 0 <= yy < 3 and 0 <= xx < 3:
                                expected[0, oc, oh, ow] += int(x[0, c, yy, xx])
                expected[0, oc, oh, ow] += int(b[oc])
    assert np.array_equal(result, expected)
    assert im2col_nchw(x, plan).shape == (2, 9, 18)
