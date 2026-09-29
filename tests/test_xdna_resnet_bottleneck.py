import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

import pytest  # noqa: E402

onnx = pytest.importorskip("onnx")

from resnet_bottleneck import plan_bottleneck_blocks  # noqa: E402

QUICKTEST_MODEL = Path("/home/takecheeze/ryzen_ai-1.8.0/venv/quicktest/test_model.onnx")


def _quicktest_model() -> str:
    """The Ryzen AI quicktest ResNet-50 (only present on the XDNA development host)."""
    if not QUICKTEST_MODEL.is_file():
        pytest.skip(f"quicktest model not available: {QUICKTEST_MODEL}")
    return str(QUICKTEST_MODEL)


def test_quicktest_resnet_has_sixteen_bottleneck_blocks():
    model = onnx.load(_quicktest_model())
    blocks = plan_bottleneck_blocks(model)
    assert len(blocks) == 16
    assert sum(block.skip_conv_index is not None for block in blocks) == 4
    assert all(len(block.main_conv_indices) == 3 for block in blocks)
