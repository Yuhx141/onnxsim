import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

import pytest  # noqa: E402

onnx = pytest.importorskip("onnx")

from resnet_bottleneck import plan_bottleneck_blocks  # noqa: E402


def test_quicktest_resnet_has_sixteen_bottleneck_blocks():
    model = onnx.load("/home/takecheeze/ryzen_ai-1.8.0/venv/quicktest/test_model.onnx")
    blocks = plan_bottleneck_blocks(model)
    assert len(blocks) == 16
    assert sum(block.skip_conv_index is not None for block in blocks) == 4
    assert all(len(block.main_conv_indices) == 3 for block in blocks)
