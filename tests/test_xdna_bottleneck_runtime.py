import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

onnx = pytest.importorskip("onnx")

from bottleneck_runtime import bind_bottleneck_block  # noqa: E402
from resnet_bottleneck import plan_bottleneck_blocks  # noqa: E402


def test_quicktest_bottleneck_bindings_reject_bias_or_downsample():
    model = onnx.load("/home/takecheeze/ryzen_ai-1.8.0/venv/quicktest/test_model.onnx")
    bindings = [bind_bottleneck_block(model, block) for block in plan_bottleneck_blocks(model)]
    assert len(bindings) == 16
    assert all(binding.has_bias for binding in bindings)
    assert not any(binding.executable_with_existing_primitive for binding in bindings)
    assert all(binding.weights.size > 0 for binding in bindings)
    assert all(binding.biases and all(bias.size > 0 for bias in binding.biases) for binding in bindings)
    assert sum(binding.skip_weight is not None for binding in bindings) == 4
    assert "bias_requantize" in bindings[0].required_postops
