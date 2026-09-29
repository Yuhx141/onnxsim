import pytest

pytest.importorskip("onnx")

from onnxsim.rpc import _protocol as proto  # noqa: E402
from onnxsim.rpc import xdna  # noqa: E402


def test_body_groups_accept_lists_and_option_dicts():
    groups = [["/a"], {"blocks": ["/b", "/c"], "chunk_cap": 17000, "depth": 2}]
    assert xdna._body_groups(groups) == groups


@pytest.mark.parametrize(
    "groups",
    [None, [], [[]], [["/a", 3]], [{"blocks": []}], [["/a"]] * 9],
)
def test_body_groups_reject_malformed(groups):
    with pytest.raises(proto.RPCError):
        xdna._body_groups(groups)


def test_resnet_network_compile_command_carries_tuning_options(tmp_path):
    stages = [["/a/0", "/a/1"], ["/b/0"]]
    options = {
        "stages": stages,
        "cols": 8,
        "split_weights": [0, 1],
        "weight_depths": [1, 2],
    }
    command, xclbin, insts, manifest = xdna._compile_command(
        {"_xdna_python": "py"}, "resnet_network", options, tmp_path
    )
    assert manifest is None and xclbin is not None and insts is not None
    assert command[1].endswith("resnet_stage_design.py")
    assert "--stem" in command
    assert command[command.index("--cols") + 1] == "8"
    assert command[command.index("--split-weights") + 1] == "0,1"
    assert command[command.index("--weight-depths") + 1] == "1,2"
    assert command.count("--stage") == 2


def test_resnet_network_rejects_mismatched_tuning_lists(tmp_path):
    options = {"stages": [["/a/0"], ["/b/0"]], "split_weights": [1]}
    with pytest.raises(proto.RPCError):
        xdna._compile_command(
            {"_xdna_python": "py"}, "resnet_network", options, tmp_path
        )
