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
