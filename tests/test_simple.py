import os
import tempfile
from typing import Optional

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper, parser

import onnxsim
from onnxsim.test_utils import export_simplify_and_check_by_python_api


def _model(body, initializer=(), opset=13, ir_version=10):
    # Pinning ir_version to 10 by default matches the older onnxruntime bundled
    # with some CI wheels (which cap at IR version 11); several tests below run
    # models through onnxruntime directly, and onnxsim's own checks do too.
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def test_onnx_simplifier():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    export_simplify_and_check_by_python_api(MockModel(), torch.randn(1, 10))


def test_mg():
    class MG(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, b):
            x = x.float()
            b = b.float()
            sh = x.shape
            x = x.view(sh[0], sh[1], -1)
            b = b.squeeze(-1)
            b = b.squeeze(-1)
            a = torch.matmul(b, x)
            preds = a.view(1, 100, sh[2], sh[3])
            return preds

    x = torch.randn([1, 256, 160, 184])
    b = torch.randn([100, 256, 1, 1])
    opt = export_simplify_and_check_by_python_api(
        MG(), (x, b), export_kwargs={"dynamo": True}
    )
    sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(
        out_names,
        {opt.graph.input[0].name: x.numpy(), opt.graph.input[1].name: b.numpy()},
    )
    assert outs[0].shape == MG()(x, b).shape


def test_transformer():
    model = torch.nn.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
    )
    model.to("cpu").to(torch.float32)
    model.eval()

    inputs = (
        torch.rand((100, 2, 256), dtype=torch.float32),
        torch.rand((15, 2, 256), dtype=torch.float32),
    )
    export_simplify_and_check_by_python_api(
        model, inputs, export_kwargs={"dynamo": True}
    )


def test_upsample():
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1_1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(16)
            self.maxpool = torch.nn.MaxPool2d(2, stride=2)
            self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.conv3_1 = torch.nn.Conv2d(32, 16, 3, padding=1)
            self.conv3_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.conv3_3 = torch.nn.Conv2d(16, 3, 3, padding=1)
            self.bn3 = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x)))))
            x2 = self.maxpool(x1)
            xup = F.interpolate(
                x2, scale_factor=2, mode="bilinear", align_corners=False
            )
            x3 = self.bn2(self.conv2_2(F.relu(self.conv2_1(xup))))
            x4 = F.relu(self.conv3_3(self.conv3_2(F.relu(self.conv3_1(x3)))))
            x5 = self.bn3(x4)
            return F.softsign(x5)

    inp = torch.rand(1, 3, 96, 96)
    net = Net()
    opt = export_simplify_and_check_by_python_api(
        net, (inp,), export_kwargs={"opset_version": 9}
    )

    u_out = None
    for n in opt.graph.node:
        if n.op_type == "Upsample":
            u_out = n.output[0]
    assert u_out is not None
    u_info = None
    for v in opt.graph.value_info:
        if v.name == u_out:
            u_info = v
    assert u_info is not None
    assert [i.dim_value for i in v.type.tensor_type.shape.dim] == [1, 3, 96, 96]


def test_concat_squeese():
    # test for https://github.com/onnxsim/onnxsim/issues/46
    class Model(torch.nn.Module):
        def forward(self, x):
            # return torch.cat((torch.mean(x, 1, keepdim=True), torch.mean(x, 1, keepdim=True)), dim=1)
            return torch.cat(
                (torch.mean(x, 1).unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
            )

    export_simplify_and_check_by_python_api(
        Model(), (torch.rand(20, 20),), export_kwargs={"opset_version": 9}
    )


def test_trilinear():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, input_tensor):
            return torch.nn.functional.interpolate(
                input_tensor, scale_factor=[4, 4, 4], mode="trilinear"
            )

    x = torch.rand(1, 8, 20, 120, 120)
    opt = export_simplify_and_check_by_python_api(
        Model(),
        (x,),
        export_kwargs={
            "opset_version": 11,
            "export_params": True,
        },
    )
    sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(out_names, {opt.graph.input[0].name: x.numpy()})
    assert outs[0].shape == (1, 8, 80, 480, 480)


def test_optional_type():
    class MyModule(torch.nn.Module):
        def forward(self, x: Optional[torch.Tensor]):
            return x is not None

    module = MyModule()
    inputs = (torch.tensor(1),)
    module = torch.jit.script(module, example_inputs=[inputs, (None,)])
    export_simplify_and_check_by_python_api(
        module,
        inputs,
        simplify_kwargs={"input_data": {"i": inputs[0].numpy()}},
        export_kwargs={"opset_version": 15, "input_names": ["i"]},
    )


def test_ext():
    class MyMod(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.param = torch.nn.Parameter(
                torch.rand(1024, 1024, 1024, dtype=torch.float16)
            )

        def forward(self, x):
            return self.param * x

    module = MyMod()
    inputs = (torch.tensor(1),)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_fn = os.path.join(tmpdirname, "tmp.onnx")
        torch.onnx.export(module, inputs, model_fn, dynamo=False, input_names=["x"])
        sim_model, check_ok = onnxsim.simplify(model_fn, check_n=0)
        module = None
        assert check_ok


def _constant_value(model, name):
    # Return the constant value produced for `name`, whether it is delivered as
    # a graph initializer or as a Constant node's value attribute, else None.
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer)
    for node in model.graph.node:
        if node.op_type == "Constant" and name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None


def test_partial_shape_evaluation_gather():
    # Partial shape evaluation for https://github.com/onnxsim/onnxsim/issues/139
    # The input's leading dimension is dynamic, but a Gather that reads only the
    # static dimensions of its shape must still be pre-computed into a constant.
    model = _model(
        """
        g (float[batch,3,4,5] x) => (int64[3] g)
        <int64[3] indices = {1, 2, 3}>
        {
          s = Shape(x)
          g = Gather<axis=0>(s, indices)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The whole Shape -> Gather chain collapses to a constant even though the
    # batch dimension is dynamic.
    assert "Shape" not in op_types
    assert "Gather" not in op_types
    value = _constant_value(sim_model, "g")
    assert value is not None
    assert list(value) == [3, 4, 5]


def test_partial_shape_evaluation_keeps_dynamic_gather():
    # A Gather that reads the dynamic dimension must NOT be folded: its value is
    # unknown until runtime.
    model = _model(
        """
        g (float[batch,3,4,5] x) => (int64[1] g)
        <int64[1] indices = {0}>
        {
          s = Shape(x)
          g = Gather<axis=0>(s, indices)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The dynamic dimension cannot be pre-computed, so the ops stay and "g" is
    # not turned into a constant.
    assert "Gather" in op_types
    assert _constant_value(sim_model, "g") is None


def test_partial_shape_evaluation_reshape_single_dynamic_dim():
    # Data propagation for Reshape: the target shape is computed at runtime from
    # the input's shape (Shape -> Gather -> Concat), and it has a single dynamic
    # entry (the batch) plus otherwise-static dimensions. Partial shape
    # evaluation propagates that shape to [batch, 60], and the Reshape pass
    # materializes it as the constant [-1, 60] -- the single -1 lets ONNX infer
    # the dynamic dim from the total element count, which is provably equivalent
    # -- so the whole Shape -> Gather -> Concat scaffolding collapses away.
    model = _model(
        """
        g (float[batch,3,4,5] x) => (float[batch,60] y)
        <int64[1] idx0 = {0}, int64[1] sixty = {60}>
        {
          s = Shape(x)
          b = Gather<axis=0>(s, idx0)
          newshape = Concat<axis=0>(b, sixty)
          y = Reshape(x, newshape)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The runtime shape computation is gone; only the Reshape (fed by a constant
    # shape) survives.
    assert "Shape" not in op_types
    assert "Gather" not in op_types
    assert "Concat" not in op_types
    reshape = [n for n in sim_model.graph.node if n.op_type == "Reshape"]
    assert len(reshape) == 1
    shape_value = _constant_value(sim_model, reshape[0].input[1])
    assert shape_value is not None
    assert list(shape_value) == [-1, 60]


def test_data_propagation_through_reshape():
    # ONNX has no data-propagation function for Reshape, so onnxsim registers one
    # (contrib_schemas): a shape tensor threaded through an element-preserving
    # Reshape must keep its propagated value so downstream shape arithmetic still
    # folds. Here Shape(x) -> Reshape(., [-1]) -> Gather([1, 2]) reads only the
    # static dims, so it must pre-compute to [3, 4] even though the batch is
    # dynamic -- which requires the value to survive the Reshape.
    model = _model(
        """
        g (float[batch,3,4] x) => (int64[2] g)
        <int64[1] flat = {-1}, int64[2] indices = {1, 2}>
        {
          s = Shape(x)
          s2 = Reshape(s, flat)
          g = Gather<axis=0>(s2, indices)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The value propagated through the Reshape, so the whole chain pre-computes.
    assert "Gather" not in op_types
    assert "Reshape" not in op_types
    value = _constant_value(sim_model, "g")
    assert value is not None
    assert list(value) == [3, 4]


def test_partial_shape_evaluation_symbolic_arithmetic_reshape():
    # Native symbolic shape evaluation (issue #532). The Reshape target is
    # computed at runtime with *arithmetic over the dynamic batch dim*:
    #   half = ReduceProd(Shape(x)) / 2 = (batch * 768) / 2 = batch * 384
    #   newshape = Concat(half, [2]) = [batch * 384, 2]
    # ONNX data propagation cannot carry a symbol through ReduceProd/Div, so the
    # #526 rewrite alone leaves the whole Shape -> ReduceProd -> Div -> Concat
    # scaffolding standing. The SymExpr evaluator keeps the dim as `384*batch`,
    # sees a single symbolic entry, and materializes the shape as the constant
    # [-1, 2] -- provably equivalent since batch*384*2 == numel(x) for every
    # batch -- so the scaffolding collapses.
    model = _model(
        """
        g (float[batch,768] x) => (float[?,2] y)
        <int64[1] two = {2}, int64[1] two2 = {2}>
        {
          s = Shape(x)
          total = ReduceProd<keepdims=1>(s)
          half = Div(total, two)
          newshape = Concat<axis=0>(half, two2)
          y = Reshape(x, newshape)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    # The runtime shape arithmetic is gone; only the Reshape (fed by a constant
    # shape) survives.
    assert "Shape" not in op_types
    assert "ReduceProd" not in op_types
    assert "Div" not in op_types
    assert "Concat" not in op_types
    reshape = [n for n in sim_model.graph.node if n.op_type == "Reshape"]
    assert len(reshape) == 1
    shape_value = _constant_value(sim_model, reshape[0].input[1])
    assert shape_value is not None
    assert list(shape_value) == [-1, 2]


def test_unfoldable_const_node_keeps_topological_order():
    # A const node (all-constant inputs) that fails to fold must keep its
    # original position. Here SequenceEmpty is treated as a const node; its
    # output feeds a non-const consumer (SequenceInsert). If a failed const node
    # were moved to the end of the graph it would land after its consumer and
    # break topological sorting, making the output fail onnx's checker (issues
    # #238, #335, #352).
    #
    # Constant folding is disabled here on purpose: SequenceEmpty produces a
    # sequence value, which the backend returns as an empty list. Folding would
    # coerce that into an empty *tensor* initializer and drop the node, which is
    # semantically wrong for a sequence. Skipping folding keeps the sequence
    # pipeline intact so we exercise the topological ordering of the preserved
    # nodes.
    model = _model(
        """
        g (float[2] x) => (float[2] y)
        {
          seq = SequenceEmpty()
          seq2 = SequenceInsert(seq, x)
          y = ConcatFromSequence<axis=0, new_axis=0>(seq2)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, skip_constant_folding=True)
    assert check_ok
    # Output must remain a valid, topologically sorted graph.
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.index("SequenceEmpty") < op_types.index("SequenceInsert")


def test_folding_does_not_duplicate_initializers():
    # Folding a const op that reads a weight (here a Transpose on an initializer)
    # produces a new initializer for the result but must not leave the original
    # operand initializer dangling in the graph. Otherwise the weight data is
    # duplicated, which can push a large model past onnx's 2GB protobuf limit
    # before the optimizer runs (issue #174).
    model = _model(
        """
        g (float[3,4] x) => (float[3,4] y)
        <float[4,3] w = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}>
        {
          wt = Transpose<perm=[1, 0]>(w)
          y = Add(x, wt)
        }
        """
    )
    onnx.checker.check_model(model)

    # Disable the onnx optimizer so the constant folding logic alone is
    # responsible for cleaning up the dangling initializer.
    sim_model, check_ok = onnxsim.simplify(model, perform_optimization=False)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # Core invariant of the fix, independent of platform: the simplified graph
    # must never carry an initializer that no node consumes. Whether the backend
    # executor is able to fold the Transpose can vary between environments, but
    # the "no dangling initializer" property must hold either way.
    op_types = [n.op_type for n in sim_model.graph.node]
    used = {i for n in sim_model.graph.node for i in n.input}
    for init in sim_model.graph.initializer:
        assert init.name in used, f"unused initializer left behind: {init.name}"

    # When the Transpose was actually folded into an initializer, the folded
    # result must replace the original weight "w" rather than duplicate it.
    if "Transpose" not in op_types:
        assert "w" not in {init.name for init in sim_model.graph.initializer}


def test_batched_constant_folding():
    # Many independent constant sub-expressions are folded in a single backend
    # Session (batched folding) instead of one Session per node. This builds a
    # wide fan of Constant -> Mul chains that are summed into one constant and
    # then added to the runtime input, and checks that every constant node is
    # folded into a single initializer and that the result stays numerically
    # correct.
    n = 40
    base = np.arange(8, dtype=np.float32)

    fanin_lines = []
    prod_names = []
    for i in range(n):
        fanin_lines.append(f"c{i} = Constant<value = float {{{float(i)}}}>()")
        fanin_lines.append(f"p{i} = Mul(base, c{i})")
        prod_names.append(f"p{i}")
    fanin_body = "\n".join(fanin_lines)
    base_literal = ", ".join(str(float(v)) for v in base)

    model = _model(
        f"""
        g (float[8] x) => (float[8] y)
        <float[8] base = {{{base_literal}}}>
        {{
          {fanin_body}
          acc = Sum({", ".join(prod_names)})
          y = Add(x, acc)
        }}
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, perform_optimization=False)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # No initializer may be left dangling regardless of backend availability.
    used = {i for node in sim_model.graph.node for i in node.input}
    for init in sim_model.graph.initializer:
        assert init.name in used, f"unused initializer left behind: {init.name}"

    op_types = [node.op_type for node in sim_model.graph.node]
    # When the backend folded the constants, the whole fan collapses to a single
    # Add reading the runtime input and one folded initializer.
    if "Mul" not in op_types and "Sum" not in op_types and "Constant" not in op_types:
        assert op_types == ["Add"]
        acc = base * (n * (n - 1) / 2)
        x_val = np.random.rand(8).astype(np.float32)
        sess = onnxruntime.InferenceSession(
            sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (out,) = sess.run(None, {"x": x_val})
        np.testing.assert_allclose(out, x_val + acc, rtol=1e-5, atol=1e-5)


def _fp8_zero_point(name):
    # A single float8_e4m3fn zero. The parser's text form encodes a float8
    # tensor literal as an integer ({0}), so this needs neither raw bytes nor an
    # ml_dtypes dependency.
    return f"float8e4m3fn {name} = {{0}}"


def test_fp8_qdq_model():
    # Regression test for GitHub issue #348. NVIDIA ModelOpt emits fp8 QDQ
    # models whose QuantizeLinear/DequantizeLinear zero points use the
    # ``float8_e4m3fn`` element type (17). onnxoptimizer's tensor-value hashing
    # passes (eliminate_common_subexpression / eliminate_duplicate_initializer)
    # cannot hash such tensors and used to abort the whole simplification with
    # "RuntimeError: no supported data type: 17". onnxsim now detects these
    # tensors and transparently skips the offending passes instead of crashing.
    weight_literal = ", ".join(str(0.1 * i) for i in range(12))
    model = _model(
        f"""
        g (float[2,4] X) => (float[2,3] Y)
        <float[4,3] W = {{{weight_literal}}},
         float w_scale = {{0.05}},
         float a_scale = {{0.1}},
         {_fp8_zero_point("a_zp")},
         {_fp8_zero_point("w_zp")},
         {_fp8_zero_point("w_zp2")}>
        {{
          X_q = QuantizeLinear(X, a_scale, a_zp)
          X_dq = DequantizeLinear(X_q, a_scale, a_zp)
          W_q = QuantizeLinear(W, w_scale, w_zp)
          W_dq = DequantizeLinear(W_q, w_scale, w_zp2)
          Y = MatMul(X_dq, W_dq)
        }}
        """,
        opset=21,
    )
    onnx.checker.check_model(model)

    # Must not raise "no supported data type: 17".
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The fp8 QDQ structure must survive simplification.
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.count("QuantizeLinear") == 2
    assert op_types.count("DequantizeLinear") == 2
    assert "MatMul" in op_types


def test_fp8_qdq_modelopt_integration():
    # NVIDIA ModelOpt used to simplify ONNX models inside its quantization
    # preprocessing with exactly ``model_simp, check = onnxsim.simplify(model)``
    # (wrapped in a try/except that fell back to the unsimplified model on
    # error). It dropped onnxsim for onnxslim, in part because onnxsim aborted
    # on ModelOpt's fp8 QDQ output with "no supported data type: 17" (issue
    # #348). This exercises that exact call shape on a richer, more
    # ModelOpt-like graph -- Conv + Gemm with both activation and weight QDQ and
    # duplicated float8 zero points -- to guard that onnxsim simplifies it
    # without crashing and preserves the QDQ structure TensorRT relies on.
    conv_idx = np.arange(8 * 3 * 3 * 3)
    conv_w = (0.01 * (conv_idx % 7 - 3)).astype(np.float32).reshape(8, 3, 3, 3)
    gemm_idx = np.arange(8 * 8)
    gemm_w = (0.02 * (gemm_idx % 5 - 2)).astype(np.float32).reshape(8, 8)

    model = _model(
        f"""
        g (float[1,3,6,6] X) => (float[1,8] Y)
        <float conv_w_scale = {{0.02}},
         float gemm_w_scale = {{0.03}},
         float act_scale = {{0.1}},
         float act_scale2 = {{0.1}},
         {_fp8_zero_point("a_zp")},
         {_fp8_zero_point("a_zp2")},
         {_fp8_zero_point("cw_zp")},
         {_fp8_zero_point("cw_zp2")},
         {_fp8_zero_point("gw_zp")},
         {_fp8_zero_point("gw_zp2")}>
        {{
          Xq = QuantizeLinear(X, act_scale, a_zp)
          Xdq = DequantizeLinear(Xq, act_scale, a_zp)
          cwq = QuantizeLinear(conv_w, conv_w_scale, cw_zp)
          cwdq = DequantizeLinear(cwq, conv_w_scale, cw_zp2)
          conv_out = Conv<kernel_shape=[3, 3]>(Xdq, cwdq)
          pooled = GlobalAveragePool(conv_out)
          flat = Flatten<axis=1>(pooled)
          flatq = QuantizeLinear(flat, act_scale2, a_zp2)
          flatdq = DequantizeLinear(flatq, act_scale2, a_zp2)
          gwq = QuantizeLinear(gemm_w, gemm_w_scale, gw_zp)
          gwdq = DequantizeLinear(gwq, gemm_w_scale, gw_zp2)
          Y = Gemm<transB=1>(flatdq, gwdq)
        }}
        """,
        initializer=[
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
        ],
        opset=21,
    )
    onnx.checker.check_model(model)

    # ModelOpt's exact former call shape. Must not raise the unsupported ONNX
    # data type 17 error; check must be True so ModelOpt keeps the simplified model.
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # Every QDQ pair must be preserved -- TensorRT needs the QDQ structure, and
    # onnxsim must not fold or dedupe it away.
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.count("QuantizeLinear") == 4
    assert op_types.count("DequantizeLinear") == 4
    assert "Conv" in op_types
    assert "Gemm" in op_types


def test_if_with_const_cond_is_folded():
    # An `If` whose condition is a constant, with branches that each just
    # produce a constant, used to crash simplify() with a segfault (exit 139):
    # onnxsim's constant folding turns the branch `Constant` nodes into subgraph
    # initializers, and the onnxoptimizer "eliminate_if_with_const_cond" pass
    # then dereferenced a null value while inlining the taken branch (the branch
    # output was now an initializer with no producing node). With the fixed pass
    # the `If` is folded away and its output becomes the taken branch's constant
    # (GitHub issue #452).
    #
    # onnx.parser's text format has no syntax for GRAPH-typed attributes
    # (then_branch/else_branch), so this stays on onnx.helper construction.
    tv = helper.make_tensor("tv", TensorProto.FLOAT, [2], [1.0, 2.0])
    fv = helper.make_tensor("fv", TensorProto.FLOAT, [2], [3.0, 4.0])
    then_b = helper.make_graph(
        [helper.make_node("Constant", [], ["to"], value=tv)],
        "tb",
        [],
        [helper.make_tensor_value_info("to", TensorProto.FLOAT, [2])],
    )
    else_b = helper.make_graph(
        [helper.make_node("Constant", [], ["fo"], value=fv)],
        "fb",
        [],
        [helper.make_tensor_value_info("fo", TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node(
        "If", ["c"], ["Y"], then_branch=then_b, else_branch=else_b
    )
    graph = helper.make_graph(
        [if_node],
        "g",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        [helper.make_tensor("c", TensorProto.BOOL, [], [True])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The constant-condition `If` (condition is True) must be folded away, and
    # the model output must be the "then" branch constant [1.0, 2.0]. Read the
    # folded value straight out of the graph rather than running the model,
    # since helper.make_model stamps the latest ONNX IR version, which the
    # bundled onnxruntime may not load.
    assert all(n.op_type != "If" for n in sim_model.graph.node)

    def _resolve_const(model, name):
        # Follow the output through initializers, Constant nodes, and Identity
        # aliases until a constant tensor is found.
        seen = set()
        while name and name not in seen:
            seen.add(name)
            for init in model.graph.initializer:
                if init.name == name:
                    return numpy_helper.to_array(init)
            alias = None
            for node in model.graph.node:
                if name not in node.output:
                    continue
                if node.op_type == "Constant":
                    (attr,) = [a for a in node.attribute if a.name == "value"]
                    return numpy_helper.to_array(attr.t)
                if node.op_type == "Identity":
                    alias = node.input[0]
                break
            name = alias
        raise AssertionError(f"output {name!r} is not a resolvable constant")

    out = _resolve_const(sim_model, sim_model.graph.output[0].name)
    assert out.tolist() == [1.0, 2.0]


def test_loop_with_const_trip_count_is_unrolled():
    # A `Loop` with a compile-time-constant trip count and no break condition
    # -- the shape emitted for a plain Python `for i in range(N): ...` with no
    # `break` -- is unrolled into N copies of its body by the onnxoptimizer
    # "eliminate_loop_with_const_trip_count" pass. This matters for
    # downstream compilers that don't support `Loop` at all (e.g. TVM's Relax
    # ONNX frontend, see docs/dlpack-executor.md).
    #
    # onnx.parser's text format has no syntax for GRAPH-typed attributes (the
    # Loop `body`), so this stays on onnx.helper construction.
    step = helper.make_tensor("step", TensorProto.FLOAT, [2], [1.0, 1.0])
    body = helper.make_graph(
        [helper.make_node("Add", ["v_in", "step"], ["v_out"])],
        "loop_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2]),
        ],
        [
            # cond_out: a direct passthrough of cond_in (never actually
            # computed, since the outer Loop's `cond` input is omitted below).
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2]),
        ],
        [step],
    )
    trip_count = helper.make_tensor("trip_count", TensorProto.INT64, [], [3])
    loop_node = helper.make_node("Loop", ["trip_count", "", "x"], ["y"], body=body)
    graph = helper.make_graph(
        [loop_node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        [trip_count],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The Loop must be gone, unrolled into three Adds (onnxsim's own constant
    # folding may further fuse/fold these, so check for absence of Loop
    # rather than an exact Add count).
    assert all(n.op_type != "Loop" for n in sim_model.graph.node)


def test_sequence_at_construct_is_folded():
    # `SequenceAt(SequenceConstruct(a, b, c), i)` with a constant index -- a
    # common PyTorch-export artifact for indexing a fixed-size Python list of
    # tensors -- folds straight to the indexed tensor, dropping the Sequence
    # type entirely. Downstream compilers (e.g. TVM's Relax ONNX frontend)
    # generally have little to no support for Sequence.
    model = _model(
        """
        g (float[2] a, float[2] b, float[2] c) => (float[2] y)
        <int64 idx = {1}>
        {
          seq = SequenceConstruct(a, b, c)
          y = SequenceAt(seq, idx)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    assert all(
        n.op_type not in ("Sequence", "SequenceConstruct", "SequenceAt")
        for n in sim_model.graph.node
    )


def test_sequence_length_construct_folds_and_unrolls_loop():
    # `SequenceLength(SequenceConstruct(...))` folds to a constant, which is
    # exactly what a `for i in range(len(some_list)): ...` loop exports as: a
    # Sequence feeding a Loop's trip count. Folding SequenceLength lets
    # eliminate_loop_with_const_trip_count unroll the Loop in turn, so the
    # whole pattern collapses to plain feed-forward ops.
    #
    # onnx.parser's text format has no syntax for GRAPH-typed attributes (the
    # Loop `body`), so this stays on onnx.helper construction.
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2])
    seq = helper.make_node("SequenceConstruct", ["a", "b", "c"], ["seq"])
    length = helper.make_node("SequenceLength", ["seq"], ["n"])

    step = helper.make_tensor("step", TensorProto.FLOAT, [2], [1.0, 1.0])
    body = helper.make_graph(
        [helper.make_node("Add", ["v_in", "step"], ["v_out"])],
        "loop_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2]),
        ],
        [step],
    )
    loop_node = helper.make_node("Loop", ["n", "", "x"], ["y"], body=body)
    graph = helper.make_graph(
        [seq, length, loop_node],
        "g",
        [a, b, c, helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Loop" not in op_types
    assert "SequenceConstruct" not in op_types
    assert "SequenceLength" not in op_types


def test_optional_get_element_of_optional_is_folded():
    # `OptionalGetElement(Optional(x))` -- a common `torch.jit.script`-export
    # artifact for an `Optional[Tensor]` argument that is known to be present
    # -- folds straight to `x`, dropping the Optional type entirely. Most
    # ONNX consumers (this includes compilers such as TVM's Relax ONNX
    # frontend) have little to no support for the Optional type.
    model = _model(
        """
        g (float[2] x) => (float[2] y)
        {
          opt = Optional(x)
          y = OptionalGetElement(opt)
        }
        """,
        opset=18,
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Optional" not in op_types
    assert "OptionalGetElement" not in op_types


def test_optional_has_element_is_folded():
    # `OptionalHasElement` folds to a constant bool whenever its emptiness is
    # already known: true for `Optional(x)`, false for an explicitly-empty
    # `Optional()` and for the op's own input being omitted entirely.
    #
    # onnx.parser's text format has no syntax for TYPE_PROTO-typed node
    # attributes (the explicit `type` on the empty `Optional`), so this stays
    # on onnx.helper construction.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    opt = helper.make_node("Optional", ["x"], ["opt"])
    opt_empty = helper.make_node(
        "Optional",
        [],
        ["opt_empty"],
        type=helper.make_tensor_type_proto(TensorProto.FLOAT, [2]),
    )
    has_present = helper.make_node("OptionalHasElement", ["opt"], ["h_present"])
    has_empty = helper.make_node("OptionalHasElement", ["opt_empty"], ["h_empty"])
    has_no_input = helper.make_node("OptionalHasElement", [], ["h_no_input"])
    graph = helper.make_graph(
        [opt, opt_empty, has_present, has_empty, has_no_input],
        "g",
        [x],
        [
            helper.make_tensor_value_info("h_present", TensorProto.BOOL, []),
            helper.make_tensor_value_info("h_empty", TensorProto.BOOL, []),
            helper.make_tensor_value_info("h_no_input", TensorProto.BOOL, []),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    onnx.checker.check_model(sim_model)
    assert all(n.op_type != "OptionalHasElement" for n in sim_model.graph.node)

    values = {}
    for init in sim_model.graph.initializer:
        values[init.name] = numpy_helper.to_array(init)
    for node in sim_model.graph.node:
        if node.op_type == "Constant":
            (attr,) = [a for a in node.attribute if a.name == "value"]
            values[node.output[0]] = numpy_helper.to_array(attr.t)

    out_names = [o.name for o in sim_model.graph.output]
    assert bool(values[out_names[0]]) is True  # h_present
    assert bool(values[out_names[1]]) is False  # h_empty
    assert bool(values[out_names[2]]) is False  # h_no_input


def test_arg_reduce_select_last_index_is_rewritten():
    # `select_last_index=1` (added at opset 12) isn't implemented by some
    # downstream ONNX consumers (e.g. TVM's Relax ONNX frontend). It's
    # rewritten to an equivalent computation over `Shape`/`Gather`/`Slice`/
    # `Sub`/`ArgMax` that never needs the attribute: flip the axis, take the
    # *first* occurrence there (select_last_index's own default), and map
    # the index back through the flip.
    model = _model(
        """
        g (float[4] x) => (int64[1] y)
        {
          y = ArgMax<axis=0, keepdims=1, select_last_index=1>(x)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "ArgMax" in op_types  # rewritten, not eliminated
    for node in sim_model.graph.node:
        if node.op_type == "ArgMax":
            select_last_index = [
                a.i for a in node.attribute if a.name == "select_last_index"
            ]
            # Absent (defaults to 0) or explicitly reset to 0 -- either is
            # fine, since the rewrite compensates by flipping the axis.
            assert not select_last_index or select_last_index[0] == 0

    # Numeric check on inputs with ties, where select_last_index actually
    # changes the result relative to the default (first-occurrence) index.
    sess_orig = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sess_sim = onnxruntime.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for v in (
        np.array([3, 5, 5, 2], dtype=np.float32),
        np.array([4, 4, 4, 4], dtype=np.float32),
        np.array([1, 2, 3, 4], dtype=np.float32),
    ):
        orig_out = sess_orig.run(None, {"x": v})[0]
        sim_out = sess_sim.run(None, {"x": v})[0]
        assert np.array_equal(orig_out, sim_out)


def _einsum_matmul_model():
    model = _model(
        """
        g (float[2,3] x, float[3,4] y) => (float[2,4] z)
        {
          z = Einsum<equation="ij,jk->ik">(x, y)
        }
        """
    )
    onnx.checker.check_model(model)
    return model


def test_extra_optimizers_opts_into_an_off_by_default_pass():
    # extra_optimizers is the counterpart to skipped_optimizers: it runs a
    # named pass in addition to the default fuse/elimination set.
    # replace_einsum_with_matmul is a real onnx-optimizer pass that is not
    # part of that default set (confirmed via the second assertion below), so
    # it only fires when named explicitly.
    model = _einsum_matmul_model()

    sim_default, ok_default = onnxsim.simplify(model, check_n=0)
    assert ok_default
    assert [n.op_type for n in sim_default.graph.node] == ["Einsum"]

    from onnxsim.onnx_simplifier import C

    assert "replace_einsum_with_matmul" not in C._list_optimizers()
    assert "replace_einsum_with_matmul" in C._list_other_optimizers()

    sim_extra, ok_extra = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["replace_einsum_with_matmul"]
    )
    assert ok_extra
    assert [n.op_type for n in sim_extra.graph.node] == ["MatMul"]


def test_extra_optimizers_unknown_name_raises():
    model = _einsum_matmul_model()
    with pytest.raises(Exception):
        onnxsim.simplify(model, check_n=0, extra_optimizers=["not_a_real_pass"])


def test_extra_optimizers_has_no_effect_when_optimization_disabled():
    # perform_optimization=False means skipped_optimizers=None internally,
    # which already disables the whole default pass set; extra_optimizers is
    # documented to have no effect in that case either.
    model = _einsum_matmul_model()
    sim, ok = onnxsim.simplify(
        model,
        check_n=0,
        perform_optimization=False,
        extra_optimizers=["replace_einsum_with_matmul"],
    )
    assert ok
    assert [n.op_type for n in sim.graph.node] == ["Einsum"]


def _matmul_weight_data():
    return [
        1.0,
        -2.0,
        0.5,
        3.0,
        0.25,
        -1.5,
        -0.75,
        2.5,
        1.0,
        0.1,
        -0.2,
        4.0,
    ]  # [4, 3]


def _matmul_model():
    weight_literal = ", ".join(str(v) for v in _matmul_weight_data())
    model = _model(
        f"""
        g (float[2,4] x) => (float[2,3] y)
        <float[4,3] w = {{{weight_literal}}}>
        {{
          y = MatMul(x, w)
        }}
        """
    )
    onnx.checker.check_model(model)
    return model


def _gemm_with_bias_model():
    weight_literal = ", ".join(str(v) for v in _matmul_weight_data())
    model = _model(
        f"""
        g (float[2,4] x) => (float[2,3] y)
        <float[4,3] w = {{{weight_literal}}}, float[3] b = {{0.1, -0.2, 0.3}}>
        {{
          y = Gemm(x, w, b)
        }}
        """
    )
    onnx.checker.check_model(model)
    return model


def _matmul_test_input():
    return np.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -3.0]], dtype=np.float32)


def test_defuse_matmul_integer_to_float_undoes_dynamic_quantize_matmul():
    # defuse_matmul_integer_to_float is the exact inverse of
    # dynamic_quantize_matmul: it folds the DynamicQuantizeLinear +
    # MatMulInteger + Cast + Mul + Mul chain that pass builds back into a
    # single plain MatMul, for consumers that cannot import the quantized
    # ops at all. Both passes are PassType::Other, off by default.
    from onnxsim.onnx_simplifier import C

    model = _matmul_model()

    assert "dynamic_quantize_matmul" not in C._list_optimizers()
    assert "dynamic_quantize_matmul" in C._list_other_optimizers()
    assert "defuse_matmul_integer_to_float" not in C._list_optimizers()
    assert "defuse_matmul_integer_to_float" in C._list_other_optimizers()

    sim_q, ok_q = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["dynamic_quantize_matmul"]
    )
    assert ok_q
    assert set(n.op_type for n in sim_q.graph.node) == {
        "DynamicQuantizeLinear",
        "MatMulInteger",
        "Cast",
        "Mul",
    }

    sim_d, ok_d = onnxsim.simplify(
        sim_q, check_n=0, extra_optimizers=["defuse_matmul_integer_to_float"]
    )
    assert ok_d
    assert [n.op_type for n in sim_d.graph.node] == ["MatMul"]

    sess_orig = onnxruntime.InferenceSession(model.SerializeToString())
    sess_defused = onnxruntime.InferenceSession(sim_d.SerializeToString())
    x = _matmul_test_input()
    y_orig = sess_orig.run(None, {"x": x})[0]
    y_defused = sess_defused.run(None, {"x": x})[0]
    # Only W went through lossy INT8 quantization (X stays float, unlike the
    # actual quantized graph), so this should be close, not bitwise equal.
    np.testing.assert_allclose(y_defused, y_orig, atol=0.05)


def test_defuse_matmul_integer_to_float_with_bias():
    from onnxsim.onnx_simplifier import C

    model = _gemm_with_bias_model()

    # The forward pass's Gemm(+bias) handling, combined with the full default
    # pass set, has an unrelated pre-existing flakiness on some weight
    # values; run it alone (skip every default pass) to build a
    # deterministic quantized fixture.
    sim_q, ok_q = onnxsim.simplify(
        model,
        check_n=0,
        skipped_optimizers=list(C._list_optimizers()),
        extra_optimizers=["dynamic_quantize_matmul"],
    )
    assert ok_q
    assert "Add" in [n.op_type for n in sim_q.graph.node]

    sim_d, ok_d = onnxsim.simplify(
        sim_q, check_n=0, extra_optimizers=["defuse_matmul_integer_to_float"]
    )
    assert ok_d
    # The default pass set is still active alongside the extra one, so the
    # MatMul + Add(bias) this pass emits is immediately re-fused into a
    # single Gemm by fuse_matmul_add_bias_into_gemm -- also a correct
    # result, just a more idiomatic one.
    assert [n.op_type for n in sim_d.graph.node] == ["Gemm"]

    sess_orig = onnxruntime.InferenceSession(model.SerializeToString())
    sess_defused = onnxruntime.InferenceSession(sim_d.SerializeToString())
    x = _matmul_test_input()
    y_orig = sess_orig.run(None, {"x": x})[0]
    y_defused = sess_defused.run(None, {"x": x})[0]
    np.testing.assert_allclose(y_defused, y_orig, atol=0.05)


def test_defuse_matmul_integer_to_float_is_off_by_default():
    model = _matmul_model()
    sim_q, ok_q = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["dynamic_quantize_matmul"]
    )
    assert ok_q
    quantized_ops = set(n.op_type for n in sim_q.graph.node)

    sim_default, ok_default = onnxsim.simplify(sim_q, check_n=0)
    assert ok_default
    assert set(n.op_type for n in sim_default.graph.node) == quantized_ops


def test_defuse_matmul_integer_to_float_leaves_plain_matmul_alone():
    model = _matmul_model()
    sim, ok = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["defuse_matmul_integer_to_float"]
    )
    assert ok
    assert [n.op_type for n in sim.graph.node] == ["MatMul"]


def test_ir3_conv_bn_fuses():
    # IR version 3 models (e.g. the opset-8 ``resnet101-v1-7``) list every
    # initializer as a graph input too, which is required before IR 4. onnxsim
    # used to skip ``remove_initializer_from_input`` for such models, so the
    # weights stayed "runtime inputs" and onnxoptimizer refused to fold them --
    # ``fuse_bn_into_conv`` never fired and the graph came out unchanged
    # (GitHub issue #543). onnxsim now bumps these to IR 4 and drops the
    # initializer inputs, so the Conv+BN pair fuses away.
    out_ch, in_ch = 4, 3
    W = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32)
    scale = np.abs(np.random.randn(out_ch).astype(np.float32)) + 1
    bias = np.random.randn(out_ch).astype(np.float32)
    mean = np.random.randn(out_ch).astype(np.float32)
    var = np.abs(np.random.randn(out_ch).astype(np.float32)) + 1

    # Initializers are *also* listed as graph inputs, as IR<4 requires.
    model = _model(
        f"""
        g (float[1,{in_ch},8,8] X, float[{out_ch},{in_ch},3,3] W,
           float[{out_ch}] scale, float[{out_ch}] bias, float[{out_ch}] mean,
           float[{out_ch}] var) => (float[1,{out_ch},6,6] Y)
        {{
          conv_out = Conv(X, W)
          Y = BatchNormalization(conv_out, scale, bias, mean, var)
        }}
        """,
        initializer=[
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(mean, "mean"),
            numpy_helper.from_array(var, "var"),
        ],
        opset=8,
        ir_version=3,
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
    # The BatchNormalization is folded into the Conv weights and disappears.
    assert all(n.op_type != "BatchNormalization" for n in sim_model.graph.node)
    assert any(n.op_type == "Conv" for n in sim_model.graph.node)


def _ir3_matmul_model(opset: int) -> onnx.ModelProto:
    # A minimal IR-3 model whose weight ``W`` is *also* listed as a graph input,
    # as IR<4 required. ``opset`` selects the ai.onnx version under test.
    W = np.ones((2, 2), dtype=np.float32)
    return _model(
        """
        g (float[2,2] X, float[2,2] W) => (float[2,2] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[numpy_helper.from_array(W, "W")],
        opset=opset,
        ir_version=3,
    )


def test_remove_initializer_from_input_skips_ancient_opset():
    # The opset-3 onnx-caffe2 ``resnet50-caffe2-v1-3`` regressed once IR<4
    # folding was enabled: bumping such a model to IR 4 lets onnxoptimizer's
    # value-baking fusions fire, and ``fuse_bn_into_conv`` emits ``Cast`` nodes
    # using the modern INT ``to`` attribute -- illegal under opset 3, where
    # ``to`` is a STRING -- so onnx/onnxruntime abort with "Mismatched attribute
    # type in 'Cast_0 : to'. Expected: 'STRING', actual: 'INT'". Opsets below
    # ``_MIN_OPSET_FOR_INITIALIZER_FOLD`` must therefore be left untouched: the
    # initializer stays a graph input (so the fusions stay disabled) and the IR
    # version is not bumped.
    from onnxsim.onnx_simplifier import remove_initializer_from_input

    model = _ir3_matmul_model(opset=3)
    out = remove_initializer_from_input(model)
    assert out.ir_version == 3
    assert "W" in [i.name for i in out.graph.input]


def test_remove_initializer_from_input_folds_modern_opset():
    # An opset >= 6 IR-3 model (e.g. the opset-8 ``resnet101-v1-7``) is safe to
    # fold: the initializer-input is dropped and the model bumped to IR 4 so the
    # freed constant becomes foldable and value-baking fusions can fire.
    from onnxsim.onnx_simplifier import remove_initializer_from_input

    model = _ir3_matmul_model(opset=8)
    out = remove_initializer_from_input(model)
    assert out.ir_version == 4
    assert "W" not in [i.name for i in out.graph.input]
