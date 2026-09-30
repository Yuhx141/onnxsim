"""Layer-engine job list of a real quantized ResNet-50 body (layer1..layer4) from its block bindings."""

from __future__ import annotations

import numpy as np

from layer_engine import Job, layout_for, valid_taps


def _full_3x3(weight: np.ndarray, taps) -> np.ndarray:
    """[oc][ic][1][ntaps] (kept taps) -> [oc][ic][3][3] with the pruned taps zero."""
    oc, ic = weight.shape[:2]
    full = np.zeros((oc, ic, 3, 3), dtype=np.int8)
    for j, tap in enumerate(taps):
        ky, kx = divmod(int(tap), 3)
        full[:, :, ky, kx] = weight[:, :, 0, j]
    return full


def jobs_from_bindings(stage_bindings):
    """stage_bindings: [[binding, ...] per stage] -> (jobs, block_output_slots).

    Slot 0 holds the network input (the pooled map). Every job writes its own slot.
    """
    jobs: list[Job] = []
    block_outputs: dict[str, int] = {}
    slot = 1
    block_in = 0
    lay = None
    for binds in stage_bindings:
        for b in binds:
            raw = b["raw_weights"]
            _, cin, h, w = b["input_shape"]
            if lay is None:
                lay = layout_for(cin, w, h)
            stride = int(b["conv2_stride"][0])
            sh1, sh2, sh3 = b["shifts"]
            c1 = Job("c1", raw["w1"], raw["b1"], block_in, slot, lay, in_flip=True, shift=int(sh1))
            slot += 1
            w2 = _full_3x3(raw["w2"], b["conv2_taps"])
            c2 = Job("c2", w2, raw["b2"], c1.out_slot, slot, c1.out_layout, stride=stride, shift=int(sh2))
            slot += 1
            if not set(c2.taps) <= {int(t) for t in b["conv2_taps"]}:
                raise ValueError("engine 3x3 taps are not a subset of the block's kept taps")
            res_slot, res_mode = block_in, 2
            block = [c1, c2]
            if raw.get("skip_weight") is not None:
                sk = Job("sk", raw["skip_weight"], raw["skip_bias"], block_in, slot, lay, stride=stride, in_flip=True,
                         shift=int(b["skip_output_shift"]), relu=False)
                slot += 1
                block.append(sk)
                res_slot, res_mode = sk.out_slot, 1
            c3 = Job("c3", raw["w3"], raw["b3"], c2.out_slot, slot, c2.out_layout, shift=int(sh3), res_slot=res_slot,
                     res_mode=res_mode, out_flip=True, ea=int(b["main_residual_shift"]), eb=int(b["skip_residual_shift"]))
            slot += 1
            block.append(c3)
            jobs += block
            block_outputs[b["block"].prefix] = c3.out_slot
            block_in, lay = c3.out_slot, c3.out_layout
    return jobs, block_outputs
