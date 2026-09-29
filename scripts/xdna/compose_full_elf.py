#!/usr/bin/env python3
"""Merge several compiled linked-stage designs into one multi-device full ELF.

Each stage keeps its own ``aie.device``; a top-level ``@main`` runtime sequence issues
``aiex.configure``/``aiex.run`` for every stage over shared host buffers, so the whole
chain executes as ONE host launch with PDI loads instead of one hardware-context switch
(~0.75 ms each) per xclbin.

Inputs are the ``<name>.prj`` directories the linked stage design leaves next to each
xclbin (they hold the lowered ``aie.mlir`` and the kernel objects).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _elems(text: str) -> tuple[int, str]:
    match = re.fullmatch(r"memref<(\d+)x(\w+)>", text.strip())
    if not match:
        raise ValueError(f"unsupported memref type {text!r}")
    return int(match.group(1)), match.group(2)


def compose(prjs: list[Path], work_dir: Path) -> tuple[str, dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    devices, stages = [], []
    for index, prj in enumerate(prjs):
        text = (prj / "aie.mlir").read_text()
        text, count = re.subn(r"aie\.device\((npu2\w*)\) \{", rf"aie.device(\1) @dev{index} {{", text, count=1)
        if count != 1:
            raise ValueError(f"{prj}: no unnamed aie.device found")
        signature = re.search(r"aie\.runtime_sequence\(([^)]*)\)", text)
        if signature is None:
            raise ValueError(f"{prj}: no runtime sequence")
        args = [part.split(":")[1].strip() for part in signature.group(1).split(",")]
        text = text.replace("aie.runtime_sequence(", f"aie.runtime_sequence @seq{index}(", 1)
        for obj in sorted(set(re.findall(r'link_with = "([^"]+)"', text))):
            shutil.copyfile(prj / obj, work_dir / f"s{index}_{obj}")
            text = text.replace(f'"{obj}"', f'"s{index}_{obj}"')
        body = text[text.index("aie.device"):text.rindex("}")].rstrip()
        devices.append("  " + body.replace("\n", "\n  "))
        stages.append([_elems(a) for a in args])

    # Shared host buffers: x (stage 0 input), packed params (all stages), y (last output),
    # and one scratch region holding every intermediate stage boundary.
    x_n, x_t = stages[0][0]
    y_n, y_t = stages[-1][2]
    p_total = sum(s[1][0] for s in stages)
    mids = [s[2][0] for s in stages[:-1]]
    mid_total = sum(mids) or 1
    def view(tag: str, src: str, src_n: int, src_t: str, off: int, n: int) -> list[str]:
        """Static subview of a 1-D host buffer, cast back to a plain memref of n elements."""
        sv = f"memref<{n}x{src_t}, strided<[1], offset: {off}>>"
        return [
            f"%{tag} = memref.subview %{src}[{off}] [{n}] [1] : memref<{src_n}x{src_t}> to {sv}",
            f"%{tag}c = memref.reinterpret_cast %{tag} to offset: [0], sizes: [{n}], strides: [1] : {sv} to memref<{n}x{src_t}>",
        ]

    main = ["  aie.device(npu2) @main {",
            f"    aie.runtime_sequence @sequence(%x: memref<{x_n}x{x_t}>, %params: memref<{p_total}x{stages[0][1][1]}>, %y: memref<{y_n}x{y_t}>, %mid: memref<{mid_total}x{x_t}>) {{"]
    param_off, mid_off = 0, 0
    for index, ((in_n, in_t), (p_n, p_t), (out_n, out_t)) in enumerate(stages):
        lines = []
        if index == 0:
            lines += view(f"a{index}", "x", x_n, x_t, 0, in_n)
        else:
            lines += view(f"a{index}", "mid", mid_total, x_t, mid_off - mids[index - 1], in_n)
        lines += view(f"p{index}", "params", p_total, p_t, param_off, p_n)
        if index == len(stages) - 1:
            lines += view(f"o{index}", "y", y_n, y_t, 0, out_n)
        else:
            lines += view(f"o{index}", "mid", mid_total, x_t, mid_off, out_n)
            mid_off += out_n
        param_off += p_n
        main.append(f"      aiex.configure @dev{index} {{")
        main += ["        " + ln for ln in lines]
        main.append(
            f"        aiex.run @seq{index} (%a{index}c, %p{index}c, %o{index}c) : "
            f"(memref<{in_n}x{in_t}>, memref<{p_n}x{p_t}>, memref<{out_n}x{out_t}>)"
        )
        main.append("      }")
    main += ["    }", "  }"]
    return "module {\n" + "\n".join(main) + "\n" + "\n".join(devices) + "\n}\n", {
        "x": x_n, "params": p_total, "y": y_n, "mid": mid_total, "param_offsets": [sum(s[1][0] for s in stages[:i]) for i in range(len(stages))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prj", type=Path, nargs="+", required=True, help="stage .prj directories in dispatch order")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--emit-mlir", type=Path, help="only write the merged MLIR here")
    parser.add_argument("--full-elf", type=Path)
    args = parser.parse_args()
    mlir, meta = compose(args.prj, args.work_dir)
    if args.emit_mlir:
        args.emit_mlir.write_text(mlir)
    print(meta)
    if args.full_elf:
        from aie.utils.compile.utils import compile_mlir_module
        args.work_dir.joinpath("aie.mlir").write_text(mlir)
        compile_mlir_module(mlir, full_elf_path=args.full_elf.resolve(), work_dir=str(args.work_dir.resolve()))


if __name__ == "__main__":
    main()
