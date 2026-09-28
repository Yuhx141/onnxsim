#!/usr/bin/env python3
"""Compile a tinygrad-generated Hexagon kernel (from capture_kernel.py) with the real Hexagon
toolchain, wrap it in a thin TVM PackedFunc ABI shim (wrapper_template.c), link both into one
.so via TVM's own tvm.contrib.hexagon.tools.link_shared, and run it for real on the phone
through TVM's existing, already-permitted Hexagon RPC session (session.load_module +
mod.get_function) -- see capture_kernel.py's docstring for why this transport, not tinygrad's
own, is what actually reaches the DSP on a production Android build.

Needs: $HEXAGON_TOOLCHAIN pointed at the Hexagon SDK's HEXAGON_Tools (the same one TVM's own
Hexagon build uses), and $PYTHONPATH/$TVM_LIBRARY_PATH set up for a TVM built with Hexagon
support (tvm.contrib.hexagon), matching whatever set up the rest of scripts/android/maskrcnn_e2e.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import tvm
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import link_shared
from tvm.rpc.tracker import Tracker

HERE = Path(__file__).parent


def find_kernel_name(kernel_src: str) -> str:
    m = re.search(r"void (r_[0-9_]+)\(", kernel_src)
    if not m:
        raise ValueError("couldn't find a tinygrad-style kernel function (void r_...(...)) in the source")
    return m.group(1)


def compile_kernel(
    hexagon_clang: Path, kernel_c: Path, out_o: Path, hex_arch: str, explicit_target: bool
) -> None:
    import subprocess

    target_args = ["--target=hexagon"] if explicit_target else []
    subprocess.run(
        [
            str(hexagon_clang), *target_args, "-c", "-O2", "-Wall", "-fno-stack-protector", "-x", "c", "-fPIC",
            "-ffreestanding", f"-mcpu=hexagon{hex_arch}", f"-mhvx={hex_arch}", "-mhvx-length=128b",
            "-o", str(out_o), str(kernel_c),
        ],
        check=True,
    )


def compile_wrapper(
    hexagon_clang: Path, wrapper_c: Path, out_o: Path, hex_arch: str, tvm_root: Path, explicit_target: bool
) -> None:
    import subprocess

    target_args = ["--target=hexagon"] if explicit_target else []
    subprocess.run(
        [
            str(hexagon_clang), *target_args, "-c", "-O2", "-fPIC", f"-mcpu=hexagon{hex_arch}", f"-mhvx={hex_arch}",
            "-mhvx-length=128b", "-I", str(tvm_root / "include"),
            "-I", str(tvm_root / "3rdparty" / "dlpack" / "include"), "-o", str(out_o), str(wrapper_c),
        ],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", type=Path, default=Path("kernel.c"), help="output of capture_kernel.py")
    p.add_argument("--workdir", type=Path, default=Path("bridge_work"))
    p.add_argument("--device", default="239dbd8f")
    p.add_argument("--hexagon-toolchain", type=Path, required=True, help="$HEXAGON_TOOLCHAIN")
    p.add_argument("--hexagon-clang", type=Path, help="override SDK compiler (useful for older targets)")
    p.add_argument("--tvm-root", type=Path, required=True, help="TVM source root (for headers)")
    p.add_argument("--hex-arch", default="v73")
    p.add_argument("--link-arch", help="link support libraries for this arch (defaults to --hex-arch)")
    p.add_argument("--hw", type=int, default=512)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=256)
    p.add_argument("--rpc-port", type=int, default=9193)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    args.workdir.mkdir(exist_ok=True)
    kernel_src = args.kernel.read_text()
    kernel_name = find_kernel_name(kernel_src)
    wrapper_name = "tinygrad_gemm"

    kernel_o = args.workdir / "kernel.o"
    wrapper_c = args.workdir / "wrapper.c"
    wrapper_o = args.workdir / "wrapper.o"
    so_path = args.workdir / "bridge.so"

    hexagon_clang = args.hexagon_clang or (args.hexagon_toolchain / "bin" / "hexagon-clang")
    compile_kernel(hexagon_clang, args.kernel, kernel_o, args.hex_arch, args.hexagon_clang is not None)

    template = (HERE / "wrapper_template.c").read_text()
    wrapper_c.write_text(template.replace("KERNEL_NAME", kernel_name).replace("WRAPPER_NAME", wrapper_name))
    compile_wrapper(hexagon_clang, wrapper_c, wrapper_o, args.hex_arch, args.tvm_root, args.hexagon_clang is not None)

    link_shared(str(so_path), [str(kernel_o), str(wrapper_o)], {"hex_arch": args.link_arch or args.hex_arch})
    print(f"linked {so_path} ({so_path.stat().st_size} bytes)")

    rng = np.random.default_rng(5)
    a_np = rng.integers(0, 100, (args.hw, args.cin)).astype(np.uint8)
    b_np = rng.integers(0, 100, (args.cin, args.cout)).astype(np.uint8)
    ref = a_np.astype(np.int64) @ b_np.astype(np.int64)
    macs = args.hw * args.cin * args.cout

    target = tvm.target.hexagon(args.hex_arch)
    tracker = Tracker(host="127.0.0.1", port=args.rpc_port)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1", "rpc_tracker_port": args.rpc_port,
            "rpc_server_port": args.rpc_port - 1120, "workspace_base": "/data/local/tmp/tg_hex_bridge",
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        with launcher.create_session() as session:
            remote_path = session.upload(str(so_path), so_path.name)
            mod = session.load_module(str(remote_path))
            fn = mod.get_function(wrapper_name)
            dev = session.device
            a = tvm.nd.array(a_np, device=dev)
            b = tvm.nd.array(b_np, device=dev)
            out = tvm.nd.array(np.zeros((args.hw, args.cout), dtype=np.int32), device=dev)
            fn(out, a, b)
            match = bool(np.array_equal(out.numpy().astype(np.int64), ref))
            print(f"correctness match on real hardware: {match}")
            times = []
            for _ in range(args.repeat):
                t0 = time.time()
                fn(out, a, b)
                times.append(time.time() - t0)
            median = float(np.median(times))
            print(f"median={median * 1e3:.3f} ms  {macs / median / 1e9:.2f} GMAC/s")
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
