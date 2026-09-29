import numpy as np, onnxruntime as ort, glob, os, collections

x = np.array(
    [((i * 2654435761) % 1000) / 1000 - 0.5 for i in range(3 * 32 * 32)], dtype="f"
).reshape(1, 3, 32, 32)
res = []
for f in sorted(glob.glob(os.environ.get("MODELS", "m") + "/*.onnx")):
    n = os.path.basename(f)[:-5]
    try:
        h = (
            ort.InferenceSession(f, providers=["CPUExecutionProvider"])
            .run(None, {"X": x})[0]
            .astype("f")
        )
    except Exception as e:
        res.append((n, "HOSTFAIL", str(e)[:60]))
        continue
    if not os.path.exists(f"out/{n}.bin") or os.path.getsize(f"out/{n}.bin") == 0:
        res.append(
            (
                n,
                "NOOUT",
                open(f"err/{n}.txt").read().strip().splitlines()[-1][:110]
                if os.path.getsize(f"err/{n}.txt")
                else "",
            )
        )
        continue
    p = np.fromfile(f"out/{n}.bin", dtype="f")
    if p.size != h.size:
        res.append((n, "SIZE", f"{p.size} vs {h.size}"))
        continue
    e = np.abs(p.reshape(h.shape) - h)
    tol = 1e-3 + 1e-3 * np.abs(h)
    bad = float((e > tol).mean())
    res.append((n, "OK" if bad == 0 else "BAD", f"bad={bad:.3f} maxerr={e.max():.3g}"))
c = collections.Counter(r[1] for r in res)
print(c)
for r in res:
    if r[1] != "OK":
        print(f"{r[1]:8s} {r[0]:22s} {r[2]}")
