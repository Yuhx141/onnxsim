# Adreno 730: padded shared-memory tile transpose miscompiles (Dawn/Vulkan)

`harness.cc` runs a WGSL transpose kernel through Dawn (the same Dawn/Tint that ORT's WebGPU EP
builds) and checks `o[c*R+r] == a[r*C+c]`. Build it against the static libs of an ORT
`--use_webgpu` build (`_deps/dawn-build`, abseil, spirv), e.g. for Android:

    clang++ -std=c++20 -static-libstdc++ -I <build>/_deps/dawn-src/include \
      -I <build>/_deps/dawn-build/gen/include harness.cc -o harness \
      -Wl,--start-group $(find <build>/_deps -name '*.a') -Wl,--end-group -landroid -llog -ldl
    ./harness k0_ort.wgsl R C ceil(R/16) ceil(C/16)

`k0_ort.wgsl` is ORT's tiled Transpose (`tile: array<array<f32, 17>, 16>`, workgroup 16x16, guarded
store, `workgroupBarrier()`). Results on Adreno 730 (Qualcomm Vulkan 512.615.0, compiler
EV031.36.08.11, vendor `qualcomm`, arch `adreno-7xx`); NVIDIA RTX 5050, RADV and lavapipe are
correct for every kernel and shape here:

| kernel | change from k0 | Adreno 730 (96x32) |
|---|---|---|
| k0_ort | (ORT's kernel) | 27% wrong (also 1x16: 50%, 3x1024: 50%, 256x256: 26%) |
| k1_nopad | tile row stride 16 instead of 17 | correct on 8 shapes incl. 1x16, 33x17, 3x1024 |
| k2_flat | flat `array<f32,272>`, stride 17 | 25% wrong (not the nested array) |
| k4_noguard | drop the two bounds `if`s | correct (needs exact-multiple shapes; not a fix) |
| k5_storebar | + `storageBarrier()` | 2.4% wrong (looks like a race) |

So the odd (17) padded stride combined with the guarded store trips the Adreno compiler; the
unpadded tile is correct. Tint has Qualcomm workarounds (matrix pass-by-pointer, std140, NClamp,
uniform vector component loads) but none for workgroup memory or barriers.
`../ort_transpose_qualcomm_unpadded_tile.patch` makes ORT use the unpadded tile when the adapter
vendor is `qualcomm`; with it the 235-op sweep passes 230/235 on the phone (the other 5 are 1-ulp
input ties that the phone CPU EP shows too).
