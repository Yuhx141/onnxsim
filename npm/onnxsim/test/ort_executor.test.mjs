import assert from "node:assert/strict";
import { makeOrtRunner } from "../../../scripts/convertmodel/ort_executor.mjs";

function fakeOrt() {
  return {
    InferenceSession: {
      async create() {
        return {
          inputNames: ["x"],
          outputNames: ["y"],
          async run(feeds) {
            assert.deepEqual(feeds.x.dims, [2]);
            assert.deepEqual(Array.from(feeds.x.data), [1, 2]);
            return { y: { type: "float32", data: new Float32Array([3, 4]), dims: [2] } };
          },
          async release() {},
        };
      },
    },
    Tensor: class {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    },
  };
}

const runner = makeOrtRunner(fakeOrt());
const input = new Float32Array([1, 2]);
const result = await runner(
  new Uint8Array([1]),
  new Uint8Array(input.buffer),
  new Float64Array([1, 1, 2]),
);
assert.deepEqual(Array.from(new Float32Array(result.data.buffer)), [3, 4]);
assert.equal(result.profile[0].name, "ort_web_run");

await assert.rejects(
  () => runner(new Uint8Array(), new Uint8Array(4), new Float64Array([1])),
  /truncated input metadata/,
);
await assert.rejects(
  () => runner(new Uint8Array(), new Uint8Array(4), new Float64Array([1, 1, 2])),
  /input blob is too short/,
);
await assert.rejects(
  () => runner(new Uint8Array(), new Uint8Array(8), new Float64Array([1, 1, 2, 1, 1, 2])),
  /too many input tensors/,
);

console.log("ort executor contract tests passed");
