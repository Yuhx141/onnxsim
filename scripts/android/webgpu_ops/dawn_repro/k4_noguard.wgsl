const tile_size = 16u;
var<workgroup> tile : array<array<f32, tile_size + 1>, tile_size>;
struct U { rows: u32, cols: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let workgroup_idx = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
  let stride = (u.rows - 1) / tile_size + 1;
  let workgroup_id_x = workgroup_idx % stride;
  let workgroup_id_y = workgroup_idx / stride;
  let input_col = workgroup_id_y * tile_size + local_id.x;
  let input_row = workgroup_id_x * tile_size + local_id.y;
  {
    tile[local_id.y][local_id.x] = a[input_row * u.cols + input_col];
  }
  workgroupBarrier();
  let output_col = workgroup_id_x * tile_size + local_id.x;
  let output_row = workgroup_id_y * tile_size + local_id.y;
  {
    o[output_row * u.rows + output_col] = tile[local_id.x][local_id.y];
  }
}
