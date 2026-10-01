// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See adaquant_entry.h for the full rationale (including why this differs
// from adaround_entry.h's own weight-only-INT4 candidate shape) and
// onnxsim/adaquant.py for the technique this ports.

#include "adaquant_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

// Rectified-sigmoid relaxation constants -- transcribed from adaround.py's
// own _ZETA/_GAMMA (adaquant.py imports the same two constants).
constexpr double kZeta = 1.1;
constexpr double kGamma = -0.1;

constexpr double kWeightNMin = -127.0;
constexpr double kWeightNMax = 127.0;
constexpr double kActNMax = 255.0;

// --- Candidate matching ----------------------------------------------------
//
// Transcribed from adaquant.py's own _find_static_qdq_candidates: joins the
// float and quantized models by node output tensor name, same as
// gptq_entry.cpp's/adaround_entry.cpp's own FindInt4MatmulCandidates, but
// matching onnxsim.quantize_static's W8A8 QDQ shape instead of
// quantize_weight_only_int4's blocked-INT4 one -- see adaquant_entry.h's
// own comment for the exact graph shape matched.

struct Candidate {
  std::string output_name;
  std::string float_x_name;
  std::string w_float_name;
  std::string wq_name;
  std::string ws_name;
  bool weight_transposed;
  std::string x_scale_name;
  std::string x_zp_name;
};

int64_t GetIntAttr(const onnx::NodeProto& node, const std::string& name,
                   int64_t fallback) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name && attr.type() == onnx::AttributeProto::INT) {
      return attr.i();
    }
  }
  return fallback;
}

// Reads an INT8 tensor's elements as plain int64_t codes, from either
// raw_data (little-endian bytes, one per element) or the packed-int32
// int32_data fallback -- mirrors ReadFloatTensor's own two-branch shape
// throughout this codebase, narrowed to INT8.
std::vector<int64_t> ReadInt8Tensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<int64_t> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = static_cast<int64_t>(
          static_cast<int8_t>(t.raw_data()[static_cast<size_t>(i)]));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.int32_data(static_cast<int>(i));
    }
  }
  return out;
}

// Same as ReadInt8Tensor, for a UINT8 tensor: raw bytes must be read
// unsigned, or a zero point >= 128 comes back negative.
std::vector<int64_t> ReadUint8Tensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<int64_t> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = static_cast<int64_t>(
          static_cast<uint8_t>(t.raw_data()[static_cast<size_t>(i)]));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.int32_data(static_cast<int>(i));
    }
  }
  return out;
}

std::vector<float> ReadFloatTensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(out.data(), t.raw_data().data(), out.size() * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                        out.size() * sizeof(float),
                                        sizeof(float));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.float_data(static_cast<int>(i));
    }
  }
  return out;
}

std::vector<Candidate> FindStaticQdqCandidates(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model,
    std::unordered_map<std::string, int>& q_init_index,
    std::unordered_map<std::string, int>& f_init_index) {
  const onnx::GraphProto& q_graph = quantized_model.graph();
  const onnx::GraphProto& f_graph = float_model.graph();

  for (int i = 0; i < q_graph.initializer_size(); ++i) {
    q_init_index.emplace(q_graph.initializer(i).name(), i);
  }
  for (int i = 0; i < f_graph.initializer_size(); ++i) {
    f_init_index.emplace(f_graph.initializer(i).name(), i);
  }

  std::vector<std::string> q_order;
  std::unordered_map<std::string, int> q_by_output;
  for (int i = 0; i < q_graph.node_size(); ++i) {
    const auto& n = q_graph.node(i);
    if (n.output_size() < 1) {
      continue;
    }
    if (!q_by_output.count(n.output(0))) {
      q_order.push_back(n.output(0));
    }
    q_by_output[n.output(0)] = i;
  }
  std::unordered_map<std::string, int> f_by_output;
  for (int i = 0; i < f_graph.node_size(); ++i) {
    const auto& n = f_graph.node(i);
    if (n.output_size() < 1) {
      continue;
    }
    f_by_output[n.output(0)] = i;
  }

  std::vector<Candidate> candidates;
  for (const auto& out_name : q_order) {
    const onnx::NodeProto& qn = q_graph.node(q_by_output[out_name]);
    if ((qn.op_type() != "MatMul" && qn.op_type() != "Gemm") ||
        qn.input_size() < 2) {
      continue;
    }
    // quantize_static keeps the layer's output name; quantize_full_qdq
    // renames the pre-requantize tensor "<name>/f". (A layer whose Relu
    // full_qdq folded into the output Q is renamed after the Relu's output,
    // matches no float layer here, and is skipped.)
    auto fit = f_by_output.find(out_name);
    static const std::string kFullQdqSuffix = "/f";
    if (fit == f_by_output.end() && out_name.size() > kFullQdqSuffix.size() &&
        out_name.compare(out_name.size() - kFullQdqSuffix.size(),
                         kFullQdqSuffix.size(), kFullQdqSuffix) == 0) {
      fit = f_by_output.find(
          out_name.substr(0, out_name.size() - kFullQdqSuffix.size()));
    }
    if (fit == f_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& fn = f_graph.node(fit->second);
    if (fn.op_type() != qn.op_type() || fn.input_size() < 2) {
      continue;
    }

    auto wfit = f_init_index.find(fn.input(1));
    if (wfit == f_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& w_float_init = f_graph.initializer(wfit->second);
    if (w_float_init.data_type() != onnx::TensorProto::FLOAT ||
        w_float_init.dims_size() != 2) {
      continue;
    }

    // Weight branch: Wdq = DequantizeLinear(Wq, Ws, [Wzp], axis=...).
    auto wdqit = q_by_output.find(qn.input(1));
    if (wdqit == q_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& wdq = q_graph.node(wdqit->second);
    if (wdq.op_type() != "DequantizeLinear" ||
        (wdq.input_size() != 2 && wdq.input_size() != 3)) {
      continue;
    }
    auto wqit = q_init_index.find(wdq.input(0));
    auto wsit = q_init_index.find(wdq.input(1));
    if (wqit == q_init_index.end() || wsit == q_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& wq_init = q_graph.initializer(wqit->second);
    const onnx::TensorProto& ws_init = q_graph.initializer(wsit->second);
    if (wq_init.data_type() != onnx::TensorProto::INT8 ||
        wq_init.dims_size() != w_float_init.dims_size()) {
      continue;
    }
    bool same_dims = true;
    for (int d = 0; d < wq_init.dims_size(); ++d) {
      if (wq_init.dims(d) != w_float_init.dims(d)) {
        same_dims = false;
        break;
      }
    }
    if (!same_dims) {
      continue;
    }
    if (wdq.input_size() == 3) {
      auto wzpit = q_init_index.find(wdq.input(2));
      if (wzpit == q_init_index.end()) {
        continue;
      }
      const onnx::TensorProto& wzp_init = q_graph.initializer(wzpit->second);
      if (wzp_init.data_type() != onnx::TensorProto::INT8 ||
          wzp_init.dims_size() != ws_init.dims_size()) {
        continue;
      }
      bool wzp_dims_match = true;
      for (int d = 0; d < wzp_init.dims_size(); ++d) {
        if (wzp_init.dims(d) != ws_init.dims(d)) {
          wzp_dims_match = false;
          break;
        }
      }
      if (!wzp_dims_match) {
        continue;
      }
      const std::vector<int64_t> wzp_codes = ReadInt8Tensor(wzp_init);
      bool any_nonzero = false;
      for (int64_t v : wzp_codes) {
        if (v != 0) {
          any_nonzero = true;
          break;
        }
      }
      if (any_nonzero) {
        continue;
      }
    }

    // Activation branch: Xdq = DequantizeLinear(Xq, Xs, Xzp), Xq =
    // QuantizeLinear(X, Xs, Xzp), both sharing the same scale/zero-point.
    auto xdqit = q_by_output.find(qn.input(0));
    if (xdqit == q_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& xdq = q_graph.node(xdqit->second);
    if (xdq.op_type() != "DequantizeLinear" || xdq.input_size() != 3) {
      continue;
    }
    auto xqit = q_by_output.find(xdq.input(0));
    if (xqit == q_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& xq_node = q_graph.node(xqit->second);
    if (xq_node.op_type() != "QuantizeLinear" || xq_node.input_size() != 3) {
      continue;
    }
    const std::string& x_scale_name = xdq.input(1);
    const std::string& x_zp_name = xdq.input(2);
    if (xq_node.input(1) != x_scale_name || xq_node.input(2) != x_zp_name) {
      continue;
    }
    auto x_scale_it = q_init_index.find(x_scale_name);
    auto x_zp_it = q_init_index.find(x_zp_name);
    if (x_scale_it == q_init_index.end() || x_zp_it == q_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& x_zp_init = q_graph.initializer(x_zp_it->second);
    if (x_zp_init.data_type() != onnx::TensorProto::UINT8) {
      continue;
    }

    const bool weight_transposed =
        qn.op_type() == "Gemm" && GetIntAttr(qn, "transB", 0) != 0;
    candidates.push_back({out_name, fn.input(0), fn.input(1), wq_init.name(),
                          ws_init.name(), weight_transposed, x_scale_name,
                          x_zp_name});
  }
  return candidates;
}

// Round-half-to-even (banker's rounding), matching numpy's own `round` --
// transcribed from gptq_entry.cpp's own RoundHalfToEven.
double RoundHalfToEven(double v) {
  const double f = std::floor(v);
  const double d = v - f;
  if (d < 0.5) {
    return f;
  }
  if (d > 0.5) {
    return f + 1.0;
  }
  const double half = f / 2.0;
  return (half == std::floor(half)) ? f : f + 1.0;
}

void SetRawInitializer(onnx::TensorProto* t, const std::string& name,
                       int32_t data_type, const std::vector<int64_t>& dims,
                       const void* data, size_t bytes, size_t elem_size) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(static_cast<onnx::TensorProto::DataType>(data_type));
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(bytes, '\0');
  std::memcpy(raw.data(), data, bytes);
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      bytes, elem_size);
  }
  t->set_raw_data(std::move(raw));
}

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's/adaround_entry.cpp's own
// ActivationRows/AccumulateActivationRows.
struct ActivationRows {
  std::vector<double> data;  // Concatenated [total_rows, K], row-major.
  int64_t k = -1;
  bool ok = false;
};

void AccumulateActivationRows(
    std::unordered_map<std::string, ActivationRows>& acc,
    const ModelExecutor& executor, const onnx::ModelProto& float_model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  if (probe_names.empty()) {
    return;
  }

  onnx::ModelProto probe_model = float_model;
  std::unordered_set<std::string> existing_outputs;
  for (const auto& o : probe_model.graph().output()) {
    existing_outputs.insert(o.name());
  }
  for (const auto& name : probe_names) {
    if (existing_outputs.insert(name).second) {
      probe_model.mutable_graph()->add_output()->set_name(name);
    }
  }

  std::unordered_map<std::string, size_t> output_index;
  for (int i = 0; i < probe_model.graph().output_size(); ++i) {
    output_index.emplace(probe_model.graph().output(i).name(),
                         static_cast<size_t>(i));
  }
  const auto& graph_inputs = probe_model.graph().input();

  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> input_dls;
    std::vector<const DLManagedTensor*> input_ptrs;
    input_dls.reserve(static_cast<size_t>(graph_inputs.size()));
    input_ptrs.reserve(static_cast<size_t>(graph_inputs.size()));
    for (const auto& gi : graph_inputs) {
      auto it = batch.find(gi.name());
      if (it == batch.end()) {
        throw std::invalid_argument(
            "ApplyAdaquant: calibration batch is missing "
            "required graph input '" +
            gi.name() + "'");
      }
      input_dls.emplace_back(
          onnxsim::dlpack::FromTensorProtoBorrowing(it->second));
      input_ptrs.push_back(input_dls.back().get());
    }

    std::vector<DLManagedTensorPtr> outputs =
        executor.Run(probe_model, input_ptrs);

    for (const auto& name : probe_names) {
      auto oit = output_index.find(name);
      if (oit == output_index.end() || oit->second >= outputs.size()) {
        continue;
      }
      const DLTensor& dl = outputs[oit->second]->dl_tensor;
      onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 2) {
        continue;
      }
      const int64_t kk = tp.dims(static_cast<int>(tp.dims_size() - 1));
      if (kk <= 0) {
        continue;
      }
      ActivationRows& rows = acc[name];
      if (!rows.ok) {
        rows.k = kk;
        rows.ok = true;
      } else if (rows.k != kk) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      rows.data.reserve(rows.data.size() + data.size());
      for (float v : data) {
        rows.data.push_back(static_cast<double>(v));
      }
    }
  }
}

// --- Small dense matmul kernels, nested-vector ("Matrix") style ------------
using Matrix = std::vector<std::vector<double>>;

// y[S, N] = x[S, K] @ w[N, K]^T.
Matrix YFromXWt(const Matrix& x, const Matrix& w) {
  const size_t num_samples = x.size();
  const size_t k = x[0].size();
  const size_t n_rows = w.size();
  Matrix y(num_samples, std::vector<double>(n_rows, 0.0));
  for (size_t s = 0; s < num_samples; ++s) {
    for (size_t r = 0; r < n_rows; ++r) {
      double acc = 0.0;
      for (size_t c = 0; c < k; ++c) {
        acc += x[s][c] * w[r][c];
      }
      y[s][r] = acc;
    }
  }
  return y;
}

// out[N, K] = a[S, N]^T @ b[S, K].
Matrix AtB(const Matrix& a, const Matrix& b) {
  const size_t num_samples = a.size();
  const size_t n_rows = a[0].size();
  const size_t k = b[0].size();
  Matrix out(n_rows, std::vector<double>(k, 0.0));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      double acc = 0.0;
      for (size_t s = 0; s < num_samples; ++s) {
        acc += a[s][r] * b[s][c];
      }
      out[r][c] = acc;
    }
  }
  return out;
}

// out[S, K] = a[S, N] @ b[N, K].
Matrix AB(const Matrix& a, const Matrix& b) {
  const size_t num_samples = a.size();
  const size_t n_rows = b.size();
  const size_t k = b[0].size();
  Matrix out(num_samples, std::vector<double>(k, 0.0));
  for (size_t s = 0; s < num_samples; ++s) {
    for (size_t r = 0; r < n_rows; ++r) {
      const double av = a[s][r];
      if (av == 0.0) {
        continue;
      }
      for (size_t c = 0; c < k; ++c) {
        out[s][c] += av * b[r][c];
      }
    }
  }
  return out;
}

struct AdaquantResult {
  Matrix codes_nk;
  double x_scale;
  int64_t x_zp;
};

// AdaQuant's own joint Adam loop -- transcribed scalar-loop-for-scalar-loop
// from adaquant.py's own _optimize_adaquant. See adaquant_entry.h's own
// accepted numerical scope note.
AdaquantResult OptimizeAdaquant(
    const Matrix& w_nk, const std::vector<double>& scale_n, const Matrix& x,
    double x_scale0, double x_zp0, int64_t num_iterations,
    double weight_learning_rate, double activation_learning_rate,
    double reg_param, double warm_start, double beta_start, double beta_end) {
  const size_t n_rows = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_samples = x.size();

  Matrix scale_nk(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      scale_nk[r][c] = scale_n[r];
    }
  }

  Matrix floor_base(n_rows, std::vector<double>(k));
  Matrix v(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const double ratio = w_nk[r][c] / scale_nk[r][c];
      const double fb = std::floor(ratio);
      const double frac = std::clamp(ratio - fb, 1e-4, 1.0 - 1e-4);
      const double sig0 =
          std::clamp((frac - kGamma) / (kZeta - kGamma), 1e-4, 1.0 - 1e-4);
      floor_base[r][c] = fb;
      v[r][c] = std::log(sig0 / (1.0 - sig0));
    }
  }
  double log_s = std::log(std::max(x_scale0, 1e-8));
  double zp = std::clamp(x_zp0, 0.0, kActNMax);

  Matrix m_v(n_rows, std::vector<double>(k, 0.0));
  Matrix v2_v(n_rows, std::vector<double>(k, 0.0));
  double m_s = 0.0, v2_s = 0.0, m_zp = 0.0, v2_zp = 0.0;
  constexpr double kAdamBeta1 = 0.9, kAdamBeta2 = 0.999, kAdamEps = 1e-8;

  const int64_t warm_start_iters =
      static_cast<int64_t>(static_cast<double>(num_iterations) * warm_start);
  const double n_elems = static_cast<double>(num_samples * n_rows);

  const Matrix y_float = YFromXWt(x, w_nk);

  for (int64_t t = 0; t < num_iterations; ++t) {
    const double s_x = std::exp(log_s);

    Matrix h(n_rows, std::vector<double>(k));
    Matrix dh_dv(n_rows, std::vector<double>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        const double s = 1.0 / (1.0 + std::exp(-v[r][c]));
        const double raw = s * (kZeta - kGamma) + kGamma;
        const bool active_h = raw > 0.0 && raw < 1.0;
        h[r][c] = std::clamp(raw, 0.0, 1.0);
        dh_dv[r][c] = active_h ? s * (1.0 - s) * (kZeta - kGamma) : 0.0;
      }
    }

    Matrix w_hat(n_rows, std::vector<double>(k));
    std::vector<std::vector<uint8_t>> active_w(n_rows, std::vector<uint8_t>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        const double raw_w = floor_base[r][c] + h[r][c];
        const double cl = std::clamp(raw_w, kWeightNMin, kWeightNMax);
        active_w[r][c] = (raw_w > kWeightNMin && raw_w < kWeightNMax) ? 1 : 0;
        w_hat[r][c] = cl * scale_nk[r][c];
      }
    }

    Matrix xq(num_samples, std::vector<double>(k));
    Matrix xdq(num_samples, std::vector<double>(k));
    std::vector<std::vector<uint8_t>> active_x(num_samples,
                                               std::vector<uint8_t>(k));
    for (size_t s = 0; s < num_samples; ++s) {
      for (size_t c = 0; c < k; ++c) {
        const double rounded = RoundHalfToEven(x[s][c] / s_x);
        const double xq_raw = rounded + zp;
        active_x[s][c] = (xq_raw > 0.0 && xq_raw < kActNMax) ? 1 : 0;
        const double clipped = std::clamp(xq_raw, 0.0, kActNMax);
        xq[s][c] = clipped;
        xdq[s][c] = (clipped - zp) * s_x;
      }
    }

    const Matrix y_hat = YFromXWt(xdq, w_hat);
    Matrix dl_dy(num_samples, std::vector<double>(n_rows));
    for (size_t s = 0; s < num_samples; ++s) {
      for (size_t r = 0; r < n_rows; ++r) {
        dl_dy[s][r] = 2.0 * (y_hat[s][r] - y_float[s][r]) / n_elems;
      }
    }
    const Matrix dl_dw_hat = AtB(dl_dy, xdq);  // [N, K]

    Matrix grad_v(n_rows, std::vector<double>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        const double dl_dh =
            dl_dw_hat[r][c] * (active_w[r][c] ? scale_nk[r][c] : 0.0);
        grad_v[r][c] = dl_dh * dh_dv[r][c];
      }
    }

    if (t >= warm_start_iters) {
      const double denom = static_cast<double>(
          std::max<int64_t>(1, num_iterations - warm_start_iters - 1));
      const double progress = static_cast<double>(t - warm_start_iters) / denom;
      const double beta = beta_start + (beta_end - beta_start) * progress;
      for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < k; ++c) {
          const double u = 2.0 * h[r][c] - 1.0;
          const double abs_u = std::fabs(u);
          const double sign_u = (u > 0.0) - (u < 0.0);
          const double dreg_dh =
              -2.0 * reg_param * beta * sign_u * std::pow(abs_u, beta - 1.0);
          grad_v[r][c] += dreg_dh * dh_dv[r][c];
        }
      }
    }

    const Matrix dl_dxdq = AB(dl_dy, w_hat);  // [S, K]
    double grad_log_s_sum = 0.0;
    double grad_zp = 0.0;
    for (size_t s = 0; s < num_samples; ++s) {
      for (size_t c = 0; c < k; ++c) {
        const double dxdq_ds =
            (xq[s][c] - zp) - (active_x[s][c] ? x[s][c] / s_x : 0.0);
        const double dxdq_dzp =
            s_x * (static_cast<double>(active_x[s][c]) - 1.0);
        grad_log_s_sum += dl_dxdq[s][c] * dxdq_ds;
        grad_zp += dl_dxdq[s][c] * dxdq_dzp;
      }
    }
    const double grad_log_s = grad_log_s_sum * s_x;

    const double bias_c1 =
        1.0 - std::pow(kAdamBeta1, static_cast<double>(t + 1));
    const double bias_c2 =
        1.0 - std::pow(kAdamBeta2, static_cast<double>(t + 1));

    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        m_v[r][c] = kAdamBeta1 * m_v[r][c] + (1.0 - kAdamBeta1) * grad_v[r][c];
        v2_v[r][c] = kAdamBeta2 * v2_v[r][c] +
                     (1.0 - kAdamBeta2) * grad_v[r][c] * grad_v[r][c];
        v[r][c] -= weight_learning_rate * (m_v[r][c] / bias_c1) /
                   (std::sqrt(v2_v[r][c] / bias_c2) + kAdamEps);
      }
    }

    m_s = kAdamBeta1 * m_s + (1.0 - kAdamBeta1) * grad_log_s;
    v2_s = kAdamBeta2 * v2_s + (1.0 - kAdamBeta2) * grad_log_s * grad_log_s;
    log_s -= activation_learning_rate * (m_s / bias_c1) /
             (std::sqrt(v2_s / bias_c2) + kAdamEps);

    m_zp = kAdamBeta1 * m_zp + (1.0 - kAdamBeta1) * grad_zp;
    v2_zp = kAdamBeta2 * v2_zp + (1.0 - kAdamBeta2) * grad_zp * grad_zp;
    zp -= activation_learning_rate * (m_zp / bias_c1) /
          (std::sqrt(v2_zp / bias_c2) + kAdamEps);
    zp = std::clamp(zp, 0.0, kActNMax);
  }

  Matrix codes(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const double s = 1.0 / (1.0 + std::exp(-v[r][c]));
      const double raw = s * (kZeta - kGamma) + kGamma;
      const double h_final = std::clamp(raw, 0.0, 1.0);
      codes[r][c] = std::clamp(floor_base[r][c] + RoundHalfToEven(h_final),
                               kWeightNMin, kWeightNMax);
    }
  }
  const double final_scale = std::exp(log_s);
  const int64_t final_zp =
      static_cast<int64_t>(std::clamp(RoundHalfToEven(zp), 0.0, 255.0));
  return {codes, final_scale, final_zp};
}

}  // namespace

onnx::ModelProto ApplyAdaquant(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations, double weight_learning_rate,
    double activation_learning_rate, double reg_param, double warm_start,
    double beta_start, double beta_end) {
  std::unordered_map<std::string, int> q_init_index;
  std::unordered_map<std::string, int> f_init_index;
  const std::vector<Candidate> candidates = FindStaticQdqCandidates(
      float_model, quantized_model, q_init_index, f_init_index);
  if (candidates.empty()) {
    return quantized_model;
  }
  const onnx::GraphProto& f_graph = float_model.graph();
  const onnx::GraphProto& q_graph = quantized_model.graph();

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.float_x_name);
  }
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, float_model, probe_names,
                           calibration_data);

  std::unordered_map<std::string, std::vector<int8_t>> optimized_codes;
  std::unordered_map<std::string, float> optimized_scale;
  std::unordered_map<std::string, uint8_t> optimized_zp;

  for (const auto& c : candidates) {
    auto ait = activations.find(c.float_x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable activation -- leave untouched.
    }
    const ActivationRows& rows = ait->second;

    const onnx::TensorProto& w_float_init =
        f_graph.initializer(f_init_index[c.w_float_name]);
    const int64_t dim0 = w_float_init.dims(0);
    const int64_t dim1 = w_float_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (rows.k != k) {
      continue;  // Activation's feature dim doesn't match K -- skip.
    }
    const int64_t num_samples =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);

    const std::vector<float> w_flat = ReadFloatTensor(w_float_init);
    Matrix w_nk(static_cast<size_t>(n_rows),
                std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(i * k + j)]
                    : w_flat[static_cast<size_t>(j * n_rows + i)]);
      }
    }

    const onnx::TensorProto& ws_init =
        q_graph.initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);
    if (static_cast<int64_t>(s_flat.size()) != n_rows) {
      continue;  // scale_n must have exactly N elements.
    }
    std::vector<double> scale_n(static_cast<size_t>(n_rows));
    for (int64_t i = 0; i < n_rows; ++i) {
      scale_n[static_cast<size_t>(i)] =
          static_cast<double>(s_flat[static_cast<size_t>(i)]);
    }

    const onnx::TensorProto& x_scale_init =
        q_graph.initializer(q_init_index[c.x_scale_name]);
    const onnx::TensorProto& x_zp_init =
        q_graph.initializer(q_init_index[c.x_zp_name]);
    const std::vector<float> x_scale_flat = ReadFloatTensor(x_scale_init);
    const std::vector<int64_t> x_zp_flat = ReadUint8Tensor(x_zp_init);
    if (x_scale_flat.empty() || x_zp_flat.empty()) {
      continue;
    }
    const double x_scale0 = static_cast<double>(x_scale_flat[0]);
    const double x_zp0 = static_cast<double>(x_zp_flat[0]);

    Matrix x(static_cast<size_t>(num_samples),
             std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < num_samples; ++r) {
      for (int64_t c2 = 0; c2 < k; ++c2) {
        x[static_cast<size_t>(r)][static_cast<size_t>(c2)] =
            rows.data[static_cast<size_t>(r * k + c2)];
      }
    }

    const AdaquantResult result = OptimizeAdaquant(
        w_nk, scale_n, x, x_scale0, x_zp0, num_iterations, weight_learning_rate,
        activation_learning_rate, reg_param, warm_start, beta_start, beta_end);

    std::vector<int8_t> codes_flat(static_cast<size_t>(dim0 * dim1));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          codes_flat[static_cast<size_t>(i * dim1 + j)] = static_cast<int8_t>(
              static_cast<int64_t>(result.codes_nk[static_cast<size_t>(i)]
                                                  [static_cast<size_t>(j)]));
        }
      }
    } else {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          codes_flat[static_cast<size_t>(i * dim1 + j)] = static_cast<int8_t>(
              static_cast<int64_t>(result.codes_nk[static_cast<size_t>(j)]
                                                  [static_cast<size_t>(i)]));
        }
      }
    }
    optimized_codes.emplace(c.wq_name, std::move(codes_flat));
    optimized_scale[c.x_scale_name] = static_cast<float>(result.x_scale);
    optimized_zp[c.x_zp_name] = static_cast<uint8_t>(result.x_zp);
  }

  if (optimized_codes.empty()) {
    return quantized_model;
  }
  onnx::ModelProto corrected = quantized_model;
  for (auto& t : *corrected.mutable_graph()->mutable_initializer()) {
    // Original name/dims are captured into local copies before
    // SetRawInitializer clears `t` -- `t.name()`/`t.dims()` are references
    // into `t`'s own fields, so passing them straight through as
    // SetRawInitializer's arguments would alias memory that function's own
    // `t->Clear()` wipes before `set_name`/dims are rebuilt from them
    // (every rewrite here keeps the existing initializer's own name and
    // shape -- codes: same shape as the INT8 weight; scale/zero_point:
    // whatever 0-d/1-element shape quantize_static originally gave them).
    const std::string name = t.name();
    const std::vector<int64_t> dims(t.dims().begin(), t.dims().end());
    auto cit = optimized_codes.find(name);
    if (cit != optimized_codes.end()) {
      SetRawInitializer(&t, name, onnx::TensorProto::INT8, dims,
                        cit->second.data(), cit->second.size(), sizeof(int8_t));
      continue;
    }
    auto sit = optimized_scale.find(name);
    if (sit != optimized_scale.end()) {
      SetRawInitializer(&t, name, onnx::TensorProto::FLOAT, dims, &sit->second,
                        sizeof(float), sizeof(float));
      continue;
    }
    auto zit = optimized_zp.find(name);
    if (zit != optimized_zp.end()) {
      SetRawInitializer(&t, name, onnx::TensorProto::UINT8, dims, &zit->second,
                        sizeof(uint8_t), sizeof(uint8_t));
    }
  }
  return corrected;
}
