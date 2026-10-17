#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>

#include <cstdlib>
#include <string>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#endif

namespace at::native {

namespace {

// ---------------------------------------------------------------------------
// Backend selection for torch._grouped_mm on XPU.
//
//   TORCH_XPU_GROUPED_MM_BACKEND = auto (default) | sycltla | onednn | fallback
//
//   auto      SYCL-TLA when it can serve the call and groups are small (see
//             kMaxAvgRowsPerGroup), else oneDNN when available, else the
//             per-group fallback.
//   sycltla   force SYCL-TLA for every call it can serve (no size heuristic).
//   onednn    force the oneDNN grouped matmul for every call it can serve.
//   fallback  always use the per-group fallback (torch.mm per group).
//
// A forced backend that cannot serve a call (unsupported layout, dtype, bias,
// or not built) falls back to the per-group path; the reason is printed once
// as a warning and, with TORCH_XPU_GROUPED_MM_LOG set, per decision.
//
// The variable is read on every call, so it can be changed at runtime.
// ---------------------------------------------------------------------------
enum class Backend { Auto, SyclTla, OneDnn, Fallback };

Backend requested_backend() {
  const char* v = std::getenv("TORCH_XPU_GROUPED_MM_BACKEND");
  if (v == nullptr || v[0] == '\0') {
    return Backend::Auto;
  }
  const std::string s(v);
  if (s == "auto") {
    return Backend::Auto;
  }
  if (s == "sycltla" || s == "sycl-tla") {
    return Backend::SyclTla;
  }
  if (s == "onednn") {
    return Backend::OneDnn;
  }
  if (s == "fallback") {
    return Backend::Fallback;
  }
  TORCH_WARN_ONCE(
      "Ignoring unknown TORCH_XPU_GROUPED_MM_BACKEND='", s,
      "'; expected auto, sycltla, onednn or fallback.");
  return Backend::Auto;
}

// The persistent SYCL-TLA kernel wins clearly when groups are small (many
// experts with a few hundred rows each: ~2x over a loop of dense GEMMs on
// PVC). For large per-group problems (e.g. DeepSeek-V3 16B with EP=8, ~6k
// rows per expert) a loop of dense GEMMs is 5-12% faster, so `auto` keeps
// those off SYCL-TLA.
constexpr int64_t kMaxAvgRowsPerGroup = 1024;

// Rows that `offs` partitions, averaged over groups: mat_a rows for 2D x 3D,
// the contraction dimension for 2D x 2D.
int64_t avg_rows_per_group(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& offs) {
  const int64_t groups = offs.numel();
  const int64_t rows = (mat_a.dim() == 2 && mat_b.dim() == 3)
      ? mat_a.size(0)
      : mat_a.size(-1);
  return groups > 0 ? rows / groups : 0;
}

// Checks shared by every fast backend.
bool common_fast_path_ok(
    const Tensor& mat_a,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    c10::ScalarType out_dtype,
    std::string& why) {
  if (!offs.has_value()) {
    why = "no offs (3D x 3D batched)";
    return false;
  }
  if (bias.has_value()) {
    why = "bias";
    return false;
  }
  if (out_dtype != mat_a.scalar_type()) {
    why = "out_dtype differs from input dtype";
    return false;
  }
  return true;
}

// ---- oneDNN backend slot ---------------------------------------------------
// Reserved for the oneDNN grouped matmul from pytorch/pytorch#195427
// (at::native::onednn::scaled_grouped_matmul, 2D x 3D). Until that lands this
// build has no oneDNN grouped kernel and the slot reports "not available".
// Wiring it in is: include <ATen/native/mkldnn/xpu/detail/oneDNN.h>, accept
// 2D x 3D here, and in run_onednn_grouped_mm() call
//   at::native::onednn::scaled_grouped_matmul(mat_a, mat_b, std::nullopt,
//       std::nullopt, std::nullopt, std::nullopt, offs, out);
bool onednn_grouped_mm_can_serve(
    const Tensor& /*mat_a*/,
    const Tensor& /*mat_b*/,
    const Tensor& /*offs*/,
    std::string& why) {
  why = "oneDNN grouped matmul not in this build (pending pytorch/pytorch#195427)";
  return false;
}

void run_onednn_grouped_mm(
    const Tensor& /*mat_a*/,
    const Tensor& /*mat_b*/,
    const Tensor& /*offs*/,
    Tensor& /*out*/) {
  TORCH_INTERNAL_ASSERT(false, "oneDNN grouped matmul backend is not built");
}

// ---------------------------------------------------------------------------

struct Decision {
  Backend backend = Backend::Fallback;
  std::string reason;      // why the fallback was chosen (empty otherwise)
  std::string reason_key;  // stable part of `reason`, for log de-duplication
};

Decision choose_backend(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    c10::ScalarType out_dtype,
    const Tensor& out) {
  const Backend req = requested_backend();
  Decision d;
  if (req == Backend::Fallback) {
    d.reason = d.reason_key = "TORCH_XPU_GROUPED_MM_BACKEND=fallback";
    return d;
  }
  std::string why;
  if (!common_fast_path_ok(mat_a, offs, bias, out_dtype, why)) {
    d.reason = d.reason_key = why;
    return d;
  }

  std::string sycltla_why;
  bool sycltla_ok = false;
  if (req == Backend::Auto || req == Backend::SyclTla) {
    sycltla_ok = sycltla::can_use_grouped_mm(mat_a, mat_b, *offs, out, &sycltla_why);
  }
  if (req == Backend::SyclTla) {
    if (sycltla_ok) {
      d.backend = Backend::SyclTla;
      return d;
    }
    TORCH_WARN_ONCE(
        "TORCH_XPU_GROUPED_MM_BACKEND=sycltla: SYCL-TLA cannot serve some calls (",
        sycltla_why, "); using the per-group fallback for those.");
    d.reason = "forced sycltla, but SYCL-TLA cannot serve it: " + sycltla_why;
    d.reason_key = "forced sycltla|" + sycltla_why;
    return d;
  }

  std::string onednn_why;
  const bool onednn_ok = (req == Backend::Auto || req == Backend::OneDnn) &&
      onednn_grouped_mm_can_serve(mat_a, mat_b, *offs, onednn_why);
  if (req == Backend::OneDnn) {
    if (onednn_ok) {
      d.backend = Backend::OneDnn;
      return d;
    }
    TORCH_WARN_ONCE(
        "TORCH_XPU_GROUPED_MM_BACKEND=onednn: ", onednn_why,
        "; using the per-group fallback.");
    d.reason = "forced onednn, but " + onednn_why;
    d.reason_key = "forced onednn|" + onednn_why;
    return d;
  }

  // auto
  const int64_t avg = avg_rows_per_group(mat_a, mat_b, *offs);
  const bool small_groups = offs->numel() > 0 && avg < kMaxAvgRowsPerGroup;
  if (sycltla_ok && small_groups) {
    d.backend = Backend::SyclTla;
    return d;
  }
  if (onednn_ok) {
    d.backend = Backend::OneDnn;
    return d;
  }
  if (!sycltla_ok) {
    d.reason = "SYCL-TLA cannot serve it: " + sycltla_why;
    d.reason_key = sycltla_why;
  } else {
    d.reason = c10::str(
        "size heuristic: avg rows/group ", avg, " >= ", kMaxAvgRowsPerGroup,
        " (TORCH_XPU_GROUPED_MM_BACKEND=sycltla forces SYCL-TLA)");
    d.reason_key = "size heuristic";
  }
  return d;
}

} // namespace

// torch._grouped_mm on XPU.
//
// Backends (selected per call, see TORCH_XPU_GROUPED_MM_BACKEND above):
//   * SYCL-TLA persistent grouped GEMM in torch-xpu-ops (sycltla::grouped_mm)
//     for the shapes autograd produces for a MoE expert layer:
//       forward      mat_a [M, K] x mat_b [E, K, N]              -> [M, N]
//       grad_input   mat_a [M, N] x mat_b.transpose(-2, -1)      -> [M, K]
//       grad_weight  mat_a.t() [K, M] x grad [M, N]              -> [E, K, N]
//   * oneDNN grouped matmul (slot for pytorch/pytorch#195427).
//   * the generic per-group fallback for everything else (3D x 2D, 3D x 3D,
//     bias, mixed dtypes, misaligned strides).
//
// The registration and test scaffolding for XPU grouped GEMM mirrors
// pytorch/pytorch#195428 (split from #181557).
//
// Set TORCH_XPU_GROUPED_MM_LOG=1 (each distinct decision once) or =2 (every
// call) to print which backend each call takes, and why, to stderr.
Tensor _grouped_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ =
      _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out =
      create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  const Decision d = choose_backend(mat_a, mat_b, offs, bias, out_dtype_, out);
  switch (d.backend) {
    case Backend::SyclTla:
      sycltla::grouped_mm(mat_a, mat_b, *offs, out); // logs its own plan
      return out;
    case Backend::OneDnn:
      if (sycltla::grouped_mm_log_level() > 0) {
        sycltla::log_grouped_mm(
            c10::str("onednn|", mat_a.dim(), "x", mat_b.dim(), "|",
                     mat_a.scalar_type(), "|", mat_a.size(-1), "|", mat_b.size(-1)),
            c10::str("oneDNN grouped matmul  a", mat_a.sizes(), " b",
                     mat_b.sizes(), " -> out", out.sizes()));
      }
      run_onednn_grouped_mm(mat_a, mat_b, *offs, out);
      return out;
    default:
      break;
  }
  if (sycltla::grouped_mm_log_level() > 0) {
    sycltla::log_grouped_mm(
        // Key on the dims that do not change with routing (not the row
        // count), so level 1 prints each distinct decision once.
        c10::str(
            "fallback|", d.reason_key, "|", mat_a.dim(), "x", mat_b.dim(), "|",
            mat_a.scalar_type(), "|",
            (mat_a.dim() == 2 && mat_b.dim() == 2) ? mat_a.size(0) : mat_a.size(-1),
            "|", mat_b.size(-1)),
        c10::str(
            "fallback per-group mm: ", d.reason, "  a", mat_a.sizes(), " b",
            mat_b.sizes(), " -> out", out.sizes()));
  }
  _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  return out;
}

} // namespace at::native
