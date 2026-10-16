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

// The persistent SYCL-TLA kernel wins clearly when groups are small (many
// experts with a few hundred rows each: ~2x over a loop of dense GEMMs on
// PVC). For very large per-group problems a loop of oneDNN GEMMs is as fast
// or slightly faster, so those stay on the fallback. Set
// TORCH_XPU_GROUPED_MM_FORCE_SYCLTLA=1 to always take the SYCL-TLA path.
constexpr int64_t kMaxAvgRowsPerGroup = 1024;

bool force_sycltla() {
  static const bool force = [] {
    const char* v = std::getenv("TORCH_XPU_GROUPED_MM_FORCE_SYCLTLA");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return force;
}

// Rows that `offs` partitions, averaged over groups: mat_a rows for 2D x 3D,
// the contraction dimension for 2D x 2D.
int64_t avg_rows_per_group(const Tensor& mat_a, const Tensor& mat_b, const Tensor& offs) {
  const int64_t groups = offs.numel();
  const int64_t rows = (mat_a.dim() == 2 && mat_b.dim() == 3)
      ? mat_a.size(0)
      : mat_a.size(-1);
  return groups > 0 ? rows / groups : 0;
}

// Picks the implementation. Returns true for SYCL-TLA; otherwise fills
// `reason` (always filled when logging is on).
bool choose_sycltla(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    c10::ScalarType out_dtype,
    const Tensor& out,
    std::string& reason,
    std::string& reason_key) {
  if (!offs.has_value()) {
    reason = reason_key = "no offs (3D x 3D batched)";
    return false;
  }
  if (bias.has_value()) {
    reason = reason_key = "bias";
    return false;
  }
  if (out_dtype != mat_a.scalar_type()) {
    reason = reason_key = "out_dtype differs from input dtype";
    return false;
  }
  if (!force_sycltla()) {
    const int64_t avg = avg_rows_per_group(mat_a, mat_b, *offs);
    if (offs->numel() == 0 || avg >= kMaxAvgRowsPerGroup) {
      reason = c10::str(
          "size heuristic: avg rows/group ", avg, " >= ", kMaxAvgRowsPerGroup,
          " (TORCH_XPU_GROUPED_MM_FORCE_SYCLTLA=1 overrides)");
      reason_key = "size heuristic";
      return false;
    }
  }
  std::string why;
  if (!sycltla::can_use_grouped_mm(mat_a, mat_b, *offs, out, &why)) {
    reason = "SYCL-TLA cannot serve it: " + why;
    reason_key = why;
    return false;
  }
  return true;
}

} // namespace

// torch._grouped_mm on XPU.
//
// The fast path is the SYCL-TLA persistent grouped GEMM that lives in
// torch-xpu-ops (sycltla::grouped_mm). It serves the shapes autograd produces
// for a MoE expert layer:
//   forward      mat_a [M, K] x mat_b [E, K, N]              -> [M, N]
//   grad_input   mat_a [M, N] x mat_b.transpose(-2, -1)      -> [M, K]
//   grad_weight  mat_a.t() [K, M] x grad [M, N]              -> [E, K, N]
// Everything else (3D x 2D / 3D x 3D, bias, mixed dtypes, misaligned
// strides, large groups per the size heuristic) goes through the generic
// per-group fallback.
//
// Set TORCH_XPU_GROUPED_MM_LOG=1 (each distinct decision once) or =2 (every
// call) to print which path each call takes, and why, to stderr.
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

  std::string reason, reason_key;
  if (choose_sycltla(mat_a, mat_b, offs, bias, out_dtype_, out, reason, reason_key)) {
    sycltla::grouped_mm(mat_a, mat_b, *offs, out);  // logs its own plan
  } else {
    if (sycltla::grouped_mm_log_level() > 0) {
      sycltla::log_grouped_mm(
          // Key on the dims that do not change with routing (not the row
          // count), so level 1 prints each distinct decision once.
          c10::str(
              "fallback|", reason_key, "|", mat_a.dim(), "x", mat_b.dim(), "|",
              mat_a.scalar_type(), "|",
              (mat_a.dim() == 2 && mat_b.dim() == 2) ? mat_a.size(0) : mat_a.size(-1),
              "|", mat_b.size(-1)),
          c10::str(
              "fallback per-group mm: ", reason, "  a", mat_a.sizes(), " b",
              mat_b.sizes(), " -> out", out.sizes()));
    }
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native
