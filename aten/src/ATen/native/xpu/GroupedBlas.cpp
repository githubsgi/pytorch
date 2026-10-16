#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>

#include <cstdlib>

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
bool prefer_sycltla_grouped_mm(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& offs) {
  static const bool force = [] {
    const char* v = std::getenv("TORCH_XPU_GROUPED_MM_FORCE_SYCLTLA");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (force) {
    return true;
  }
  constexpr int64_t kMaxAvgRowsPerGroup = 1024;
  const int64_t groups = offs.numel();
  // Rows that `offs` partitions: mat_a rows for 2D x 3D, the contraction
  // dimension for 2D x 2D.
  const int64_t rows = (mat_a.dim() == 2 && mat_b.dim() == 3)
      ? mat_a.size(0)
      : mat_a.size(-1);
  return groups > 0 && rows / groups < kMaxAvgRowsPerGroup;
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
// strides) goes through the generic per-group fallback.
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

  const bool use_sycltla = offs.has_value() && !bias.has_value() &&
      out_dtype_ == mat_a.scalar_type() &&
      prefer_sycltla_grouped_mm(mat_a, mat_b, *offs) &&
      sycltla::can_use_grouped_mm(mat_a, mat_b, *offs, out);
  if (use_sycltla) {
    sycltla::grouped_mm(mat_a, mat_b, *offs, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native
