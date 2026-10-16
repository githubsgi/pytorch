#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#endif

namespace at::native {

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
      sycltla::can_use_grouped_mm(mat_a, mat_b, *offs, out);
  if (use_sycltla) {
    sycltla::grouped_mm(mat_a, mat_b, *offs, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native
