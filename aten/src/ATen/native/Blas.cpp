#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

template<typename scalar_t>
bool gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

template<typename scalar_t>
bool gemm(char transa, char transb, int64_t m, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb, scalar_t beta, scalar_t *c, int64_t ldc);


constexpr inline bool lda_cond(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda > std::max<int64_t>(1L, m);
}

Tensor &addmv_impl_cpu(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta_, Scalar alpha_) {
  auto r_stride = result.stride(0);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, mat.scalar_type(), "addmv_impl_cpu", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    bool is_fast = false;
    if (mat.stride(0) == 1 && lda_cond(mat.size(0), mat.size(1), mat.stride(1))) {
      is_fast = gemv<scalar_t>('n', mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else if (mat.stride(1) == 1 && lda_cond(mat.size(1), mat.size(0), mat.stride(0))) {
      is_fast = gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, mat.data_ptr<scalar_t>(), mat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else {
      Tensor cmat = mat.contiguous();
      is_fast = gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, cmat.data_ptr<scalar_t>(), cmat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }

    // In THE FAST PATH of gemv (x,0).mv(0) does not handle beta, whereas gemv does for case where (x,0).mv(0,y).
    // But in the naive fall back implementation, this is not the case.
    if (is_fast && vec.size(0) == 0 && mat.size(0) != 0) {
      if (beta == scalar_t(0)) {
        result.zero_();
      } else if (beta != scalar_t(1)) {
        result.mul_(beta);
      }
    }
  });
  return result;
}

Tensor &addmv_out(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  { // scope of NoNamesGuard

  at::NoNamesGuard guard;
  result.resize_({mat.size(0)});

  Tensor self_ = self;
  if (self.dim() == 0 || self.size(0) == 1) {
    self_ = self.expand({mat.size(0)});
  }

  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self_.dim() == 1),
    "vector + matrix @ vector expected, got ", self_.dim(), ", ", mat.dim(), ", ", vec.dim());
  TORCH_CHECK((mat.size(1) == vec.size(0) && mat.size(0) == self_.size(0)),
    "size mismatch, get ", self_.size(0), ", ", mat.size(0), "x", mat.size(1), ",", vec.size(0));

  if (!result.is_same(self_)) {
    at::native::copy_(result, self_);
  }

  if (result.numel() != 0) {
    at::_addmv_impl_(result, self_, mat, vec, beta, alpha);
  }

  } // scope of NoNamesGuard
  at::namedinference::propagate_names_for_addmv(result, mat, vec, self);
  return result;
}

Tensor addmv(const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  Tensor result = at::empty({mat.size(0)}, mat.options());
  return native::addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor &addmv_(Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  return native::addmv_out(self, self, mat, vec, beta, alpha);
}

Tensor &mv_out(Tensor& result, const Tensor &self, const Tensor &vec) {
  return native::addmv_out(result, result, self, vec, 0, 1);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result = at::empty({self.size(0)}, self.options());
  return native::mv_out(result, self, vec);
}

Tensor &addmm_impl_cpu(Tensor& result, const Tensor &a, const Tensor &b, const Tensor &c, Scalar beta_, Scalar alpha_) {
  auto r_stride = result.stride(0);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, b.scalar_type(), "addmm_impl_cpu", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    gemm<scalar_t>('t', 't', a.size(0), b.size(1), a.size(1), alpha, a.data_ptr<scalar_t>(), a.size(1), b.data_ptr<scalar_t>(), 
        b.size(1), beta, result.data_ptr<scalar_t>(), r_stride);
  });
  return result;
}

Tensor &addmm_out(Tensor& result, const Tensor &a, const Tensor &b, const Tensor &c, Scalar beta, Scalar alpha) {
  { // scope of NoNamesGuard

  at::NoNamesGuard guard;
  result.resize_({a.size(0), b.size(1)});

  TORCH_CHECK((a.dim() == 2 && b.dim() == 2),
    "matrix + matrix @ matrix expected, got ", a.dim(), ", ", b.dim(), ", ", c.dim());
  TORCH_CHECK((a.size(1) == b.size(0) && a.size(0) == c.size(0) && b.size(1) == c.size(1)),
    "size mismatch, get ", a.size(0), ", ", a.size(1), "x", b.size(0), ",", b.size(1));

  if (!result.is_same(c)) {
    at::native::copy_(result, a);
  }

  if (result.numel() != 0) {
    at::_addmm_impl_(result, a, b, c, beta, alpha);
  }

  } // scope of NoNamesGuard
  // at::namedinference::propagate_names_for_addmm(result, b, c, a);
  return result;
}

Tensor addmm(const Tensor &a, const Tensor &b, const Tensor &c, Scalar beta, Scalar alpha) {
  Tensor result = at::empty({a.size(0), b.size(1)}, a.options());
  return native::addmm_out(result, a, b, c, beta, alpha);
}

Tensor &addmm_(Tensor &a, const Tensor &b, const Tensor &c, Scalar beta, Scalar alpha) {
  return native::addmm_out(a, a, b, c, beta, alpha);
}

Tensor &mm_out(Tensor& result, const Tensor &a, const Tensor &b) {
  return native::addmm_out(result, a, b, result, 0, 1);
}

Tensor mm(const Tensor &a, const Tensor &b) {
  Tensor result = at::empty({a.size(0), b.size(1)}, a.options());
  return native::mm_out(result, a, b);
}

}}  // namespace at::native
