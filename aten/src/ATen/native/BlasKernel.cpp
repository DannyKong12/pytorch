#include <limits>
#include <algorithm>
#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_BUILD_WITH_BLAS()
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
extern "C" void cgemm_(char *transa, char *transb, int *m, int *n, int *k, c10::complex<float> *alpha, c10::complex<float> *a, int *lda, c10::complex<float> *b, int *ldb, c10::complex<float> *beta, c10::complex<float> *c, int *ldc);
extern "C" void zgemm_(char *transa, char *transb, int *m, int *n, int *k, c10::complex<double> *alpha, c10::complex<double> *a, int *lda, c10::complex<double> *b, int *ldb, c10::complex<double> *beta, c10::complex<double> *c, int *ldc);

#endif // AT_BUILD_WITH_BLAS

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif


namespace at { namespace native {

namespace blas_impl {

template <typename scalar_t>
bool scal_use_fast_path(int64_t n, int64_t incx) {
  return false;
}

template <typename scalar_t>
bool gemv_use_fast_path(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  return false;
}

template <typename scalar_t>
bool gemm_use_fast_path(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  return false;
}

template <typename scalar_t>
void scal_fast_path(int *n, scalar_t *a, scalar_t *x, int *incx) {
  TORCH_INTERNAL_ASSERT(false, "scal_fast_path shouldn't be called for this configuration");
}

template <typename scalar_t>
void gemv_fast_path(char *trans, int *m, int *n, scalar_t *alpha, scalar_t *a, int *lda, scalar_t *x, int *incx, scalar_t *beta, scalar_t *y, int *incy) {
  TORCH_INTERNAL_ASSERT(false, "gemv_fast_path shouldn't be called for this configuration");
}

template <typename scalar_t>
void gemm_fast_path(char *transa, char *transb, int m, int n, int k, scalar_t alpha, scalar_t *a, int lda, scalar_t *b, int ldb, scalar_t beta, scalar_t *c, int ldc) {
  TORCH_INTERNAL_ASSERT(false, "gemm_fast_path shouldn't be called for this configuration");
}

#define INSTANTIATE(scalar_t)                                                                                                                                                     \
template bool scal_use_fast_path<scalar_t>(int64_t n, int64_t incx);                                                                                                              \
template bool gemv_use_fast_path<scalar_t>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy);                                                                        \
template void gemv_fast_path<scalar_t>(char *trans, int *m, int *n, scalar_t *alpha, scalar_t *a, int *lda, scalar_t *x, int *incx, scalar_t *beta, scalar_t *y, int *incy);      \
template void gemm_fast_path<scalar_t>(char *trans, char *transb, int m, int n, int k,  scalar_t alpha, scalar_t *a, int lda, scalar_t *b, int ldb, scalar_t beta, scalar_t *c, int ldc);      \
template void scal_fast_path<scalar_t>(int *n, scalar_t *a, scalar_t *x, int *incx);

#if AT_BUILD_WITH_BLAS()
template <>
bool scal_use_fast_path<double>(int64_t n, int64_t incx) {
  auto intmax = std::numeric_limits<int>::max();
  return n <= intmax && incx <= intmax;
}

template <>
bool scal_use_fast_path<float>(int64_t n, int64_t incx) {
  return scal_use_fast_path<double>(n, incx);
}

template <>
void scal_fast_path<double>(int *n, double *a, double *x, int *incx) {
  dscal_(n, a, x, incx);
}

template <>
void scal_fast_path<float>(int *n, float *a, float *x, int *incx) {
  sscal_(n, a, x, incx);
}

template <>
bool gemv_use_fast_path<float>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  auto intmax = std::numeric_limits<int>::max();
  return (m <= intmax) && (n <= intmax) && (lda <= intmax) &&
         (incx > 0) && (incx <= intmax) && (incy > 0) && (incy <= intmax);
}

template <>
bool gemv_use_fast_path<double>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  return gemv_use_fast_path<float>(m, n, lda, incx, incy);
}

template <>
bool gemm_use_fast_path<float>(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  // TODO: Should Perform Checks
  return true;
}

template <>
bool gemm_use_fast_path<double>(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  return gemm_use_fast_path<float>(m, n, k, lda, ldb, ldc);
}

template <>
bool gemm_use_fast_path<c10::complex<float>>(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  return gemm_use_fast_path<float>(m, n, k, lda, ldb, ldc);
}

template <>
bool gemm_use_fast_path<c10::complex<double>>(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  return gemm_use_fast_path<float>(m, n, k, lda, ldb, ldc);
}

template <>
void gemv_fast_path<double>(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy) {
  dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemv_fast_path<float>(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy) {
  sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemm_fast_path<double>(char *transa, char *transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_fast_path<float>(char *transa, char *transb, int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_fast_path<c10::complex<double>>(char *transa, char *transb, int m, int n, int k, c10::complex<double> alpha, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb, c10::complex<double> beta, c10::complex<double> *c, int ldc) {
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

template <>
void gemm_fast_path<c10::complex<float>>(char *transa, char *transb, int m, int n, int k, c10::complex<float> alpha, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb, c10::complex<float> beta, c10::complex<float> *c, int ldc) {
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}
#else
INSTANTIATE(float);
INSTANTIATE(double);
#endif // AT_BUILD_WITH_BLAS

INSTANTIATE(uint8_t);
INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int);
INSTANTIATE(int64_t);
INSTANTIATE(c10::BFloat16);
#undef INSTANTIATE

} // namespace blas_impl

template <typename scalar_t>
inline void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if (n == 1) incx = 1;
  if (blas_impl::scal_use_fast_path<scalar_t>(n, incx)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    blas_impl::scal_fast_path<scalar_t>(&i_n, &a, x, &i_incx);
    return;
  }
  for (int64_t i = 0; i < n; i++) {
    if (a == scalar_t(0)) {
      x[i * incx] = 0;
    } else {
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
bool gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  if(n == 1) lda = m;

  if (blas_impl::gemv_use_fast_path<scalar_t>(m, n, lda, incx, incy)) {
    TORCH_CHECK(lda >= std::max<int64_t>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    blas_impl::gemv_fast_path<scalar_t>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return true;
  }

  if ((trans == 'T') || (trans == 't')) {
    for (int64_t i = 0; i < n; i++)
    {
      scalar_t sum = 0;
      scalar_t *row_ = a + lda * i;
      for (int64_t j = 0; j < m; j++) {
        sum += x[j * incx] * row_[j];
      }
      if (beta == scalar_t(0)) {
        y[i * incy] = alpha * sum;
      } else {
        y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    }
  } else {
    if (beta != scalar_t(1)) scal<scalar_t>(m, beta, y, incy);

    for (int64_t j = 0; j < n; j++) {
      scalar_t *column_ = a + lda * j;
      scalar_t z = alpha * x[j * incx];
      for (int64_t i = 0; i < m; i++) {
        y[i * incy] += z * column_[i];
      }
    }
  }
  return false;
}

template<typename scalar_t>
bool gemm(char transa, char transb, int64_t m, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb, scalar_t beta, scalar_t *c, int64_t ldc) {
  // TODO: CHECK STUFF
  //

  if (blas_impl::gemm_use_fast_path<scalar_t>(m, n, k, lda, ldb, ldc)) {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;
    blas_impl::gemm_fast_path<scalar_t>(&transa, &transb, i_m, i_n, i_k, alpha, a, i_lda, b, i_ldb, beta, c, i_ldc);
    return true;
  }

  TORCH_CHECK(false, "This type is currently not supported by mm");
  return false;
}


#define INSTANTIATE(scalar_t, _) \
template bool gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy); \
template bool gemm<scalar_t>(char transa, char transb, int64_t m, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb, scalar_t beta, scalar_t *c, int64_t ldc);
AT_FORALL_SCALAR_TYPES_AND(BFloat16, INSTANTIATE);
AT_FORALL_COMPLEX_TYPES(INSTANTIATE);
#undef INSTANTIATE

}} // namespace at::native
