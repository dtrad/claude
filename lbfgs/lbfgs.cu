// lbfgs.cu
//
// Implementation of the two-loop recursion L-BFGS optimizer declared in
// lbfgs.h. All vector arithmetic goes through cuBLAS so the class works
// unmodified for float or double (see real_t typedef in lbfgs.h).

#include "lbfgs.h"
#include <cstdio>
#include <cmath>

// ---- small helpers to make cuBLAS calls precision-agnostic ----
// If you switch real_t to float, these will need the S-prefixed cuBLAS
// calls instead (cublasSdot, cublasSaxpy, cublasSscal). Left as explicit
// overload pairs rather than a template so the choice is visible at the
// call site during code review.

static inline cublasStatus_t cublasDotWrap(cublasHandle_t h, int n,
                                            const double* x, const double* y,
                                            double* result) {
    return cublasDdot(h, n, x, 1, y, 1, result);
}
static inline cublasStatus_t cublasDotWrap(cublasHandle_t h, int n,
                                            const float* x, const float* y,
                                            float* result) {
    return cublasSdot(h, n, x, 1, y, 1, result);
}
static inline cublasStatus_t cublasAxpyWrap(cublasHandle_t h, int n,
                                             const double* alpha,
                                             const double* x, double* y) {
    return cublasDaxpy(h, n, alpha, x, 1, y, 1);
}
static inline cublasStatus_t cublasAxpyWrap(cublasHandle_t h, int n,
                                             const float* alpha,
                                             const float* x, float* y) {
    return cublasSaxpy(h, n, alpha, x, 1, y, 1);
}
static inline cublasStatus_t cublasScalWrap(cublasHandle_t h, int n,
                                             const double* alpha, double* x) {
    return cublasDscal(h, n, alpha, x, 1);
}
static inline cublasStatus_t cublasScalWrap(cublasHandle_t h, int n,
                                             const float* alpha, float* x) {
    return cublasSscal(h, n, alpha, x, 1);
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            throw std::runtime_error("CUDA error in LBFGS");                 \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                     \
        cublasStatus_t st = (call);                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__, \
                    __LINE__);                                               \
            throw std::runtime_error("cuBLAS error in LBFGS");               \
        }                                                                    \
    } while (0)

// Elementwise kernel: out[i] = x[i] * precond[i]
// Used only when a diagonal preconditioner is supplied for H0.
__global__ void elementwiseScaleKernel(const real_t* x, const real_t* precond,
                                        real_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * precond[i];
}

LBFGS::LBFGS(int n, int m, cublasHandle_t handle)
    : n_(n), m_(m), count_(0), head_(0), handle_(handle) {
    s_hist_.resize(m_, nullptr);
    y_hist_.resize(m_, nullptr);
    rho_hist_.resize(m_, real_t(0));
    for (int i = 0; i < m_; ++i) {
        CUDA_CHECK(cudaMalloc(&s_hist_[i], sizeof(real_t) * n_));
        CUDA_CHECK(cudaMalloc(&y_hist_[i], sizeof(real_t) * n_));
    }
    CUDA_CHECK(cudaMalloc(&scratch_d_, sizeof(real_t) * n_));
    alpha_host_ = new real_t[m_];
}

LBFGS::~LBFGS() {
    for (int i = 0; i < m_; ++i) {
        if (s_hist_[i]) cudaFree(s_hist_[i]);
        if (y_hist_[i]) cudaFree(y_hist_[i]);
    }
    if (scratch_d_) cudaFree(scratch_d_);
    delete[] alpha_host_;
}

real_t LBFGS::dotProduct(const real_t* a_d, const real_t* b_d) const {
    real_t result = 0;
    CUBLAS_CHECK(cublasDotWrap(handle_, n_, a_d, b_d, &result));
    return result;
}

void LBFGS::axpy(real_t alpha, const real_t* x_d, real_t* y_d) const {
    CUBLAS_CHECK(cublasAxpyWrap(handle_, n_, &alpha, x_d, y_d));
}

void LBFGS::scal(real_t alpha, real_t* x_d) const {
    CUBLAS_CHECK(cublasScalWrap(handle_, n_, &alpha, x_d));
}

void LBFGS::copy(const real_t* src_d, real_t* dst_d) const {
    CUDA_CHECK(cudaMemcpy(dst_d, src_d, sizeof(real_t) * n_,
                           cudaMemcpyDeviceToDevice));
}

void LBFGS::setDiagonalPreconditioner(const real_t* precond_d) {
    precond_d_ = const_cast<real_t*>(precond_d);
}

void LBFGS::reset() {
    count_ = 0;
    head_ = 0;
}

void LBFGS::updateHistory(const real_t* s_d, const real_t* y_d) {
    real_t sy = dotProduct(s_d, y_d);

    // Curvature condition. If this fails, the pair would make H_k
    // indefinite, so skip it rather than store it. In elastic FWI this
    // is most common right after a frequency-band change or a very
    // short line-search step -- worth logging if it happens often, since
    // persistent failures usually mean the line search tolerance is too
    // loose rather than being a one-off.
    const real_t curvature_eps = real_t(1e-10);
    if (sy <= curvature_eps * dotProduct(s_d, s_d)) {
        std::fprintf(stderr,
                      "[LBFGS] skipping history update: s^T y = %g "
                      "(curvature condition failed)\n",
                      (double)sy);
        return;
    }

    int slot = head_;
    copy(s_d, s_hist_[slot]);
    copy(y_d, y_hist_[slot]);
    rho_hist_[slot] = real_t(1) / sy;

    head_ = (head_ + 1) % m_;
    if (count_ < m_) count_++;
}

void LBFGS::computeDirection(const real_t* grad_d, real_t* dir_d) {
    // Standard two-loop recursion (Nocedal & Wright, Algorithm 7.4).
    // q starts as the gradient; dir_d doubles as q/r storage throughout.
    copy(grad_d, dir_d);

    if (count_ == 0) {
        // No curvature information yet: steepest descent step.
        scal(real_t(-1), dir_d);
        return;
    }

    // Walk history newest -> oldest.
    // slot indices: most recent pair is at (head_ - 1 + m_) % m_
    std::vector<int> order(count_);
    for (int k = 0; k < count_; ++k) {
        order[k] = (head_ - 1 - k + 2 * m_) % m_;
    }

    for (int k = 0; k < count_; ++k) {
        int slot = order[k];
        real_t alpha = rho_hist_[slot] * dotProduct(s_hist_[slot], dir_d);
        alpha_host_[slot] = alpha;
        axpy(-alpha, y_hist_[slot], dir_d); // q -= alpha * y_k
    }

    // Initial Hessian scaling H0 = gamma_k * I  (or gamma_k * diag(precond)
    // if a preconditioner was supplied). gamma_k = s_{k-1}^T y_{k-1} /
    // y_{k-1}^T y_{k-1} is the standard Nocedal-Wright scaling -- it keeps
    // the very first L-BFGS step (before any real curvature has been
    // accumulated in a useful shape) on a sane scale.
    int newest = (head_ - 1 + m_) % m_;
    real_t sy = dotProduct(s_hist_[newest], y_hist_[newest]);
    real_t yy = dotProduct(y_hist_[newest], y_hist_[newest]);
    real_t gamma = (yy > real_t(0)) ? (sy / yy) : real_t(1);

    if (precond_d_) {
        int threads = 256;
        int blocks = (n_ + threads - 1) / threads;
        elementwiseScaleKernel<<<blocks, threads>>>(dir_d, precond_d_,
                                                      scratch_d_, n_);
        CUDA_CHECK(cudaGetLastError());
        copy(scratch_d_, dir_d);
    }
    scal(gamma, dir_d); // dir_d now holds r = H0 * q

    for (int k = count_ - 1; k >= 0; --k) {
        int slot = order[k];
        real_t beta = rho_hist_[slot] * dotProduct(y_hist_[slot], dir_d);
        axpy(alpha_host_[slot] - beta, s_hist_[slot], dir_d); // r += (a-b)*s_k
    }

    // dir_d currently holds H_k * grad; negate for a descent direction.
    scal(real_t(-1), dir_d);
}
