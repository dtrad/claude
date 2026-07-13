// line_search.cu
//
// Bracketing + zoom strong Wolfe line search (Nocedal & Wright Alg 3.5/3.6).

#include "line_search.h"
#include <cstdio>
#include <cmath>

static inline cublasStatus_t dotw(cublasHandle_t h, int n, const double* x,
                                   const double* y, double* r) {
    return cublasDdot(h, n, x, 1, y, 1, r);
}
static inline cublasStatus_t dotw(cublasHandle_t h, int n, const float* x,
                                   const float* y, float* r) {
    return cublasSdot(h, n, x, 1, y, 1, r);
}

// trial = m + alpha * dir
__global__ void formTrialKernel(const real_t* m, const real_t* dir,
                                 real_t alpha, real_t* trial, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) trial[i] = m[i] + alpha * dir[i];
}

static real_t directionalDeriv(cublasHandle_t handle, int n,
                                const real_t* grad_d, const real_t* dir_d) {
    real_t result = 0;
    dotw(handle, n, grad_d, dir_d, &result);
    return result;
}

static void formTrial(int n, const real_t* m_d, const real_t* dir_d,
                       real_t alpha, real_t* trial_d) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    formTrialKernel<<<blocks, threads>>>(m_d, dir_d, alpha, trial_d, n);
    cudaDeviceSynchronize();
}

// Zoom phase: bisection/interpolation between alpha_lo and alpha_hi.
// Kept as plain bisection here rather than cubic interpolation --
// simpler to verify correctness against your existing NLCG results
// first; swap in interpolation later if line-search call count becomes
// a bottleneck (each iteration here costs one forward+adjoint solve,
// which for 3D elastic FWI is the dominant cost, so keeping the zoom
// loop itself cheap doesn't matter much until iteration count does).
static LineSearchResult zoom(int n, cublasHandle_t handle, const EvalFn& eval,
                              const real_t* m_d, const real_t* dir_d,
                              real_t f0, real_t g0, real_t c1, real_t c2,
                              real_t alpha_lo, real_t f_lo, real_t alpha_hi,
                              real_t* m_trial_d, real_t* grad_trial_d,
                              int max_iter) {
    for (int i = 0; i < max_iter; ++i) {
        real_t alpha = real_t(0.5) * (alpha_lo + alpha_hi);
        formTrial(n, m_d, dir_d, alpha, m_trial_d);
        real_t f = eval(m_trial_d, grad_trial_d);

        if (f > f0 + c1 * alpha * g0 || f >= f_lo) {
            alpha_hi = alpha;
        } else {
            real_t g = directionalDeriv(handle, n, grad_trial_d, dir_d);
            if (std::fabs(g) <= -c2 * g0) {
                return {alpha, f, true};
            }
            if (g * (alpha_hi - alpha_lo) >= 0) {
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha;
            f_lo = f;
        }
    }
    // Max zoom iterations hit: return best point found. Log this --
    // repeated zoom exhaustion usually means c1/c2 are too tight for
    // how noisy your misfit function is (common at low frequencies
    // early in a multiscale FWI schedule).
    std::fprintf(stderr,
                  "[line_search] zoom exhausted max_iter=%d, returning "
                  "alpha=%g\n",
                  max_iter, (double)alpha_lo);
    formTrial(n, m_d, dir_d, alpha_lo, m_trial_d);
    real_t f = eval(m_trial_d, grad_trial_d);
    return {alpha_lo, f, false};
}

LineSearchResult strongWolfeLineSearch(int n, cublasHandle_t handle,
                                        const EvalFn& eval, const real_t* m_d,
                                        const real_t* dir_d,
                                        const real_t* grad_d, real_t f0,
                                        real_t* m_trial_d,
                                        real_t* grad_trial_d, real_t c1,
                                        real_t c2, real_t initial_step,
                                        int max_iter) {
    real_t g0 = directionalDeriv(handle, n, grad_d, dir_d);
    if (g0 >= 0) {
        std::fprintf(stderr,
                      "[line_search] warning: direction is not a descent "
                      "direction (g0=%g). Falling back to negative "
                      "gradient scaling.\n",
                      (double)g0);
    }

    real_t alpha_prev = real_t(0);
    real_t f_prev = f0;
    real_t alpha = initial_step;

    for (int i = 0; i < max_iter; ++i) {
        formTrial(n, m_d, dir_d, alpha, m_trial_d);
        real_t f = eval(m_trial_d, grad_trial_d);

        if (f > f0 + c1 * alpha * g0 || (i > 0 && f >= f_prev)) {
            return zoom(n, handle, eval, m_d, dir_d, f0, g0, c1, c2,
                        alpha_prev, f_prev, alpha, m_trial_d, grad_trial_d,
                        max_iter);
        }

        real_t g = directionalDeriv(handle, n, grad_trial_d, dir_d);
        if (std::fabs(g) <= -c2 * g0) {
            return {alpha, f, true};
        }
        if (g >= 0) {
            return zoom(n, handle, eval, m_d, dir_d, f0, g0, c1, c2, alpha,
                        f, alpha_prev, m_trial_d, grad_trial_d, max_iter);
        }

        alpha_prev = alpha;
        f_prev = f;
        alpha = alpha * real_t(2.0); // simple expansion; cap externally if
                                      // your model updates need a hard
                                      // step-length ceiling for stability
    }

    std::fprintf(stderr,
                  "[line_search] max_iter reached without satisfying "
                  "Wolfe conditions, returning last trial alpha=%g\n",
                  (double)alpha_prev);
    formTrial(n, m_d, dir_d, alpha_prev, m_trial_d);
    real_t f = eval(m_trial_d, grad_trial_d);
    return {alpha_prev, f, false};
}
