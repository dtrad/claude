// lbfgs.h
//
// Limited-memory BFGS optimizer for CUDA-resident model vectors.
// Designed to drop into an elastic FWI loop in place of nonlinear CG.
//
// Design assumptions (adjust to match your code):
//   - The model vector m is a single contiguous device array of length n,
//     typically a concatenation of parameter classes, e.g.
//         n = nVp + nVs + nRho   (or however many classes you invert for)
//   - Gradients g are device arrays of the same length n, already scaled
//     by whatever adjoint-state normalization you use.
//   - All BLAS-like ops (dot, axpy, scal) go through cuBLAS so this works
//     for float or double without hand-rolled reduction kernels.
//
// L-BFGS does NOT know about parameter classes. It treats m/g as flat
// vectors. If Vp/Vs/Rho live on very different scales, cross-talk shows
// up as the optimizer effectively "seeing" one parameter class more than
// another in the curvature estimate. Two common fixes, both left as
// hooks below rather than baked in, since the right choice is model-
// dependent:
//   1. Non-dimensionalize each parameter class before it enters L-BFGS
//      (e.g. divide by a reference velocity/density so all classes are
//      O(1)), and undo the scaling only when you write out models.
//   2. Supply a diagonal preconditioner P (per-parameter-class scaling)
//      applied to the initial Hessian H0 in the two-loop recursion --
//      see `setDiagonalPreconditioner`.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Precision switch: change to float if your FWI runs single precision.
using real_t = double;

class LBFGS {
public:
    // n  = length of the flattened model vector
    // m  = history size (typically 5-20; elastic FWI often uses 8-10
    //      because each vector is large and memory-limited)
    LBFGS(int n, int m, cublasHandle_t handle);
    ~LBFGS();

    // Compute the search direction dir = -H_k * grad using the two-loop
    // recursion. On the first call (no history yet) this reduces to
    // steepest descent, dir = -grad, which is the standard L-BFGS
    // initialization.
    //
    // grad_d and dir_d are device pointers of length n. dir_d is
    // overwritten.
    void computeDirection(const real_t* grad_d, real_t* dir_d);

    // Push a new (s, y) pair into the circular history after a step is
    // accepted:
    //   s_k = x_{k+1} - x_k   (the actual model update taken)
    //   y_k = g_{k+1} - g_k   (gradient change)
    // Call this AFTER the line search has accepted a step, not before.
    // Internally checks the curvature condition s^T y > 0 (required for
    // positive-definiteness) and skips the update if it fails rather
    // than corrupting the Hessian approximation -- this happens
    // occasionally in FWI when the line search takes a marginal step
    // near a non-convex region of the misfit function.
    void updateHistory(const real_t* s_d, const real_t* y_d);

    // Optional: supply a diagonal preconditioner (device array, length n)
    // used to scale the initial Hessian H0 = gamma_k * diag(precond) in
    // the two-loop recursion, instead of the default H0 = gamma_k * I.
    // Useful for correcting for gross scale differences between Vp, Vs,
    // Rho blocks of the model vector without having to non-dimensionalize
    // the whole optimization. Pass nullptr to disable (default).
    void setDiagonalPreconditioner(const real_t* precond_d);

    void reset(); // clears history, e.g. after a model-space jump
                  // (multiscale frequency continuation, mesh change, etc.)

    int historyCount() const { return count_; }

private:
    int n_;                 // model vector length
    int m_;                 // max history pairs
    int count_;              // number of valid pairs currently stored
    int head_;               // circular buffer insertion index

    cublasHandle_t handle_;

    std::vector<real_t*> s_hist_;  // device pointers, m_ slots of length n_
    std::vector<real_t*> y_hist_;  // device pointers, m_ slots of length n_
    std::vector<real_t> rho_hist_; // host-side, 1/(y_k^T s_k), one per slot

    real_t* precond_d_ = nullptr; // optional diagonal preconditioner
    real_t* scratch_d_ = nullptr; // length-n scratch buffer for the
                                   // two-loop recursion's q/r vector

    real_t* alpha_host_;          // host buffer, length m_, two-loop coeffs

    real_t dotProduct(const real_t* a_d, const real_t* b_d) const;
    void axpy(real_t alpha, const real_t* x_d, real_t* y_d) const; // y += a*x
    void scal(real_t alpha, real_t* x_d) const;                    // x *= a
    void copy(const real_t* src_d, real_t* dst_d) const;
};
