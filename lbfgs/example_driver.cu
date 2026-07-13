// example_driver.cpp
//
// Shows how to wire LBFGS + strongWolfeLineSearch into an elastic FWI
// loop in place of nonlinear CG. The parts marked TODO are where your
// existing modeling/adjoint-gradient code plugs in -- everything else
// (history management, line search, convergence check) is generic.

#include "lbfgs.h"
#include "line_search.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------
// TODO: replace with your actual elastic modeling + adjoint-state
// gradient computation. Signature matches EvalFn from line_search.h:
// given a trial model (device array, length n = nVp + nVs [+ nRho]),
// fill grad_out_d with the gradient and return the data misfit.
//
// This is presumably close to what your current NLCG loop already
// calls each iteration -- the optimizer swap shouldn't require changing
// this function at all.
// ---------------------------------------------------------------------
real_t evalElasticMisfitAndGradient(const real_t* model_trial_d,
                                     real_t* grad_out_d, int n) {
    // 1. Unpack model_trial_d into Vp, Vs, (Rho) fields as your forward
    //    solver expects.
    // 2. Run forward elastic modeling for all sources.
    // 3. Compute data residuals against observed data.
    // 4. Run adjoint-state propagation to form the gradient.
    // 5. Pack the gradient back into grad_out_d in the same layout as
    //    model_trial_d.
    // 6. Return the scalar misfit (e.g. 0.5 * ||residual||^2).
    //
    // Placeholder so this file is self-contained and shows the shape of
    // the call:
    real_t misfit = 0.0;
    // ... your existing modeling/gradient code here ...
    return misfit;
}

// out[i] = a[i] - b[i]
__global__ void subtractKernel(const real_t* a, const real_t* b, real_t* out,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

int main(int argc, char** argv) {
    // ---- problem size: replace with your actual model dimensions ----
    int nVp = 0, nVs = 0; // TODO: set from your grid (e.g. nx*ny*nz)
    int n = nVp + nVs;    // add nRho here too if you invert for density

    int history_size = 8; // typical for large 3D elastic FWI; increase
                           // toward 15-20 if you have memory headroom,
                           // since more history generally means better
                           // curvature approximation and can further
                           // help cross-talk

    int max_iterations = 100;
    real_t grad_tol = 1e-6; // convergence: ||grad||_2 / ||grad0||_2

    cublasHandle_t handle;
    cublasCreate(&handle);

    LBFGS optimizer(n, history_size, handle);

    // ---- device buffers ----
    real_t *model_d, *grad_d, *dir_d, *s_d, *y_d;
    real_t *model_trial_d, *grad_trial_d, *grad_prev_d;
    cudaMalloc(&model_d, sizeof(real_t) * n);
    cudaMalloc(&grad_d, sizeof(real_t) * n);
    cudaMalloc(&dir_d, sizeof(real_t) * n);
    cudaMalloc(&s_d, sizeof(real_t) * n);
    cudaMalloc(&y_d, sizeof(real_t) * n);
    cudaMalloc(&model_trial_d, sizeof(real_t) * n);
    cudaMalloc(&grad_trial_d, sizeof(real_t) * n);
    cudaMalloc(&grad_prev_d, sizeof(real_t) * n);

    // TODO: initialize model_d with your starting Vp/Vs (e.g. from
    // tomography / smoothed well logs), copied from host or generated
    // directly on device.

    // ---- optional: diagonal preconditioner for cross-talk mitigation ----
    // Uncomment and fill if you want per-parameter-class scaling on H0
    // rather than (or in addition to) non-dimensionalizing the model
    // vector itself. A simple starting point is the inverse of the
    // per-class RMS of the initial gradient, broadcast across each
    // block of the vector -- cheap to compute and often enough to get
    // Vp and Vs updates onto comparable scales.
    //
    // real_t* precond_d;
    // cudaMalloc(&precond_d, sizeof(real_t) * n);
    // ... fill precond_d ...
    // optimizer.setDiagonalPreconditioner(precond_d);

    EvalFn eval = [n](const real_t* m, real_t* g) {
        return evalElasticMisfitAndGradient(m, g, n);
    };

    real_t f = eval(model_d, grad_d);
    real_t grad0_norm;
    cublasDdot(handle, n, grad_d, 1, grad_d, 1, &grad0_norm);
    grad0_norm = std::sqrt(grad0_norm);

    for (int iter = 0; iter < max_iterations; ++iter) {
        optimizer.computeDirection(grad_d, dir_d);

        cudaMemcpy(grad_prev_d, grad_d, sizeof(real_t) * n,
                   cudaMemcpyDeviceToDevice);

        LineSearchResult ls = strongWolfeLineSearch(
            n, handle, eval, model_d, dir_d, grad_d, f, model_trial_d,
            grad_trial_d,
            /*c1=*/1e-4, /*c2=*/0.9, /*initial_step=*/1.0, /*max_iter=*/20);

        if (!ls.success) {
            std::fprintf(stderr,
                          "[FWI] iter %d: line search did not converge, "
                          "step=%g -- consider tightening c2 or checking "
                          "gradient scaling\n",
                          iter, (double)ls.step);
        }

        // s_k = model_trial_d - model_d, y_k = grad_trial_d - grad_prev_d
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        subtractKernel<<<blocks, threads>>>(model_trial_d, model_d, s_d, n);
        subtractKernel<<<blocks, threads>>>(grad_trial_d, grad_prev_d, y_d,
                                             n);
        cudaDeviceSynchronize();

        optimizer.updateHistory(s_d, y_d);

        cudaMemcpy(model_d, model_trial_d, sizeof(real_t) * n,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(grad_d, grad_trial_d, sizeof(real_t) * n,
                   cudaMemcpyDeviceToDevice);
        f = ls.misfit;

        real_t gnorm;
        cublasDdot(handle, n, grad_d, 1, grad_d, 1, &gnorm);
        gnorm = std::sqrt(gnorm);

        std::printf("iter %3d  misfit %.6e  |grad| %.3e  step %.3e  "
                    "history %d\n",
                    iter, (double)f, (double)gnorm, (double)ls.step,
                    optimizer.historyCount());

        if (gnorm / grad0_norm < grad_tol) {
            std::printf("[FWI] converged at iter %d\n", iter);
            break;
        }
    }

    cudaFree(model_d);
    cudaFree(grad_d);
    cudaFree(dir_d);
    cudaFree(s_d);
    cudaFree(y_d);
    cudaFree(model_trial_d);
    cudaFree(grad_trial_d);
    cudaFree(grad_prev_d);
    cublasDestroy(handle);
    return 0;
}
