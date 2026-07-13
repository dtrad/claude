// line_search.h
//
// Strong Wolfe line search. L-BFGS, unlike NLCG with a simple backtracking
// line search, needs curvature information (the second Wolfe condition)
// from the line search to guarantee the next (s, y) pair satisfies
// s^T y > 0. Plugging L-BFGS into an existing backtracking-only line
// search from an NLCG implementation is the most common way people end up
// with an optimizer that silently degrades toward steepest descent
// (because updateHistory keeps rejecting pairs) -- worth swapping the
// line search at the same time as the optimizer, not after.
//
// This is a fairly standard implementation (Nocedal & Wright, Algorithm
// 3.5 / 3.6: bracketing + zoom). It calls back into your existing
// forward-modeling + adjoint-gradient code through the EvalFn callback,
// so it doesn't need to know anything about elastic FWI internals.

#pragma once

#include <functional>
#include "lbfgs.h"

// Evaluates misfit and gradient at a trial model m_trial_d (device
// pointer, length n). Must fill grad_out_d (device pointer, length n)
// and return the scalar misfit value.
//
// This is where your existing elastic forward-modeling + adjoint-state
// gradient computation plugs in -- see example_driver.cpp.
using EvalFn = std::function<real_t(const real_t* m_trial_d, real_t* grad_out_d)>;

struct LineSearchResult {
    real_t step;         // accepted step length alpha
    real_t misfit;        // misfit at accepted point
    bool success;        // false if max iterations exceeded
};

// m_d       : current model (device, length n) -- NOT modified
// dir_d     : search direction from LBFGS::computeDirection (device, length n)
// grad_d    : gradient at m_d (device, length n)
// f0        : misfit at m_d (already computed by caller, avoids recompute)
// m_trial_d : scratch buffer (device, length n), caller-allocated,
//             will hold the accepted trial model on return
// grad_trial_d : scratch buffer (device, length n), will hold the
//             gradient at the accepted trial model on return -- this is
//             exactly the gradient you need for the next updateHistory()
//             call, so no extra gradient evaluation is needed after the
//             line search returns.
LineSearchResult strongWolfeLineSearch(int n, cublasHandle_t handle,
                                        const EvalFn& eval,
                                        const real_t* m_d,
                                        const real_t* dir_d,
                                        const real_t* grad_d,
                                        real_t f0,
                                        real_t* m_trial_d,
                                        real_t* grad_trial_d,
                                        real_t c1 = real_t(1e-4),
                                        real_t c2 = real_t(0.9),
                                        real_t initial_step = real_t(1.0),
                                        int max_iter = 20);
