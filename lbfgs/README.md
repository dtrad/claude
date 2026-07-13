# L-BFGS for elastic FWI (replacing NLCG)

No CUDA toolkit was available in the sandbox this was written in, so
none of this has been compiled or run — treat it as a correctly-structured
starting point to build and debug against your actual toolchain, not
tested code. The Makefile below assumes a standard CUDA 11/12 install
with cuBLAS.

## Files

- `lbfgs.h` / `lbfgs.cu` — L-BFGS optimizer core: two-loop recursion,
  circular (s, y) history buffer on device, cuBLAS-based vector ops.
- `line_search.h` / `line_search.cu` — strong Wolfe line search
  (bracketing + zoom). L-BFGS needs this, not backtracking-only —
  see the note at the top of `line_search.h`.
- `example_driver.cu` — shows the optimizer loop wired to a placeholder
  `evalElasticMisfitAndGradient`. This is the one function you need to
  fill in with your existing forward-modeling + adjoint-state gradient
  code.

## Integration steps

1. In `example_driver.cu`, implement `evalElasticMisfitAndGradient` by
   calling your existing modeling operators. The interface is
   deliberately minimal: trial model in, gradient out, misfit returned.
   If your current NLCG loop already has a function shaped like this,
   you likely just move the body over unchanged.
2. Set `nVp`, `nVs` (and `nRho` if you invert for density) from your
   actual grid, and initialize `model_d` with your starting model.
3. Pick a history size. 8–10 is a reasonable starting point for large
   3D elastic problems — each history pair costs `2 * n * sizeof(real_t)`
   of device memory, so at 3D scale this is worth sizing deliberately
   rather than defaulting to the textbook m=20.
4. Build and run against a problem you already have NLCG results for,
   and compare convergence curves before trusting it on new models.

## On reducing cross-talk specifically

L-BFGS's better curvature approximation generally does help with
Vp/Vs/density cross-talk relative to NLCG, but it isn't a complete fix
by itself. Two additional levers, both hooked into the code but not
active by default:

- **Non-dimensionalize the model vector** before it enters the
  optimizer (e.g. divide each parameter class by a reference value so
  Vp, Vs, and Rho are all O(1)), and undo the scaling only when writing
  models out. This is usually the highest-leverage single change.
- **Diagonal preconditioner on H0** via
  `LBFGS::setDiagonalPreconditioner` — scales the initial Hessian
  per-parameter-class instead of leaving it isotropic. A cheap starting
  point is the inverse RMS of the initial gradient computed separately
  for each parameter block.

Neither is enabled in `example_driver.cu` by default — they're commented
hooks so you can add them once the base loop is verified against NLCG.

## Known gaps to fill in

- `evalElasticMisfitAndGradient` is a stub — this is the actual physics
  and is yours to fill in.
- The zoom phase in `line_search.cu` uses plain bisection rather than
  cubic interpolation. Simpler to verify correctness first; worth
  upgrading if line-search iteration count becomes a bottleneck (each
  iteration there costs one extra forward+adjoint solve).
- No unit tests. Given each gradient evaluation is your full elastic
  forward+adjoint solve, the useful sanity check is comparing L-BFGS
  vs. NLCG on a small 2D synthetic model where you already know the
  expected result, before moving to 3D.
# claude
