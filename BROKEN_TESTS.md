# Broken Tests Summary

This document summarizes the 9 tests marked as "broken" in the BallArithmetic.jl test suite. These represent known limitations rather than bugs.

## Summary

| Category | Count | Reason |
|----------|-------|--------|
| Interval Iterative Methods | 4 | Convergence not guaranteed for interval arithmetic |
| Extension Not Loaded | 3 | Optional dependencies (DoubleFloats, MultiFloats) |
| Ill-Conditioned Numerics | 1 | LAPACK failure on edge case |
| Repeated Eigenvalues | 1 | Iterative refinement limitation |

---

## 1. Interval Gauss-Seidel Method (Horáček)

**File:** `test/test_horacek_methods.jl:41`

**Test:** `@testset "Gauss-Seidel Method"`

**Issue:** Interval Gauss-Seidel iteration may not converge even for diagonally dominant matrices due to interval wrapping effects.

**Reason:** In interval arithmetic, each iteration can introduce overestimation (wrapping effect), causing the interval widths to grow instead of shrink. This is a fundamental limitation of naive interval iterative methods, not a bug.

**Reference:** Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis, Section 3.2

---

## 2. Interval Jacobi Method (Horáček)

**File:** `test/test_horacek_methods.jl:56`

**Test:** `@testset "Jacobi Method"`

**Issue:** Same as Gauss-Seidel - interval Jacobi iteration may not converge.

**Reason:** Interval wrapping effect prevents convergence. The spectral radius condition `ρ(D⁻¹(L+U)) < 1` is necessary but not sufficient for interval convergence.

---

## 3-4. Method Comparison (Gauss-Seidel & Jacobi)

**File:** `test/test_horacek_methods.jl:380, 383`

**Test:** `@testset "Method Comparison"`

**Issue:** When comparing solution methods, iterative methods (Gauss-Seidel, Jacobi) may not converge while direct methods (Gaussian elimination) succeed.

**Reason:** Same fundamental limitation of interval iterative methods. Direct methods are preferred for interval linear systems.

---

## 5. V3 Neumann Failure Mode (Sylvester Resolvent)

**File:** `test/test_pseudospectra/test_sylvester_resolvent.jl:409`

**Test:** `@testset "V3 Neumann failure mode"`

**Issue:** LAPACK exception when computing resolvent bounds for ill-conditioned triangular matrices.

**Reason:** The test deliberately creates an ill-conditioned case (large off-diagonal relative to eigenvalue separation) where the Neumann series bound fails (α ≥ 1). In rare cases, the fallback triangular inversion also triggers LAPACK numerical issues.

**Note:** This is an edge case specifically designed to test failure handling.

---

## 6. RefSyEv with Repeated Eigenvalues

**File:** `test/test_eigenvalues/test_iterative_schur_refinement.jl:326`

**Test:** `@testset "RefSyEv handles multiple eigenvalues"`

**Issue:** Iterative Schur refinement may not converge to ultra-high precision for matrices with repeated (multiple) eigenvalues.

**Reason:** When eigenvalues are repeated, the eigenspaces are not uniquely defined, making iterative refinement fundamentally more challenging. The algorithm still produces correct results but may not achieve the target precision (1e-30) within the iteration limit.

**Reference:** This is a known limitation of Newton-based eigenvalue refinement methods.

---

## 7. Precision Cascade SVD

**File:** `test/test_svd/test_precision_cascade_svd.jl:22`

**Test:** `@testset "Precision Cascade SVD"`

**Issue:** Test is skipped when MultiFloats.jl extension is not loaded.

**Reason:** The precision cascade SVD uses MultiFloats.jl for extended precision arithmetic. When this optional dependency is not installed, the test is marked as broken/skipped.

**Fix:** Install MultiFloats.jl: `] add MultiFloats`

---

## 8. GenericLinearAlgebra SVD

**File:** `test/test_svd/test_gla_svd.jl:19`

**Test:** `@testset "GenericLinearAlgebra SVD"`

**Issue:** Test is skipped when GenericLinearAlgebra.jl extension is not properly loaded.

**Reason:** The GLA SVD extension provides native BigFloat SVD. The test checks if the extension method is available before running.

**Fix:** Ensure GenericLinearAlgebra.jl is installed: `] add GenericLinearAlgebra`

---

## 9. Iterative Refinement Extensions (Double64)

**File:** `test/test_decompositions/test_iterative_refinement_ext.jl:20`

**Test:** `@testset "Iterative Refinement Extensions"`

**Issue:** Test is skipped when DoubleFloats.jl is not available.

**Reason:** Double64 iterative refinement requires DoubleFloats.jl for ~106-bit precision arithmetic. When not installed, these tests are skipped.

**Fix:** Install DoubleFloats.jl: `] add DoubleFloats`

---

## Recommendations

1. **For interval linear systems:** Use direct methods (`interval_gaussian_elimination`, `krawczyk_linear_system`) instead of iterative methods when rigorous enclosures are needed.

2. **For repeated eigenvalues:** Accept slightly lower precision or increase iteration limits. The results are still mathematically correct, just not at ultra-high precision.

3. **For optional extensions:** Install the optional dependencies if you need:
   - `DoubleFloats.jl` - for Double64 iterative refinement
   - `MultiFloats.jl` - for precision cascade SVD
   - `GenericLinearAlgebra.jl` - for native BigFloat SVD

---

*Last updated: 2026-01-30*
