# Certified Matrix Decompositions

BallArithmetic.jl provides rigorous certification for several matrix decompositions.
Each method computes both a numerical approximation and a **rigorous error bound**.

## Overview

| Decomposition | Function | Description |
|---------------|----------|-------------|
| SVD | `ogita_svd_refine` | Singular value decomposition |
| Schur | `iterative_schur_refinement` | Schur form |
| LU | `verified_lu` | LU factorization |
| Cholesky | `verified_cholesky` | Cholesky factorization |
| QR | `verified_qr` | QR factorization |
| Polar | `verified_polar` | Polar decomposition |
| Takagi | `verified_takagi` | Takagi factorization |

## Recommended Extensions

BallArithmetic supports several extension packages that provide different performance characteristics:

| Extension | Package | Use Case |
|-----------|---------|----------|
| **GenericLinearAlgebra** | `GenericLinearAlgebra.jl` | **Recommended** - Native BigFloat SVD, fastest and most accurate |
| **MultiFloats** | `MultiFloats.jl` | SIMD-accelerated multi-precision (Float64x2, Float64x4) |
| **DoubleFloats** | `DoubleFloats.jl` | Fast Double64 arithmetic (~106 bits) |

## Rigorous vs Non-Rigorous Results

Each decomposition method returns:

- **Non-rigorous residual:** The computed `‖A - UΣVᴴ‖_F` using standard floating-point
- **Rigorous bound:** A mathematically guaranteed upper bound on the true error

The rigorous bound accounts for:
- Rounding errors in all computations
- Potential instabilities in the algorithm
- Accumulated error from multiple operations

For certification purposes, **always use the rigorous bound**.

## Other Matrix Decompositions (100×100 Matrix)

Beyond SVD, BallArithmetic provides verified versions of standard matrix decompositions.

### Standard (Float64 → BigFloat)

These use Float64 computation with BigFloat certification:

| Decomposition | Time | Residual Norm | Rigorous Bound |
|---------------|------|---------------|----------------|
| Cholesky (SPD) | 1.7s | 2.7×10⁻¹⁶ | ✓ |
| QR | 2.1s | 3.1×10⁻¹⁵ | ✓ |
| Polar | 3.3s | 4.8×10⁻¹⁶ | ✓ |
| LU | 5.6s | 1.3×10⁻¹⁴ | ✓ |
| Takagi | 10.3s | 1.7×10⁻¹⁵ | ✓ |

### GLA-Based (Native BigFloat)

With GenericLinearAlgebra, we achieve **60+ orders of magnitude better residuals**:

| Decomposition | Time | Residual Norm | Improvement |
|---------------|------|---------------|-------------|
| **Cholesky (GLA)** | **0.55s** | **5.2×10⁻⁷⁴** | 10⁵⁸× better |
| **LU (GLA)** | **0.91s** | **5.9×10⁻⁷⁵** | 10⁶¹× better |
| **QR (GLA)** | **0.91s** | **5.4×10⁻⁷⁵** | 10⁶⁰× better |
| **SVD (GLA)** | 5.15s | 4.5×10⁻⁷⁴ | 10⁶⁰× better |

**GLA functions:**
```julia
using BallArithmetic, GenericLinearAlgebra

result_lu = verified_lu(A)              # in BigFloat, the default
result_qr = verified_qr(A)
result_chol = verified_cholesky(A_spd)
result_svd = verified_svd(A)            # a VerifySvdAllResult
result_fast = verified_lu(Float64, A)
result_d64 = verified_lu(A; oracle = Double64)   # starting factorisation in Double64
```

**Usage examples:**

```julia
using BallArithmetic
using LinearAlgebra

A = randn(100, 100) + 5I

# LU decomposition
result_lu = verified_lu(A)
# result_lu.L, result_lu.U are BallMatrix enclosures
# result_lu.residual_norm is the rigorous error bound

# QR decomposition
result_qr = verified_qr(A)
# result_qr.Q, result_qr.R are BallMatrix enclosures
# result_qr.orthogonality_defect bounds ‖QᴴQ - I‖

# Cholesky (for symmetric positive definite)
A_spd = A'A + 100I
result_chol = verified_cholesky(A_spd)
# result_chol.G encloses the upper triangular factor, A_spd = G'G

# Polar decomposition A = QP
result_polar = verified_polar(A)
# result_polar.Q (unitary), result_polar.P (Hermitian positive definite)

# Takagi factorization (for complex symmetric)
A_symm = A + transpose(A)
result_takagi = verified_takagi(A_symm)
# A = U Σ Uᵀ (not Uᴴ!)
```

## Schur Complement Bounds (Oishi 2023 / Rump-Oishi 2024)

For block-structured matrices, the Schur complement method provides efficient σ_min bounds:

```julia
result = rump_oishi_2024_sigma_min_bound(G; block_size=m)
```

This implements:
- **Oishi 2023:** Base Schur complement algorithm
- **Rump-Oishi 2024:** Improved ψ(N) formula that works for ‖N‖ ≥ 1

Key advantages:
- Works on matrices too large for full SVD
- Exploits block diagonal dominance structure
- Automatic optimal block size selection

## References

- Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value decomposition based on matrix multiplication"
- Rump, S.M. & Ogita, S. (2024), "A Note on Oishi's Lower Bound for the Smallest Singular Value of Linearized Galerkin Equations"
- Oishi, S. (2023), "Lower bounds for the smallest singular values of generalized asymptotic diagonal dominant matrices"
