# Certified Matrix Decompositions

BallArithmetic.jl provides rigorous certification for several matrix decompositions.
Each method computes both a numerical approximation and a **rigorous error bound**.

## Overview

| Decomposition | Function | Description |
|---------------|----------|-------------|
| SVD | `ogita_svd_refine` | Singular value decomposition |
| SVD Cascade | `ogita_svd_cascade` | Multi-precision SVD cascade |
| SVD (GLA) | `ogita_svd_cascade_gla` | Native BigFloat SVD |
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

## SVD Certification Methods

The SVD is fundamental for resolvent norm certification and condition number estimation.
We provide multiple methods with different speed/accuracy trade-offs.

### Method Comparison

All benchmarks performed on:
- **CPU:** AMD Ryzen 5 5600 6-Core Processor
- **RAM:** 128 GB
- **Julia:** 1.11+
- **BigFloat precision:** 256 bits

#### Small Matrix (100×100)

| Method | Time | Non-rigorous Residual | Rigorous σ_min Bound | Speedup |
|--------|------|----------------------|---------------------|---------|
| **GLA (no refine)** | **4.2s** | **4×10⁻⁷⁴** | ✓ | **6.6×** |
| D64 cascade (F64→D64→BF) | 11.7s | 1×10⁻¹² | ✓ | 2.4× |
| Minimal cascade (F64→MF3→BF) | 12.2s | 1×10⁻¹² | ✓ | 2.3× |
| Full cascade (F64→D64→MF3→MF4→BF) | 15.0s | 1×10⁻¹³ | ✓ | 1.8× |
| Pure Ogita (F64→BF×5) | 27.7s | 1×10⁻¹² | ✓ | 1.0× |

**Key observations:**
- GLA achieves **60+ orders of magnitude** better residual than cascade methods
- GLA is **6.6× faster** than pure Ogita refinement
- All methods produce valid rigorous bounds for σ_min
- Simpler cascades (D64 only) outperform complex multi-level cascades

#### Medium Matrix (200×200)

| Method | Time | Non-rigorous Residual | Rigorous Bound | Speedup |
|--------|------|----------------------|----------------|---------|
| Cascade (F64→D64→MF3→MF4→BF×2) | 196s | 3.3×10⁻¹² | ✓ | 2.1× |
| Pure Ogita (F64→BF×5) | 407s | 5.0×10⁻¹² | ✓ | 1.0× |

#### Large Matrix (500×500)

| Method | Time | Non-rigorous Residual | Rigorous Bound | Speedup |
|--------|------|----------------------|----------------|---------|
| Cascade (F64→D64→MF3→MF4→BF×2) | 3234s | 1.4×10⁻¹¹ | ✓ | 1.9× |
| Pure Ogita (F64→BF×5) | 6279s | 1.6×10⁻¹¹ | ✓ | 1.0× |

### Recommendations

1. **Use GenericLinearAlgebra** when available:
   ```julia
   using BallArithmetic, GenericLinearAlgebra

   T_bf = Complex{BigFloat}.(T)
   z_bf = Complex{BigFloat}(z)
   result = ogita_svd_cascade_gla(T_bf, z_bf)

   # result.σ_min is a rigorous lower bound
   # result.residual_norm is the non-rigorous residual
   ```

2. **Without GLA**, use the simple D64 cascade:
   ```julia
   using BallArithmetic, MultiFloats, DoubleFloats

   result = ogita_svd_cascade(T_bf, z_bf;
       f64_iters=1, d64_iters=1, mf3_iters=0, mf4_iters=0, bf_iters=2)
   ```

3. **For pure BigFloat** (no extensions):
   ```julia
   using BallArithmetic

   result = ogita_svd_refine(A, U, Σ, V; max_iterations=5)
   ```

## Rigorous vs Non-Rigorous Results

Each decomposition method returns:

- **Non-rigorous residual:** The computed `‖A - UΣVᴴ‖_F` using standard floating-point
- **Rigorous bound:** A mathematically guaranteed upper bound on the true error

The rigorous bound accounts for:
- Rounding errors in all computations
- Potential instabilities in the algorithm
- Accumulated error from multiple operations

For certification purposes, **always use the rigorous bound**.

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
