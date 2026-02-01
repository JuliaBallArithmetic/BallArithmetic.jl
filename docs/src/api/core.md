# API - Core Types

Core ball arithmetic types and basic operations.

## Ball Type

```@docs
Ball
BallF64
BallComplexF64
mid
rad
midtype
radtype
inf
sup
ball_hull
intersect_ball
```

## Array Types

```@docs
BallArray
BallMatrix
BallVector
```

## Norm Bounds

```@docs
upper_bound_norm
upper_bound_L1_opnorm
upper_bound_L2_opnorm
upper_bound_L_inf_opnorm
collatz_upper_bound_L2_opnorm
svd_bound_L2_opnorm
rump_oishi_2024_triangular_bound
backward_singular_value_bound
```

### L2 Operator Norm: Method Comparison

The package provides several methods for computing rigorous upper bounds on `‖A‖₂`:

| Method | Complexity | Accuracy | Description |
|--------|------------|----------|-------------|
| `svd_bound_L2_opnorm` | O(n³) | Exact (~0%) | Computes rigorous SVD, returns largest σ |
| `collatz_upper_bound_L2_opnorm` | O(k·n²) | 0-500%* | Iterative power method on \|A\|ᵀ\|A\| |
| `sqrt(‖·‖₁·‖·‖∞)` | O(n²) | 87-564% | Uses `‖A‖₂ ≤ √(‖A‖₁·‖A‖∞)` |
| `upper_bound_L2_opnorm` | O(k·n²) | min of above | Takes minimum of Collatz and sqrt |

*Collatz accuracy depends heavily on matrix structure:
- **Tridiagonal/banded**: ~0% overestimation (essentially exact)
- **Hilbert-like matrices**: ~0% overestimation
- **Diagonally dominant**: ~26% overestimation
- **Random dense matrices**: 200-500% overestimation

#### Benchmark Results (n×n matrices)

```
Matrix type               Collatz   sqrt(‖·‖₁·‖·‖∞)   SVD bound
─────────────────────────────────────────────────────────────────
Random (n=50)             186%      249%              0%
Diagonally dominant       26%       36%               0%
Tridiagonal               0.1%      0.1%              0%
Symmetric positive def.   140%      230%              0%
Low rank + noise          54%       143%              0%
Hilbert-like              0%        117%              0%
```

#### Recommendations

- **General use**: `upper_bound_L2_opnorm` (fast, takes best of cheap bounds)
- **Accuracy critical**: `svd_bound_L2_opnorm` (exact but O(n³))
- **Structured matrices**: `collatz_upper_bound_L2_opnorm` often gives near-exact results

## Oishi 2023 Schur Complement Bounds

Lower bounds on minimum singular values using the Schur complement method
from Oishi (2023), "Lower bounds for the smallest singular values of
generalized asymptotic diagonal dominant matrices".

```@docs
Oishi2023Result
oishi_2023_sigma_min_bound
oishi_2023_optimal_block_size
```

## Rump-Oishi 2024 Improved Schur Complement Bounds

Improved bounds from Rump & Oishi (2024), "A Note on Oishi's Lower Bound for
the Smallest Singular Value of Linearized Galerkin Equations".

### Key Improvements over Oishi 2023

1. **Removes conditions 1 & 2**: No longer requires `‖A⁻¹B‖ < 1` and `‖CA⁻¹‖ < 1`.
   Only condition 3 (`‖Dd⁻¹(Df - CA⁻¹B)‖ < 1`) is needed.

2. **Uses exact ψ(N) formula**: Instead of `1/(1-‖N‖)`, uses
   `ψ(μ) = √(1 + 2αμ√(1-α²) + α²μ²)` which is tighter and works for any `μ ≥ 0`.

3. **Fast γ bound** (optional): Uses `π(N) = √(‖N‖₁·‖N‖∞)` to quickly check
   if the method will succeed, avoiding expensive matrix products and reducing
   complexity from O((n-m)²m) to O((n-m)m²).

### Example

```julia
using BallArithmetic, LinearAlgebra

# Create a diagonally dominant matrix (Example 3 from paper)
k, n = 0.9, 50
G_mid = [i == j ? Float64(i) : k^abs(i-j) for i in 1:n, j in 1:n]
G = BallMatrix(G_mid, zeros(n, n))

# Compute bounds with optimal block size
best_m, result = rump_oishi_2024_optimal_block_size(G; max_m=30)

if result.verified
    println("σ_min(G) ≥ ", result.sigma_min_lower)
    println("‖G⁻¹‖ ≤ ", result.G_inv_upper)
    println("Optimal block size: m = ", best_m)
    println("Used fast γ bound: ", result.used_fast_gamma)
end
```

### API Reference

```@docs
RumpOishi2024Result
rump_oishi_2024_sigma_min_bound
rump_oishi_2024_optimal_block_size
psi_schur_factor
pi_norm
```
