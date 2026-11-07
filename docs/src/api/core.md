# API - Core Types

Core ball arithmetic types and basic operations.

```@meta
CurrentModule = BallArithmetic
```

## The Module

```@docs
BallArithmetic
```

## Ball Type

```@docs
Ball
BallF64
BallComplexF64
mid
rad
mag
mig
midtype
radtype
inf
sup
ball_hull
intersect_ball
in0
BallArithmetic.:±
BallArithmetic._ball_from_bounds
```

## Directed Rounding

```@docs
abs_up
```

## Array Types

```@docs
BallArray
BallMatrix
BallVector
HermitianBallMatrix
DenseBallMatrix
DenseBallVector
BallArithmetic._zero_radius
BallArithmetic._check_axes
```

The radii are not checked on construction, since the scan would cost more than the arithmetic it
guards; the axes of the midpoint and radius containers are, that test being `O(1)`. Radii of
external provenance go through one of the two predicates below first.

```@docs
isvalid_enclosure
check_enclosure(::BallArray)
```

## The `Base` interface of the ball types

What follows is the arithmetic and the array interface that `Ball`, `BallArray`, `BallMatrix` and
`BallVector` add to `Base`. Every operation returns an enclosure: the midpoint is the
floating-point result and the radius carries the radii of the operands together with a bound on
the rounding of the midpoint, accumulated with the directed operations of
[Arithmetic and Rounding](arithmetic.md). The comparisons are the exception, and the one place
where a result is *not* a proof of anything about the enclosed sets: they compare midpoints, so
that balls can be sorted, and a rigorous "certainly less than" is `sup(a) < inf(b)`.

### Scalar arithmetic

```@docs
Base.:+(::Ball)
Base.:+(::Ball{T}, ::Ball{T}) where T
Base.:/(::Ball, ::Ball)
Base.inv(::Ball{T, T}) where T<:AbstractFloat
Base.inv(::Ball{T, Complex{T}}) where T<:AbstractFloat
Base.sqrt(::Ball{T}) where T<:AbstractFloat
Base.abs(::Ball{T, T}) where T<:AbstractFloat
Base.conj(::Ball)
```

### Containment, comparison and conversion

```@docs
Base.in(::Union{AbstractFloat, Complex{<:AbstractFloat}}, ::Ball)
Base.in(::Ball{T, T}, ::Ball{T, T}) where T<:AbstractFloat
Base.in(::Ball{T, Complex{T}}, ::Ball{T, Complex{T}}) where T<:AbstractFloat
Base.isless(::Ball{T, T}, ::Ball{T, T}) where T<:AbstractFloat
Base.isless(::Ball{T, T}, ::Number) where T<:AbstractFloat
Base.:<(::Ball{T, T}, ::Ball{T, T}) where T<:AbstractFloat
Base.:<=(::Ball{T, T}, ::Ball{T, T}) where T<:AbstractFloat
Base.:>(::Ball{T, T}, ::Ball{T, T}) where T<:AbstractFloat
Base.:>=(::Ball{T, T}, ::Ball{T, T}) where T<:AbstractFloat
Base.convert(::Type{Ball{T, CT}}, ::Ball) where {T, CT}
Base.convert(::Type{Ball{T, CT}}, ::Number) where {T, CT}
```

### The array interface

```@docs
Base.size(::BallArray)
Base.length(::BallArray)
Base.getindex(::BallArray, ::Int)
Base.setindex!(::BallArray, ::Any, ::Any...)
Base.copy(::BallArray)
Base.real(::BallArray{T, N, T}) where {T<:AbstractFloat, N}
Base.imag(::BallArray{T, N, T}) where {T<:AbstractFloat, N}
Base.zeros(::Type{B}, ::NTuple{N, Integer}) where {B<:Ball, N}
Base.ones(::Type{B}, ::NTuple{N, Integer}) where {B<:Ball, N}
Base.fill(::Ball, ::Vararg{Int, N}) where N
```

### Matrix and vector arithmetic

```@docs
Base.adjoint(::BallMatrix)
Base.transpose(::BallMatrix)
Base.:+(::BallMatrix{T}, ::BallMatrix{T}) where T<:AbstractFloat
Base.:+(::BallMatrix{T}, ::AbstractMatrix{<:Union{Complex{T}, T}}) where T<:AbstractFloat
Base.:+(::AbstractMatrix{<:Union{Complex{T}, T}}, ::BallMatrix{T}) where T<:AbstractFloat
Base.:+(::BallMatrix{T}, ::LinearAlgebra.UniformScaling) where T
Base.:+(::LinearAlgebra.UniformScaling, ::BallMatrix)
Base.:+(::BallMatrix{T}, ::LinearAlgebra.UniformScaling{Ball{T, NT}}) where {T, NT<:Union{Complex{T}, T}}
Base.:+(::BallVector{T}, ::BallVector{T}) where T<:AbstractFloat
Base.:-(::Ball)
Base.:-(::Ball{T, CT} where CT<:Union{Complex{T}, T}, ::Ball{T, CT} where CT<:Union{Complex{T}, T}) where T
Base.:-(::BallMatrix{T}, ::BallMatrix{T}) where T<:AbstractFloat
Base.:-(::BallMatrix{T}, ::AbstractMatrix{<:Union{Complex{T}, T}}) where T<:AbstractFloat
Base.:-(::AbstractMatrix{<:Union{Complex{T}, T}}, ::BallMatrix{T}) where T<:AbstractFloat
Base.:-(::BallMatrix{T}, ::LinearAlgebra.UniformScaling) where T
Base.:-(::BallMatrix{T}, ::LinearAlgebra.UniformScaling{Ball{T, NT}}) where {T, NT<:Union{Complex{T}, T}}
Base.:-(::LinearAlgebra.UniformScaling, ::BallMatrix{T}) where T
Base.:-(::LinearAlgebra.UniformScaling{Ball{T, NT}}, ::BallMatrix{T}) where {T, NT<:Union{Complex{T}, T}}
Base.:-(::BallVector{T}, ::BallVector{T}) where T<:AbstractFloat
Base.:*(::Ball{T, CT} where CT<:Union{Complex{T}, T}, ::Ball{T, CT} where CT<:Union{Complex{T}, T}) where T
Base.:*(::Number, ::BallMatrix{T}) where T
Base.:*(::BallMatrix, ::Number)
Base.:*(::Ball{T, NT}, ::BallMatrix{T}) where {T, NT<:Union{Complex{T}, T}}
Base.:*(::BallMatrix{T}, ::Ball{T, NT}) where {T, NT<:Union{Complex{T}, T}}
Base.:*(::BallMatrix{T, S}, ::BallMatrix{T, S}) where {S, T<:AbstractFloat}
Base.:*(::BallMatrix{T, S}, ::AbstractMatrix{S}) where {S, T<:AbstractFloat}
Base.:*(::AbstractMatrix{S}, ::BallMatrix{T, S}) where {S, T<:AbstractFloat}
Base.:*(::BallMatrix{T, Complex{T}}, ::BallMatrix{T, T}) where T<:AbstractFloat
Base.:*(::BallMatrix{T, T}, ::BallMatrix{T, Complex{T}}) where T<:AbstractFloat
Base.:*(::BallMatrix{T, Complex{T}}, ::BallMatrix{T, Complex{T}}) where T<:AbstractFloat
Base.:*(::AbstractMatrix{T}, ::BallMatrix{T, Complex{T}}) where T<:AbstractFloat
Base.:*(::AbstractMatrix{Complex{T}}, ::BallMatrix{T, T}) where T<:AbstractFloat
Base.:*(::AbstractMatrix{Complex{T}}, ::BallMatrix{T, Complex{T}}) where T<:AbstractFloat
Base.:*(::Number, ::BallVector{T}) where T
Base.:*(::Ball{T, NT}, ::BallVector{T}) where {T, NT<:Union{Complex{T}, T}}
Base.:*(::BallMatrix, ::BallVector)
Base.:*(::AbstractMatrix, ::BallVector)
Base.:*(::BallMatrix, ::AbstractVector)
```

## Whether a Result Is Proved

Every result type whose computation can fail to prove what it claims carries one boolean field,
`certified`. `iscertified` reads it whatever the type, and reads it as well under the names the
same flag had in version 0.3.0, `success`, `verified`, `is_regular` and
`precomputation_success`, which still resolve with a deprecation warning.

```@docs
iscertified
```

## Norm Bounds

```@docs
upper_bound_norm
upper_bound_L1_opnorm
upper_bound_L2_opnorm
upper_bound_L_inf_opnorm
collatz_upper_bound_L2_opnorm
svd_bound_L2_opnorm
BallArithmetic.svd_bound_L2_opnorm_inverse
triangular_inverse_two_norm_bound
rump_oishi_2024_triangular_bound
BallArithmetic.upper_abs
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

if result.certified
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
psi_squared
pi_norm
```
