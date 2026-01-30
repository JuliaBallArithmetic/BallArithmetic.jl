"""
    precision_cascade_svd.jl

Precision cascade SVD refinement for efficient high-precision certification.

For large matrices (n ≥ 200), this provides ~2x speedup over pure BigFloat refinement
by performing most iterations in cheaper precisions (Float64, Double64, MultiFloat)
and only using BigFloat for final certification.

# Schedule
The default cascade performs 1 iteration at each precision level:
1. Float64 SVD (LAPACK) - initial approximation
2. 1× Float64 Ogita iteration
3. 1× Double64 iteration (~106 bits)
4. 1× Float64x3 iteration (~159 bits)
5. 1× Float64x4 iteration (~212 bits)
6. 2× BigFloat iterations (256 bits) - final certification

# Performance
- n=30: No speedup (BigFloat is fast enough)
- n=200: ~2x speedup
- Larger matrices: Expected greater speedup due to O(n³) scaling

# References
- Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value
  decomposition based on matrix multiplication", J. Comput. Appl. Math. 369, 112512.
"""

# Note: DoubleFloats and MultiFloats are optional dependencies.
# This module provides the implementation; the actual cascade function
# is enabled via extensions when these packages are loaded.

"""
    PrecisionCascadeSVDResult{T}

Result from precision cascade SVD refinement.
"""
struct PrecisionCascadeSVDResult{T, UT, ΣT, VT}
    U::UT                    # Refined left singular vectors
    Σ::ΣT                    # Refined singular values (vector)
    V::VT                    # Refined right singular vectors
    residual_norm::T         # Final residual norm ||A - UΣV'||
    σ_min::T                 # Certified minimum singular value (Σ[end] - residual_norm)
    final_precision::Int     # Final precision used (bits)
end

"""
    _ogita_iteration!(A, U, Σ, V) -> residual_norm

Perform a single Ogita SVD refinement iteration in-place.

Modifies U, Σ, V to improve the SVD approximation of A.
Returns the Frobenius norm of the residual A - U*Σ*V'.
"""
function _ogita_iteration!(A::AbstractMatrix{T}, U::AbstractMatrix{T},
                           Σ::AbstractVector{RT}, V::AbstractMatrix{T}) where {T, RT<:Real}
    n = size(A, 1)
    I_n = Matrix{T}(I, n, n)

    # Compute residuals
    B = I_n - U' * U  # Orthogonality defect of U
    C = I_n - V' * V  # Orthogonality defect of V
    D = U' * A * V - Diagonal(Σ)  # Off-diagonal residual

    # Update singular values
    for i in 1:n
        Σ[i] = abs(real(D[i, i] + Σ[i] * (1 - B[i, i]/2 - C[i, i]/2)))
    end

    # Compute correction matrices E, F
    E = zeros(T, n, n)
    F = zeros(T, n, n)
    δ = 2 * eps(RT) * maximum(Σ)

    for j in 1:n
        for i in 1:n
            if i == j
                E[i, i] = B[i, i] / 2
                F[i, i] = C[i, i] / 2
            else
                σ_diff = Σ[j] - Σ[i]
                σ_sum = Σ[j] + Σ[i]

                if abs(σ_diff) > δ
                    E[i, j] = (D[i, j] + Σ[j] * B[i, j]) / σ_diff
                else
                    E[i, j] = B[i, j] / 2
                end

                if abs(σ_sum) > δ
                    F[i, j] = (D[j, i]' + Σ[j] * C[i, j]) / σ_sum
                else
                    F[i, j] = C[i, j] / 2
                end
            end
        end
    end

    # Update U, V
    U .= U * (I_n + E)
    V .= V * (I_n + F)

    # Newton-Schulz re-orthogonalization (2 steps for stability)
    for _ in 1:2
        UtU = U' * U
        U .= U * (T(3) * I_n - UtU) / T(2)
        VtV = V' * V
        V .= V * (T(3) * I_n - VtV) / T(2)
    end

    # Compute residual norm (Frobenius as upper bound for spectral)
    residual = A - U * Diagonal(Σ) * V'
    return sqrt(real(sum(abs2, residual)))
end

# Placeholder for the cascade function - actual implementation in DoubleFloatsExt
# when both DoubleFloats and MultiFloats are available.

"""
    ogita_svd_cascade(T_bf::Matrix{Complex{BigFloat}}, z_bf::Complex{BigFloat};
                      f64_iters=1, d64_iters=1, mf3_iters=1, mf4_iters=1, bf_iters=2)

Precision cascade SVD refinement for the shifted matrix `T_bf - z_bf * I`.

Performs Ogita's iterative SVD refinement through a cascade of increasing precisions:
Float64 → Double64 → Float64x3 → Float64x4 → BigFloat

This is more efficient than pure BigFloat refinement for large matrices (n ≥ 200)
because most iterations happen in cheaper precisions.

# Arguments
- `T_bf`: Matrix in BigFloat precision
- `z_bf`: Shift value in BigFloat precision
- `f64_iters`: Number of Float64 Ogita iterations (default: 1)
- `d64_iters`: Number of Double64 iterations (default: 1)
- `mf3_iters`: Number of Float64x3 iterations (default: 1)
- `mf4_iters`: Number of Float64x4 iterations (default: 1)
- `bf_iters`: Number of final BigFloat iterations (default: 2)

# Returns
`PrecisionCascadeSVDResult` with refined SVD and certified σ_min.

# Performance
- n=200: ~2x speedup over pure BigFloat (5 iterations)
- Accuracy: relative difference ~1e-10 compared to pure BigFloat

# Example
```julia
using BallArithmetic, DoubleFloats, MultiFloats
setprecision(BigFloat, 256)

T = randn(200, 200) + 5I
T_bf = Complex{BigFloat}.(T)
z_bf = Complex{BigFloat}(6.0, 0.0)

result = ogita_svd_cascade(T_bf, z_bf)
println("σ_min = ", Float64(result.σ_min))
```

# Note
Requires DoubleFloats.jl and MultiFloats.jl to be loaded.
"""
function ogita_svd_cascade end  # Implemented in MultiFloatsExt

"""
    svd_bigfloat(A::AbstractMatrix{T}) where T<:Union{BigFloat, Complex{BigFloat}}

Compute SVD of a BigFloat matrix using GenericLinearAlgebra's native implementation.

This is faster than Float64 SVD + Ogita refinement for matrices larger than ~50×50.

# Performance (256-bit BigFloat):
- 50×50:   ~2.5x faster than Ogita refinement
- 100×100: ~3x faster than Ogita refinement

# Note
Requires GenericLinearAlgebra.jl to be loaded.
"""
function svd_bigfloat end  # Implemented in GenericLinearAlgebraExt

"""
    ogita_svd_cascade_gla(T_bf::Matrix{Complex{BigFloat}}, z_bf::Complex{BigFloat};
                          refine_iterations=0)

SVD certification using GenericLinearAlgebra's native BigFloat SVD.

This is the fastest and most accurate method for BigFloat SVD certification.
GenericLinearAlgebra computes SVD directly at full precision, achieving
residuals ~1e-74 (at 256 bits) without needing refinement.

# Arguments
- `T_bf`: Matrix in BigFloat precision
- `z_bf`: Shift value in BigFloat precision
- `refine_iterations`: Number of Ogita refinement iterations (default: 0)

# Returns
`PrecisionCascadeSVDResult` with SVD and certified σ_min.

# Performance (100×100, 256-bit)
- 6.6x faster than pure Ogita refinement
- 2.8x faster than precision cascades
- 60+ orders of magnitude better residual

# Note
Requires GenericLinearAlgebra.jl to be loaded.
"""
function ogita_svd_cascade_gla end  # Implemented in GenericLinearAlgebraExt
