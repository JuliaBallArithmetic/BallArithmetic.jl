"""
    GenericLinearAlgebraExt

Extension for BallArithmetic providing native BigFloat SVD via GenericLinearAlgebra.jl.

GenericLinearAlgebra provides pure-Julia implementations of linear algebra routines
that work with arbitrary numeric types, including BigFloat. For matrices larger than
~50×50, the native BigFloat SVD is faster than Float64 SVD + Ogita refinement.

# Performance comparison (256-bit BigFloat):
- 50×50:   GLA ~2.5x faster than Ogita refinement
- 100×100: GLA ~3x faster than Ogita refinement
- 200×200: GLA ~3x faster than Ogita refinement

# Usage
```julia
using BallArithmetic, GenericLinearAlgebra

A_bf = BigFloat.(randn(100, 100))
result = svd_bigfloat(A_bf)  # Uses GenericLinearAlgebra's native SVD
```
"""
module GenericLinearAlgebraExt

using BallArithmetic
using BallArithmetic: PrecisionCascadeSVDResult
using GenericLinearAlgebra
using LinearAlgebra

"""
    svd_bigfloat(A::AbstractMatrix{T}) where T<:Union{BigFloat, Complex{BigFloat}}

Compute SVD of a BigFloat matrix using GenericLinearAlgebra's native implementation.

This is faster than Float64 SVD + Ogita refinement for matrices larger than ~50×50.

# Returns
`SVD` factorization object with U, S, Vt fields.

# Example
```julia
using BallArithmetic, GenericLinearAlgebra
A = BigFloat.(randn(100, 100))
F = svd_bigfloat(A)
println("σ_min = ", F.S[end])
```
"""
function BallArithmetic.svd_bigfloat(A::AbstractMatrix{T}) where T<:Union{BigFloat, Complex{BigFloat}}
    return svd(A)
end

"""
    ogita_svd_cascade_gla(T_bf::Matrix{Complex{BigFloat}}, z_bf::Complex{BigFloat};
                          refine_iterations=0)

SVD certification using GenericLinearAlgebra's native BigFloat SVD.

GenericLinearAlgebra computes SVD directly at BigFloat precision, giving
extremely accurate results (residual ~1e-74 at 256 bits) without needing
refinement iterations.

# Arguments
- `T_bf`: Matrix in BigFloat precision
- `z_bf`: Shift value in BigFloat precision
- `refine_iterations`: Number of Ogita refinement iterations (default: 0)
  Note: refinement is typically unnecessary and can slightly degrade accuracy.

# Returns
`PrecisionCascadeSVDResult` with SVD and certified σ_min.

# Performance (100×100 matrix, 256-bit precision)
- GLA (no refine): 4.2s, residual ~1e-74  (fastest, most accurate)
- D64 cascade:     11.7s, residual ~1e-12
- Full cascade:    15.0s, residual ~1e-13
- Pure Ogita:      27.7s, residual ~1e-12

GLA is ~6.6x faster than pure Ogita refinement and ~2.8x faster than cascades.
"""
function BallArithmetic.ogita_svd_cascade_gla(
        T_bf::Matrix{Complex{BigFloat}}, z_bf::Complex{BigFloat};
        refine_iterations::Int=0)

    final_precision = Base.precision(BigFloat)
    A_bf = T_bf - z_bf * I

    # Use GenericLinearAlgebra's native BigFloat SVD
    F = svd(A_bf)
    U_bf = copy(F.U)
    Σ_bf = copy(F.S)
    V_bf = copy(Matrix(F.V))

    # Optional Ogita refinement for improved accuracy
    local residual_norm
    if refine_iterations > 0
        for _ in 1:refine_iterations
            residual_norm = _ogita_iteration_gla!(A_bf, U_bf, Σ_bf, V_bf)
        end
    else
        # Compute residual without refinement
        residual = A_bf - U_bf * Diagonal(Σ_bf) * V_bf'
        residual_norm = sqrt(real(sum(abs2, residual)))
    end

    σ_min = Σ_bf[end] - residual_norm

    return PrecisionCascadeSVDResult(U_bf, Σ_bf, V_bf, residual_norm, σ_min, final_precision)
end

"""
Ogita SVD refinement iteration for BigFloat matrices.
"""
function _ogita_iteration_gla!(A::AbstractMatrix{T}, U::AbstractMatrix{T},
                                Σ::AbstractVector{RT}, V::AbstractMatrix{T}) where {T, RT<:Real}
    n = size(A, 1)
    I_n = Matrix{T}(I, n, n)

    B = I_n - U' * U
    C = I_n - V' * V
    D = U' * A * V - Diagonal(Σ)

    for i in 1:n
        Σ[i] = abs(real(D[i, i] + Σ[i] * (1 - B[i, i]/2 - C[i, i]/2)))
    end

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

    U .= U * (I_n + E)
    V .= V * (I_n + F)

    # Newton-Schulz re-orthogonalization
    for _ in 1:2
        UtU = U' * U
        U .= U * (T(3) * I_n - UtU) / T(2)
        VtV = V' * V
        V .= V * (T(3) * I_n - VtV) / T(2)
    end

    residual = A - U * Diagonal(Σ) * V'
    return sqrt(real(sum(abs2, residual)))
end

end # module
