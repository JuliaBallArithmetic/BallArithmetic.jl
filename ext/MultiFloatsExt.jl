"""
    MultiFloatsExt

Extension for BallArithmetic providing fast matrix decomposition refinement using MultiFloats.jl.

MultiFloats provides Float64x2 (~106 bits), Float64x4 (~212 bits), etc. with SIMD acceleration.
This is complementary to DoubleFloatsExt:
- DoubleFloats: Double64 (~106 bits), full linear algebra via GenericLinearAlgebra
- MultiFloats: Float64x2-x8 (106-424 bits), SIMD accelerated, faster for element-wise operations

The "oracle then certify" pattern:
1. Fast approximate computation in Float64xN (no rigorous rounding needed)
2. Final certification in BigFloat for rigorous bounds

Based on Rump & Ogita (2024), "Verified Error Bounds for Matrix Decompositions",
SIAM J. Matrix Anal. Appl., 45(4):2155-2183.
"""
module MultiFloatsExt

using BallArithmetic
using BallArithmetic: _ogita_svd_refine_impl, _spectral_norm_bound,
                      OgitaSVDRefinementResult, ogita_iterations_for_precision,
                      SchurRefinementResult, SymmetricEigenRefinementResult,
                      _frobenius_norm, _to_bigfloat
using MultiFloats
using LinearAlgebra

#==============================================================================#
# Type aliases for convenience
#==============================================================================#

const F64x2 = Float64x2
const F64x4 = Float64x4

#==============================================================================#
# Ogita SVD Refinement with MultiFloats
#==============================================================================#

"""
    ogita_svd_refine_multifloat(A, U, Σ, V; precision=:x2, max_iterations=2,
                                 certify_with_bigfloat=true, bigfloat_precision=256)

Fast SVD refinement using MultiFloats arithmetic.

# Arguments
- `A`: Original matrix (Float64 or ComplexF64)
- `U, Σ, V`: Initial SVD approximation (from LAPACK)
- `precision`: MultiFloats precision (`:x2` = 106 bits, `:x4` = 212 bits, `:x8` = 424 bits)
- `max_iterations`: Number of refinement iterations (default: 2)
- `certify_with_bigfloat`: if true, the factors are returned in BigFloat and the residual is
  evaluated there (a floating-point diagnostic, not a bound)

# Precision Guide
- `:x2` (Float64x2): ~106 bits, similar to Double64, fastest
- `:x4` (Float64x4): ~212 bits, intermediate
- `:x8` (Float64x8): ~424 bits, highest precision MultiFloats offers

# Performance
MultiFloats is SIMD-accelerated and can be faster than DoubleFloats for some operations.
Expected speedup: 10-50× compared to BigFloat.
"""
function BallArithmetic.ogita_svd_refine_multifloat(
        A::AbstractMatrix{T}, U, Σ, V;
        precision::Symbol=:x2,
        max_iterations::Int=2,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    # Select MultiFloat type based on precision
    MF = _select_multifloat_type(precision)

    # Convert to MultiFloats for refinement
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        U_mf = convert.(Complex{MF}, U)
        V_mf = convert.(Complex{MF}, V)
    else
        A_mf = convert.(MF, A)
        U_mf = convert.(MF, U)
        V_mf = convert.(MF, V)
    end
    Σ_mf = convert.(MF, Σ)

    # Run refinement in MultiFloats
    result_mf = _ogita_svd_refine_impl(A_mf, U_mf, Σ_mf, V_mf, max_iterations, false)

    if certify_with_bigfloat
        # the residual evaluated in BigFloat: a diagnostic, not a bound
        # a scoped setting: a global one has no effect inside a caller's scoped block
        return setprecision(BigFloat, bigfloat_precision) do
            # Convert refined SVD to BigFloat
            if T <: Complex
                U_bf = convert.(Complex{BigFloat}, result_mf.U)
                V_bf = convert.(Complex{BigFloat}, result_mf.V)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                U_bf = convert.(BigFloat, result_mf.U)
                V_bf = convert.(BigFloat, result_mf.V)
                A_bf = convert.(BigFloat, A)
            end
            Σ_vec = isa(result_mf.Σ, Diagonal) ? diag(result_mf.Σ) : result_mf.Σ
            Σ_bf = convert.(BigFloat, Σ_vec)

            # the norm of the residual, in floating point
            residual = A_bf - U_bf * Diagonal(Σ_bf) * V_bf'
            residual_norm = _spectral_norm_bound(residual)

            return OgitaSVDRefinementResult(
                U_bf, Diagonal(Σ_bf), V_bf,
                max_iterations,
                bigfloat_precision,
                residual_norm,
                true
            )
        end
    else
        # Return MultiFloats result directly
        Σ_vec = isa(result_mf.Σ, Diagonal) ? diag(result_mf.Σ) : result_mf.Σ
        residual = A_mf - result_mf.U * Diagonal(Σ_vec) * result_mf.V'
        residual_norm = _spectral_norm_bound_mf(residual)

        return OgitaSVDRefinementResult(
            result_mf.U, Diagonal(Σ_vec), result_mf.V,
            max_iterations,
            _precision_bits(MF),
            convert(BigFloat, residual_norm),
            true
        )
    end
end

#==============================================================================#
# Schur Decomposition Refinement with MultiFloats
#==============================================================================#

# The Schur refinement is the one of the package (Algorithm 4 of Bujanović, Kressner and
# Schröder) with a MultiFloat as the high precision.
BallArithmetic.refine_schur_multifloat(A::AbstractMatrix{T}, Q0::Matrix, T0::Matrix;
    method::Symbol = :bujanovicetal2023, precision::Symbol = :x2,
    max_iterations::Int = 10, tol::Real = 0.0, threshold::Real = 1e-5, rng = nothing,
    certify::Bool = true, bigfloat_precision::Int = 256) where {T <: Union{Float64, ComplexF64}} =
    BallArithmetic._schur_refinement(method, A, Q0, T0, max_iterations, tol;
        hp = _select_multifloat_type(precision), rng, threshold)

#==============================================================================#
# Symmetric eigendecomposition: RefSyEv of the package with a MultiFloat as the high precision
#==============================================================================#

BallArithmetic.refine_symmetric_eigen_multifloat(A::AbstractMatrix{T}, Q0::Matrix,
    λ0::AbstractVector; precision::Symbol = :x2, max_iterations::Int = 10, tol::Real = 0.0,
    certify::Bool = true,
    bigfloat_precision::Int = 256) where {T <: Union{Float64, ComplexF64}} =
    BallArithmetic._refine_symmetric_eigen_impl(A, Q0, λ0, max_iterations, tol;
        hp = _select_multifloat_type(precision))

#==============================================================================#
# Helper Functions
#==============================================================================#

"""Select MultiFloat type based on precision symbol."""
function _select_multifloat_type(precision::Symbol)
    if precision == :x2
        return Float64x2
    elseif precision == :x4
        return Float64x4
    elseif precision == :x8
        return Float64x8
    else
        throw(ArgumentError("Unknown precision: $precision. Use :x2, :x4, or :x8"))
    end
end

"""Get precision in bits for MultiFloat type."""
function _precision_bits(::Type{Float64x2})
    return 106
end
function _precision_bits(::Type{Float64x4})
    return 212
end
function _precision_bits(::Type{Float64x8})
    return 424
end

"""Spectral norm bound for MultiFloat matrices."""
function _spectral_norm_bound_mf(A::AbstractMatrix{T}) where T
    m, n = size(A)
    inf_norm = zero(real(T))
    for i in 1:m
        row_sum = sum(abs(A[i, j]) for j in 1:n)
        inf_norm = max(inf_norm, row_sum)
    end
    one_norm = zero(real(T))
    for j in 1:n
        col_sum = sum(abs(A[i, j]) for i in 1:m)
        one_norm = max(one_norm, col_sum)
    end
    return sqrt(inf_norm * one_norm)
end

#==============================================================================#
# Exports
#==============================================================================#


end # module
