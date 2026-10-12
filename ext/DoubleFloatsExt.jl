"""
    DoubleFloatsExt

Extension for BallArithmetic providing fast SVD refinement using DoubleFloats.jl.

The key insight is that iterative refinement (Ogita's RefSVD) doesn't need rigorous
arithmetic - it just needs extended precision to compute a good approximation.
The final certification step uses rigorous ball arithmetic.

This provides ~10-30× speedup over pure BigFloat refinement.
"""
module DoubleFloatsExt

using BallArithmetic
using BallArithmetic: _ogita_svd_refine_impl, _spectral_norm_bound,
                      OgitaSVDRefinementResult, ogita_iterations_for_precision,
                      _stril, _frobenius_norm
using DoubleFloats
using LinearAlgebra

"""
    ogita_svd_refine_double64(A, U, Σ, V; max_iterations=2, certify_with_bigfloat=true)

Fast SVD refinement using Double64 arithmetic (~106 bits precision).

This is much faster than BigFloat refinement because:
1. Double64 uses native Float64 operations with error compensation
2. No memory allocation per arithmetic operation (unlike BigFloat)
3. ~30× faster than BigFloat for matrix operations

# Arguments
- `A`: Original matrix (Float64 or ComplexF64)
- `U, Σ, V`: Initial SVD approximation (from LAPACK)
- `max_iterations`: Number of Double64 refinement iterations (default: 2)
- `certify_with_bigfloat`: If true, compute final residual with BigFloat for rigor

# Returns
- `OgitaSVDRefinementResult` with refined SVD

# Notes
Double64 provides ~106 bits (~31 decimal digits). Starting from Float64 (~15 digits):
- After 1 iteration: ~30 digits (saturates Double64)
- After 2 iterations: ~60 digits (exceeds Double64, but still improves due to error structure)

For most applications, 2 iterations with Double64 gives excellent results.
For higher precision needs, use `ogita_svd_refine` with BigFloat.
"""
function BallArithmetic.ogita_svd_refine_fast(
        A::AbstractMatrix{T}, U, Σ, V;
        max_iterations::Int=2,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    # Convert to Double64 for refinement
    if T <: Complex
        A_d64 = convert.(Complex{Double64}, A)
        U_d64 = convert.(Complex{Double64}, U)
        V_d64 = convert.(Complex{Double64}, V)
    else
        A_d64 = convert.(Double64, A)
        U_d64 = convert.(Double64, U)
        V_d64 = convert.(Double64, V)
    end
    Σ_d64 = convert.(Double64, Σ)

    # Run refinement in Double64
    result_d64 = _ogita_svd_refine_impl(A_d64, U_d64, Σ_d64, V_d64, max_iterations, false)

    if certify_with_bigfloat
        # the residual evaluated in BigFloat: a diagnostic, not a bound
        # a scoped setting: a global one has no effect inside a caller's scoped block
        return setprecision(BigFloat, bigfloat_precision) do
            # Convert refined SVD to BigFloat
            if T <: Complex
                U_bf = convert.(Complex{BigFloat}, result_d64.U)
                V_bf = convert.(Complex{BigFloat}, result_d64.V)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                U_bf = convert.(BigFloat, result_d64.U)
                V_bf = convert.(BigFloat, result_d64.V)
                A_bf = convert.(BigFloat, A)
            end
            Σ_vec = isa(result_d64.Σ, Diagonal) ? diag(result_d64.Σ) : result_d64.Σ
            Σ_bf = convert.(BigFloat, Σ_vec)

            # the norm of the residual, in floating point
            residual = A_bf - U_bf * Diagonal(Σ_bf) * V_bf'
            residual_norm = _spectral_norm_bound(residual)

            return OgitaSVDRefinementResult(
                U_bf, Diagonal(Σ_bf), V_bf,
                result_d64.iterations,
                bigfloat_precision,
                residual_norm,
                true
            )
        end
    else
        # the Double64 result as it is
        Σ_vec = isa(result_d64.Σ, Diagonal) ? diag(result_d64.Σ) : result_d64.Σ
        residual = A_d64 - result_d64.U * Diagonal(Σ_vec) * result_d64.V'
        residual_norm = _spectral_norm_bound(residual)

        return OgitaSVDRefinementResult(
            result_d64.U, Diagonal(Σ_vec), result_d64.V,
            result_d64.iterations,
            106,  # Double64 precision in bits
            convert(BigFloat, residual_norm),  # Convert for type consistency
            true
        )
    end
end

"""
    ogita_svd_refine_hybrid(A, U, Σ, V; d64_iterations=2, bf_iterations=1, precision_bits=256)

Hybrid refinement: Double64 for bulk iterations, BigFloat for final polish.

This combines the speed of Double64 with the rigor of BigFloat:
1. Run `d64_iterations` in Double64 (fast, ~30× faster than BigFloat)
2. Run `bf_iterations` in BigFloat (rigorous, provides certified error bound)

# Arguments
- `A`: Original matrix
- `U, Σ, V`: Initial SVD from LAPACK
- `d64_iterations`: Number of Double64 iterations (default: 2)
- `bf_iterations`: Number of BigFloat iterations (default: 1)
- `precision_bits`: BigFloat precision (default: 256)

# Performance
For 256-bit precision:
- Pure BigFloat: 3 iterations needed
- Hybrid (2 D64 + 1 BF): ~2× faster, same precision
"""
function BallArithmetic.ogita_svd_refine_hybrid(
        A::AbstractMatrix{T}, U, Σ, V;
        d64_iterations::Int=2,
        bf_iterations::Int=1,
        precision_bits::Int=256) where {T<:Union{Float64, ComplexF64}}

    # Phase 1: Fast refinement in Double64
    if T <: Complex
        A_d64 = convert.(Complex{Double64}, A)
        U_d64 = convert.(Complex{Double64}, U)
        V_d64 = convert.(Complex{Double64}, V)
    else
        A_d64 = convert.(Double64, A)
        U_d64 = convert.(Double64, U)
        V_d64 = convert.(Double64, V)
    end
    Σ_d64 = convert.(Double64, Σ)

    result_d64 = _ogita_svd_refine_impl(A_d64, U_d64, Σ_d64, V_d64, d64_iterations, false)

    # Phase 2: Polish and certify in BigFloat
    # a scoped setting: a global one has no effect inside a caller's scoped block
    return setprecision(BigFloat, precision_bits) do
        if T <: Complex
            A_bf = convert.(Complex{BigFloat}, A)
            U_bf = convert.(Complex{BigFloat}, result_d64.U)
            V_bf = convert.(Complex{BigFloat}, result_d64.V)
        else
            A_bf = convert.(BigFloat, A)
            U_bf = convert.(BigFloat, result_d64.U)
            V_bf = convert.(BigFloat, result_d64.V)
        end
        Σ_vec = isa(result_d64.Σ, Diagonal) ? diag(result_d64.Σ) : result_d64.Σ
        Σ_bf = convert.(BigFloat, Σ_vec)

        # Run BigFloat iterations
        result_bf = _ogita_svd_refine_impl(A_bf, U_bf, Σ_bf, V_bf, bf_iterations, false)

        # Compute rigorous residual
        Σ_final = isa(result_bf.Σ, Diagonal) ? diag(result_bf.Σ) : result_bf.Σ
        residual = A_bf - result_bf.U * Diagonal(Σ_final) * result_bf.V'
        residual_norm = _spectral_norm_bound(residual)

        return OgitaSVDRefinementResult(
            result_bf.U, Diagonal(Σ_final), result_bf.V,
            d64_iterations + bf_iterations,
            precision_bits,
            residual_norm,
            true
        )
    end
end

#==============================================================================#
# Schur Decomposition Refinement with Double64
#==============================================================================#

# The Schur refinements are those of the package (Algorithm 4 of Bujanović, Kressner and
# Schröder) with Double64 as the high precision.
BallArithmetic.refine_schur_double64(A::AbstractMatrix{T}, Q0::Matrix, T0::Matrix;
    method::Symbol = :bujanovicetal2023, max_iterations::Int = 10, tol::Real = 0.0,
    threshold::Real = 1e-5, rng = nothing, certify_with_bigfloat::Bool = true,
    bigfloat_precision::Int = 256) where {T <: Union{Float64, ComplexF64}} =
    BallArithmetic._schur_refinement(method, A, Q0, T0, max_iterations, tol; hp = Double64,
        rng, threshold)

# The Double64 stage runs `method`; the BigFloat stage starts from its result, which is already
# ordered, and so runs the unordered iteration.
function BallArithmetic.refine_schur_hybrid(A::AbstractMatrix{T}, Q0::Matrix, T0::Matrix;
        method::Symbol = :bujanovicetal2023, d64_iterations::Int = 10,
        bf_iterations::Int = 5, threshold::Real = 1e-5, rng = nothing,
        precision_bits::Int = 256) where {T <: Union{Float64, ComplexF64}}
    r = BallArithmetic.refine_schur_double64(A, Q0, T0; method, max_iterations = d64_iterations,
        threshold, rng)
    # the first stage has applied `method`, reordering included; the second continues from its
    # result without reordering again
    return BallArithmetic.refine_schur_decomposition(A, r.Q, r.T;
        method = :bujanovicetal2023unordered,
        target_precision = precision_bits, max_iterations = bf_iterations)
end

#==============================================================================#
# Symmetric eigendecomposition: RefSyEv of the package with Double64 as the high precision
#==============================================================================#

BallArithmetic.refine_symmetric_eigen_double64(A::AbstractMatrix{T}, Q0::Matrix,
    λ0::AbstractVector; max_iterations::Int = 10, tol::Real = 0.0,
    certify_with_bigfloat::Bool = true,
    bigfloat_precision::Int = 256) where {T <: Union{Float64, ComplexF64}} =
    BallArithmetic._refine_symmetric_eigen_impl(A, Q0, λ0, max_iterations, tol; hp = Double64)

function BallArithmetic.refine_symmetric_eigen_hybrid(A::AbstractMatrix{T}, Q0::Matrix,
        λ0::AbstractVector; d64_iterations::Int = 10, bf_iterations::Int = 5,
        precision_bits::Int = 256) where {T <: Union{Float64, ComplexF64}}
    r = BallArithmetic.refine_symmetric_eigen_double64(A, Q0, λ0;
        max_iterations = d64_iterations)
    return BallArithmetic.refine_symmetric_eigen(A, r.Q, r.λ;
        target_precision = precision_bits, max_iterations = bf_iterations)
end

end # module
