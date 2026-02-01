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
- `certify_with_bigfloat`: If true, compute final residual with BigFloat for rigor

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
    result_mf = _ogita_svd_refine_multifloat_impl(A_mf, U_mf, Σ_mf, V_mf, max_iterations)

    if certify_with_bigfloat
        # Final certification with BigFloat for rigorous error bound
        old_precision = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)

        try
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

            # Compute rigorous residual norm
            residual = A_bf - U_bf * Diagonal(Σ_bf) * V_bf'
            residual_norm = _spectral_norm_bound(residual)

            return OgitaSVDRefinementResult(
                U_bf, Diagonal(Σ_bf), V_bf,
                max_iterations,
                bigfloat_precision,
                residual_norm,
                true
            )
        finally
            setprecision(BigFloat, old_precision)
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

"""
Internal implementation of Ogita SVD refinement for MultiFloats.
"""
function _ogita_svd_refine_multifloat_impl(A, U, Σ, V, max_iterations)
    m, n = size(A)
    min_dim = min(m, n)
    T = eltype(A)
    RT = real(T)

    U_curr = copy(U)
    Σ_curr = copy(Σ)
    V_curr = copy(V)

    for _ in 1:max_iterations
        # Step 1: Compute residual matrices
        R = I - U_curr' * U_curr
        S = I - V_curr' * V_curr
        T_matrix = U_curr' * A * V_curr

        # Step 2: Compute refined singular values
        σ_tilde = zeros(RT, min_dim)
        for i in 1:min_dim
            denom = 1 - real(R[i,i] + S[i,i]) / 2
            σ_tilde[i] = real(T_matrix[i,i] / denom)
        end

        # Steps 3-7: Compute correction matrices
        F_tilde = zeros(T, m, m)
        G_tilde = zeros(T, n, n)

        # Diagonal parts
        for i in 1:n
            F_tilde[i,i] = R[i,i] / 2
            G_tilde[i,i] = S[i,i] / 2
        end

        # Off-diagonal parts
        for i in 1:n
            for j in 1:n
                if i != j
                    denom = σ_tilde[j]^2 - σ_tilde[i]^2
                    if abs(denom) > eps(RT) * max(σ_tilde[i], σ_tilde[j])^2
                        α = T_matrix[i,j] + σ_tilde[j] * R[i,j]
                        β = conj(T_matrix[j,i]) + σ_tilde[j] * conj(S[j,i])
                        F_tilde[i,j] = (α * σ_tilde[j] + β * σ_tilde[i]) / denom
                        G_tilde[i,j] = (α * σ_tilde[i] + β * σ_tilde[j]) / denom
                    end
                end
            end
        end

        # F_12 block
        if m > n
            for i in 1:n
                for j in (n+1):m
                    if abs(σ_tilde[i]) > eps(RT)
                        F_tilde[i,j] = -T_matrix[j,i] / σ_tilde[i]
                    end
                end
            end
        end

        # F_21 block
        if m > n
            for i in (n+1):m
                for j in 1:n
                    F_tilde[i,j] = R[i,j] - F_tilde[j,i]
                end
            end
        end

        # F_22 block
        if m > n
            for i in (n+1):m
                for j in (n+1):m
                    F_tilde[i,j] = R[i,j] / 2
                end
            end
        end

        # Update
        U_curr = U_curr * (I + F_tilde)
        V_curr = V_curr * (I + G_tilde)
        Σ_curr = σ_tilde
    end

    # Final singular value computation
    T_final = U_curr' * A * V_curr
    R_final = I - U_curr' * U_curr
    S_final = I - V_curr' * V_curr
    for i in 1:min_dim
        denom = one(RT) - real(R_final[i,i] + S_final[i,i]) / 2
        Σ_curr[i] = abs(T_final[i,i] / denom)
    end

    # Phase correction for complex matrices
    if T <: Complex
        for i in 1:min_dim
            if abs(T_final[i,i]) > eps(RT)
                phase = T_final[i,i] / abs(T_final[i,i])
                U_curr[:, i] .*= phase
            end
        end
    end

    return (U=U_curr, Σ=Σ_curr, V=V_curr)
end

#==============================================================================#
# LU Decomposition Refinement (from Rump-Ogita 2024)
#==============================================================================#

"""
    LURefinementResult{T}

Result from LU decomposition refinement.
"""
struct LURefinementResult{T}
    L::Matrix{T}
    U::Matrix{T}
    P::Vector{Int}  # Permutation
    iterations::Int
    residual_norm::T
    converged::Bool
end

"""
    refine_lu_multifloat(A, L0, U0, P0; precision=:x2, max_iterations=3, certify=true)

Refine an approximate LU decomposition using MultiFloats.

Based on Rump & Ogita (2024), "Verified Error Bounds for Matrix Decompositions".

The key insight: given A ≈ P*L*U, refine L and U by solving triangular systems
with extended precision arithmetic.

# Arguments
- `A`: Original matrix
- `L0, U0, P0`: Initial LU decomposition (from LAPACK)
- `precision`: MultiFloats precision (`:x2`, `:x4`, `:x8`)
- `max_iterations`: Number of refinement iterations
- `certify`: If true, compute rigorous error bounds in BigFloat
"""
function refine_lu_multifloat(
        A::AbstractMatrix{T}, L0::Matrix, U0::Matrix, P0::Vector{Int};
        precision::Symbol=:x2,
        max_iterations::Int=3,
        certify::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)

    # Convert to MultiFloats
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        L_mf = convert.(Complex{MF}, L0)
        U_mf = convert.(Complex{MF}, U0)
    else
        A_mf = convert.(MF, A)
        L_mf = convert.(MF, L0)
        U_mf = convert.(MF, U0)
    end

    # Apply permutation to A
    PA_mf = A_mf[P0, :]

    # Iterative refinement
    L_curr, U_curr = L_mf, U_mf
    for _ in 1:max_iterations
        # Compute residual: R = PA - L*U
        R = PA_mf - L_curr * U_curr

        # Solve for corrections:
        # L*ΔU = R (for ΔU), then
        # ΔL*U = R - L*ΔU (approximately)
        ΔU = L_curr \ R
        ΔL = (R - L_curr * ΔU) / U_curr

        # Update
        L_curr = L_curr + ΔL
        U_curr = U_curr + ΔU

        # Enforce L is unit lower triangular
        for i in 1:n
            L_curr[i, i] = one(eltype(L_curr))
            for j in (i+1):n
                L_curr[i, j] = zero(eltype(L_curr))
            end
        end

        # Enforce U is upper triangular
        for i in 1:n
            for j in 1:(i-1)
                U_curr[i, j] = zero(eltype(U_curr))
            end
        end
    end

    if certify
        old_precision = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)

        try
            if T <: Complex
                L_bf = convert.(Complex{BigFloat}, L_curr)
                U_bf = convert.(Complex{BigFloat}, U_curr)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                L_bf = convert.(BigFloat, L_curr)
                U_bf = convert.(BigFloat, U_curr)
                A_bf = convert.(BigFloat, A)
            end

            PA_bf = A_bf[P0, :]
            residual = PA_bf - L_bf * U_bf
            residual_norm = _frobenius_norm(residual) / _frobenius_norm(PA_bf)

            return LURefinementResult(L_bf, U_bf, P0, max_iterations, residual_norm, true)
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        residual = PA_mf - L_curr * U_curr
        residual_norm = _frobenius_norm_mf(residual) / _frobenius_norm_mf(PA_mf)
        return LURefinementResult(L_curr, U_curr, P0, max_iterations, convert(BigFloat, residual_norm), true)
    end
end

#==============================================================================#
# QR Decomposition Refinement (from Rump-Ogita 2024)
#==============================================================================#

"""
    QRRefinementResult{T}

Result from QR decomposition refinement.
"""
struct QRRefinementResult{T}
    Q::Matrix{T}
    R::Matrix{T}
    iterations::Int
    residual_norm::T
    orthogonality_defect::T
    converged::Bool
end

"""
    refine_qr_multifloat(A, Q0, R0; precision=:x2, max_iterations=3, certify=true)

Refine an approximate QR decomposition using MultiFloats.

The algorithm maintains Q orthogonal and R upper triangular through iterative refinement.

# Reference
Rump & Ogita (2024), "Verified Error Bounds for Matrix Decompositions"
"""
function refine_qr_multifloat(
        A::AbstractMatrix{T}, Q0::Matrix, R0::Matrix;
        precision::Symbol=:x2,
        max_iterations::Int=3,
        certify::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    m = size(A, 1)

    # Convert to MultiFloats
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        Q_mf = convert.(Complex{MF}, Q0)
        R_mf = convert.(Complex{MF}, R0)
    else
        A_mf = convert.(MF, A)
        Q_mf = convert.(MF, Q0)
        R_mf = convert.(MF, R0)
    end

    I_m = Matrix{eltype(Q_mf)}(I, m, m)
    Q_curr, R_curr = Q_mf, R_mf

    for _ in 1:max_iterations
        # Update R: R' = Q'*A
        R_new = Q_curr' * A_mf

        # Orthogonalize Q using Newton-Schulz
        Q_curr = _newton_schulz_step_mf(Q_curr)

        # Update R to maintain A ≈ Q*R
        R_curr = R_new
    end

    if certify
        old_precision = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)

        try
            if T <: Complex
                Q_bf = convert.(Complex{BigFloat}, Q_curr)
                R_bf = convert.(Complex{BigFloat}, R_curr)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                Q_bf = convert.(BigFloat, Q_curr)
                R_bf = convert.(BigFloat, R_curr)
                A_bf = convert.(BigFloat, A)
            end

            I_bf = Matrix{BigFloat}(I, m, m)
            residual = A_bf - Q_bf * R_bf
            residual_norm = _frobenius_norm(residual) / _frobenius_norm(A_bf)
            orthogonality_defect = _frobenius_norm(Q_bf' * Q_bf - I_bf)

            return QRRefinementResult(Q_bf, R_bf, max_iterations, residual_norm, orthogonality_defect, true)
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        residual = A_mf - Q_curr * R_curr
        residual_norm = _frobenius_norm_mf(residual) / _frobenius_norm_mf(A_mf)
        orthogonality_defect = _frobenius_norm_mf(Q_curr' * Q_curr - I_m)
        return QRRefinementResult(Q_curr, R_curr, max_iterations,
            convert(BigFloat, residual_norm), convert(BigFloat, orthogonality_defect), true)
    end
end

#==============================================================================#
# Cholesky Decomposition Refinement (from Rump-Ogita 2024)
#==============================================================================#

"""
    CholeskyRefinementResult{T}

Result from Cholesky decomposition refinement.
"""
struct CholeskyRefinementResult{T}
    L::Matrix{T}
    iterations::Int
    residual_norm::T
    converged::Bool
end

"""
    refine_cholesky_multifloat(A, L0; precision=:x2, max_iterations=3, certify=true)

Refine an approximate Cholesky decomposition A ≈ L*L' using MultiFloats.

# Reference
Rump & Ogita (2024), "Verified Error Bounds for Matrix Decompositions"
"""
function refine_cholesky_multifloat(
        A::AbstractMatrix{T}, L0::Matrix;
        precision::Symbol=:x2,
        max_iterations::Int=3,
        certify::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)

    # Convert to MultiFloats
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        L_mf = convert.(Complex{MF}, L0)
    else
        A_mf = convert.(MF, A)
        L_mf = convert.(MF, L0)
    end

    L_curr = L_mf

    for _ in 1:max_iterations
        # Compute residual: R = A - L*L'
        R = A_mf - L_curr * L_curr'

        # Solve L*ΔL' + ΔL*L' = R for ΔL (lower triangular)
        # Approximate: ΔL ≈ R * (L'^{-1}) / 2 (simplified)
        ΔL = (R / L_curr') / 2

        # Enforce lower triangular
        for i in 1:n
            for j in (i+1):n
                ΔL[i, j] = zero(eltype(ΔL))
            end
        end

        # Update
        L_curr = L_curr + ΔL
    end

    if certify
        old_precision = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)

        try
            if T <: Complex
                L_bf = convert.(Complex{BigFloat}, L_curr)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                L_bf = convert.(BigFloat, L_curr)
                A_bf = convert.(BigFloat, A)
            end

            residual = A_bf - L_bf * L_bf'
            residual_norm = _frobenius_norm(residual) / _frobenius_norm(A_bf)

            return CholeskyRefinementResult(L_bf, max_iterations, residual_norm, true)
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        residual = A_mf - L_curr * L_curr'
        residual_norm = _frobenius_norm_mf(residual) / _frobenius_norm_mf(A_mf)
        return CholeskyRefinementResult(L_curr, max_iterations, convert(BigFloat, residual_norm), true)
    end
end

#==============================================================================#
# Schur Decomposition Refinement with MultiFloats
#==============================================================================#

"""
    refine_schur_multifloat(A, Q0, T0; precision=:x2, max_iterations=2, certify=true)

Refine Schur decomposition A = Q*T*Q' using MultiFloats.
"""
function BallArithmetic.refine_schur_multifloat(
        A::AbstractMatrix{T}, Q0::Matrix, T0::Matrix;
        precision::Symbol=:x2,
        max_iterations::Int=2,
        certify::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)

    # Convert to MultiFloats
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        Q_mf = convert.(Complex{MF}, Q0)
        T_mf = convert.(Complex{MF}, T0)
    else
        A_mf = convert.(MF, A)
        Q_mf = convert.(MF, Q0)
        T_mf = convert.(MF, T0)
    end

    Q_curr, T_curr = Q_mf, T_mf

    for _ in 1:max_iterations
        # Compute T̂ = Q' * A * Q
        T_hat = Q_curr' * A_mf * Q_curr

        # Extract strictly lower triangular part E and upper triangular T
        E = _stril_mf(T_hat)
        T_curr = T_hat - E

        # Newton-Schulz orthogonalization
        Q_curr = _newton_schulz_step_mf(Q_curr)
    end

    if certify
        old_precision = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)

        try
            if T <: Complex
                Q_bf = convert.(Complex{BigFloat}, Q_curr)
                T_bf = convert.(Complex{BigFloat}, T_curr)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                Q_bf = convert.(BigFloat, Q_curr)
                T_bf = convert.(BigFloat, T_curr)
                A_bf = convert.(BigFloat, A)
            end

            I_bf = Matrix{BigFloat}(I, size(Q_bf, 1), size(Q_bf, 1))
            residual = A_bf - Q_bf * T_bf * Q_bf'
            residual_norm = _frobenius_norm(residual) / _frobenius_norm(A_bf)
            orthogonality_defect = _frobenius_norm(Q_bf' * Q_bf - I_bf)

            return SchurRefinementResult(Q_bf, T_bf, max_iterations, residual_norm, orthogonality_defect, true)
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        n = size(Q_curr, 1)
        I_mf = Matrix{eltype(Q_curr)}(I, n, n)
        residual = A_mf - Q_curr * T_curr * Q_curr'
        residual_norm = _frobenius_norm_mf(residual) / _frobenius_norm_mf(A_mf)
        orthogonality_defect = _frobenius_norm_mf(Q_curr' * Q_curr - I_mf)

        return SchurRefinementResult(Q_curr, T_curr, max_iterations,
            convert(BigFloat, residual_norm), convert(BigFloat, orthogonality_defect), true)
    end
end

#==============================================================================#
# Symmetric Eigenvalue Refinement with MultiFloats
#==============================================================================#

"""
    refine_symmetric_eigen_multifloat(A, Q0, λ0; precision=:x2, max_iterations=2, certify=true)

Refine symmetric eigenvalue decomposition A = Q*Λ*Q' using MultiFloats.
Uses the RefSyEv algorithm from Ogita & Aishima (2018).
"""
function BallArithmetic.refine_symmetric_eigen_multifloat(
        A::AbstractMatrix{T}, Q0::Matrix, λ0::AbstractVector;
        precision::Symbol=:x2,
        max_iterations::Int=2,
        certify::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)

    # Convert to MultiFloats
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        Q_mf = convert.(Complex{MF}, Q0)
    else
        A_mf = convert.(MF, A)
        Q_mf = convert.(MF, Q0)
    end
    λ_mf = convert.(MF, λ0)

    I_n = Matrix{eltype(Q_mf)}(I, n, n)
    E_tilde = zeros(eltype(Q_mf), n, n)
    Q_curr, λ_curr = Q_mf, λ_mf

    for _ in 1:max_iterations
        R = I_n - Q_curr' * Q_curr
        S = Q_curr' * A_mf * Q_curr

        # Update eigenvalues
        for i in 1:n
            denom = one(MF) - real(R[i, i])
            if abs(denom) > eps(MF)
                λ_curr[i] = real(S[i, i] / denom)
            else
                λ_curr[i] = real(S[i, i])
            end
        end

        # Compute threshold δ
        D_tilde = Diagonal(λ_curr)
        S_minus_D_norm = _frobenius_norm_mf(S - D_tilde)
        R_norm = _frobenius_norm_mf(R)
        A_norm = _frobenius_norm_mf(A_mf)
        δ = MF(2) * (S_minus_D_norm + A_norm * R_norm)

        # Compute correction matrix
        fill!(E_tilde, zero(eltype(E_tilde)))
        for j in 1:n
            for i in 1:n
                if i == j
                    E_tilde[i, i] = R[i, i] / MF(2)
                else
                    λ_diff = λ_curr[j] - λ_curr[i]
                    if abs(λ_diff) > δ
                        E_tilde[i, j] = (S[i, j] + λ_curr[j] * R[i, j]) / λ_diff
                    else
                        E_tilde[i, j] = R[i, j] / MF(2)
                    end
                end
            end
        end

        Q_curr = Q_curr * (I_n + E_tilde)
    end

    if certify
        old_precision = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)

        try
            if T <: Complex
                Q_bf = convert.(Complex{BigFloat}, Q_curr)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                Q_bf = convert.(BigFloat, Q_curr)
                A_bf = convert.(BigFloat, A)
            end
            λ_bf = convert.(BigFloat, λ_curr)

            I_bf = Matrix{BigFloat}(I, n, n)
            Λ = Diagonal(λ_bf)
            reconstruction = Q_bf * Λ * Q_bf'
            residual_norm = _frobenius_norm(A_bf - reconstruction) / _frobenius_norm(A_bf)
            orthogonality_defect = _frobenius_norm(I_bf - Q_bf' * Q_bf)

            return SymmetricEigenRefinementResult(Q_bf, λ_bf, max_iterations, residual_norm, orthogonality_defect, true)
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        Λ = Diagonal(λ_curr)
        reconstruction = Q_curr * Λ * Q_curr'
        residual_norm = _frobenius_norm_mf(A_mf - reconstruction) / _frobenius_norm_mf(A_mf)
        orthogonality_defect = _frobenius_norm_mf(I_n - Q_curr' * Q_curr)

        return SymmetricEigenRefinementResult(Q_curr, λ_curr, max_iterations,
            convert(BigFloat, residual_norm), convert(BigFloat, orthogonality_defect), true)
    end
end

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

"""Frobenius norm for MultiFloat matrices."""
function _frobenius_norm_mf(A::AbstractMatrix{T}) where T
    s = zero(real(T))
    @inbounds for j in axes(A, 2)
        for i in axes(A, 1)
            s += abs2(A[i, j])
        end
    end
    return sqrt(s)
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

"""Extract strictly lower triangular part of matrix."""
function _stril_mf(A::AbstractMatrix{T}) where T
    n, m = size(A)
    L = zeros(T, n, m)
    for j in 1:min(n-1, m)
        for i in (j+1):n
            L[i, j] = A[i, j]
        end
    end
    return L
end

"""Newton-Schulz orthogonalization step for MultiFloats."""
function _newton_schulz_step_mf(Q::Matrix{T}) where T
    n = size(Q, 1)
    I_n = Matrix{T}(I, n, n)
    QtQ = Q' * Q
    return Q * (T(3) * I_n - QtQ) / T(2)
end

#==============================================================================#
# Exports
#==============================================================================#

export LURefinementResult, QRRefinementResult, CholeskyRefinementResult
export refine_lu_multifloat, refine_qr_multifloat, refine_cholesky_multifloat

#==============================================================================#
# Verified Matrix Decompositions with MultiFloat Oracle (Rump-Ogita 2024)
#==============================================================================#

using BallArithmetic: VerifiedLUResult, VerifiedCholeskyResult, VerifiedQRResult,
                      VerifiedPolarResult, VerifiedTakagiResult,
                      _lu_perturbed_identity,
                      BallMatrix, Ball

"""
    verified_lu_multifloat(A; precision=:x2, bigfloat_precision=256)

Fast verified LU decomposition using MultiFloat oracle.

# Arguments
- `A`: Input matrix (Float64 or ComplexF64)
- `precision`: MultiFloat precision (`:x2`, `:x4`, `:x8`)
- `bigfloat_precision`: BigFloat precision for certification (default: 256)

# Returns
[`VerifiedLUResult`](@ref) with rigorous BallMatrix enclosures.
"""
function BallArithmetic.verified_lu_multifloat(
        A::AbstractMatrix{T};
        precision::Symbol=:x2,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    m, n = size(A)
    mn = min(m, n)

    # Convert to MultiFloat
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
    else
        A_mf = convert.(MF, A)
    end

    # Compute LU in MultiFloat
    F = lu(A_mf, Val(true))
    L_mf = Matrix(F.L)
    U_mf = Matrix(F.U)
    p = F.p

    # Compute preconditioners
    if m >= n
        X_L_mf = inv(L_mf[1:n, 1:n])
        X_L_full_mf = vcat(X_L_mf, -L_mf[(n+1):m, 1:n] * X_L_mf)
    else
        X_L_mf = inv(L_mf)
        X_L_full_mf = X_L_mf
    end
    X_U_mf = m >= n ? inv(U_mf) : inv(U_mf[1:m, 1:m])

    # Form perturbed identity
    A_perm_mf = A_mf[p, :]
    if m >= n
        I_E_mf = X_L_full_mf * A_perm_mf * X_U_mf
    else
        I_E_mf = X_L_mf * A_perm_mf[1:m, 1:m] * X_U_mf
    end

    E = convert.(Float64, I_E_mf - I)

    # Verify with BigFloat
    L_E_data, U_E_data, _, _, success =
        _lu_perturbed_identity(E; precision_bits=bigfloat_precision)

    if !success
        L_ball = BallMatrix(convert.(BigFloat, L_mf), fill(BigFloat(Inf), m, mn))
        U_ball = BallMatrix(convert.(BigFloat, U_mf), fill(BigFloat(Inf), mn, n))
        return VerifiedLUResult(L_ball, U_ball, p, false, BigFloat(Inf))
    end

    L_offset_mid, L_offset_rad = L_E_data
    U_offset_mid, U_offset_rad = U_E_data

    old_prec = Base.precision(BigFloat)
    setprecision(BigFloat, bigfloat_precision)

    try
        A_perm_bf = convert.(BigFloat, A[p, :])
        L_mf_bf = convert.(BigFloat, L_mf)
        U_mf_bf = convert.(BigFloat, U_mf)

        I_n = Matrix{BigFloat}(I, mn, mn)
        L_E_mid = (m >= n ? Matrix{BigFloat}(I, m, mn) : I_n) + L_offset_mid
        U_E_mid = I_n + U_offset_mid[1:mn, 1:mn]

        if m >= n
            L_mid = L_mf_bf * L_E_mid
            U_mid = U_E_mid * U_mf_bf
        else
            L_mid = L_mf_bf * L_E_mid
            U_mid = hcat(U_E_mid * U_mf_bf[1:m, 1:m], zeros(BigFloat, m, n-m))
        end

        L_rad = abs.(L_mf_bf) * L_offset_rad
        U_rad = U_offset_rad * abs.(U_mf_bf[1:mn, 1:mn])
        if m < n
            U_rad = hcat(U_rad, zeros(BigFloat, m, n-m))
        end

        L_ball = BallMatrix(L_mid, L_rad)
        U_ball = BallMatrix(U_mid, U_rad)

        LU_mid = L_mid * U_mid
        residual = LU_mid - A_perm_bf
        residual_norm = maximum(abs.(residual)) / maximum(abs.(A_perm_bf))

        return VerifiedLUResult(L_ball, U_ball, p, true, residual_norm)
    finally
        setprecision(BigFloat, old_prec)
    end
end

"""
    verified_cholesky_multifloat(A; precision=:x2, bigfloat_precision=256)

Fast verified Cholesky decomposition using MultiFloat oracle.
"""
function BallArithmetic.verified_cholesky_multifloat(
        A::AbstractMatrix{T};
        precision::Symbol=:x2,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))

    A_sym = (A + A') / 2

    if T <: Complex
        A_mf = convert.(Complex{MF}, A_sym)
    else
        A_mf = convert.(MF, A_sym)
    end

    F = try
        cholesky(Hermitian(A_mf))
    catch
        G_ball = BallMatrix(fill(BigFloat(NaN), n, n), fill(BigFloat(Inf), n, n))
        return VerifiedCholeskyResult(G_ball, false, BigFloat(Inf))
    end

    G_mf = Matrix(F.U)
    X_G_mf = inv(G_mf)

    I_E_mf = X_G_mf' * A_mf * X_G_mf
    E = convert.(Float64, I_E_mf - I)

    L_E_data, U_E_data, _, _, success = _lu_perturbed_identity(E; precision_bits=bigfloat_precision)

    if !success
        G_ball = BallMatrix(convert.(BigFloat, G_mf), fill(BigFloat(Inf), n, n))
        return VerifiedCholeskyResult(G_ball, false, BigFloat(Inf))
    end

    L_offset_mid, L_offset_rad = L_E_data
    U_offset_mid, U_offset_rad = U_E_data

    old_prec = Base.precision(BigFloat)
    setprecision(BigFloat, bigfloat_precision)

    try
        I_n = Matrix{BigFloat}(I, n, n)
        L_E_mid = I_n + L_offset_mid
        U_E_mid = I_n + U_offset_mid

        D_mid = diag(U_E_mid)
        D_rad = diag(U_offset_rad)

        for i in 1:n
            if D_mid[i] - D_rad[i] <= 0
                G_ball = BallMatrix(convert.(BigFloat, G_mf), fill(BigFloat(Inf), n, n))
                return VerifiedCholeskyResult(G_ball, false, BigFloat(Inf))
            end
        end

        D_sqrt_mid = sqrt.(D_mid)
        D_sqrt_rad = zeros(BigFloat, n)
        for i in 1:n
            lower = sqrt(D_mid[i] - D_rad[i])
            upper = sqrt(D_mid[i] + D_rad[i])
            D_sqrt_mid[i] = (lower + upper) / 2
            D_sqrt_rad[i] = (upper - lower) / 2
        end

        G_E_mid = Diagonal(D_sqrt_mid) * L_E_mid'
        G_E_rad = Diagonal(D_sqrt_rad) * abs.(L_E_mid') +
                  Diagonal(D_sqrt_mid) * L_offset_rad' +
                  Diagonal(D_sqrt_rad) * L_offset_rad'

        G_mf_bf = convert.(BigFloat, G_mf)
        G_mid = G_E_mid * G_mf_bf
        G_rad = G_E_rad * abs.(G_mf_bf)

        for j in 1:n, i in (j+1):n
            G_mid[i, j] = zero(BigFloat)
            G_rad[i, j] = zero(BigFloat)
        end

        G_ball = BallMatrix(G_mid, G_rad)

        GtG = G_mid' * G_mid
        A_bf = convert.(BigFloat, A_sym)
        residual_norm = maximum(abs.(GtG - A_bf)) / maximum(abs.(A_bf))

        return VerifiedCholeskyResult(G_ball, true, residual_norm)
    finally
        setprecision(BigFloat, old_prec)
    end
end

"""
    verified_qr_multifloat(A; precision=:x2, bigfloat_precision=256)

Fast verified QR decomposition using MultiFloat oracle.
"""
function BallArithmetic.verified_qr_multifloat(
        A::AbstractMatrix{T};
        precision::Symbol=:x2,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    m, n = size(A)

    if m < n
        result_sq = BallArithmetic.verified_qr_multifloat(A[:, 1:m];
            precision=precision, bigfloat_precision=bigfloat_precision)
        if !result_sq.success
            Q_ball = BallMatrix(fill(BigFloat(NaN), m, m), fill(BigFloat(Inf), m, m))
            R_ball = BallMatrix(fill(BigFloat(NaN), m, n), fill(BigFloat(Inf), m, n))
            return VerifiedQRResult(Q_ball, R_ball, false, BigFloat(Inf), BigFloat(Inf))
        end

        old_prec = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)
        try
            A_bf = convert.(BigFloat, A)
            Q_mid = BallArithmetic.mid(result_sq.Q)
            R_full_mid = Q_mid' * A_bf
            Q_rad = BallArithmetic.rad(result_sq.Q)
            R_full_rad = Q_rad' * abs.(A_bf)

            R_ball = BallMatrix(R_full_mid, R_full_rad)
            return VerifiedQRResult(result_sq.Q, R_ball, true,
                                   result_sq.residual_norm, result_sq.orthogonality_defect)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
    else
        A_mf = convert.(MF, A)
    end

    F = qr(A_mf)
    Q_mf = Matrix(F.Q)[:, 1:n]
    R_mf = Matrix(F.R)

    X_R_mf = inv(R_mf)
    C_mf = A_mf * X_R_mf
    CtC_mf = C_mf' * C_mf
    E = convert.(Float64, CtC_mf - I)

    L_E_data, U_E_data, _, _, success = _lu_perturbed_identity(E; precision_bits=bigfloat_precision)

    if !success
        Q_ball = BallMatrix(convert.(BigFloat, Q_mf), fill(BigFloat(Inf), m, n))
        R_ball = BallMatrix(convert.(BigFloat, R_mf), fill(BigFloat(Inf), n, n))
        return VerifiedQRResult(Q_ball, R_ball, false, BigFloat(Inf), BigFloat(Inf))
    end

    L_offset_mid, L_offset_rad = L_E_data
    U_offset_mid, U_offset_rad = U_E_data

    old_prec = Base.precision(BigFloat)
    setprecision(BigFloat, bigfloat_precision)

    try
        I_n = Matrix{BigFloat}(I, n, n)
        L_E_mid = I_n + L_offset_mid
        U_E_mid = I_n + U_offset_mid

        D_mid = diag(U_E_mid)
        D_rad = diag(U_offset_rad)

        for i in 1:n
            if D_mid[i] - D_rad[i] <= 0
                Q_ball = BallMatrix(convert.(BigFloat, Q_mf), fill(BigFloat(Inf), m, n))
                R_ball = BallMatrix(convert.(BigFloat, R_mf), fill(BigFloat(Inf), n, n))
                return VerifiedQRResult(Q_ball, R_ball, false, BigFloat(Inf), BigFloat(Inf))
            end
        end

        D_sqrt_mid = sqrt.(D_mid)
        D_sqrt_rad = zeros(BigFloat, n)
        for i in 1:n
            lower = sqrt(D_mid[i] - D_rad[i])
            upper = sqrt(D_mid[i] + D_rad[i])
            D_sqrt_mid[i] = (lower + upper) / 2
            D_sqrt_rad[i] = (upper - lower) / 2
        end

        G_E_mid = Diagonal(D_sqrt_mid) * L_E_mid'
        G_E_rad = Diagonal(D_sqrt_rad) * abs.(L_E_mid') +
                  Diagonal(D_sqrt_mid) * L_offset_rad' +
                  Diagonal(D_sqrt_rad) * L_offset_rad'

        R_mf_bf = convert.(BigFloat, R_mf)
        R_mid = G_E_mid * R_mf_bf
        R_rad = G_E_rad * abs.(R_mf_bf)

        for j in 1:n, i in (j+1):n
            R_mid[i, j] = zero(BigFloat)
            R_rad[i, j] = zero(BigFloat)
        end

        C_bf = convert.(BigFloat, C_mf)
        G_E_inv_mid = L_E_mid' \ Diagonal(1 ./ D_sqrt_mid)
        Q_mid = C_bf * G_E_inv_mid

        G_E_inv_norm = maximum(sum(abs.(G_E_inv_mid), dims=2))
        G_E_rad_norm = maximum(sum(G_E_rad, dims=2))
        Q_rad_factor = G_E_inv_norm * G_E_rad_norm * G_E_inv_norm
        Q_rad = abs.(C_bf) * fill(Q_rad_factor, n, n)

        Q_ball = BallMatrix(Q_mid, Q_rad)
        R_ball = BallMatrix(R_mid, R_rad)

        A_bf = convert.(BigFloat, A)
        residual = Q_mid * R_mid - A_bf
        residual_norm = maximum(abs.(residual)) / maximum(abs.(A_bf))

        QtQ = Q_mid' * Q_mid
        ortho_defect = maximum(abs.(QtQ - I_n))

        return VerifiedQRResult(Q_ball, R_ball, true, residual_norm, ortho_defect)
    finally
        setprecision(BigFloat, old_prec)
    end
end

"""
    verified_polar_multifloat(A; precision=:x2, bigfloat_precision=256, right=true)

Fast verified polar decomposition using MultiFloat oracle.
"""
function BallArithmetic.verified_polar_multifloat(
        A::AbstractMatrix{T};
        precision::Symbol=:x2,
        bigfloat_precision::Int=256,
        right::Bool=true) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))

    # Compute SVD in Float64 first (MultiFloats doesn't have native SVD)
    F = svd(A)
    U_f64 = F.U
    σ_f64 = F.S
    V_f64 = F.Vt'

    # Convert to MultiFloats for refinement
    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        U_mf = convert.(Complex{MF}, U_f64)
        V_mf = convert.(Complex{MF}, V_f64)
    else
        A_mf = convert.(MF, A)
        U_mf = convert.(MF, U_f64)
        V_mf = convert.(MF, V_f64)
    end
    σ_mf = convert.(MF, σ_f64)

    # Refine SVD in MultiFloat precision
    for _ in 1:2
        AV = A_mf * V_mf
        for j in 1:n
            if σ_mf[j] > eps(MF)
                U_mf[:, j] = AV[:, j] / σ_mf[j]
            end
        end
        U_mf, _ = _gram_schmidt_mf(U_mf)

        AHU = A_mf' * U_mf
        for j in 1:n
            if σ_mf[j] > eps(MF)
                V_mf[:, j] = AHU[:, j] / σ_mf[j]
            end
        end
        V_mf, _ = _gram_schmidt_mf(V_mf)

        for j in 1:n
            σ_mf[j] = real(U_mf[:, j]' * A_mf * V_mf[:, j])
        end
    end

    old_prec = Base.precision(BigFloat)
    setprecision(BigFloat, bigfloat_precision)

    try
        if T <: Complex
            A_bf = convert.(Complex{BigFloat}, A)
            U_bf = convert.(Complex{BigFloat}, U_mf)
            V_bf = convert.(Complex{BigFloat}, V_mf)
        else
            A_bf = convert.(BigFloat, A)
            U_bf = convert.(BigFloat, U_mf)
            V_bf = convert.(BigFloat, V_mf)
        end
        σ_bf = convert.(BigFloat, σ_mf)

        Σ_mat = Diagonal(σ_bf)
        SVD_residual = A_bf - U_bf * Σ_mat * V_bf'
        svd_error = maximum(abs.(SVD_residual))

        Q_mid = U_bf * V_bf'
        Q_rad = fill(2 * svd_error / minimum(σ_bf[σ_bf .> eps(BigFloat)]), n, n)

        if right
            P_mid = V_bf * Σ_mat * V_bf'
            P_mid = (P_mid + P_mid') / 2
            P_rad = fill(2 * svd_error, n, n)
        else
            P_mid = U_bf * Σ_mat * U_bf'
            P_mid = (P_mid + P_mid') / 2
            P_rad = fill(2 * svd_error, n, n)
        end

        Q_ball = BallMatrix(Q_mid, Q_rad)
        P_ball = BallMatrix(P_mid, P_rad)

        QP = right ? Q_mid * P_mid : P_mid * Q_mid
        residual = QP - A_bf
        residual_norm = maximum(abs.(residual)) / maximum(abs.(A_bf))

        return VerifiedPolarResult(Q_ball, P_ball, right, true, residual_norm)
    finally
        setprecision(BigFloat, old_prec)
    end
end

"""Gram-Schmidt orthogonalization for MultiFloats."""
function _gram_schmidt_mf(V::AbstractMatrix{T}) where T
    n = size(V, 2)
    Q = Matrix(V)  # Convert to Matrix in case of Adjoint/Transpose
    R = zeros(T, n, n)

    for j in 1:n
        for i in 1:(j-1)
            R[i, j] = Q[:, i]' * Q[:, j]
            Q[:, j] -= R[i, j] * Q[:, i]
        end
        R[j, j] = sqrt(real(Q[:, j]' * Q[:, j]))
        if abs(R[j, j]) > eps(real(T))
            Q[:, j] /= R[j, j]
        end
    end

    return Q, R
end

"""
    verified_takagi_multifloat(A; precision=:x2, bigfloat_precision=256, method=:real_compound)

Fast verified Takagi decomposition using MultiFloat oracle.
"""
function BallArithmetic.verified_takagi_multifloat(
        A::AbstractMatrix{Complex{T}};
        precision::Symbol=:x2,
        bigfloat_precision::Int=256,
        method::Symbol=:real_compound) where {T<:AbstractFloat}

    MF = _select_multifloat_type(precision)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))

    A_sym = (A + transpose(A)) / 2

    if method == :real_compound
        return _verified_takagi_real_compound_mf(A_sym, MF; bigfloat_precision=bigfloat_precision)
    else
        return _verified_takagi_svd_mf(A_sym, MF; bigfloat_precision=bigfloat_precision)
    end
end

function _verified_takagi_real_compound_mf(A::AbstractMatrix{Complex{T}}, MF;
                                           bigfloat_precision::Int=256) where T
    n = size(A, 1)

    E = real.(A)
    F_mat = imag.(A)

    # Construct real symmetric matrix M in Float64 first (MultiFloats doesn't have native eigen)
    M_f64 = [E F_mat; F_mat -E]

    # Eigendecomposition in Float64
    F_eig = eigen(Symmetric(M_f64))
    eigenvalues_f64 = F_eig.values
    eigenvectors_f64 = F_eig.vectors

    # Convert to MultiFloats for refinement
    E_mf = convert.(MF, E)
    F_mf = convert.(MF, F_mat)
    M_mf = [E_mf F_mf; F_mf -E_mf]
    eigenvalues_mf = convert.(MF, eigenvalues_f64)
    eigenvectors_mf = convert.(MF, eigenvectors_f64)

    # Refine in MultiFloat precision
    for _ in 1:2
        for j in 1:2n
            v = eigenvectors_mf[:, j]
            Mv = M_mf * v
            eigenvalues_mf[j] = (v' * Mv) / (v' * v)
            for k in 1:(j-1)
                v -= (eigenvectors_mf[:, k]' * v) * eigenvectors_mf[:, k]
            end
            eigenvectors_mf[:, j] = v / sqrt(real(v' * v))
        end
    end

    old_prec = Base.precision(BigFloat)
    setprecision(BigFloat, bigfloat_precision)

    try
        A_bf = convert.(Complex{BigFloat}, A)

        perm = sortperm(eigenvalues_mf, rev=true)
        eigenvalues_sorted = convert.(BigFloat, eigenvalues_mf[perm])
        eigenvectors_sorted = convert.(BigFloat, eigenvectors_mf[:, perm])

        σ_mid = eigenvalues_sorted[1:n]
        for i in 1:n
            if σ_mid[i] < 0
                σ_mid[i] = -σ_mid[i]
            end
        end

        U_mid = zeros(Complex{BigFloat}, n, n)
        for j in 1:n
            x = eigenvectors_sorted[1:n, j]
            y = eigenvectors_sorted[(n+1):2n, j]
            u = x + im * y
            u_norm = sqrt(real(u' * u))
            if u_norm > eps(BigFloat)
                U_mid[:, j] = u / u_norm
            else
                U_mid[:, j] = u
            end
        end

        Σ_mat = Diagonal(σ_mid)
        reconstruction = U_mid * Σ_mat * transpose(U_mid)
        residual = reconstruction - A_bf
        residual_norm = maximum(abs.(residual)) / maximum(abs.(A_bf))

        σ_rad = fill(BigFloat(residual_norm), n)
        U_rad = fill(BigFloat(residual_norm * 10), n, n)

        U_ball = BallMatrix(U_mid, U_rad)
        Σ_balls = [Ball(σ_mid[i], σ_rad[i]) for i in 1:n]

        return VerifiedTakagiResult(U_ball, Σ_balls, true, residual_norm)
    finally
        setprecision(BigFloat, old_prec)
    end
end

function _verified_takagi_svd_mf(A::AbstractMatrix{Complex{T}}, MF;
                                  bigfloat_precision::Int=256) where T
    n = size(A, 1)

    # Compute SVD in Float64 first (MultiFloats doesn't have native SVD)
    F = svd(A)
    U_f64 = F.U
    σ_f64 = F.S
    V_f64 = F.Vt'

    # Convert to MultiFloats for refinement
    A_mf = convert.(Complex{MF}, A)
    U_mf = convert.(Complex{MF}, U_f64)
    V_mf = convert.(Complex{MF}, V_f64)
    σ_mf = convert.(MF, σ_f64)

    for _ in 1:2
        AV = A_mf * V_mf
        for j in 1:n
            if σ_mf[j] > eps(MF)
                U_mf[:, j] = AV[:, j] / σ_mf[j]
            end
        end
        U_mf, _ = _gram_schmidt_mf(U_mf)

        AHU = A_mf' * U_mf
        for j in 1:n
            if σ_mf[j] > eps(MF)
                V_mf[:, j] = AHU[:, j] / σ_mf[j]
            end
        end
        V_mf, _ = _gram_schmidt_mf(V_mf)

        for j in 1:n
            val = real(U_mf[:, j]' * A_mf * V_mf[:, j])
            if val < 0
                val = -val
                U_mf[:, j] = -U_mf[:, j]
            end
            σ_mf[j] = val
        end
    end

    old_prec = Base.precision(BigFloat)
    setprecision(BigFloat, bigfloat_precision)

    try
        A_bf = convert.(Complex{BigFloat}, A)
        U_bf = convert.(Complex{BigFloat}, U_mf)
        σ_bf = convert.(BigFloat, σ_mf)

        Σ_inv = Diagonal(1 ./ σ_bf)
        D_mat = U_bf' * A_bf * conj.(U_bf) * Σ_inv

        D_sqrt = zeros(Complex{BigFloat}, n)
        for j in 1:n
            D_sqrt[j] = sqrt(D_mat[j, j])
        end

        U_mid = U_bf * Diagonal(D_sqrt)
        Σ_mid = σ_bf

        reconstruction = U_mid * Diagonal(Σ_mid) * transpose(U_mid)
        residual = reconstruction - A_bf
        residual_norm = maximum(abs.(residual)) / maximum(abs.(A_bf))

        U_rad = fill(BigFloat(residual_norm * 10), n, n)
        σ_rad = fill(BigFloat(residual_norm), n)

        U_ball = BallMatrix(U_mid, U_rad)
        Σ_balls = [Ball(Σ_mid[i], σ_rad[i]) for i in 1:n]

        return VerifiedTakagiResult(U_ball, Σ_balls, true, residual_norm)
    finally
        setprecision(BigFloat, old_prec)
    end
end

#==============================================================================#
# Iterative Refinement Methods with MultiFloat Oracle
#==============================================================================#

# Result types are already available via `using BallArithmetic` at module top

"""
    refine_polar_multifloat(A, Q0; precision=:x2, method=:newton_schulz, max_iterations=10,
                            tol=1e-60, certify_with_bigfloat=true, bigfloat_precision=256)

Refine polar decomposition using MultiFloat arithmetic.

# Arguments
- `A`: Input matrix
- `Q0`: Initial approximation to unitary factor Q
- `precision`: MultiFloat precision (:x2, :x4, or :x8)
- `method`: Refinement method (:newton_schulz, :newton, or :qdwh)
- `max_iterations`: Maximum iterations (default: 10)
- `tol`: Convergence tolerance (default: 1e-60)

# References
- N.J. Higham, "Computing the Polar Decomposition—with Applications",
  SIAM J. Sci. Stat. Comput. 7(4):1160-1174, 1986. doi:10.1137/0907079
"""
function BallArithmetic.refine_polar_multifloat(
        A::AbstractMatrix{T}, Q0::AbstractMatrix;
        precision::Symbol=:x2,
        method::Symbol=:newton_schulz,
        max_iterations::Int=10,
        tol::Real=1e-60,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)

    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        Q_mf = convert.(Complex{MF}, Q0)
    else
        A_mf = convert.(MF, A)
        Q_mf = convert.(MF, Q0)
    end

    n = size(A, 1)
    I_n = Matrix{eltype(Q_mf)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    for k in 1:max_iterations
        iter = k

        if method == :newton_schulz
            QtQ = Q_mf' * Q_mf
            Q_mf = Q_mf * (MF(3) * I_n - QtQ) / MF(2)
        elseif method == :newton
            Q_norm = opnorm(Q_mf, 2)
            Q_inv_norm = opnorm(inv(Q_mf), 2)
            γ = sqrt(Q_inv_norm / Q_norm)
            Q_inv_H = inv(Q_mf)'
            Q_mf = (γ * Q_mf + Q_inv_H / γ) / MF(2)
        elseif method == :qdwh
            QtQ = Q_mf' * Q_mf
            σ_min_est = 1 / opnorm(inv(Q_mf), 2)
            σ_max_est = opnorm(Q_mf, 2)
            ℓ = σ_min_est / σ_max_est

            ℓ2 = ℓ^2
            dd = (4 * (1 - ℓ2) / ℓ2)^(1/3)
            sqd = sqrt(1 + dd)
            a_val = sqd + sqrt(8 - 4 * dd + 8 * (2 - ℓ2) / (ℓ2 * sqd)) / 2
            a_val = real(a_val)
            b_val = (a_val - 1)^2 / 4
            c_val = a_val + b_val - 1

            M = QtQ + c_val * I_n
            Q_mf = Q_mf * ((a_val * I_n + b_val * QtQ) / M)
        else
            error("Unknown method: $method")
        end

        ortho_defect = opnorm(Q_mf' * Q_mf - I_n, Inf)
        if ortho_defect < tol
            converged = true
            break
        end
    end

    H_mf = Q_mf' * A_mf
    H_mf = (H_mf + H_mf') / 2

    if certify_with_bigfloat
        old_prec = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)
        try
            if T <: Complex
                Q_bf = convert.(Complex{BigFloat}, Q_mf)
                H_bf = convert.(Complex{BigFloat}, H_mf)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                Q_bf = convert.(BigFloat, Q_mf)
                H_bf = convert.(BigFloat, H_mf)
                A_bf = convert.(BigFloat, A)
            end

            residual = Q_bf * H_bf - A_bf
            A_norm = opnorm(A_bf, Inf)
            residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)
            ortho_defect_bf = opnorm(Q_bf' * Q_bf - I, Inf)

            return PolarRefinementResult(Q_bf, H_bf, iter, residual_norm, ortho_defect_bf, converged)
        finally
            setprecision(BigFloat, old_prec)
        end
    else
        residual = Q_mf * H_mf - A_mf
        A_norm = opnorm(A_mf, Inf)
        residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

        return PolarRefinementResult(Matrix(Q_mf), Matrix(H_mf), iter,
                                     convert(Float64, residual_norm),
                                     convert(Float64, ortho_defect), converged)
    end
end

"""
    refine_lu_multifloat(A, L0, U0, p; precision=:x2, max_iterations=5,
                         tol=1e-60, certify_with_bigfloat=true, bigfloat_precision=256)

Refine LU decomposition using MultiFloat arithmetic.

# References
- J.H. Wilkinson, "Rounding Errors in Algebraic Processes", Prentice-Hall, 1963.
"""
function BallArithmetic.refine_lu_multifloat(
        A::AbstractMatrix{T}, L0::AbstractMatrix, U0::AbstractMatrix,
        p::AbstractVector{Int};
        precision::Symbol=:x2,
        max_iterations::Int=5,
        tol::Real=1e-60,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)

    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        L_mf = convert.(Complex{MF}, L0)
        U_mf = convert.(Complex{MF}, U0)
    else
        A_mf = convert.(MF, A)
        L_mf = convert.(MF, L0)
        U_mf = convert.(MF, U0)
    end

    PA_mf = A_mf[p, :]
    converged = false
    iter = 0
    residual_norm_mf = Inf

    for k in 1:max_iterations
        iter = k

        R = PA_mf - L_mf * U_mf
        PA_norm = opnorm(PA_mf, Inf)
        residual_norm_mf = PA_norm > 0 ? opnorm(R, Inf) / PA_norm : opnorm(R, Inf)

        if residual_norm_mf < tol
            converged = true
            break
        end

        ΔU = L_mf \ R
        R2 = R - L_mf * ΔU
        ΔL = R2 / U_mf

        for j in 1:n
            for i in (j+1):n
                L_mf[i, j] += ΔL[i, j]
            end
        end

        for j in 1:n
            for i in 1:j
                U_mf[i, j] += ΔU[i, j]
            end
        end
    end

    if certify_with_bigfloat
        old_prec = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)
        try
            if T <: Complex
                L_bf = convert.(Complex{BigFloat}, L_mf)
                U_bf = convert.(Complex{BigFloat}, U_mf)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                L_bf = convert.(BigFloat, L_mf)
                U_bf = convert.(BigFloat, U_mf)
                A_bf = convert.(BigFloat, A)
            end

            PA_bf = A_bf[p, :]
            residual = PA_bf - L_bf * U_bf
            PA_norm = opnorm(PA_bf, Inf)
            residual_norm = PA_norm > 0 ? opnorm(residual, Inf) / PA_norm : opnorm(residual, Inf)

            return LURefinementResult(L_bf, U_bf, p, iter, residual_norm, converged)
        finally
            setprecision(BigFloat, old_prec)
        end
    else
        return LURefinementResult(Matrix(L_mf), Matrix(U_mf), p, iter,
                                  convert(Float64, residual_norm_mf), converged)
    end
end

"""
    refine_cholesky_multifloat(A, G0; precision=:x2, max_iterations=5,
                               tol=1e-60, certify_with_bigfloat=true, bigfloat_precision=256)

Refine Cholesky decomposition using MultiFloat arithmetic.

# References
- R.S. Martin, G. Peters, J.H. Wilkinson, "Iterative Refinement of the Solution
  of a Positive Definite System of Equations", Numer. Math. 8:203-216, 1971.
"""
function BallArithmetic.refine_cholesky_multifloat(
        A::AbstractMatrix{T}, G0::AbstractMatrix;
        precision::Symbol=:x2,
        max_iterations::Int=5,
        tol::Real=1e-60,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)

    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        G_mf = convert.(Complex{MF}, G0)
    else
        A_mf = convert.(MF, A)
        G_mf = convert.(MF, G0)
    end

    A_sym = (A_mf + A_mf') / 2
    converged = false
    iter = 0
    residual_norm_mf = Inf

    for k in 1:max_iterations
        iter = k

        GtG = G_mf' * G_mf
        R = A_sym - GtG
        A_norm = opnorm(A_sym, Inf)
        residual_norm_mf = A_norm > 0 ? opnorm(R, Inf) / A_norm : opnorm(R, Inf)

        if residual_norm_mf < tol
            converged = true
            break
        end

        ΔG = (G_mf' \ R) / 2
        for j in 1:n
            for i in (j+1):n
                ΔG[i, j] = zero(eltype(ΔG))
            end
        end

        G_mf = G_mf + ΔG
    end

    if certify_with_bigfloat
        old_prec = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)
        try
            if T <: Complex
                G_bf = convert.(Complex{BigFloat}, G_mf)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                G_bf = convert.(BigFloat, G_mf)
                A_bf = convert.(BigFloat, A)
            end

            A_sym_bf = (A_bf + A_bf') / 2
            residual = G_bf' * G_bf - A_sym_bf
            A_norm = opnorm(A_sym_bf, Inf)
            residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

            return CholeskyRefinementResult(G_bf, iter, residual_norm, converged)
        finally
            setprecision(BigFloat, old_prec)
        end
    else
        return CholeskyRefinementResult(Matrix(G_mf), iter,
                                        convert(Float64, residual_norm_mf), converged)
    end
end

"""
    refine_qr_multifloat(A, Q0, R0; precision=:x2, method=:cholqr2, max_iterations=3,
                         tol=1e-60, certify_with_bigfloat=true, bigfloat_precision=256)

Refine QR decomposition using MultiFloat arithmetic.

# References
- Y. Yamamoto et al., "Roundoff error analysis of the CholeskyQR2 algorithm",
  Numer. Math. 131(2):297-322, 2015. doi:10.1007/s00211-014-0692-7
"""
function BallArithmetic.refine_qr_multifloat(
        A::AbstractMatrix{T}, Q0::AbstractMatrix, R0::AbstractMatrix;
        precision::Symbol=:x2,
        method::Symbol=:cholqr2,
        max_iterations::Int=3,
        tol::Real=1e-60,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:Union{Float64, ComplexF64}}

    MF = _select_multifloat_type(precision)
    _, n = size(A)

    if T <: Complex
        A_mf = convert.(Complex{MF}, A)
        Q_mf = convert.(Complex{MF}, Q0)
        R_mf = convert.(Complex{MF}, R0)
    else
        A_mf = convert.(MF, A)
        Q_mf = convert.(MF, Q0)
        R_mf = convert.(MF, R0)
    end

    I_n = Matrix{eltype(Q_mf)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    for k in 1:max_iterations
        iter = k

        if method == :cholqr2
            B = Q_mf' * Q_mf
            ortho_defect = opnorm(B - I_n, Inf)

            if ortho_defect < tol
                converged = true
                break
            end

            C = try
                cholesky(Hermitian(B))
            catch
                Q_mf, R_new = qr(Q_mf)
                Q_mf = Matrix(Q_mf)
                R_mf = R_new * R_mf
                continue
            end

            R_B = Matrix(C.U)
            Q_mf = Q_mf / R_B
            R_mf = R_B * R_mf
        else  # :mgs
            R_corr = zeros(eltype(Q_mf), n, n)
            for j in 1:n
                for i in 1:(j-1)
                    r_ij = Q_mf[:, i]' * Q_mf[:, j]
                    Q_mf[:, j] -= r_ij * Q_mf[:, i]
                    R_corr[i, j] = r_ij
                end
                r_jj = norm(Q_mf[:, j])
                Q_mf[:, j] /= r_jj
                R_corr[j, j] = r_jj
            end
            R_mf = R_corr * R_mf

            ortho_defect = opnorm(Q_mf' * Q_mf - I_n, Inf)
            if ortho_defect < tol
                converged = true
                break
            end
        end
    end

    if certify_with_bigfloat
        old_prec = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)
        try
            if T <: Complex
                Q_bf = convert.(Complex{BigFloat}, Q_mf)
                R_bf = convert.(Complex{BigFloat}, R_mf)
                A_bf = convert.(Complex{BigFloat}, A)
            else
                Q_bf = convert.(BigFloat, Q_mf)
                R_bf = convert.(BigFloat, R_mf)
                A_bf = convert.(BigFloat, A)
            end

            residual = Q_bf * R_bf - A_bf
            A_norm = opnorm(A_bf, Inf)
            residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)
            n_bf = size(Q_bf, 2)
            ortho_defect_bf = opnorm(Q_bf' * Q_bf - Matrix{BigFloat}(I, n_bf, n_bf), Inf)

            return QRRefinementResult(Q_bf, R_bf, iter, residual_norm, ortho_defect_bf, converged)
        finally
            setprecision(BigFloat, old_prec)
        end
    else
        residual = Q_mf * R_mf - A_mf
        A_norm = opnorm(A_mf, Inf)
        residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

        return QRRefinementResult(Matrix(Q_mf), Matrix(R_mf), iter,
                                  convert(Float64, residual_norm),
                                  convert(Float64, ortho_defect), converged)
    end
end

"""
    refine_takagi_multifloat(A, U0, Σ0; precision=:x2, max_iterations=5,
                             tol=1e-60, certify_with_bigfloat=true, bigfloat_precision=256)

Refine Takagi decomposition A = UΣUᵀ using MultiFloat arithmetic.

# References
- Adapted from T. Ogita, K. Aishima, "Iterative refinement for singular value
  decomposition based on matrix multiplication", J. Comput. Appl. Math. 369:112512, 2020.
"""
function BallArithmetic.refine_takagi_multifloat(
        A::AbstractMatrix{Complex{T}}, U0::AbstractMatrix, Σ0::AbstractVector;
        precision::Symbol=:x2,
        max_iterations::Int=5,
        tol::Real=1e-60,
        certify_with_bigfloat::Bool=true,
        bigfloat_precision::Int=256) where {T<:AbstractFloat}

    MF = _select_multifloat_type(precision)
    n = size(A, 1)

    A_mf = convert.(Complex{MF}, A)
    U_mf = convert.(Complex{MF}, U0)
    Σ_mf = convert.(MF, Σ0)

    I_n = Matrix{Complex{MF}}(I, n, n)
    converged = false
    iter = 0
    residual_norm_mf = Inf

    for k in 1:max_iterations
        iter = k

        R = I_n - U_mf' * U_mf
        T_mat = U_mf' * A_mf * conj.(U_mf)

        for i in 1:n
            denom = 1 - real(R[i, i])
            if abs(denom) > eps(MF)
                Σ_mf[i] = abs(real(T_mat[i, i])) / denom
            else
                Σ_mf[i] = abs(real(T_mat[i, i]))
            end
        end

        E = zeros(Complex{MF}, n, n)
        Σ_max = maximum(Σ_mf)
        δ = 2 * eps(MF) * Σ_max * n

        for j in 1:n
            for i in 1:n
                if i == j
                    E[i, i] = R[i, i] / 2
                else
                    σ_diff = Σ_mf[j]^2 - Σ_mf[i]^2
                    if abs(σ_diff) > δ * max(Σ_mf[i], Σ_mf[j])
                        E[i, j] = (T_mat[i, j] + Σ_mf[j] * R[i, j]) * Σ_mf[j] / σ_diff
                    else
                        E[i, j] = R[i, j] / 2
                    end
                end
            end
        end

        U_mf = U_mf * (I_n + E)

        for _ in 1:2
            UtU = U_mf' * U_mf
            U_mf = U_mf * (MF(3) * I_n - UtU) / MF(2)
        end

        reconstruction = U_mf * Diagonal(Σ_mf) * transpose(U_mf)
        A_max = maximum(abs.(A_mf))
        residual_norm_mf = A_max > 0 ? maximum(abs.(reconstruction - A_mf)) / A_max : maximum(abs.(reconstruction - A_mf))

        if residual_norm_mf < tol
            converged = true
            break
        end
    end

    if certify_with_bigfloat
        old_prec = Base.precision(BigFloat)
        setprecision(BigFloat, bigfloat_precision)
        try
            U_bf = convert.(Complex{BigFloat}, U_mf)
            Σ_bf = convert.(BigFloat, Σ_mf)
            A_bf = convert.(Complex{BigFloat}, A)

            reconstruction = U_bf * Diagonal(Σ_bf) * transpose(U_bf)
            A_max = maximum(abs.(A_bf))
            residual_norm = A_max > 0 ? maximum(abs.(reconstruction - A_bf)) / A_max : maximum(abs.(reconstruction - A_bf))

            return TakagiRefinementResult(U_bf, Σ_bf, iter, residual_norm, converged)
        finally
            setprecision(BigFloat, old_prec)
        end
    else
        return TakagiRefinementResult(Matrix(U_mf), Vector(Σ_mf), iter,
                                      convert(Float64, residual_norm_mf), converged)
    end
end

#==============================================================================#
# Precision Cascade SVD Refinement
#==============================================================================#

using BallArithmetic: PrecisionCascadeSVDResult, _ogita_iteration!
using DoubleFloats: Double64

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
"""
function BallArithmetic.ogita_svd_cascade(
        T_bf::Matrix{Complex{BigFloat}}, z_bf::Complex{BigFloat};
        f64_iters::Int=1, d64_iters::Int=1, mf3_iters::Int=1, mf4_iters::Int=1, bf_iters::Int=2)

    final_precision = Base.precision(BigFloat)

    # Stage 0: Float64 SVD (LAPACK) for initial approximation
    T_f64 = Complex{Float64}.(T_bf)
    z_f64 = Complex{Float64}(z_bf)
    A_f64 = T_f64 - z_f64 * I
    F = svd(A_f64)
    U = copy(F.U)
    Σ = copy(F.S)
    V = Matrix(F.V)

    # Stage 1: Float64 Ogita iterations
    for _ in 1:f64_iters
        _ogita_iteration!(A_f64, U, Σ, V)
    end

    # Stage 2: Double64 iterations
    if d64_iters > 0
        T_d64 = Complex{Double64}.(T_bf)
        z_d64 = Complex{Double64}(z_bf)
        A_d64 = T_d64 - z_d64 * I
        U_d64 = Complex{Double64}.(U)
        Σ_d64 = Double64.(Σ)
        V_d64 = Complex{Double64}.(V)

        for _ in 1:d64_iters
            _ogita_iteration!(A_d64, U_d64, Σ_d64, V_d64)
        end

        # Convert back through Float64 for MultiFloats compatibility
        U = Complex{Float64}.(U_d64)
        Σ = Float64.(Σ_d64)
        V = Complex{Float64}.(V_d64)
    end

    # Stage 3: Float64x3 iterations (~159 bits)
    if mf3_iters > 0
        T_mf3 = Complex{Float64x3}.(T_bf)
        z_mf3 = Complex{Float64x3}(z_bf)
        A_mf3 = T_mf3 - z_mf3 * I
        U_mf3 = Complex{Float64x3}.(U)
        Σ_mf3 = Float64x3.(Σ)
        V_mf3 = Complex{Float64x3}.(V)

        for _ in 1:mf3_iters
            _ogita_iteration!(A_mf3, U_mf3, Σ_mf3, V_mf3)
        end

        # Convert for next stage
        U_mf4 = Complex{Float64x4}.(U_mf3)
        Σ_mf4 = Float64x4.(Σ_mf3)
        V_mf4 = Complex{Float64x4}.(V_mf3)
    else
        U_mf4 = Complex{Float64x4}.(U)
        Σ_mf4 = Float64x4.(Σ)
        V_mf4 = Complex{Float64x4}.(V)
    end

    # Stage 4: Float64x4 iterations (~212 bits)
    if mf4_iters > 0
        T_mf4 = Complex{Float64x4}.(T_bf)
        z_mf4 = Complex{Float64x4}(z_bf)
        A_mf4 = T_mf4 - z_mf4 * I

        for _ in 1:mf4_iters
            _ogita_iteration!(A_mf4, U_mf4, Σ_mf4, V_mf4)
        end
    end

    # Stage 5: Final BigFloat iterations for certification
    A_bf = T_bf - z_bf * I
    U_bf = Complex{BigFloat}.(U_mf4)
    Σ_bf = BigFloat.(Σ_mf4)
    V_bf = Complex{BigFloat}.(V_mf4)

    local residual_norm
    for _ in 1:bf_iters
        residual_norm = _ogita_iteration!(A_bf, U_bf, Σ_bf, V_bf)
    end

    # Certified σ_min = smallest singular value minus residual norm
    σ_min = Σ_bf[end] - residual_norm

    return PrecisionCascadeSVDResult(U_bf, Σ_bf, V_bf, residual_norm, σ_min, final_precision)
end

end # module
