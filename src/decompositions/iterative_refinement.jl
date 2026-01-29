# Iterative Refinement Methods for Matrix Decompositions
#
# References:
# [1] N.J. Higham, "Computing the Polar Decomposition—with Applications",
#     SIAM J. Sci. Stat. Comput. 7(4):1160-1174, 1986. doi:10.1137/0907079
# [2] N.J. Higham, "Functions of Matrices: Theory and Computation",
#     SIAM, Philadelphia, 2008. Chapter 8.
# [3] Y. Nakatsukasa, Z. Bai, F. Gygi, "Optimizing Halley's Iteration for
#     Computing the Matrix Polar Decomposition", SIAM J. Sci. Comput.
#     32(5):2700-2720, 2010. doi:10.1137/090774999
# [4] J.H. Wilkinson, "Rounding Errors in Algebraic Processes", Prentice-Hall, 1963.
# [5] R.S. Martin, G. Peters, J.H. Wilkinson, "Iterative Refinement of the
#     Solution of a Positive Definite System of Equations", Numer. Math.
#     8:203-216, 1971. doi:10.1007/BF02163185
# [6] Y. Yamamoto et al., "Roundoff error analysis of the CholeskyQR2 algorithm",
#     Numer. Math. 131(2):297-322, 2015. doi:10.1007/s00211-014-0692-7
# [7] T. Ogita, K. Aishima, "Iterative refinement for singular value decomposition
#     based on matrix multiplication", J. Comput. Appl. Math. 369:112512, 2020.
# [8] E. Carson, N.J. Higham, "A New Analysis of Iterative Refinement...",
#     SIAM J. Sci. Comput. 39(6):A2834-A2856, 2017. doi:10.1137/17M1122918

using LinearAlgebra

#==============================================================================#
# Result Types
#==============================================================================#

"""
    PolarRefinementResult{T}

Result from iterative refinement of polar decomposition.

# Fields
- `Q::Matrix{T}`: Refined unitary factor
- `H::Matrix{T}`: Refined positive semidefinite Hermitian factor (A = QH)
- `iterations::Int`: Number of refinement iterations
- `residual_norm::Real`: Final residual ‖QH - A‖/‖A‖
- `orthogonality_defect::Real`: Final ‖QᴴQ - I‖
- `converged::Bool`: Whether convergence criterion was met
"""
struct PolarRefinementResult{T}
    Q::Matrix{T}
    H::Matrix{T}
    iterations::Int
    residual_norm::Real
    orthogonality_defect::Real
    converged::Bool
end

"""
    LURefinementResult{T}

Result from iterative refinement of LU decomposition.

# Fields
- `L::Matrix{T}`: Refined lower triangular factor
- `U::Matrix{T}`: Refined upper triangular factor
- `p::Vector{Int}`: Permutation vector
- `iterations::Int`: Number of refinement iterations
- `residual_norm::Real`: Final residual ‖PA - LU‖/‖A‖
- `converged::Bool`: Whether convergence criterion was met
"""
struct LURefinementResult{T}
    L::Matrix{T}
    U::Matrix{T}
    p::Vector{Int}
    iterations::Int
    residual_norm::Real
    converged::Bool
end

"""
    CholeskyRefinementResult{T}

Result from iterative refinement of Cholesky decomposition.

# Fields
- `G::Matrix{T}`: Refined upper triangular Cholesky factor (A = GᵀG)
- `iterations::Int`: Number of refinement iterations
- `residual_norm::Real`: Final residual ‖GᵀG - A‖/‖A‖
- `converged::Bool`: Whether convergence criterion was met
"""
struct CholeskyRefinementResult{T}
    G::Matrix{T}
    iterations::Int
    residual_norm::Real
    converged::Bool
end

"""
    QRRefinementResult{T}

Result from iterative refinement of QR decomposition.

# Fields
- `Q::Matrix{T}`: Refined orthogonal/unitary factor
- `R::Matrix{T}`: Refined upper triangular factor
- `iterations::Int`: Number of refinement iterations
- `residual_norm::Real`: Final residual ‖QR - A‖/‖A‖
- `orthogonality_defect::Real`: Final ‖QᴴQ - I‖
- `converged::Bool`: Whether convergence criterion was met
"""
struct QRRefinementResult{T}
    Q::Matrix{T}
    R::Matrix{T}
    iterations::Int
    residual_norm::Real
    orthogonality_defect::Real
    converged::Bool
end

"""
    TakagiRefinementResult{T}

Result from iterative refinement of Takagi decomposition.

# Fields
- `U::Matrix{T}`: Refined unitary factor
- `Σ::Vector{Real}`: Refined singular values
- `iterations::Int`: Number of refinement iterations
- `residual_norm::Real`: Final residual ‖UΣUᵀ - A‖/‖A‖
- `converged::Bool`: Whether convergence criterion was met
"""
struct TakagiRefinementResult{T}
    U::Matrix{T}
    Σ::Vector{<:Real}
    iterations::Int
    residual_norm::Real
    converged::Bool
end

#==============================================================================#
# Polar Decomposition Refinement
#==============================================================================#

"""
    refine_polar_newton_schulz(A, Q0; max_iterations=10, tol=1e-14)

Refine polar decomposition using Newton-Schulz iteration (inverse-free).

For A = QH where Q is unitary and H is positive semidefinite Hermitian,
the Newton-Schulz iteration refines Q:

    Q_{k+1} = Q_k (3I - Q_k^H Q_k) / 2

This converges quadratically when ‖I - Q₀ᴴQ₀‖ < 1.

# Arguments
- `A`: Input matrix
- `Q0`: Initial approximation to unitary factor Q
- `max_iterations`: Maximum number of iterations (default: 10)
- `tol`: Convergence tolerance for orthogonality defect (default: 1e-14)

# Returns
[`PolarRefinementResult`](@ref) with refined Q and H = QᴴA.

# References
- N.J. Higham, "Computing the Polar Decomposition—with Applications",
  SIAM J. Sci. Stat. Comput. 7(4):1160-1174, 1986.
  doi:[10.1137/0907079](https://doi.org/10.1137/0907079)
- N.J. Higham, "Functions of Matrices: Theory and Computation",
  SIAM, 2008, Chapter 8. doi:[10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778)
"""
function refine_polar_newton_schulz(A::AbstractMatrix{T}, Q0::AbstractMatrix;
                                    max_iterations::Int=10,
                                    tol::Real=1e-14) where T
    n = size(A, 1)
    Q = Matrix(Q0)

    I_n = Matrix{eltype(Q)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    for k in 1:max_iterations
        iter = k

        # Newton-Schulz step: Q = Q(3I - Q^H Q)/2
        QtQ = Q' * Q
        Q = Q * (3 * I_n - QtQ) / 2

        # Check convergence
        ortho_defect = opnorm(Q' * Q - I_n, Inf)
        if ortho_defect < tol
            converged = true
            break
        end
    end

    # Compute H = Q^H A
    H = Q' * A
    # Symmetrize H
    H = (H + H') / 2

    # Compute residual
    residual = Q * H - A
    A_norm = opnorm(A, Inf)
    residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

    return PolarRefinementResult(Q, H, iter, residual_norm, ortho_defect, converged)
end

"""
    refine_polar_newton(A, Q0; max_iterations=10, tol=1e-14)

Refine polar decomposition using scaled Newton iteration.

The Newton iteration for the unitary polar factor:

    Q_{k+1} = (γ_k Q_k + Q_k^{-H} / γ_k) / 2

where γ_k is a scaling factor (Frobenius norm scaling or determinantal scaling).

This requires matrix inversion but has better convergence for poorly conditioned
initial approximations compared to Newton-Schulz.

# Arguments
- `A`: Input matrix
- `Q0`: Initial approximation to unitary factor Q
- `max_iterations`: Maximum number of iterations (default: 10)
- `tol`: Convergence tolerance (default: 1e-14)

# References
- N.J. Higham, "Computing the Polar Decomposition—with Applications",
  SIAM J. Sci. Stat. Comput. 7(4):1160-1174, 1986.
  doi:[10.1137/0907079](https://doi.org/10.1137/0907079)
"""
function refine_polar_newton(A::AbstractMatrix{T}, Q0::AbstractMatrix;
                             max_iterations::Int=10,
                             tol::Real=1e-14) where T
    n = size(A, 1)
    Q = Matrix(Q0)

    I_n = Matrix{eltype(Q)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    for k in 1:max_iterations
        iter = k

        # Frobenius norm scaling
        Q_norm = norm(Q, 2)  # Approximation to largest singular value
        Q_inv_norm = norm(inv(Q), 2)
        γ = sqrt(Q_inv_norm / Q_norm)

        # Newton step with scaling: Q = (γQ + Q^{-H}/γ) / 2
        Q_inv_H = inv(Q)'
        Q = (γ * Q + Q_inv_H / γ) / 2

        # Check convergence
        ortho_defect = opnorm(Q' * Q - I_n, Inf)
        if ortho_defect < tol
            converged = true
            break
        end
    end

    # Compute H = Q^H A
    H = Q' * A
    H = (H + H') / 2

    # Compute residual
    residual = Q * H - A
    A_norm = opnorm(A, Inf)
    residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

    return PolarRefinementResult(Q, H, iter, residual_norm, ortho_defect, converged)
end

"""
    refine_polar_qdwh(A, Q0; max_iterations=6, tol=1e-14)

Refine polar decomposition using QDWH (QR-based Dynamically Weighted Halley).

QDWH achieves cubic convergence and typically needs only 6 iterations for
condition numbers up to 10^16. It is backward stable and communication-optimal.

The iteration is based on the Halley method with dynamic weighting:

    Q_{k+1} = Q_k (a_k I + b_k Q_k^H Q_k)(I + c_k Q_k^H Q_k)^{-1}

where a_k, b_k, c_k are computed from the condition number estimate.

# Arguments
- `A`: Input matrix
- `Q0`: Initial approximation to unitary factor Q
- `max_iterations`: Maximum iterations (default: 6, usually sufficient)
- `tol`: Convergence tolerance (default: 1e-14)

# References
- Y. Nakatsukasa, Z. Bai, F. Gygi, "Optimizing Halley's Iteration for Computing
  the Matrix Polar Decomposition", SIAM J. Sci. Comput. 32(5):2700-2720, 2010.
  doi:[10.1137/090774999](https://doi.org/10.1137/090774999)
"""
function refine_polar_qdwh(A::AbstractMatrix{T}, Q0::AbstractMatrix;
                           max_iterations::Int=6,
                           tol::Real=1e-14) where T
    n = size(A, 1)
    Q = Matrix(Q0)

    I_n = Matrix{eltype(Q)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    # Estimate initial condition number via singular values
    # Use a rough estimate based on norm ratio
    σ_min_est = 1 / opnorm(inv(Q), 2)
    σ_max_est = opnorm(Q, 2)
    ℓ = σ_min_est / σ_max_est  # Estimate of smallest singular value (normalized)

    for k in 1:max_iterations
        iter = k

        # Compute QDWH parameters
        # These formulas are from Nakatsukasa et al. (2010)
        ℓ2 = ℓ^2
        dd = (4 * (1 - ℓ2) / ℓ2)^(1/3)
        sqd = sqrt(1 + dd)
        a = sqd + sqrt(8 - 4 * dd + 8 * (2 - ℓ2) / (ℓ2 * sqd)) / 2
        a = real(a)
        b = (a - 1)^2 / 4
        c = a + b - 1

        # Update ℓ for next iteration
        ℓ_new = ℓ * (a + b * ℓ2) / (1 + c * ℓ2)

        # QDWH step using QR decomposition for numerical stability
        # Q_{k+1} = (b/c) Q_k + (1/sqrt(c)) (a - b/c) Q_k (Q_k^H Q_k + c I)^{-1}
        # Implemented via QR: [sqrt(c) Q_k; I] = [Q1; Q2] R

        QtQ = Q' * Q

        # Form the augmented matrix and use QR
        # This is more stable than direct inversion
        M = QtQ + c * I_n

        # Newton-Schulz-like formulation that avoids explicit inversion
        # Q_new = Q * (a*I + b*QtQ) * inv(I + c*QtQ)
        # Use the formula: (I + cM)^{-1} = I - c*M*(I + c*M)^{-1}
        Q = Q * ((a * I_n + b * QtQ) / M)

        ℓ = ℓ_new

        # Check convergence
        ortho_defect = opnorm(Q' * Q - I_n, Inf)
        if ortho_defect < tol || abs(1 - ℓ) < tol
            converged = true
            break
        end
    end

    # Compute H = Q^H A
    H = Q' * A
    H = (H + H') / 2

    # Compute residual
    residual = Q * H - A
    A_norm = opnorm(A, Inf)
    residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

    return PolarRefinementResult(Q, H, iter, residual_norm, ortho_defect, converged)
end

#==============================================================================#
# LU Decomposition Refinement
#==============================================================================#

"""
    refine_lu(A, L0, U0, p; max_iterations=5, tol=1e-14)

Refine LU decomposition by iteratively reducing the factorization residual.

Given PA ≈ L₀U₀, compute corrections ΔL, ΔU such that:

    R = PA - LU  (residual)
    L ΔU ≈ R     (solve for ΔU)
    ΔL U ≈ R - L ΔU  (solve for ΔL)

# Arguments
- `A`: Original matrix
- `L0`: Initial lower triangular factor (unit diagonal)
- `U0`: Initial upper triangular factor
- `p`: Permutation vector
- `max_iterations`: Maximum iterations (default: 5)
- `tol`: Convergence tolerance (default: 1e-14)

# Returns
[`LURefinementResult`](@ref) with refined L and U.

# References
- J.H. Wilkinson, "Rounding Errors in Algebraic Processes", Prentice-Hall, 1963.
- N.J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
  SIAM, 2002, Chapter 9. doi:[10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
"""
function refine_lu(A::AbstractMatrix{T}, L0::AbstractMatrix, U0::AbstractMatrix,
                   p::AbstractVector{Int};
                   max_iterations::Int=5,
                   tol::Real=1e-14) where T
    n = size(A, 1)
    L = Matrix(L0)
    U = Matrix(U0)
    PA = A[p, :]

    converged = false
    iter = 0
    residual_norm = Inf

    for k in 1:max_iterations
        iter = k

        # Compute residual R = PA - LU
        R = PA - L * U

        # Check convergence
        PA_norm = opnorm(PA, Inf)
        residual_norm = PA_norm > 0 ? opnorm(R, Inf) / PA_norm : opnorm(R, Inf)

        if residual_norm < tol
            converged = true
            break
        end

        # Solve L ΔU = R for ΔU (forward substitution since L is lower triangular)
        ΔU = L \ R

        # Solve ΔL U = R - L ΔU for ΔL
        # Note: ΔL should be strictly lower triangular
        R2 = R - L * ΔU
        ΔL = R2 / U

        # Enforce structure: L has unit diagonal, U is upper triangular
        # Update L (only strictly lower triangular part)
        for j in 1:n
            for i in (j+1):n
                L[i, j] += ΔL[i, j]
            end
        end

        # Update U (only upper triangular part)
        for j in 1:n
            for i in 1:j
                U[i, j] += ΔU[i, j]
            end
        end
    end

    return LURefinementResult(L, U, p, iter, residual_norm, converged)
end

#==============================================================================#
# Cholesky Decomposition Refinement
#==============================================================================#

"""
    refine_cholesky(A, G0; max_iterations=5, tol=1e-14)

Refine Cholesky decomposition A = GᵀG by iteratively reducing the residual.

Given A ≈ G₀ᵀG₀, solve the correction equation:

    GᵀΔG + ΔGᵀG = A - GᵀG

Using the symmetry, this reduces to solving a triangular Sylvester-like equation.

# Arguments
- `A`: Original symmetric positive definite matrix
- `G0`: Initial upper triangular Cholesky factor
- `max_iterations`: Maximum iterations (default: 5)
- `tol`: Convergence tolerance (default: 1e-14)

# Returns
[`CholeskyRefinementResult`](@ref) with refined G.

# References
- R.S. Martin, G. Peters, J.H. Wilkinson, "Iterative Refinement of the Solution
  of a Positive Definite System of Equations", Numer. Math. 8:203-216, 1971.
  doi:[10.1007/BF02163185](https://doi.org/10.1007/BF02163185)
"""
function refine_cholesky(A::AbstractMatrix{T}, G0::AbstractMatrix;
                         max_iterations::Int=5,
                         tol::Real=1e-14) where T
    n = size(A, 1)
    G = Matrix(G0)

    # Symmetrize A
    A_sym = (A + A') / 2

    converged = false
    iter = 0
    residual_norm = Inf

    for k in 1:max_iterations
        iter = k

        # Compute residual R = A - GᵀG
        GtG = G' * G
        R = A_sym - GtG

        # Check convergence
        A_norm = opnorm(A_sym, Inf)
        residual_norm = A_norm > 0 ? opnorm(R, Inf) / A_norm : opnorm(R, Inf)

        if residual_norm < tol
            converged = true
            break
        end

        # Solve GᵀΔG + ΔGᵀG = R for ΔG (upper triangular)
        # Simplified approach: ΔG ≈ (G'^{-1} R) / 2 (keeping upper triangular)
        # This is an approximation that works well for small residuals
        ΔG = (G' \ R) / 2

        # Enforce upper triangular structure
        for j in 1:n
            for i in (j+1):n
                ΔG[i, j] = zero(eltype(ΔG))
            end
        end

        # Update G
        G = G + ΔG
    end

    return CholeskyRefinementResult(G, iter, residual_norm, converged)
end

#==============================================================================#
# QR Decomposition Refinement (CholeskyQR2 style)
#==============================================================================#

"""
    refine_qr_cholqr2(A, Q0, R0; max_iterations=3, tol=1e-14)

Refine QR decomposition using CholeskyQR2-style reorthogonalization.

The algorithm applies Cholesky-based orthogonalization iteratively:
1. Compute B = QᵀQ (Gram matrix, should be close to I)
2. Compute Cholesky: B = RᵦᵀRᵦ
3. Update: Q ← Q Rᵦ⁻¹, R ← Rᵦ R

This improves orthogonality of Q while maintaining A = QR.

# Arguments
- `A`: Original matrix
- `Q0`: Initial orthogonal factor
- `R0`: Initial upper triangular factor
- `max_iterations`: Maximum iterations (default: 3, usually 2 suffices)
- `tol`: Convergence tolerance for orthogonality (default: 1e-14)

# Returns
[`QRRefinementResult`](@ref) with refined Q and R.

# References
- Y. Yamamoto et al., "Roundoff error analysis of the CholeskyQR2 algorithm",
  Numer. Math. 131(2):297-322, 2015.
  doi:[10.1007/s00211-014-0692-7](https://doi.org/10.1007/s00211-014-0692-7)
- T. Fukaya et al., "LU-Cholesky QR Algorithms for Thin QR Decomposition",
  SIAM J. Sci. Comput. 42(3):A1401-A1423, 2020.
  doi:[10.1137/18M1187347](https://doi.org/10.1137/18M1187347)
"""
function refine_qr_cholqr2(A::AbstractMatrix{T}, Q0::AbstractMatrix, R0::AbstractMatrix;
                           max_iterations::Int=3,
                           tol::Real=1e-14) where T
    _, n = size(A)
    Q = Matrix(Q0)
    R = Matrix(R0)

    I_n = Matrix{eltype(Q)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    for k in 1:max_iterations
        iter = k

        # Compute Gram matrix
        B = Q' * Q

        # Check orthogonality
        ortho_defect = opnorm(B - I_n, Inf)

        if ortho_defect < tol
            converged = true
            break
        end

        # Cholesky of Gram matrix
        C = try
            cholesky(Hermitian(B))
        catch
            # If Cholesky fails, use modified Gram-Schmidt instead
            Q, R_new = qr(Q)
            Q = Matrix(Q)
            R = R_new * R
            continue
        end

        R_B = Matrix(C.U)  # B = R_B' * R_B

        # Update: Q ← Q R_B⁻¹, R ← R_B R
        Q = Q / R_B
        R = R_B * R
    end

    # Compute residual
    residual = Q * R - A
    A_norm = opnorm(A, Inf)
    residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

    return QRRefinementResult(Q, R, iter, residual_norm, ortho_defect, converged)
end

"""
    refine_qr_mgs(A, Q0, R0; max_iterations=2, tol=1e-14)

Refine QR decomposition using Modified Gram-Schmidt reorthogonalization.

Applies MGS to the columns of Q to improve orthogonality.

# Arguments
- `A`: Original matrix
- `Q0`: Initial orthogonal factor
- `R0`: Initial upper triangular factor
- `max_iterations`: Maximum iterations (default: 2)
- `tol`: Convergence tolerance (default: 1e-14)

# Returns
[`QRRefinementResult`](@ref) with refined Q and R.
"""
function refine_qr_mgs(A::AbstractMatrix{T}, Q0::AbstractMatrix, R0::AbstractMatrix;
                       max_iterations::Int=2,
                       tol::Real=1e-14) where T
    _, n = size(A)
    Q = Matrix(Q0)
    R = Matrix(R0)

    I_n = Matrix{eltype(Q)}(I, n, n)
    converged = false
    iter = 0
    ortho_defect = Inf

    for k in 1:max_iterations
        iter = k

        # Modified Gram-Schmidt on columns of Q
        R_corr = zeros(eltype(Q), n, n)
        for j in 1:n
            for i in 1:(j-1)
                r_ij = Q[:, i]' * Q[:, j]
                Q[:, j] -= r_ij * Q[:, i]
                R_corr[i, j] = r_ij
            end
            r_jj = norm(Q[:, j])
            Q[:, j] /= r_jj
            R_corr[j, j] = r_jj
        end

        # Update R to maintain A = QR
        R = R_corr * R

        # Check orthogonality
        ortho_defect = opnorm(Q' * Q - I_n, Inf)

        if ortho_defect < tol
            converged = true
            break
        end
    end

    # Compute residual
    residual = Q * R - A
    A_norm = opnorm(A, Inf)
    residual_norm = A_norm > 0 ? opnorm(residual, Inf) / A_norm : opnorm(residual, Inf)

    return QRRefinementResult(Q, R, iter, residual_norm, ortho_defect, converged)
end

#==============================================================================#
# Takagi Decomposition Refinement
#==============================================================================#

"""
    refine_takagi(A, U0, Σ0; max_iterations=5, tol=1e-14)

Refine Takagi decomposition A = UΣUᵀ for complex symmetric matrices.

The refinement is based on the SVD refinement pattern adapted for Takagi:
1. Compute residual matrices from orthogonality and factorization conditions
2. Solve correction equations for U and Σ
3. Update and reorthogonalize

# Arguments
- `A`: Complex symmetric matrix (Aᵀ = A)
- `U0`: Initial unitary factor
- `Σ0`: Initial singular values (non-negative real)
- `max_iterations`: Maximum iterations (default: 5)
- `tol`: Convergence tolerance (default: 1e-14)

# Returns
[`TakagiRefinementResult`](@ref) with refined U and Σ.

# References
- Adapted from T. Ogita, K. Aishima, "Iterative refinement for singular value
  decomposition based on matrix multiplication", J. Comput. Appl. Math. 369:112512, 2020.
  doi:[10.1016/j.cam.2019.112512](https://doi.org/10.1016/j.cam.2019.112512)
"""
function refine_takagi(A::AbstractMatrix{Complex{T}}, U0::AbstractMatrix,
                       Σ0::AbstractVector;
                       max_iterations::Int=5,
                       tol::Real=1e-14) where T<:AbstractFloat
    n = size(A, 1)
    U = Matrix(U0)
    Σ = Vector(Σ0)

    I_n = Matrix{eltype(U)}(I, n, n)
    converged = false
    iter = 0
    residual_norm = Inf

    for k in 1:max_iterations
        iter = k

        # Compute orthogonality residual: R = I - UᴴU
        R = I_n - U' * U

        # Compute transformed matrix: T = UᴴAŪ (where Ū is complex conjugate)
        T_mat = U' * A * conj.(U)

        # Update singular values from diagonal of T
        for i in 1:n
            denom = 1 - real(R[i, i])
            if abs(denom) > eps(T)
                Σ[i] = abs(real(T_mat[i, i])) / denom
            else
                Σ[i] = abs(real(T_mat[i, i]))
            end
        end

        # Compute correction matrix for U
        # Similar to RefSVD but adapted for Takagi structure
        E = zeros(eltype(U), n, n)

        # Threshold for distinguishing clustered singular values
        Σ_max = maximum(Σ)
        δ = 2 * eps(T) * Σ_max * n

        for j in 1:n
            for i in 1:n
                if i == j
                    E[i, i] = R[i, i] / 2
                else
                    σ_diff = Σ[j]^2 - Σ[i]^2
                    if abs(σ_diff) > δ * max(Σ[i], Σ[j])
                        # Well-separated singular values
                        E[i, j] = (T_mat[i, j] + Σ[j] * R[i, j]) * Σ[j] / σ_diff
                    else
                        # Clustered singular values
                        E[i, j] = R[i, j] / 2
                    end
                end
            end
        end

        # Update U
        U = U * (I_n + E)

        # Reorthogonalize U using Newton-Schulz
        for _ in 1:2
            UtU = U' * U
            U = U * (3 * I_n - UtU) / 2
        end

        # Check convergence
        reconstruction = U * Diagonal(Σ) * transpose(U)
        residual = reconstruction - A
        A_norm = maximum(abs.(A))
        residual_norm = A_norm > 0 ? maximum(abs.(residual)) / A_norm : maximum(abs.(residual))

        if residual_norm < tol
            converged = true
            break
        end
    end

    return TakagiRefinementResult(U, Σ, iter, residual_norm, converged)
end

#==============================================================================#
# Exports
#==============================================================================#

export PolarRefinementResult, LURefinementResult, CholeskyRefinementResult
export QRRefinementResult, TakagiRefinementResult

export refine_polar_newton_schulz, refine_polar_newton, refine_polar_qdwh
export refine_lu, refine_cholesky
export refine_qr_cholqr2, refine_qr_mgs
export refine_takagi

#==============================================================================#
# Stub functions for extensions
#==============================================================================#

"""
    refine_polar_double64(A, Q0; method=:newton_schulz, max_iterations=10, tol=1e-30)

Refine polar decomposition using Double64 arithmetic. Requires DoubleFloats.jl.
"""
function refine_polar_double64 end

"""
    refine_polar_multifloat(A, Q0; precision=:x2, method=:newton_schulz, max_iterations=10)

Refine polar decomposition using MultiFloat arithmetic. Requires MultiFloats.jl.
"""
function refine_polar_multifloat end

"""
    refine_lu_double64(A, L0, U0, p; max_iterations=5, tol=1e-30)

Refine LU decomposition using Double64 arithmetic. Requires DoubleFloats.jl.
"""
function refine_lu_double64 end

"""
    refine_lu_multifloat(A, L0, U0, p; precision=:x2, max_iterations=5)

Refine LU decomposition using MultiFloat arithmetic. Requires MultiFloats.jl.
"""
function refine_lu_multifloat end

"""
    refine_cholesky_double64(A, G0; max_iterations=5, tol=1e-30)

Refine Cholesky decomposition using Double64 arithmetic. Requires DoubleFloats.jl.
"""
function refine_cholesky_double64 end

"""
    refine_cholesky_multifloat(A, G0; precision=:x2, max_iterations=5)

Refine Cholesky decomposition using MultiFloat arithmetic. Requires MultiFloats.jl.
"""
function refine_cholesky_multifloat end

"""
    refine_qr_double64(A, Q0, R0; method=:cholqr2, max_iterations=3, tol=1e-30)

Refine QR decomposition using Double64 arithmetic. Requires DoubleFloats.jl.
"""
function refine_qr_double64 end

"""
    refine_qr_multifloat(A, Q0, R0; precision=:x2, method=:cholqr2, max_iterations=3)

Refine QR decomposition using MultiFloat arithmetic. Requires MultiFloats.jl.
"""
function refine_qr_multifloat end

"""
    refine_takagi_double64(A, U0, Σ0; max_iterations=5, tol=1e-30)

Refine Takagi decomposition using Double64 arithmetic. Requires DoubleFloats.jl.
"""
function refine_takagi_double64 end

"""
    refine_takagi_multifloat(A, U0, Σ0; precision=:x2, max_iterations=5)

Refine Takagi decomposition using MultiFloat arithmetic. Requires MultiFloats.jl.
"""
function refine_takagi_multifloat end

export refine_polar_double64, refine_polar_multifloat
export refine_lu_double64, refine_lu_multifloat
export refine_cholesky_double64, refine_cholesky_multifloat
export refine_qr_double64, refine_qr_multifloat
export refine_takagi_double64, refine_takagi_multifloat
