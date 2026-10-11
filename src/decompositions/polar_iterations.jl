# Floating-point algorithms for the polar factor and for a thin QR decomposition, each as its
# source states it. They return approximations with no error bound; enclosures of the factors are
# `verified_polar` and `verified_qr` (Rump and Ogita (2024)).
#
#   N. J. Higham, "Computing the polar decomposition, with applications", SIAM J. Sci. Stat.
#     Comput. 7(4) (1986) 1160-1174, doi 10.1137/0907079: Algorithm Polar.
#   N. J. Higham, Functions of Matrices: Theory and Computation, SIAM, 2008,
#     doi 10.1137/1.9780898717778: iteration (8.20), Newton-Schulz.
#   Y. Nakatsukasa, Z. Bai and F. Gygi, "Optimizing Halley's iteration for computing the matrix
#     polar decomposition", SIAM J. Matrix Anal. Appl. 31(5) (2010) 2700-2720,
#     doi 10.1137/090774999: the QDWH algorithm.
#   Y. Yamamoto, Y. Nakatsukasa, Y. Yanagisawa and T. Fukaya, "Roundoff error analysis of the
#     CholeskyQR2 algorithm", Electron. Trans. Numer. Anal. 44 (2015) 306-326: CholeskyQR2.
#
# The routines `refine_polar_*`, `refine_qr_*`, `refine_lu`, `refine_cholesky` and
# `refine_takagi` that stood here until 2026-10-11 ran these iterations from an approximate factor
# in place of the matrix, or had no source; started from a factorisation perturbed by 1e-8 they
# returned a residual of that size.

"""
    PolarIterationResult

Result of [`polar_newton`](@ref), [`polar_newton_schulz`](@ref) and [`polar_qdwh`](@ref): the
computed factors of `A ≈ U H`, `U` with orthonormal columns and `H` Hermitian, the number of
iterations, and whether the stopping test of the algorithm was met. The factors are
floating-point approximations.
"""
struct PolarIterationResult{T}
    U::Matrix{T}
    H::Matrix{T}
    iterations::Int
    converged::Bool
end

# U ≈ polar factor of the square or tall A from that of its triangular factor: A = QR,
# R = U_R H_R, A = (Q U_R) H_R (Higham (1986), Section 3.1)
function _polar_via_triangular(iteration, A::AbstractMatrix)
    m, n = size(A)
    m >= n || throw(DimensionMismatch("the polar iterations need at least as many rows as columns"))
    S = float(eltype(A))
    if m == n
        U, k, converged = iteration(Matrix{S}(A))
    else
        F = qr(Matrix{S}(A))
        Ur, k, converged = iteration(Matrix(F.R))
        U = Matrix(F.Q) * Ur
    end
    H1 = U' * A
    return PolarIterationResult(U, (H1 + H1') / 2, k, converged)       # step (3) of Algorithm Polar
end

"""
    polar_newton(A; maxiter = 100, tol = nothing) -> PolarIterationResult

Algorithm Polar of Higham (1986) for a nonsingular square matrix `A`, giving `A ≈ U H`:

1. `X₀ = A`;
2. repeat `Y_k = X_k⁻¹`; `γ_k = 1` when close to convergence, otherwise
   `γ_k = √(β_k/α_k)` with `α_k = √(‖X_k‖₁‖X_k‖_∞)` and `β_k = √(‖Y_k‖₁‖Y_k‖_∞)`;
   `X_{k+1} = (γ_k X_k + Y_k*/γ_k)/2`; until converged;
3. `U = X_{k+1}`, `H = (U*A + (U*A)*)/2`.

The convergence test is (3.17), `‖X_{k+1} − X_k‖₁ ≤ δ_n ‖X_k‖₁`, with `δ_n = tol`; the paper
asks for a small multiple of the unit roundoff depending on `n`, and the default is `8n` times
it. "Close to convergence" is taken as a relative change below `10⁻²`, the value at which the
paper reverts to the unaccelerated iteration. A matrix with more rows than columns is reduced
to its triangular factor first (Section 3.1 of the paper).

The result is an approximation with no error bound; see [`verified_polar`](@ref).
"""
function polar_newton(A::AbstractMatrix; maxiter::Integer = 100, tol = nothing)
    return _polar_via_triangular(A) do X
        n = size(X, 1)
        δ = tol === nothing ? 4n * eps(real(eltype(X))) : tol
        close = false
        for k in 1:maxiter
            Y = inv(X)
            γ = one(real(eltype(X)))
            if !close
                α = sqrt(opnorm(X, 1) * opnorm(X, Inf))
                β = sqrt(opnorm(Y, 1) * opnorm(Y, Inf))
                γ = sqrt(β / α)
            end
            Xn = (γ .* X .+ Y' ./ γ) ./ 2
            change = opnorm(Xn - X, 1) / opnorm(X, 1)
            X = Xn
            change <= δ && return X, k, true
            close = close || change <= 1e-2
        end
        return X, maxiter, false
    end
end

"""
    polar_newton_schulz(A; maxiter = 100, tol = nothing) -> PolarIterationResult

The Newton-Schulz iteration (8.20) of Higham (2008),

    X_{k+1} = X_k (3I − X_k* X_k)/2,      X₀ = A,

which converges quadratically to the unitary polar factor when every singular value of `A` lies
in `(0, √3)`, and applies to rectangular `A` of full column rank. Throws when the computed
`‖A‖₂` is not below `√3`. The iteration stops when the relative change of the iterate in the
1-norm is at most `tol` (default `8n` times the unit roundoff), the test of
[`polar_newton`](@ref); the book states no test for this iteration.

The result is an approximation with no error bound; see [`verified_polar`](@ref).
"""
function polar_newton_schulz(A::AbstractMatrix; maxiter::Integer = 100, tol = nothing)
    m, n = size(A)
    m >= n || throw(DimensionMismatch("polar_newton_schulz needs at least as many rows as columns"))
    X = Matrix{float(eltype(A))}(A)
    opnorm(X, 2) < sqrt(3) ||
        throw(ArgumentError("polar_newton_schulz needs every singular value of A below √3"))
    δ = tol === nothing ? 4n * eps(real(eltype(X))) : tol
    k, converged = 0, false
    while k < maxiter && !converged
        k += 1
        Xn = X * (3I - X' * X) / 2
        converged = opnorm(Xn - X, 1) <= δ * opnorm(X, 1)
        X = Xn
    end
    H1 = X' * A
    return PolarIterationResult(X, (H1 + H1') / 2, k, converged)
end

# (3.15) of Nakatsukasa, Bai and Gygi
function _qdwh_h(ℓ::T) where {T <: AbstractFloat}
    d = cbrt(4 * (1 - ℓ^2) / ℓ^4)
    return sqrt(1 + d) + sqrt(8 - 4d + 8 * (2 - ℓ^2) / (ℓ^2 * sqrt(1 + d))) / 2
end

# an estimate of ‖A⁻¹‖₁: the LAPACK estimator for the BLAS types, the norm of the inverse otherwise
function _inverse_one_norm_estimate(A::Matrix{S}) where {S}
    if S <: LinearAlgebra.BlasFloat
        a1 = opnorm(A, 1)
        rc = LinearAlgebra.LAPACK.gecon!('1', lu(A).factors, a1)
        return 1 / (rc * a1)
    end
    return opnorm(inv(A), 1)
end

"""
    polar_qdwh(A; maxiter = 20) -> PolarIterationResult

The QDWH algorithm of Nakatsukasa, Bai and Gygi (2010), the QR-based dynamically weighted Halley
iteration, for a nonsingular square `A`:

1. `X₀ = A/α`, `ℓ₀ = β/α`;
2. repeat: `a_k = h(ℓ_k)`, `b_k = (a_k − 1)²/4`, `c_k = a_k + b_k − 1`, with `h` of (3.15);
   the QR decomposition `[√c_k X_k; I] = [Q₁; Q₂] R`;
   `X_{k+1} = (b_k/c_k) X_k + (1/√c_k)(a_k − b_k/c_k) Q₁ Q₂*`;
   `ℓ_{k+1} = ℓ_k (a_k + b_k ℓ_k²)/(1 + c_k ℓ_k²)`; until convergence;
3. `U = X_k`.

The estimates are those of Section 4 of the paper: `α = ‖A‖_F`, and `β = 1/(γ√n)` (4.7) with
`γ` the LAPACK estimate of `‖A⁻¹‖₁` (for element types outside the BLAS, the 1-norm of the
computed inverse). The stopping test is (3.21), `‖X_k − X_{k−1}‖_F ≤ (4ε)^{1/3}`, `ε` the
machine epsilon. The paper bounds the number of iterations by six for condition numbers up to
`10¹⁶` in double precision. A matrix with more rows than columns is reduced to its triangular
factor first, as in [`polar_newton`](@ref).

The result is an approximation with no error bound; see [`verified_polar`](@ref).
"""
function polar_qdwh(A::AbstractMatrix; maxiter::Integer = 20)
    return _polar_via_triangular(A) do X0
        n = size(X0, 1)
        S = eltype(X0)
        T = real(S)
        α = norm(X0)                                    # the Frobenius norm
        β = 1 / (_inverse_one_norm_estimate(X0) * sqrt(T(n)))
        X = X0 ./ α
        ℓ = min(β / α, one(T))
        stop = cbrt(4 * eps(T))
        Id = Matrix{S}(I, n, n)
        for k in 1:maxiter
            a = _qdwh_h(ℓ)
            b = (a - 1)^2 / 4
            c = a + b - 1
            Q = Matrix(qr(vcat(sqrt(c) .* X, Id)).Q)
            Xn = (b / c) .* X .+ ((a - b / c) / sqrt(c)) .* (Q[1:n, :] * Q[(n + 1):2n, :]')
            ℓ = min(ℓ * (a + b * ℓ^2) / (1 + c * ℓ^2), one(T))
            change = norm(Xn - X)
            X = Xn
            change <= stop && return X, k, true
        end
        return X, maxiter, false
    end
end

"""
    cholesky_qr2(X) -> (Z, U)

The CholeskyQR2 algorithm of Yamamoto, Nakatsukasa, Yanagisawa and Fukaya (2015) for the thin QR
decomposition `X ≈ Z U` of a matrix with at least as many rows as columns (Section 2.2):

    A = X*X,   R = chol(A),   Y = X R⁻¹,
    B = Y*Y,   S = chol(B),   Z = Y S⁻¹,   U = S R.

Under their assumption (2.1), `8 κ₂(X) √(mnu + n(n+1)u) ≤ 1`, so `κ₂(X)` at most of order
`u^(-1/2)`, the paper proves that the loss of orthogonality of `Z` and the relative residual
are of order `u`. For a larger condition number the Cholesky factorisation can break down, and
the exception of `cholesky` is then thrown.

The result is an approximation with no error bound; see [`verified_qr`](@ref).
"""
function cholesky_qr2(X::AbstractMatrix)
    m, n = size(X)
    m >= n || throw(DimensionMismatch("cholesky_qr2 needs at least as many rows as columns"))
    Xf = Matrix{float(eltype(X))}(X)
    R = cholesky(Hermitian(Xf' * Xf)).U
    Y = Xf / R
    S = cholesky(Hermitian(Y' * Y)).U
    return Y / S, Matrix(S * R)
end

export PolarIterationResult, polar_newton, polar_newton_schulz, polar_qdwh, cholesky_qr2
