# Interval Gaussian elimination for a square real interval system A x = b.
#
# J. Horáček, "Interval linear and nonlinear systems", PhD thesis, Charles University, Prague,
# 2019: Section 5.6.1 (Algorithm 5.9, the elimination; Algorithm 5.10, the backward
# substitution) and Section 8.4.1 (the determinant).

using LinearAlgebra

"""
    GaussianEliminationResult{VT, MT}

The result of [`interval_gaussian_elimination`](@ref).

# Fields
- `solution::VT`: a ball vector containing the solution of `Â x = b̂` for every `Â` in `A` and
  `b̂` in `b`; its radii are infinite when not `certified`
- `certified::Bool`: whether the elimination ran to the end with no pivot containing zero. Then
  every matrix of `A` is nonsingular. `false` says only that the elimination stopped: it is not
  a proof that `A` contains a singular matrix
- `L::MT`, `U::MT`: ball matrices, unit lower triangular and upper triangular. When `certified`,
  for every `Â` in `A` the rows of `Â` taken in the order `p` factor as `Â[p, :] = L̂ Û` with
  `L̂` in `L` and `Û` in `U`
- `p::Vector{Int}`: the order of the rows
"""
struct GaussianEliminationResult{VT, MT}
    solution::VT
    certified::Bool
    L::MT
    U::MT
    p::Vector{Int}
end

# the largest lower bound of |x| over the ball, zero when it contains zero
_ge_mig(x::Ball{T, T}) where {T} = max(sub_down(abs(mid(x)), rad(x)), zero(T))

_ge_ball_matrix(M::Matrix{<:Ball}) = BallMatrix(mid.(M), rad.(M))

"""
    interval_gaussian_elimination(A::BallMatrix, b::BallVector; pivoting = :partial)
        -> GaussianEliminationResult

Gaussian elimination and backward substitution in ball arithmetic for the square real interval
system `A x = b`: Algorithms 5.9 and 5.10 of Horáček (2019).

Elimination, for `k = 1, …, n − 1`: a row `j ≥ k` whose entry in column `k` does not contain
zero is brought to position `k`; for every row `j > k` the entry `a_jk` is set to zero and

    a_j,k+1:n := a_j,k+1:n − (a_jk / a_kk) a_k,k+1:n,        b_j := b_j − (a_jk / a_kk) b_k.

If no such row exists the elimination stops, and the result is not `certified`. Backward
substitution: `x_i := (b_i − Σ_{j>i} a_ij x_j) / a_ii` for `i = n, …, 1`.

Why the result is an enclosure: for a fixed `Â` in `A` and `b̂` in `b`, Gaussian elimination on
them with the same order of the rows evaluates the same formulae on numbers contained in the
balls, and every operation on balls contains the result of the operation on their points; a
pivot that does not contain zero is not zero for `Â`. So `Â` is nonsingular, its solution is in
`solution`, and its factors are in `L` and `U`.

`pivoting = :partial` chooses, among the rows whose entry does not contain zero, the one whose
entry is farthest from zero (the largest lower bound of its absolute value). Step 2 of
Algorithm 5.9 asks only for a row whose entry does not contain zero and leaves the choice free;
the thesis then says that it selects "a pivot with the smallest mignitude". `pivoting = :none`
keeps the rows in their order and stops at the first pivot that contains zero.

The enclosure may be much wider than the hull of the solution set: the radii grow with the
number of operations, as Example 5.11 of the thesis shows. The thesis recommends
preconditioning by the inverse of the midpoint ([`compute_preconditioner`](@ref)).

# Reference

J. Horáček, "Interval linear and nonlinear systems", PhD thesis, Charles University, Prague,
2019, Section 5.6.1.
"""
function interval_gaussian_elimination(A::BallMatrix{T}, b::BallVector{T};
        pivoting::Symbol = :partial) where {T}
    n = size(A, 1)
    (n == size(A, 2) && length(b) == n) ||
        throw(DimensionMismatch("A must be square and b of the same order"))
    (eltype(mid(A)) <: Real && eltype(mid(b)) <: Real) ||
        throw(ArgumentError("interval_gaussian_elimination is for real interval systems"))
    pivoting in (:partial, :none) || throw(ArgumentError("pivoting must be :partial or :none"))

    BT = Ball{T, T}
    U = BT[Ball(mid(A)[i, j], rad(A)[i, j]) for i in 1:n, j in 1:n]
    y = BT[Ball(mid(b)[i], rad(b)[i]) for i in 1:n]
    L = BT[Ball(i == j ? one(T) : zero(T), zero(T)) for i in 1:n, j in 1:n]
    p = collect(1:n)
    failure() = GaussianEliminationResult(BallVector(zeros(T, n), fill(T(Inf), n)), false,
        _ge_ball_matrix(L), _ge_ball_matrix(U), p)
    (all(x -> isfinite(mid(x)) && isfinite(rad(x)), U) &&
     all(x -> isfinite(mid(x)) && isfinite(rad(x)), y)) || return failure()

    for k in 1:n
        # steps 2 and 3 of Algorithm 5.9: a row whose entry in column k does not contain zero
        row = k
        if pivoting === :partial
            best = _ge_mig(U[k, k])
            for i in (k + 1):n
                m = _ge_mig(U[i, k])
                if m > best
                    best, row = m, i
                end
            end
        end
        _ge_mig(U[row, k]) > 0 || return failure()
        if row != k
            for j in 1:n
                U[k, j], U[row, j] = U[row, j], U[k, j]
            end
            for j in 1:(k - 1)
                L[k, j], L[row, j] = L[row, j], L[k, j]
            end
            y[k], y[row] = y[row], y[k]
            p[k], p[row] = p[row], p[k]
        end
        # step 4
        for i in (k + 1):n
            mult = U[i, k] / U[k, k]
            L[i, k] = mult
            for j in (k + 1):n
                U[i, j] = U[i, j] - mult * U[k, j]
            end
            U[i, k] = Ball(zero(T), zero(T))
            y[i] = y[i] - mult * y[k]
        end
    end

    # Algorithm 5.10
    x = Vector{BT}(undef, n)
    for i in n:-1:1
        s = y[i]
        for j in (i + 1):n
            s = s - U[i, j] * x[j]
        end
        x[i] = s / U[i, i]
    end
    all(v -> isfinite(mid(v)) && isfinite(rad(v)), x) || return failure()
    return GaussianEliminationResult(BallVector(mid.(x), rad.(x)), true, _ge_ball_matrix(L),
        _ge_ball_matrix(U), p)
end

"""
    interval_gaussian_elimination_det(A::BallMatrix) -> Ball

A ball containing the determinant of every matrix of the real interval matrix `A`, by
Section 8.4.1 of Horáček (2019): the product of the diagonal of the row echelon form computed by
[`interval_gaussian_elimination`](@ref), with the sign of the permutation of the rows. The ball
has infinite radius when the elimination stops at a pivot that contains zero; the elimination
is then inconclusive, and the determinant may or may not vanish on `A`.
"""
function interval_gaussian_elimination_det(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    eltype(mid(A)) <: Real ||
        throw(ArgumentError("interval_gaussian_elimination_det is for real interval matrices"))
    result = interval_gaussian_elimination(A, BallVector(zeros(T, n), zeros(T, n)))
    result.certified || return Ball(zero(T), T(Inf))
    det = result.U[1, 1]
    for i in 2:n
        det = det * result.U[i, i]
    end
    # the sign of the permutation, by sorting a copy with transpositions
    p = copy(result.p)
    negative = false
    for i in 1:n
        while p[i] != i
            j = p[i]
            p[i], p[j] = p[j], p[i]
            negative = !negative
        end
    end
    return negative ? -det : det
end

"""
    is_regular_gaussian_elimination(A::BallMatrix) -> Bool

`true` when [`interval_gaussian_elimination`](@ref) runs to the end on the real interval matrix
`A`, which proves that every matrix of `A` is nonsingular. `false` is inconclusive.
"""
function is_regular_gaussian_elimination(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    return interval_gaussian_elimination(A, BallVector(zeros(T, n), zeros(T, n))).certified
end

export GaussianEliminationResult
export interval_gaussian_elimination, interval_gaussian_elimination_det
export is_regular_gaussian_elimination
