# M-matrices and H-matrices, by the definitions of
#
#   R. S. Varga, Geršgorin and His Circles, Springer Series in Computational Mathematics 36,
#   Springer, Berlin, 2004, doi 10.1007/978-3-642-17798-9, Chapter 5 and Appendix C.

using LinearAlgebra: diag

"""
Returns a matrix containing the off-diagonal elements
"""
function off_diagonal_abs(A::BallMatrix)
    B = deepcopy(upper_abs(A))
    for i in diagind(B)
        B[i] = 0.0
    end
    return BallMatrix(B)
end

"""
Computes a vector containing lower bounds for the diagonal elements of |A|
"""
diagonal_abs_lower_bound(A::BallMatrix{T}) where {T} =
    T[sub_down(_abs_lo(A.c[i, i]), A.r[i, i]) for i in axes(A.c, 1)]

# Definition C.3 for a family of matrices of Z^{n×n}: `dlo[i] ≤ a_ii ≤ dhi[i]` and
# `0 ≤ −a_ij ≤ off[i, j]` for i ≠ j. With μ := max_i dhi[i], every member is μI − B with
# 0 ≤ B ≤ B̄ entrywise, B̄ having μ − dlo[i] on the diagonal and `off` elsewhere, and the spectral
# radius of a nonnegative matrix does not decrease when an entry increases ((iii) of Theorem C.2
# there), so ρ(B̄) < μ gives ρ(B) < μ for every member.
function _nonsingular_M_matrix(dlo::Vector{T}, dhi::Vector{T}, off::Matrix{T}) where {T}
    all(isfinite, dlo) && all(isfinite, dhi) && all(isfinite, off) || return false
    μ = maximum(dhi)
    μ > 0 || return false
    B = copy(off)
    for i in eachindex(dlo)
        B[i, i] = sub_up(μ, dlo[i])
    end
    iszero(B) && return true
    return collatz_upper_bound(BallMatrix(B)) < μ
end

"""
    is_M_matrix(A::BallMatrix) -> Bool

`true` when every matrix of the ball `A` is proved to be a nonsingular M-matrix, by the
definition of Varga (2004), Definition 5.4 (repeated as Definition C.3): a real matrix `A` with
`a_ij ≤ 0` for all `i ≠ j`, written `A = μI − B` with `μ` real and `B ≥ 0` entrywise, is a
nonsingular M-matrix if `ρ(B) < μ`. Then `A⁻¹ ≥ 0` entrywise (Proposition C.4 there).

The two conditions are checked for the whole ball: the upper ends of the off-diagonal entries
are not positive, and with `μ` the largest upper end of a diagonal entry, an upper bound of the
spectral radius of a matrix that dominates every `B` entrywise is below `μ`
([`collatz_upper_bound`](@ref)). `false` means that this was not proved, and is returned for a
complex matrix, an M-matrix being real.

For the property that needs no sign condition, that the comparison matrix is a nonsingular
M-matrix, see [`is_H_matrix`](@ref).

# Reference

R. S. Varga, *Geršgorin and His Circles*, Springer Series in Computational Mathematics 36,
Springer, Berlin, 2004, doi 10.1007/978-3-642-17798-9, Definition 5.4 and Appendix C.
"""
function is_M_matrix(A::BallMatrix{T}) where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("is_M_matrix needs a square matrix"))
    eltype(A.c) <: Real || return false
    n = size(A, 1)
    off = zeros(T, n, n)
    for j in 1:n, i in 1:n
        i == j && continue
        add_up(A.c[i, j], A.r[i, j]) <= 0 || return false
        off[i, j] = add_up(abs(A.c[i, j]), A.r[i, j])
    end
    return _nonsingular_M_matrix(T[sub_down(A.c[i, i], A.r[i, i]) for i in 1:n],
        T[add_up(A.c[i, i], A.r[i, i]) for i in 1:n], off)
end

"""
    is_H_matrix(A::BallMatrix) -> Bool

`true` when every matrix of the ball `A`, real or complex, is proved to be a nonsingular
H-matrix: its comparison matrix `M(A)`, with `|a_ii|` on the diagonal and `−|a_ij|` off it, is a
nonsingular M-matrix (Varga (2004), Definition C.6), which makes `A` nonsingular
(Proposition C.7 there). The test is that of [`is_M_matrix`](@ref) on the comparison matrices
of the ball; `false` means that this was not proved.

# Reference

R. S. Varga, *Geršgorin and His Circles*, Springer Series in Computational Mathematics 36,
Springer, Berlin, 2004, doi 10.1007/978-3-642-17798-9, Appendix C, (C.4), Definition C.6 and
Proposition C.7.
"""
function is_H_matrix(A::BallMatrix{T}) where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("is_H_matrix needs a square matrix"))
    n = size(A, 1)
    off = upper_abs(A)
    for i in 1:n
        off[i, i] = zero(T)
    end
    return _nonsingular_M_matrix(diagonal_abs_lower_bound(A),
        T[add_up(_abs_hi(A.c[i, i]), A.r[i, i]) for i in 1:n], Matrix{T}(off))
end
