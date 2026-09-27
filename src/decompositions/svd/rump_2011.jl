# Verified bounds for singular values, by
#
#   @Article{Rump2011,
#     author  = {Rump, Siegfried M.},
#     title   = {Verified bounds for singular values, in particular for the spectral norm of a
#                matrix and its inverse},
#     journal = {BIT Numerical Mathematics},
#     year    = {2011},
#     volume  = {51},
#     pages   = {367--384},
#     doi     = {10.1007/s10543-010-0295-z},
#   }
#
# Theorem 3.1 is the two-frame enclosure; Miyajima (2014) cites it as his Theorem 5. Theorem 3.2
# and its (3.15) bound the spectral norm from ONE frame.
#
# THEOREM 3.1, verbatim. For A, U, V in K^{n by n}, put Sigma := U^H A V and assume
#
#     ||I - U^H U|| <= alpha < 1,     ||I - V^H V|| <= beta < 1,                          (3.5)
#
# and write Sigma = D + E with D diagonal. Then there is a NUMBERING
# nu : {1..n} -> {1..n} with
#
#     (|D_ii| - ||E||) / sqrt((1+alpha)(1+beta))  <=  sigma_{nu(i)}(A)
#                                                 <=  (|D_ii| + ||E||) / sqrt((1-alpha)(1-beta)),   (3.6)
#
# and in particular
#
#     ||A|| <= (max_i |D_ii| + ||E||) / sqrt((1-alpha)(1-beta)).                          (3.7)
#
# Two things about this statement decide how it can honestly be used.
#
# The frames are SQUARE. Miyajima's Remark 3 shows the economy version of the same split loses the
# upper bound: for m = 2n, A = [I_n; 0], U the block swap and V = I_n every truncated residual
# vanishes while every singular value is 1. So this route is for square A only, and the economy
# enclosure is Theorem 7 of Miyajima (2014) with its residual measured against A.
#
# (3.7), the spectral-norm corollary, is not a separate routine: it is the first entry of
# `_rump2011_thm3_1`, since the intervals come back ordered by |D_ii| and the largest singular
# value is the largest whatever nu is, and the package already has `upper_bound_L2_opnorm` for
# bounding a norm directly. A wrapper returning that entry would be a third way to say it.
#
# The conclusion holds only UP TO the numbering nu, which the theorem does not determine. What is
# proved is a statement about the two multisets, not that the i-th interval holds sigma_i. Pairing
# them in decreasing order is legitimate only once the intervals are pairwise disjoint, in which
# case the numbering is forced; where they overlap, all that is available is that the union of the
# overlapping intervals contains as many singular values as it has members. `_rump2011_thm3_1`
# returns the intervals in decreasing order of |D_ii| and its docstring says exactly this.

using LinearAlgebra

# A Ball from an outward-rounded pair of bounds. Note the cost of the representation: the
# midpoint is rounded, so the radius absorbs about one ulp of it, which at sigma = 5.8 is 8.9e-16
# against radii of order 1e-14. The interval stays valid, but two enclosures whose bounds differ
# by less than that cannot be ranked by their radii.
function _ball_from_bounds(lo::T, hi::T, ::Type{T}) where {T}
    isfinite(lo) && isfinite(hi) || return Ball(zero(T), T(Inf))
    mid_ = (lo + hi) / 2
    rad_ = setrounding(T, RoundUp) do
        max(hi - mid_, mid_ - lo)
    end
    return Ball(mid_, rad_)
end

"""
    _rump2011_thm3_1(A::BallMatrix) -> Vector{Ball}

Theorem 3.1 of Rump (2011), cited as Theorem 5 by Miyajima (2014). With `U`, `V` square frames
from a numerical SVD of `A`, `Σ = UᴴAV = D + E` with `D` diagonal, `α = ‖I − UᴴU‖₂` and
`β = ‖I − VᴴV‖₂` both below one,

    (|Dᵢᵢ| − ‖E‖₂)/√((1+α)(1+β))  ≤  σ_ν(i)(A)  ≤  (|Dᵢᵢ| + ‖E‖₂)/√((1−α)(1−β)).

**The numbering matters.** The theorem asserts the inequalities for *some* permutation `ν` which
it does not determine, so what is proved relates the two multisets, not the `i`-th interval to
`σᵢ`. The intervals are returned in decreasing order of `|Dᵢᵢ|`; reading the `i`-th of them as an
enclosure of `σᵢ(A)` is justified only when they are pairwise disjoint, which forces the
numbering. Where they overlap, what holds is that a union of `k` overlapping intervals contains
`k` singular values counted with multiplicity. [`_miyajima2014_thm7`](@ref) has no such caveat,
since its bounds are anchored to `Σ̂ᵢᵢ` of the SVD itself.

`A` must be square: Miyajima's Remark 3 shows this split loses the upper bound on an economy
frame. Returns `Inf` radii where the orthogonality condition fails.
"""
function _rump2011_thm3_1(A::BallMatrix{T}) where {T}
    m, n = size(A)
    m == n ||
        throw(ArgumentError("_rump2011_thm3_1 needs a square matrix: Theorem 3.1 is stated for " *
                            "square frames, and Remark 3 of Miyajima (2014) shows the economy " *
                            "form of this split loses the upper bound"))
    F = _svd_candidate(A)
    F === nothing && return [Ball(zero(T), T(Inf)) for _ in 1:n]
    Ub, Vb = BallMatrix(F.U), BallMatrix(F.V)
    alpha = upper_bound_L2_opnorm(Ub' * Ub - I)
    beta = upper_bound_L2_opnorm(Vb' * Vb - I)
    (alpha < 1 && beta < 1) || return [Ball(F.S[i], T(Inf)) for i in 1:n]

    Sig = Ub' * A * Vb                       # Sigma = U^H A V
    D = T[abs(mid(Sig)[i, i]) for i in 1:n]
    Dr = T[rad(Sig)[i, i] for i in 1:n]
    # E is Sigma off the diagonal; the diagonal radius of Sigma is charged to |D_ii| instead
    Em = copy(mid(Sig))
    Er = copy(rad(Sig))
    for i in 1:n
        Em[i, i] = zero(eltype(Em))
        Er[i, i] = zero(T)
    end
    normE = upper_bound_L2_opnorm(BallMatrix(Em, Er))

    den_lo = setrounding(T, RoundUp) do
        sqrt((one(T) + alpha) * (one(T) + beta))
    end
    den_hi = setrounding(T, RoundDown) do
        sqrt((one(T) - alpha) * (one(T) - beta))
    end
    ord = sortperm(D; rev = true)
    return [_ball_from_bounds(
                setrounding(T, RoundDown) do
                    (D[p] - Dr[p] - normE) / den_lo
                end,
                setrounding(T, RoundUp) do
                    (D[p] + Dr[p] + normE) / den_hi
                end, T) for p in ord]
end
