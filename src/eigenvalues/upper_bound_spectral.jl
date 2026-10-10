"""
    collatz_upper_bound(A::BallMatrix; iterates = 10)

A rigorous upper bound on the spectral radius of `|M|` for every `M` in the ball `A`, hence on the
spectral radius of every such `M`, by Collatz's inclusion: for a nonnegative matrix `B`,

    ρ(B) ≤ max_i (Bx)_i / x_i      for every positive vector x,

applied to `B = `[`upper_abs`](@ref)`(A)` with `x` chosen by `iterates` power iterations from the
vector of ones. Any number of iterations gives a valid bound; the result is `Inf`, not `NaN`, when
the quotient is not finite (the vector is kept strictly positive, so a zero row or a nilpotent
matrix is handled).

# References

L. Collatz, *Einschließungssatz für die charakteristischen Zahlen von Matrizen*, Math. Z. **48**
(1942) 221-226.

S. M. Rump, *Verified bounds for singular values, in particular for the spectral norm of a matrix
and its inverse*, BIT Numer. Math. **51** (2011) 367-384, doi 10.1007/s10543-010-0294-0, where the
inclusion is used in the form (3.3) for `BᵀB`; see [`collatz_upper_bound_L2_opnorm`](@ref).
"""
function collatz_upper_bound(A::BallMatrix{T}; iterates = 10) where {T}
    m, k = size(A)
    m == k || throw(DimensionMismatch("collatz_upper_bound needs a square matrix, got $m by $k"))
    absA = upper_abs(A)
    return _collatz_quotient_up(x -> absA * x, m, real(eltype(A.c)), iterates)
end
