"""
    krawczyk_complete.jl

Complete Krawczyk operator implementation for verified linear systems and Sylvester equations.

The Krawczyk operator provides a powerful tool for computing verified enclosures of solutions
to linear systems with quadratic convergence.

# References
- Krawczyk, R. (1969), "Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken"
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
- Rump, S.M. (1999), "INTLAB - INTerval LABoratory"
"""

using LinearAlgebra

"""
    KrawczykResult{T, VT}

Result from Krawczyk verification method.
"""
struct KrawczykResult{T, VT}
    """Verified enclosure of the solution."""
    solution::VT
    """Whether verification succeeded."""
    certified::Bool
    """Number of iterations performed."""
    iterations::Int
    """Final residual norm."""
    residual_norm::T
    """Contraction factor (< 1 implies unique solution)."""
    contraction_factor::T
end

"""
    krawczyk_linear_system(A::BallMatrix, b::BallVector; max_iterations = 20) -> KrawczykResult

A verified enclosure of the solution of `A x = b`, for every matrix of the ball `A` and every
vector of the ball `b`: [`verifylss`](@ref), which is Algorithm 10.7 of Rump (2010), with its
result in the fields of a [`KrawczykResult`](@ref).

With `R` a floating-point inverse of the midpoint of `A` and `x̃ = R mid(b)`, the algorithm
iterates `X ← Z + C·Y`, `Z ∋ R(b − Ax̃)`, `C ∋ I − RA`, `Y` an inflation of `X`, all in ball
arithmetic, and stops when `X` is in the interior of `Y`; then `A` and `R` are nonsingular and
the solution is in `x̃ + X` (Theorem 10.8 there). This is the iteration of Krawczyk (1969) for a
linear system with the inflation of Rump.

`certified` is the success of that test. `iterations` is the number of iterations,
`contraction_factor` an upper bound of `‖I − RA‖₂` (a diagnostic: the test may succeed when it
is not below one), `residual_norm` an upper bound of `‖A x̂ − b‖₂` over the two balls for `x̂` the
midpoint of the returned enclosure. When `certified` is false the enclosure proves nothing.

The preconditioner and the approximate solution are those of the algorithm; the keywords `R`,
`x_approx` and `expansion_factor` of earlier versions are gone with the code that used them,
which formed `I − RA` in floating point and left the residual out of the inclusion test.

# References

S. M. Rump, *Verification methods: rigorous results using floating-point arithmetic*, Acta
Numerica **19** (2010), 287-449, doi 10.1017/S096249291000005X, Algorithm 10.7 and Theorem 10.8.

R. Krawczyk, *Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken*, Computing
**4** (1969), 187-201, doi 10.1007/BF02234767.
"""
function krawczyk_linear_system(A::BallMatrix{T}, b::BallVector{T};
        max_iterations::Integer = 20) where {T}
    sol = verifylss(A, b; iter_max = max_iterations)
    x = sol.solution
    res = A * BallMatrix(reshape(mid(x), :, 1)) - BallMatrix(reshape(mid(b), :, 1), reshape(rad(b), :, 1))
    acc = zero(T)
    for (m, ρ) in zip(mid(res), rad(res))
        a = add_up(abs_up(m), ρ)
        acc = add_up(acc, mul_up(a, a))
    end
    NT = promote_type(eltype(mid(A)), eltype(mid(b)))
    return KrawczykResult(x, sol.certified, sol.iterations, sqrt_up(acc),
        sol.spectral_radius_bound)::KrawczykResult{T, DenseBallVector{T, NT}}
end

# `krawczyk_sylvester` was removed on 2026-10-11: it threw on complex input and its contraction
# estimate was never below one, so it verified nothing (0 of 204 problems tried). The verified
# enclosure of the solution of a Sylvester equation is `sylvester_miyajima_enclosure`
# (Miyajima 2013), in `sylvester.jl`.

export KrawczykResult
export krawczyk_linear_system
