##
# Verified solution of the linear system A x = b, A and b balls.
#
# The algorithm is `_rump2010_alg10_7`, Algorithm 10.7 with Theorem 10.8 of
#
#   @Article{Rump2010,
#     author    = {Rump, Siegfried M.},
#     title     = {Verification methods: Rigorous results using floating-point arithmetic},
#     journal   = {Acta Numerica},
#     year      = {2010},
#     volume    = {19},
#     pages     = {287--449},
#     doi       = {10.1017/S096249291000005X},
#     publisher = {Cambridge University Press},
#   }
#
# which is the algorithm behind INTLAB's `verifylss`, and which Rump attributes to Rump (1980).
# The strict interior containment the iteration tests is `in0` of footnote 19 there.
#
# The skeleton of this file came from IntervalLinearAlgebra.jl,
# https://github.com/JuliaIntervals/IntervalLinearAlgebra.jl/blob/main/src/linear_systems/verify.jl,
# whose version departed from Algorithm 10.7 in two places, both corrected here: the iteration
# matrix was formed as `I - R*mid(A)` in floating point, which is not an enclosure of `I - RA`,
# and the containment test was `in`, which is not strict.

"""
    VerifyLssResult{T, VT}

Result from epsilon-inflation linear system verification.

# Fields
- `solution::VT`: Enclosure of the solution (BallVector or BallMatrix)
- `certified::Bool`: Whether the solution is mathematically certified
- `iterations::Int`: Number of iterations performed
- `spectral_radius_bound::T`: Upper bound on ‖I - RA‖₂ (convergence requires < 1)
- `condition_number::T`: Approximate condition number of A

# Convergence Diagnostics

If `certified == false`, check:
- `spectral_radius_bound ≥ 1`: Iteration matrix doesn't contract, method cannot converge
- `condition_number > 1/eps(T)`: Matrix is numerically singular
- `iterations == iter_max`: May need more iterations or smaller inflation parameters
"""
struct VerifyLssResult{T<:AbstractFloat, VT}
    solution::VT
    certified::Bool
    iterations::Int
    spectral_radius_bound::T
    condition_number::T
end

# For backward compatibility, allow tuple destructuring
function Base.iterate(r::VerifyLssResult, state=1)
    if state == 1
        return (r.solution, 2)
    elseif state == 2
        return (r.certified, 3)
    else
        return nothing
    end
end
Base.length(::VerifyLssResult) = 2

"""
    _rump2010_alg10_7(A::BallMatrix{T}, b::BallVector{T};
                      r=0.1, ϵ=1e-20, iter_max=20) where {T<:AbstractFloat}

Algorithm 10.7 of Rump (2010), the algorithm behind INTLAB's `verifylss`: an enclosure of the
solution of the square linear system ``Ax = b`` by epsilon-inflation. Not exported; reached
through [`verifylss`](@ref) with `method = :rump2010`.

# Input

* `A`        -- square matrix of size n × n
* `b`        -- vector of length n or matrix of size n × m
* `r`        -- relative inflation, default 10%
* `ϵ`        -- absolute inflation, default 1e-20
* `iter_max` -- maximum number of iterations

# Output

Returns an [`VerifyLssResult`](@ref) containing:
* `solution` -- enclosure of the solution of the linear system
* `certified` -- Boolean flag, if `true`, then solution is *certified* to contain the true
  solution; if `false`, certification failed (check diagnostics)
* `iterations` -- number of iterations performed
* `spectral_radius_bound` -- upper bound on ‖I - RA‖₂ (must be < 1 for convergence)
* `condition_number` -- approximate condition number of A

The result can be destructured as `(x, cert) = verifylss(A, b)`.

# Algorithm

This is Algorithm 10.7 of

> S. M. Rump, *Verification methods: rigorous results using floating-point arithmetic*,
> Acta Numerica **19** (2010) 287-449, doi 10.1017/S096249291000005X,

which Rump attributes to Rump (1980) and which is the algorithm behind INTLAB's `verifylss`,
transcribed step for step:

```
R  = inv(mid(A))                     # approximate inverse
xs = R * mid(b)                      # approximate solution
C  = I - R * A                       # iteration matrix, R against the BALL A
Z  = R * (b - A * xs)
X  = Z
for iter in 1:15
    Y = X * [0.9, 1.1] + [-1e-20, 1e-20]
    X = Z + C * Y
    in0(X, Y) && return xs + X
end
```

Two points decide whether it proves anything, and both are easy to get wrong.

`C` is `I` minus `R` times the **ball** `A`, not `I - R*mid(A)` in floating point: the product
`R*mid(A)` is itself rounded, and an enclosure of `I - RA` has to carry that rounding. On a
50x50 `randn` matrix with an exact midpoint, the float form `BallMatrix(I - R*mid(A),
abs.(R)*rad(A))` claims radius `0.0` while the entries of `I - RA` differ from its midpoint by
up to `4.9e-15`, so it is not an enclosure and nothing downstream of it is proved. This
routine previously did that.

The containment test is [`in0`](@ref), strict interior containment, per Rump's footnote 19:
`in0(X, Y)` checks `X ⊂ int(Y)` componentwise. `in`, which allows equality, does not give the
contraction the argument rests on.

# What success proves

Theorem 10.8: if the iteration ends successfully then every matrix in the ball `A` is
nonsingular, and the returned enclosure contains
``Σ(A, b) = {x : Ax = b for some A ∈ A, b ∈ b}``. Nonsingularity is a conclusion, not an
assumption, which is why Rump uses this to enclose `W⁻¹BW` while proving `W` invertible.

The note following Theorem 10.8 records that the result holds for a matrix right-hand side
``b ∈ IR^{n×k}`` as well, which is the method taken by the `BallMatrix` right-hand side below.

# Convergence

An inclusion is computed essentially if and only if ``ρ(|I − RA|) < 1``;
`spectral_radius_bound` reports ``‖I − RA‖₂``, an upper bound for it, and a value ``≥ 1`` means
the iteration cannot contract.

# Example

```julia
A = BallMatrix(randn(5, 5))
b = BallVector(randn(5))
result = verifylss(A, b)
if result.certified
    println("Certified solution: ", result.solution)
else
    println("Verification failed:")
    println("  Spectral radius bound: ", result.spectral_radius_bound)
    println("  Condition number: ", result.condition_number)
end
```
"""
function _rump2010_alg10_7(A::BallMatrix{T}, b::BallVector{T};
        r = 0.1, ϵ = 1e-20, iter_max = 20) where {T <: AbstractFloat}
    n = size(A, 1)
    r1 = Ball(T(1), T(r))
    ϵ1 = fill(Ball(T(0), T(ϵ)), length(b))

    # Compute approximate inverse of midpoint
    A_mid = mid(A)
    R = try
        inv(A_mid)
    catch e
        if e isa SingularException
            # Matrix is exactly singular - return failed result with infinite diagnostics
            inf_solution = fill(Ball(T(0), T(Inf)), length(b))
            return VerifyLssResult(inf_solution, false, 0, T(Inf), T(Inf))
        end
        rethrow(e)
    end

    # C = I - R*A of Algorithm 10.7, with R against the BALL A so that the rounding of the
    # product is inside the enclosure. Computing I - R*mid(A) in floating point and calling
    # abs.(R)*rad(A) its radius is not an enclosure of I - RA, and every claim below rests on
    # this one being an enclosure.
    C = I - BallMatrix(R) * A
    spectral_radius = upper_bound_L2_opnorm(C)

    # Approximate condition number for diagnostics
    cond_A = opnorm(A_mid, 2) * opnorm(R, 2)

    # Warn if spectral radius bound suggests non-convergence
    if spectral_radius >= 1
        @warn "Epsilon-inflation unlikely to converge: ‖I - RA‖₂ ≥ $(spectral_radius) ≥ 1"
    end

    xs = R * mid(b)
    # R as a degenerate ball, which is exact: the float product Matrix*BallMatrix has no
    # rigorous method for complex entries, and R is a point matrix in any case
    z = BallMatrix(R) * (b - (A * BallVector(xs)))
    x = z

    iterations = 0
    for iter in 1:iter_max
        iterations = iter
        y = r1 * x + ϵ1
        x = z + C * y
        # in0, not in: Rump's footnote 19 requires X ⊂ int(Y) componentwise
        if in0(x, y)
            return VerifyLssResult(BallVector(xs + x), true, iterations,
                spectral_radius, cond_A)
        end
    end

    return VerifyLssResult(BallVector(xs + x), false, iterations, spectral_radius, cond_A)
end

function _rump2010_alg10_7(A::BallMatrix{T}, B::BallMatrix{T};
        r = 0.1, ϵ = 1e-20, iter_max = 20) where {T <: AbstractFloat}
    r1 = Ball(T(1), T(r))
    ϵ1 = fill(Ball(T(0), T(ϵ)), size(B))

    # Compute approximate inverse of midpoint
    A_mid = mid(A)
    R = try
        inv(A_mid)
    catch e
        if e isa SingularException
            # Matrix is exactly singular - return failed result with infinite diagnostics
            inf_solution = fill(Ball(T(0), T(Inf)), size(B))
            return VerifyLssResult(BallMatrix(inf_solution), false, 0, T(Inf), T(Inf))
        end
        rethrow(e)
    end

    # see the one-argument method: R against the BALL A, so the rounding of the product is
    # enclosed; the float form is not an enclosure of I - RA
    C = I - BallMatrix(R) * A
    spectral_radius = upper_bound_L2_opnorm(C)
    cond_A = opnorm(A_mid, 2) * opnorm(R, 2)

    if spectral_radius >= 1
        @warn "Epsilon-inflation unlikely to converge: ‖I - RA‖₂ ≥ $(spectral_radius) ≥ 1"
    end

    xs = R * mid(B)
    # R as a degenerate ball: exact, and the only form defined for complex entries
    z = BallMatrix(R) * (B - (A * BallMatrix(xs)))
    x = z

    iterations = 0
    for iter in 1:iter_max
        iterations = iter
        y = r1 * x + ϵ1
        x = z + C * y
        # in0, not in: Rump's footnote 19 requires X ⊂ int(Y) componentwise
        if in0(x, y)
            return VerifyLssResult(BallMatrix(xs + x), true, iterations,
                spectral_radius, cond_A)
        end
    end

    return VerifyLssResult(BallMatrix(xs + x), false, iterations, spectral_radius, cond_A)
end

"""
    verifylss(A::BallMatrix, b; method = :rump2010, r = 0.1, ϵ = 1e-20, iter_max = 20)
        -> VerifyLssResult

Verified enclosure of the solution of `A x = b`, with `b` a `BallVector` or a `BallMatrix`.
When the returned `certified` is `true`, every matrix in the ball `A` is nonsingular and the
enclosure contains the solution for every `A ∈ A` and `b ∈ b`; nonsingularity is a conclusion
of the method, not a hypothesis.

This function selects an algorithm and does nothing else. Each algorithm is a separate
unexported function named for the paper it implements:

| `method` | routine | what it is |
|---|---|---|
| `:rump2010` | [`_rump2010_alg10_7`](@ref) | Algorithm 10.7 of Rump (2010), the algorithm behind INTLAB's `verifylss` |

# Reference

S. M. Rump, *Verification methods: rigorous results using floating-point arithmetic*,
Acta Numerica **19** (2010) 287-449, doi 10.1017/S096249291000005X. Algorithm 10.7 and
Theorem 10.8; Rump attributes the algorithm to Rump (1980).
"""
function verifylss(A::BallMatrix, b; method::Symbol = :rump2010, kwargs...)
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("verifylss expects a square matrix"))
    size(A, 2) == size(b, 1) ||
        throw(DimensionMismatch("verifylss: A has $(size(A, 2)) columns, b has $(size(b, 1)) rows"))
    method === :rump2010 && return _rump2010_alg10_7(A, b; kwargs...)
    throw(ArgumentError("verifylss: unknown method $(repr(method)); " *
                        "the implemented methods are :rump2010"))
end
