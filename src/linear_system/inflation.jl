##
# routines to give a rigorous solution of the real linear system Ax=B

# this file is sourced from https://github.com/JuliaIntervals/IntervalLinearAlgebra.jl/blob/main/src/linear_systems/verify.jl

"""
    EpsilonInflationResult{T, VT}

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
struct EpsilonInflationResult{T<:AbstractFloat, VT}
    solution::VT
    certified::Bool
    iterations::Int
    spectral_radius_bound::T
    condition_number::T
end

# For backward compatibility, allow tuple destructuring
function Base.iterate(r::EpsilonInflationResult, state=1)
    if state == 1
        return (r.solution, 2)
    elseif state == 2
        return (r.certified, 3)
    else
        return nothing
    end
end
Base.length(::EpsilonInflationResult) = 2

"""
    epsilon_inflation(A::BallMatrix{T}, b::BallVector{T};
                      r=0.1, ϵ=1e-20, iter_max=20) where {T<:AbstractFloat}

Gives an enclosure of the solution of the square linear system ``Ax=b``
using the ϵ-inflation algorithm, see Algorithm 10.7 of Neumaier (1990).

# Input

* `A`        -- square matrix of size n × n
* `b`        -- vector of length n or matrix of size n × m
* `r`        -- relative inflation, default 10%
* `ϵ`        -- absolute inflation, default 1e-20
* `iter_max` -- maximum number of iterations

# Output

Returns an [`EpsilonInflationResult`](@ref) containing:
* `solution` -- enclosure of the solution of the linear system
* `certified` -- Boolean flag, if `true`, then solution is *certified* to contain the true
  solution; if `false`, certification failed (check diagnostics)
* `iterations` -- number of iterations performed
* `spectral_radius_bound` -- upper bound on ‖I - RA‖₂ (must be < 1 for convergence)
* `condition_number` -- approximate condition number of A

For backward compatibility, the result can be destructured as `(x, cert) = epsilon_inflation(A, b)`.

# Algorithm

Given the real system ``Ax=b`` and an approximate solution ``x̃``, we initialize
``x₀ = [x̃, x̃]``. At each iteration the algorithm computes the inflation

``y = xₖ * [1 - r, 1 + r] .+ [-ϵ, ϵ]``

and the update

``xₖ₊₁ = Z + (I - CA)y``,

where ``Z = C(b - Ax₀)`` and ``C`` is an approximate inverse of ``A``. If the condition
``xₖ₊₁ ⊂ y`` is met, then ``xₖ₊₁`` is a proved enclosure of ``A⁻¹b``.

# Convergence Requirements

The method converges if and only if ``ρ(I - RA) < 1`` where R is the approximate inverse.
This is checked and reported in `spectral_radius_bound`. If this bound is ≥ 1, the
iteration cannot contract and verification will fail.

# Example

```julia
A = BallMatrix(randn(5, 5))
b = BallVector(randn(5))
result = epsilon_inflation(A, b)
if result.certified
    println("Certified solution: ", result.solution)
else
    println("Verification failed:")
    println("  Spectral radius bound: ", result.spectral_radius_bound)
    println("  Condition number: ", result.condition_number)
end
```
"""
function epsilon_inflation(A::BallMatrix{T}, b::BallVector{T};
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
            return EpsilonInflationResult(inf_solution, false, 0, T(Inf), T(Inf))
        end
        rethrow(e)
    end

    # Compute convergence diagnostics
    # Spectral radius bound: ‖I - RA‖₂ must be < 1 for convergence
    C = BallMatrix(I - R * A_mid, abs.(R) * rad(A))
    spectral_radius = upper_bound_L2_opnorm(C)

    # Approximate condition number for diagnostics
    cond_A = opnorm(A_mid, 2) * opnorm(R, 2)

    # Warn if spectral radius bound suggests non-convergence
    if spectral_radius >= 1
        @warn "Epsilon-inflation unlikely to converge: ‖I - RA‖₂ ≥ $(spectral_radius) ≥ 1"
    end

    xs = R * mid(b)
    z = R * (b - (A * BallVector(xs)))
    x = z

    iterations = 0
    for iter in 1:iter_max
        iterations = iter
        y = r1 * x + ϵ1
        x = z + C * y
        if all(in.(x, y))
            return EpsilonInflationResult(xs + x, true, iterations, spectral_radius, cond_A)
        end
    end

    return EpsilonInflationResult(xs + x, false, iterations, spectral_radius, cond_A)
end

function epsilon_inflation(A::BallMatrix{T}, B::BallMatrix{T};
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
            return EpsilonInflationResult(BallMatrix(inf_solution), false, 0, T(Inf), T(Inf))
        end
        rethrow(e)
    end

    # Compute convergence diagnostics
    C = BallMatrix(I - R * A_mid, abs.(R) * rad(A))
    spectral_radius = upper_bound_L2_opnorm(C)
    cond_A = opnorm(A_mid, 2) * opnorm(R, 2)

    if spectral_radius >= 1
        @warn "Epsilon-inflation unlikely to converge: ‖I - RA‖₂ ≥ $(spectral_radius) ≥ 1"
    end

    xs = R * mid(B)
    z = R * (B - (A * BallMatrix(xs)))
    x = z

    iterations = 0
    for iter in 1:iter_max
        iterations = iter
        y = r1 * x + ϵ1
        x = z + C * y
        if all(in.(x, y))
            return EpsilonInflationResult(xs + x, true, iterations, spectral_radius, cond_A)
        end
    end

    return EpsilonInflationResult(xs + x, false, iterations, spectral_radius, cond_A)
end
