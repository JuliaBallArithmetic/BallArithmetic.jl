##
# routines to give a rigorous solution of the real linear system Ax=B

# this file is sourced from https://github.com/JuliaIntervals/IntervalLinearAlgebra.jl/blob/main/src/linear_systems/verify.jl

"""
    epsilon_inflation(A::BallMatrix{T}, b::BallVector{T};
                      r=0.1, ϵ=1e-20, iter_max=20) where {T<:AbstractFloat}

Gives an enclosure of the solution of the square linear system ``Ax=b``
using the ϵ-inflation algorithm,  see algorithm 10.7 of [[RUM10]](@ref)

### Input

* `A`        -- square matrix of size n × n
* `b`        -- vector of length n or matrix of size n × m
* `r`        -- relative inflation, default 10%
* `ϵ`        -- absolute inflation, default 1e-20
* `iter_max` -- maximum number of iterations

### Output

* `x`    -- enclosure of the solution of the linear system
* `cert` -- Boolean flag, if `cert==true`, then `x` is *certified* to contain the true
solution of the linear system, if `cert==false`, then the algorithm could not prove that ``x``
actually contains the true solution.

### Algorithm

Given the real system ``Ax=b`` and an approximate solution ``̃x``, we initialize
``x₀ = [̃x, ̃x]``. At each iteration the algorithm computes the inflation

``y = xₖ * [1 - r, 1 + r] .+ [-ϵ, ϵ]``

and the update

``xₖ₊₁ = Z + (I - CA)y``,

where ``Z = C(b - Ax₀)`` and ``C`` is an approximate inverse of ``A``. If the condition
``xₖ₊₁ ⊂ y `` is met, then ``xₖ₊₁`` is a proved enclosure of ``A⁻¹b`` and `cert` is set to
true. If the condition is not met by the maximum number of iterations, the
latest computed enclosure is returned, but ``cert`` is set to false, meaning the algorithm
could not prove that the enclosure contains the true solution. For interval systems,
``̃x`` is obtained considering the midpoint of ``A`` and ``b``.

### Notes

- This algorithm is meant for *real* linear systems, or interval systems with
very tiny intervals. For interval linear systems with wider intervals, see the
[`solve`](@ref) function.

### Examples

"""
function epsilon_inflation(A::BallMatrix{T}, b::BallVector{T};
        r = 0.1, ϵ = 1e-20, iter_max = 20) where {T <: AbstractFloat}
    r1 = Ball(1, r)
    ϵ1 = fill(Ball(0, ϵ), length(b))
    R = inv(mid(A))

    C = I - R * A
    xs = R * mid(b)
    z = R * (b - (A * BallVector(xs)))
    x = z

    for _ in 1:iter_max
        y = r1 * x + ϵ1
        x = z + C * y
        if all(in.(x, y))
            return xs + x, true
        end
    end
    return xs + x, false
end

function epsilon_inflation(A::BallMatrix{T}, B::BallMatrix{T};
        r = 0.1, ϵ = 1e-20, iter_max = 20) where {T <: AbstractFloat}
    r1 = Ball(1, r)

    m, k = size(B)

    ϵ1 = fill(Ball(0, ϵ), length(k))
    R = inv(mid(A))

    C = I - R * A
    xs = R * mid(B)
    z = R * (B - (A * BallMatrix(xs)))
    x = z

    for _ in 1:iter_max
        y = r1 * x + ϵ1
        x = z + C * y
        if all(in.(x, y))
            return xs + x, true
        end
    end
    return xs + x, false
end
