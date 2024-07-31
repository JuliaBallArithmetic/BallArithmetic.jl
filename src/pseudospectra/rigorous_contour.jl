export Enclosure, bound_enclosure

struct Enclosure
    λ::Any
    points::Vector{ComplexF64}
    bounds::Vector{Ball{Float64, Float64}}
    radiuses::Vector{Float64}
    gradient::Vector{ComplexF64}
    loop_closure::Bool
end

function Enclosure(λ, points, bounds, radiuses, loop_closure)
    return Enclosure(λ,
        points,
        bounds,
        radiuses,
        zeros(ComplexF64, length(points)),
        loop_closure)
end

include("utilities.jl")
include("priori_circle_strategy.jl")
include("contour_strategies.jl")
include("refine_enclosure.jl")
include("full_strategy.jl")

function bound_resolvent(E::Enclosure, errF, errT, norm_Z, norm_Z_inv)
    N = length(E.points)
    errZ = @up (norm_Z.c + norm_Z.r - 1.0)
    errZinv = @up (norm_Z_inv.c + norm_Z_inv.r - 1.0)

    ϵ = Ball(maximum([errF; errT; errZ; errZinv]))

    val = 0.0

    for i in 1:N
        sigma = E.bounds[i]
        z = Ball(E.points[i], E.radiuses[i])
        boundT = 1 / sigma
        @info boundT
        check = ϵ * abs(z) * (1.0 + ϵ)^2 * boundT
        @info check
        if @up check.c + check.r <= 0.5
            temp = (1.0 + ϵ)^2 * (1.0 + 2.0 * ϵ * abs(z)) * boundT
            bound = temp / (1 - ϵ * temp)
            val = max(val, @up bound.c + bound.r)
        else
            @warn "We need a better Schur decomposition", ϵ
        end
    end
    return val
    #min_sing_val = minimum([@down x.c - x.r for x in E.bounds])
    #schur_bound = @up 1.0 / min_sing_val
end

function check_enclosure(E::Enclosure)
    check_overlap = true
    for i in 1:(length(E.points) - 1)
        check_overlap = abs(E.points[i + 1] - E.points[i]) <
                        E.radiuses[i] + E.radiuses[i + 1]
        if check_overlap == false
            return false
        end
    end
    check_overlap = abs(E.points[1] - E.points[end]) < E.radiuses[1] + E.radiuses[end]
    return check_overlap
end
