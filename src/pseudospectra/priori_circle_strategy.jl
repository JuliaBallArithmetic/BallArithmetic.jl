function _compute_exclusion_circle(T, λ, r; max_steps, rel_steps)
    return _compute_exclusion_set(T, r; max_steps, rel_steps, λ)
end

function _compute_exclusion_circle_level_set_ode(T,
        λ,
        ϵ;
        max_steps,
        rel_steps,
        max_initial_newton)
    z = λ + ϵ
    #@info z

    for j in 1:max_initial_newton
        #@info "Newton step $j"
        K = svd(T - z * I)
        z, σ = _newton_step(z, K, ϵ)
        #@info σ

        if (σ - ϵ) / ϵ < 1 / 256
            break
        end
    end
    r = abs(λ - z)

    return _compute_exclusion_circle(T, λ, r; max_steps, rel_steps)
end

"""
_compute_exclusion_circle_level_set_priori(T,
λ,
ϵ;
rel_pearl_size,
max_initial_newton)
This method bounds the resolvent on a circle centered at `λ`
that intersects in at least one point `z0` the `ϵ` level set.
This intersection is found by a Newton step, and fixes the radius
of the circle,

The value of `rel_pearl_size` gives us the relative radius of
the pearls with respect to the radius of the circle

Some rule of thumbs for the number of SVD computations:
if rel_pearl_size is 1/32, we are going to compute and certify 160 svds,
if rel_pearl_size is 1/64 we are going to compute and certify 320 svds.
In other words, the time of the computation scales linearly with the quality
of the pearl necklace
"""
function _compute_exclusion_circle_level_set_priori(T,
        λ,
        ϵ;
        rel_pearl_size,
        max_initial_newton)
    z = λ + ϵ

    for j in 1:max_initial_newton
        @debug "Newton step $j"
        K = svd(T - z * I)
        z = _newton_step(z, K, ϵ)
        σ = K.S[end]

        @info σ

        if (σ - ϵ) / ϵ < 1 / 256
            break
        end
    end

    r = abs(λ - z)

    @info "radius" r

    pearl_radius = r * rel_pearl_size
    @debug "pearl radius" pearl_radius

    dist_points = (pearl_radius * 8) / 5

    @debug "distance between points" dist_points
    # this N bounds from above 2π/dist_points , i.e., the number of equispaced
    # points on the circumference

    N = ceil(8 * r / dist_points)

    @info "number  of steps" N
    # for j in 0:(N - 1)
    #     z = λ + r * exp(2 * π * im * j / N)
    #     push!(out_z, z)

    #     K = svd(T - z * I)
    #     z_ball = Ball(z, pearl_radius)

    #     bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
    #     push!(out_bound, bound)
    #     push!(out_radiuses, pearl_radius)
    # end

    #return Enclosure(λ, out_z, out_bound, out_radiuses, true)
    return _certify_circle(T, λ, r, N)
end

function _certify_circle(T, λ, r, N)
    out_z = []
    out_bound = []
    out_radiuses = []
    out_gradient = []

    θ = 2 * pi / N
    l = r * sqrt(sin(θ)^2 + (1 - cos(θ))^2)
    @info l

    pearl_radius = (513 / 1024) * l

    for j in 0:(N - 1)
        z = λ + r * exp(im * j * θ)
        push!(out_z, z)

        K = svd(T - z * I)
        z_ball = Ball(z, pearl_radius)

        u = K.U[:, end]
        v = K.V[:, end]

        bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
        push!(out_bound, bound)
        push!(out_radiuses, pearl_radius)
        push!(out_gradient, (u' * v))
    end
    return Enclosure(λ, out_z, out_bound, out_radiuses, out_gradient, true)
end
