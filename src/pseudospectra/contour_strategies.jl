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

function _compute_central_exclusion_circle(T, r; max_steps, rel_steps, λ = 0 + im * 0)
    eigvals = diag(T)

    out_z = []
    out_bound = []
    out_radiuses = []

    loop_closure = false

    z = λ + r
    z0 = z
    r_guaranteed_1 = 0.0

    r_guaranteed = r_guaranteed_1

    for t_step in 1:max_steps
        z_old = z
        r_old = r_guaranteed

        K = svd(T - z * I)

        τ = minimum(abs.(eigvals .- z)) / rel_steps

        z = z + τ * im * (z - λ) / abs(z - λ)
        z = z - (abs(z - λ)^2 - r^2) / conj(z - λ)

        push!(out_z, z)

        r_guaranteed = 5 * abs(z_old - z) / 8

        if t_step == 1
            r_guaranteed_1 = r_guaranteed
        end

        if t_step > 1
            @assert r_guaranteed + r_old > abs(z_old - z)
        end

        # we certify in a ball around z_old
        z_ball = Ball(z_old, r_guaranteed)
        bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
        push!(out_bound, bound)
        push!(out_radiuses, r_guaranteed)

        loop_closure = abs(z_old - z0) < r_guaranteed + r_guaranteed_1

        if t_step > 10 && loop_closure
            @info t_step, "Loop closure"
            break
        end
    end
    return Enclosure(λ, out_z, out_bound, out_radiuses, loop_closure)
end

function _compute_enclosure_eigval(T, λ, ϵ; max_initial_newton, max_steps, rel_steps)
    eigvals = diag(T)

    out_z = []
    out_bound = []
    radiuses = []
    log_z = []

    z = λ + 4 * sign(real(λ)) * ϵ

    # we first use the newton method to approach the level set
    for j in 1:max_initial_newton
        K = svd(T - z * I)
        z, σ = _newton_step(z, K, ϵ)

        if (σ - ϵ) < ϵ / 256
            break
        end
    end

    # for j in 1:max_initial_newton
    #     K = svd(T - z * I)
    #     τ = minimum(abs.(eigvals .- z))/rel_steps
    #     z, σ = _follow_level_set(z, τ, K)
    #     z, σ = _newton_step(z, K, ϵ, τ)
    # end

    z0 = z
    r_guaranteed_1 = 0.0

    #push!(out_z, z)
    #push!(log_z, log(z - λ))

    for t_step in 1:max_steps

        #@info t_step, max_steps

        z_old = z

        K = svd(T - z * I)

        τ = minimum(abs.(eigvals .- z)) / rel_steps

        z, σ = _follow_level_set(z, τ, K)
        z, σ = _newton_step(z, K, ϵ)

        #        @info σ
        push!(out_z, z)
        push!(log_z, log(z - λ))

        r_guaranteed = 5 * abs(z_old - z) / 8

        if t_step == 1
            r_guaranteed_1 = r_guaranteed
        end

        # we certify the SVD on a ball around z_old

        z_ball = Ball(z_old, r_guaranteed)
        bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
        push!(out_bound, bound)
        push!(radiuses, r_guaranteed)

        # if the first point is inside the certification ball, we have found a loop closure
        #@info "r_guaranteed+r_guaranteed_1", r_guaranteed+r_guaranteed_1, "dist to start", abs(z_old-z0)

        angle = 0.0
        if length(log_z) > 2
            test = [log_z[i + 1] - log_z[i] for i in 1:(length(log_z) - 1)]

            angle = imag(sum(test))
        end
        #@info angle

        check_loop_closure = abs(z_old - z0) < (r_guaranteed + r_guaranteed_1)

        if t_step > 10 && check_loop_closure
            #@info t_step, "Loop closure"
            break
        end
    end
    return Enclosure(λ, out_z, out_bound, radiuses, true)
end

function _compute_enclosure_ode(T, λ, ϵ;
        max_initial_newton,
        max_steps,
        max_adaptive = 10,
        target = 128 * ϵ,
        α = 1.1)
    out_z = [] # contains the center of the pearl_necklace
    out_radius = [] # contains the radiuses of the pearl necklace
    out_sing = [] # contains the enclosure of the minimum singular value

    cross_x = 0
    cross_y = 0

    up = 0
    right = 0

    z0 = λ + 2 * sign(real(λ)) * target

    @info z0, abs(z0 - λ) / target

    K = svd(T - z0 * I)
    σ = K.S[end]

    for _ in 1:max_initial_newton
        if abs(σ - target) < target / 256
            break
        end

        z0 = _newton_step(z0, K, target)
        K = svd(T - z0 * I)
        σ = K.S[end]
        @info z0, σ, abs(z0 - λ) / target
    end

    up0 = imag(z0 - λ) > 0
    right0 = real(z0 - λ) > 0

    R = abs(z0 - λ)
    L = σ - ϵ

    r0 = L
    @info "We start at $z0 with σ $σ and target $target, r0 $r0"

    @info R / r0

    # we start by finding a r0 that makes possible to certify
    # the initial ball
    for _ in 1:max_initial_newton
        z_ball = Ball(z0, r0)
        σ_cert = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]

        if σ_cert.c - σ_cert.r > ϵ
            push!(out_z, z0)
            push!(out_radius, r0)
            push!(out_sing, σ_cert)
            break
        end
        r0 = r0 / 2
    end

    @warn "Expected steps" (2 * pi * R)/r0

    z = z0
    r = r0

    up = up0
    right = right0

    @info "z0 $z0 r0 $r0, $(2*π*abs(z0-λ)/r0), $σ,  "

    for i in 1:max_steps
        if i % 10 == 0
            @info i
        end
        # this block computes the next point adaptively
        # we first try the new radius can be taken bigger than the actual radius
        # r_new = α * r
        r_new = r
        for _ in 1:max_adaptive
            τ = 0.75 * (r + r_new)
            #@info "r, r0", r, r0
            #@info τ, 0.75 * (2 * r0)
            z_new = _follow_level_set(z, τ, K)
            K = svd(T - z_new * I)
            z_new = _newton_step(z_new, K, target)

            K = svd(T - z_new * I)
            σ = K.S[end]

            z_ball = Ball(z_new, r_new)
            σ_cert = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
            if σ_cert.c - σ_cert.r > ϵ && abs(z_new - z) < r_new + r
                #@info (z_new - λ) / abs(z_new - λ), r_new, abs(σ_cert.c - target)
                #@info abs(z_new - λ)
                #@info i, (2 * π * R) / abs(z - z_new)

                push!(out_z, z_new)
                push!(out_radius, r_new)
                push!(out_sing, σ_cert)

                up_new = imag(z_new - λ) > 0
                right_new = real(z_new - λ) > 0

                if right_new == true
                    if up_new != up
                        cross_x += sign(imag(z_new - z))
                        #@info cross_x
                    end
                end

                if up_new == true
                    if right_new != right
                        cross_y += sign(imag(z_new - z))
                        #@info cross_y
                    end
                end

                z = z_new
                r = r_new
                up = up_new
                right = right_new

                break
            end
            #@info "adaptive, smaller radius"
            r_new = r_new / α
        end

        if abs(z - z0) < r + r0 && abs(cross_x) == 1 && abs(cross_y) == 1
            @info "Loop closure"
            break
        end
    end
    return Enclosure(λ, out_z, out_sing, out_radius, true)
end
