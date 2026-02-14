export Enclosure, bound_enclosure

struct Enclosure
    λ::Any
    points::Vector{ComplexF64}
    bounds::Vector{Ball{Float64, Float64}}
    radiuses::Vector{Float64}
    loop_closure::Bool
end

function _compute_exclusion_circle(T, λ, r; max_steps, rel_steps, svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    return _compute_exclusion_set(T, r; max_steps, rel_steps, λ, svd_method, apply_vbd)
end

function _compute_exclusion_circle_level_set_ode(T,
        λ,
        ϵ;
        max_steps,
        rel_steps,
        max_initial_newton,
        svd_method::SVDMethod = MiyajimaM1(),
        apply_vbd::Bool = false)
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

    return _compute_exclusion_circle(T, λ, r; max_steps, rel_steps, svd_method, apply_vbd)
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
        max_initial_newton,
        svd_method::SVDMethod = MiyajimaM1(),
        apply_vbd::Bool = false)
    out_z = []
    out_bound = []
    out_radiuses = []

    z = λ + ϵ

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

    pearl_radius = r * rel_pearl_size
    #@info "pearl radius" pearl_radius

    dist_points = (pearl_radius * 8) / 5

    #@info "distance between points" dist_points
    # this N bounds from above 2π/dist_points , i.e., the number of equispaced
    # points on the circumference

    N = ceil(8 * r / dist_points)

    # #@info N
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
    return _certify_circle(T, λ, r, N; svd_method, apply_vbd)
end

function _certify_circle(T, λ, r, N; svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    out_z = []
    out_bound = []
    out_radiuses = []

    pearl_radius = 5 * (r * 2 * π / N) / 8

    for j in 0:(N - 1)
        z = λ + r * exp(2 * π * im * j / N)
        push!(out_z, z)

        K = svd(T - z * I)
        z_ball = Ball(z, pearl_radius)

        bound = _certify_svd(BallMatrix(T) - z_ball * I, K, svd_method; apply_vbd)[end]
        push!(out_bound, bound)
        push!(out_radiuses, pearl_radius)
    end
    return Enclosure(λ, out_z, out_bound, out_radiuses, true)
end

function _compute_exclusion_set(T, r; max_steps, rel_steps, λ = 0 + im * 0, svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
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
        bound = _certify_svd(BallMatrix(T) - z_ball * I, K, svd_method; apply_vbd)[end]
        push!(out_bound, bound)
        push!(out_radiuses, r_guaranteed)
        #print("test")

        ##@info "test"
        ##@info "r_guarantee", r_guaranteed
        ##@info "r_guarantee_1", r_guaranteed_1
        ##@info "dist to start", abs(z_old-z0)

        loop_closure = abs(z_old - z0) < r_guaranteed + r_guaranteed_1

        if t_step > 10 && loop_closure
            #@info t_step, "Loop closure"
            break
        end
    end
    return Enclosure(λ, out_z, out_bound, out_radiuses, loop_closure)
end

function bound_resolvent(E::Enclosure)
    min_sing_val = minimum([@down x.c - x.r for x in E.bounds])
    return @up 1.0 / min_sing_val
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

function check_loop(λ, E::Enclosure)
end

function _follow_level_set(z::ComplexF64, τ::Float64, K::SVD)
    u = K.U[:, end]
    v = K.V[:, end]
    σ = K.S[end]

    # follow the level set
    grad = v' * u
    ort = im * grad
    z = z + τ * ort / abs(ort)

    return z, σ
end

function _newton_step(z, K::SVD, ϵ)
    u = K.U[:, end]
    v = K.V[:, end]
    σ = K.S[end]

    # gradient descent, better estimate
    z = z + (σ - ϵ) / (u' * v)
    return z, σ
end

# function newton_level_set(z, T, ϵ; τ=ϵ / 16)
#     K = svd(z * I - T)
#     return _newton_step(z, K, ϵ, τ)
# end

function _compute_enclosure_eigval(T, λ, ϵ; max_initial_newton, max_steps, rel_steps, svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    #@info "Enclosing ", λ
    #@info "Level set", ϵ

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
        bound = _certify_svd(BallMatrix(T) - z_ball * I, K, svd_method; apply_vbd)[end]
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

# function _certify_circle(T, r1, r, ϵ)

#     out_z = []
#     out_bound = []

#     N = ceil(2*π*r1/(r*ϵ))

#     @info N

#     dθ = 2*π/N

#     z = r1*exp(0)

#     for i in 0:N
#         z_old = z
#         z = r1*exp(im*i*dθ)

#         K = svd(T - z * I)

#         push!(out_z, z)

#         z_ball = Ball(z_old, 1.5 * abs(z_old - z))
#         bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
#         push!(out_bound, bound)
#     end

#     return (out_z, out_bound)
# end

"""
    compute_enclosure(A::BallMatrix, r1, r2, ϵ; max_initial_newton = 30,
            max_steps = Int64(ceil(256 * π)), rel_steps = 16,
            svd_method = MiyajimaM1(), apply_vbd = false)

    Given a BallMatrix `A`, this method follows the level lines of level `ϵ`
around the eigenvalues with modulus bound between `r1` and `r2`.

The keyword arguments
    - max_initial_newton: maximum number of newton steps to reach the level lines
    - max_steps: maximum number of steps following the contour
    - rel_steps: relative integration step for the Euler method
    - svd_method: SVD certification method (MiyajimaM1(), MiyajimaM4(), or RumpOriginal())
    - apply_vbd: whether to apply verified block diagonalization for tighter bounds

The method outputs an array of tuples:
    - the first element is the eigenvalue we are enclosing
    (in the case of the excluding circles, it is 0.0 or the maximum modulus of the eigenvalues)
    - the second element is an upper bound on the resolvent norm
    - the third element is a list of points on the enclosing line; the resolvent is rigorously
    bound on circles centered at each point and of radius 5/8 the distance to the previous point
"""
function compute_enclosure(A::BallMatrix{T}, r1, r2, ϵ; max_initial_newton = 30,
        max_steps = Int64(ceil(256 * π)), rel_steps = 16,
        svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false) where T
    if real(T) === BigFloat
        @warn "compute_enclosure casts BigFloat matrices to Float64 for the Schur " *
              "decomposition. Eigenvalue information below ~1e-16 will be lost." maxlog=1
    end
    F = schur(Complex{Float64}.(A.c))

    bZ = BallMatrix(F.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(F.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    #@info "Schur unitary error", errF
    #@info "Schur reconstruction error", errT

    eigvals = diag(F.T)[[r1 < abs(x) < r2 for x in diag(F.T)]]

    #@info "Certifying around", eigvals

    output = []

    for λ in eigvals
        E = _compute_enclosure_eigval(F.T, λ, ϵ; max_initial_newton,
            max_steps, rel_steps, svd_method, apply_vbd)

        # bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])

        # if bound < 0.0
        #     @warn "Smaller rel_step required"
        # end

        # @info "resolvent upper bound", bound
        # @info "σ", bounds[i]

        push!(output, E)
    end

    # encloses the eigenvalues inside r1
    eigvals_smaller_than_r1 = diag(F.T)[[abs(x) < r1 for x in diag(F.T)]]

    if !isempty(eigvals_smaller_than_r1)
        @info "Computing exclusion circle ", r1

        E = _compute_exclusion_set(F.T, r1; max_steps, rel_steps, svd_method, apply_vbd)
        #bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        #@info bound, i
        #@info "σ", bounds[i]
        #@info bound_resolvent(E)

        push!(output, E)
    end

    # # encloses the eigenvalues outside r2
    eigvals_bigger_than_r2 = diag(F.T)[[abs(x) > r2 for x in diag(F.T)]]

    if !isempty(eigvals_bigger_than_r2)
        #@info "Computing exclusion circle ", r2

        E = _compute_exclusion_set(F.T, r2; max_steps, rel_steps, svd_method, apply_vbd)
        #max_abs_eigenvalue = maximum(abs.(diag(F.T)))
        #bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        #@info bound, i
        #@info "σ", bounds[i]

        push!(output, E)
        #     r = minimum([abs(λ)-r2 for λ in eigvals_bigger_than_r2])/5
        #     @info "Gap between r2, $r2 and smallest eigenvalue outside, $r"
        #     curve, bound = _certify_circle(F.T, r2, r, ϵ)
        #     push!(output, (max_abs_eigenvalue, curve, bound))
    end

    return output
end
