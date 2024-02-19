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

function _newton_step(z, K::SVD, ϵ, τ)
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

function _compute_enclosure_eigval(T, λ, ϵ; max_initial_newton, max_steps, rel_steps)
    @info "Enclosing ", λ
    @info "Level set", ϵ

    eigvals = diag(T)

    out_z = []
    out_bound = []

    z = λ + 4 * sign(real(λ)) * ϵ

    # we first use the newton method to approach the level set
    for j in 1:max_initial_newton
        K = svd(T - z * I)
        z, σ = _newton_step(z, K, ϵ, ϵ)

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

    push!(out_z, z)

    for t_step in 1:max_steps

        #@info t_step, max_steps

        z_old = z

        K = svd(T - z * I)

        τ = minimum(abs.(eigvals .- z)) / rel_steps

        z, σ = _follow_level_set(z, τ, K)
        z, σ = _newton_step(z, K, ϵ, τ)

        #        @info σ
        push!(out_z, z)

        r_guaranteed = 5 * abs(z_old - z) / 8

        if t_step == 1
            r_guaranteed_1 = r_guaranteed
        end

        # we certify the SVD on a ball around z_old

        z_ball = Ball(z_old, r_guaranteed)
        bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
        push!(out_bound, bound)

        # if the first point is inside the certification ball, we have found a loop closure
        #@info "r_guaranteed+r_guaranteed_1", r_guaranteed+r_guaranteed_1, "dist to start", abs(z_old-z0)

        check_loop_closure = abs(z_old - z0) < (r_guaranteed + r_guaranteed_1)

        if t_step > 10 && check_loop_closure
            @info t_step, "Loop closure"
            break
        end
    end
    return out_z, out_bound
end

function _compute_exclusion_set(T, r; max_steps, rel_steps)
    eigvals = diag(T)

    out_z = []
    out_bound = []

    z = r
    z0 = z
    r_guaranteed_1 = 0.0

    push!(out_z, z)

    for t_step in 1:max_steps
        z_old = z

        K = svd(T - z * I)

        τ = minimum(abs.(eigvals .- z)) / rel_steps

        z = z + τ * im * z / abs(z)
        z = z - (abs(z)^2 - r^2) / conj(z)

        push!(out_z, z)

        r_guaranteed = 5 * abs(z_old - z) / 8

        if t_step == 1
            r_guaranteed_1 = r_guaranteed
        end

        # we certify in a ball around z_old
        z_ball = Ball(z_old, r_guaranteed)
        bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
        push!(out_bound, bound)

        #print("test")

        #@info "test"
        #@info "r_guarantee", r_guaranteed
        #@info "r_guarantee_1", r_guaranteed_1
        #@info "dist to start", abs(z_old-z0)

        check_loop_closure = abs(z_old - z0) < r_guaranteed + r_guaranteed_1

        if t_step > 10 && check_loop_closure
            @info t_step, "Loop closure"
            break
        end
    end
    return out_z, out_bound
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
            max_steps = Int64(ceil(256 * π)), rel_steps = 16)

    Given a BallMatrix `A`, this method follows the level lines of level `ϵ`
around the eigenvalues with modulus bound between `r1` and `r2`.

The keyword arguments
    - max_initial_newton: maximum number of newton steps to reach the level lines
    - max_steps: maximum number of steps following the contour
    - rel_steps: relative integration step for the Euler method

The method outputs an array of truples:
    - the first element is the eigenvalue we are enclosing
    (in the case of the excluding circles, it is 0.0 or the maximum modulus of the eigenvalues)
    - the second element is an upper bound on the resolvent norm
    - the third element is a list of points on the enclosing line; the resolvent is rigorously
    bound on circles centered at each point and of radius 5/8 the distance to the previous point
"""
function compute_enclosure(A::BallMatrix, r1, r2, ϵ; max_initial_newton = 30,
        max_steps = Int64(ceil(256 * π)), rel_steps = 16)
    F = schur(Complex{Float64}.(A.c))

    bZ = BallMatrix(F.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(F.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    @info "Schur unitary error", errF
    @info "Schur reconstruction error", errT

    eigvals = diag(F.T)[[r1 < abs(x) < r2 for x in diag(F.T)]]

    @info "Certifying around", eigvals

    output = []

    for λ in eigvals
        curve, bounds = _compute_enclosure_eigval(F.T, λ, ϵ; max_initial_newton,
            max_steps, rel_steps)

        bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])

        if bound < 0.0
            @warn "Smaller rel_step required"
        end

        @info "resolvent upper bound", bound
        @info "σ", bounds[i]

        push!(output, (λ, bound, curve, bounds))
    end

    # encloses the eigenvalues inside r1
    eigvals_smaller_than_r1 = diag(F.T)[[abs(x) < r1 for x in diag(F.T)]]

    if !isempty(eigvals_smaller_than_r1)
        @info "Computing exclusion circle ", r1

        curve, bounds = _compute_exclusion_set(F.T, r1; max_steps, rel_steps)
        bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        @info bound, i
        @info "σ", bounds[i]

        push!(output, (0.0, bound, curve, bounds))
    end

    # # encloses the eigenvalues outside r2
    eigvals_bigger_than_r2 = diag(F.T)[[abs(x) > r2 for x in diag(F.T)]]

    if !isempty(eigvals_bigger_than_r2)
        @info "Computing exclusion circle ", r2

        curve, bounds = _compute_exclusion_set(F.T, r2; max_steps, rel_steps)
        max_abs_eigenvalue = maximum(abs.(diag(F.T)))
        bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        @info bound, i
        @info "σ", bounds[i]

        push!(output, (max_abs_eigenvalue, bound, curve, bounds))
        #     r = minimum([abs(λ)-r2 for λ in eigvals_bigger_than_r2])/5
        #     @info "Gap between r2, $r2 and smallest eigenvalue outside, $r"
        #     curve, bound = _certify_circle(F.T, r2, r, ϵ)
        #     push!(output, (max_abs_eigenvalue, curve, bound))
    end

    return output
end
