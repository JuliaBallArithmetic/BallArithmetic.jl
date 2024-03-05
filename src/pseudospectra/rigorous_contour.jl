export Enclosure, bound_enclosure

struct Enclosure
    λ::Any
    points::Vector{ComplexF64}
    bounds::Vector{Ball{Float64, Float64}}
    radiuses::Vector{Float64}
    loop_closure::Bool
end

include("contour_strategies.jl")

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
        check = ϵ * abs(z) * (1.0 + ϵ)^2 * boundT
        if @up check.c + check.r <= 0.5
            temp = (1.0 + ϵ)^2 * (1.0 + 2.0 * ϵ * abs(z)) * boundT
            bound = temp / (1 - ϵ * temp)
            val = max(val, @up bound.c + bound.r)
        else
            @warn "We need a better Schur decomposition"
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

    sigma_Z = svdbox(bZ)

    norm_Z = sigma_Z[1]
    norm_Z_inv = 1.0 / sigma_Z[end]

    @info "Schur unitary error", errF
    @info "Schur reconstruction error", errT
    @info "Norm Z, Z⁻¹", norm_Z, norm_Z_inv
    eigvals = diag(F.T)[[r1 < abs(x) < r2 for x in diag(F.T)]]

    @info "Certifying around", eigvals

    output = []

    for λ in eigvals
        E = _compute_enclosure_eigval(F.T, λ, ϵ; max_initial_newton,
            max_steps, rel_steps)

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

        E = _compute_exclusion_set(F.T, r1; max_steps, rel_steps)
        #bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        #@info bound, i
        #@info "σ", bounds[i]
        @info bound_resolvent(E)

        push!(output, E)
    end

    # # encloses the eigenvalues outside r2
    eigvals_bigger_than_r2 = diag(F.T)[[abs(x) > r2 for x in diag(F.T)]]

    if !isempty(eigvals_bigger_than_r2)
        @info "Computing exclusion circle ", r2

        E = _compute_exclusion_set(F.T, r2; max_steps, rel_steps)
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

    return output, errF, errT, norm_Z, norm_Z_inv
end

function compute_enclosure_circles(A::BallMatrix, r1, r2, ϵ; max_initial_newton = 30,
        max_steps = Int64(ceil(256 * π)), rel_steps = 16, rel_pearl_size = 1 / 32,
        rel_pearl_size_r1 = 1 / 32, rel_pearl_size_r2 = 1 / 32)
    F = schur(Complex{Float64}.(A.c))

    bZ = BallMatrix(F.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(F.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    sigma_Z = svdbox(bZ)

    norm_Z = sigma_Z[1]
    norm_Z_inv = 1.0 / sigma_Z[end]

    @info "Schur unitary error", errF
    @info "Schur reconstruction error", errT
    @info "Norm Z, Z⁻¹", norm_Z, norm_Z_inv

    max_abs_eigenvalues = maximum(abs.(diag(F.T)))

    eigvals = diag(F.T)[[r1 < abs(x) < r2 for x in diag(F.T)]]

    output = []

    for λ in eigvals
        @info "Certifying around", λ
        E = _compute_exclusion_circle_level_set_priori(
            F.T, λ, ϵ; rel_pearl_size, max_initial_newton)

        # bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])

        # if bound < 0.0
        #     @warn "Smaller rel_step required"
        # end

        # @info "resolvent upper bound", bound
        # @info "σ", bounds[i]

        push!(output, E)
    end

    min_abs_validated_eigenvalues = 0.0
    max_abs_validated_eigenvalues = max_abs_eigenvalues

    if !isempty(eigvals)
        abs_eigvals = abs.(eigvals)
        min_abs_validated_eigenvalues = minimum(abs_eigvals)
        max_abs_validated_eigenvalues = maximum(abs_eigvals)
    end

    # encloses the eigenvalues inside r1
    eigvals_smaller_than_r1 = diag(F.T)[[abs(x) < r1 for x in diag(F.T)]]

    if !isempty(eigvals_smaller_than_r1)
        max_abs_eigvals_smaller_than_r1 = maximum(abs.(eigvals_smaller_than_r1))

        r1tent = (max_abs_eigvals_smaller_than_r1 + min_abs_validated_eigenvalues) / 2.0

        @info "Computing exclusion circle ", r1
        @info "to optimize, we take a circle equidistant from the eigenvalues", r1tent

        E = _compute_exclusion_set(F.T, r1tent; max_steps, rel_steps)

        # dist = (min_abs_validated_eigenvalues - max_abs_eigvals_smaller_than_r1) *
        #        rel_pearl_size_r1
        # N = ceil(8 * r1tent / dist)
        # @info N

        # E = _certify_circle(F.T, 0.0, r1tent, N)
        # #bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        #@info bound, i
        #@info "σ", bounds[i]
        #@info bound_resolvent(E)

        push!(output, E)
    end

    # # encloses the eigenvalues outside r2
    eigvals_bigger_than_r2 = diag(F.T)[[abs(x) > r2 for x in diag(F.T)]]

    if !isempty(eigvals_bigger_than_r2)
        min_abs_eigvals_bigger_than_r2 = minimum(abs.(eigvals_bigger_than_r2))

        r2tent = (min_abs_eigvals_bigger_than_r2 + max_abs_validated_eigenvalues) / 2.0

        @info "Computing exclusion circle ", r2
        @info "to optimize, we take a circle equidistant from the eigenvalues", r2tent

        E = _compute_exclusion_set(F.T, r2tent; max_steps, rel_steps)

        # dist = (min_abs_eigvals_bigger_than_r2 - max_abs_validated_eigenvalues) *
        #        rel_pearl_size_r2
        # N = ceil(8 * r1tent / dist)
        # @info N

        # E = _certify_circle(F.T, 0.0, r2tent, N)
        #bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        #@info bound, i
        #@info "σ", bounds[i]
        #@info bound_resolvent(E)

        push!(output, E)

        # @info "Computing exclusion circle ", r2

        # E = _compute_exclusion_set(F.T, r2; max_steps, rel_steps)
        # #max_abs_eigenvalue = maximum(abs.(diag(F.T)))
        # #bound, i = findmax([@up 1.0 / (@down x.c - x.r) for x in bounds])
        # #@info bound, i
        # #@info "σ", bounds[i]

        # push!(output, E)
        # #     r = minimum([abs(λ)-r2 for λ in eigvals_bigger_than_r2])/5
        # #     @info "Gap between r2, $r2 and smallest eigenvalue outside, $r"
        # #     curve, bound = _certify_circle(F.T, r2, r, ϵ)
        # #     push!(output, (max_abs_eigenvalue, curve, bound))
    end

    return output, errF, errT, norm_Z, norm_Z_inv
end
