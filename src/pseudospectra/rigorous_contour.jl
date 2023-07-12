function follow_level_set(z::ComplexF64, τ::Float64, K::SVD)
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

function newton_level_set(z, T, ϵ; τ=ϵ / 16)
    K = svd(z * I - T)
    return _newton_step(z, K, ϵ, τ)
end


function compute_enclosure(A::BallMatrix, r1, r2, ϵ; max_initial_newton = 100, τ=ϵ / 16,
    max_steps=Int64(ceil(4π / τ)))
    F = schur(Complex{Float64}.(A.c))

    eigvals = diag(F.T)[[r1 < abs(x) < r2 for x in diag(F.T)]]

    @info "Certifying around", eigvals

    output = []

    for λ in eigvals
        curve, bound = _compute_enclosure_triangular(F.T, λ, ϵ; max_initial_newton, τ,
            max_steps)
        push!(output, (λ, curve, bound))
    end
    return output
end

function _compute_enclosure_triangular(T, λ, ϵ; max_initial_newton, τ,
    max_steps)
    @info "Enclosing ", λ
    @info "Level set", ϵ

    out_z = []
    out_bound = []

    z = λ + 4 * sign(real(λ)) * ϵ

    # we first use the newton method to approach the level set
    for j in 1:max_initial_newton
        K = svd(T- z*I)
        z, σ = _newton_step(z, K, ϵ, ϵ)
        
        if (σ - ϵ) < ϵ / 256
            break
        end
    end

    for _ in 1:max_steps
        z_old = z

        K = svd(T - z * I)
        z, σ = follow_level_set(z, τ, K)
        z, σ = _newton_step(z, K, ϵ, τ)

        push!(out_z, z)

        z_ball = Ball(z_old, 1.5 * abs(z_old - z))
 
        bound = _certify_svd(BallMatrix(T) - z_ball*I, K)[end]
        push!(out_bound, bound)
    end
    return out_z, out_bound
end
