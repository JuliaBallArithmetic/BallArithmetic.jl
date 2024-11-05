export collatz_upper_bound_L2_opnorm,
       upper_bound_L1_opnorm, upper_bound_L_inf_opnorm, upper_bound_L2_opnorm

"""
    upper_abs(A)

Return a floating point matrix `B` whose entries are bigger
or equal (componentwise) any of the entries of `A`
"""
function upper_abs(A::BallMatrix{T}) where {T}
    absA = setrounding(T, RoundUp) do
        return abs.(A.c) + A.r
    end
    return absA
end

"""
    collatz_upper_bound_L2_opnorm(A::BallMatrix; iterates=10)

Give a rigorous upper bound on the ℓ² norm of the matrix `A`
by using the Collatz theorem.

We use Perron theory here: if for two matrices with `B` positive
`|A| < B` we have ρ(A)<=ρ(B) by Wielandt's theorem
[Wielandt's theorem](https://mathworld.wolfram.com/WielandtsTheorem.html)

The keyword argument `iterates` is used to establish how many
times we are iterating the vector of ones before we use Collatz's
estimate.
"""
function collatz_upper_bound_L2_opnorm(A::BallMatrix{T}; iterates = 10) where {T}
    m, k = size(A)
    x_old = ones(k)
    x_new = x_old

    absA = upper_abs(A)
    #@info opnorm(absA, Inf)

    # using Collatz theorem
    lam = setrounding(T, RoundUp) do
        for _ in 1:iterates
            x_old = x_new
            x_new = absA' * absA * x_old
            #@info maximum(x_new ./ x_old)
        end
        lam = 0.0
        for i in 1:k
            if x_old[i] != 0.0
                lam = max(lam, x_new[i] / x_old[i])
            end
        end
        return lam
    end
    return sqrt_up(lam)
end

using LinearAlgebra

"""
    upper_bound_L1_opnorm(A::BallMatrix{T})

Returns a rigorous upper bound on the ℓ¹-norm of the ball matrix `A`
"""
function upper_bound_L1_opnorm(A::BallMatrix{T}) where {T}
    norm = setrounding(T, RoundUp) do
        return opnorm(A.c, 1) + opnorm(A.r, 1)
    end
    return norm
end

"""
    upper_bound_L_inf_opnorm(A::BallMatrix{T})

Returns a rigorous upper bound on the ℓ-∞-norm of the ball matrix `A`
"""
function upper_bound_L_inf_opnorm(A::BallMatrix{T}) where {T}
    norm = setrounding(T, RoundUp) do
        return opnorm(A.c, Inf) + opnorm(A.r, Inf)
    end
    return norm
end

"""
    upper_bound_L_inf_opnorm(A::BallMatrix{T})

Returns a rigorous upper bound on the ℓ²-norm of the ball matrix `A`
using the best between the Collatz bound and the interpolation bound
"""
function upper_bound_L2_opnorm(A::BallMatrix{T}) where {T}
    norm1 = upper_bound_L1_opnorm(A)
    norminf = upper_bound_L_inf_opnorm(A)
    norm_prod = @up norm1 * norminf

    return min(collatz_upper_bound_L2_opnorm(A), sqrt_up(norm_prod))
end

"""
    svd_bound_L2_opnorm(A::BallMatrix{T})

Returns a rigorous upper bound on the ℓ²-norm of the ball matrix `A`
using the rigorous enclosure for the singular values implemented in
svd/svd.jl
"""
function svd_bound_L2_opnorm(A::BallMatrix{T}) where {T}
    σ = svdbox(A)

    top = σ[1]

    return @up top.c + top.r
end

"""
    svd_bound_L2_opnorm_inverse(A::BallMatrix)

Returns a rigorous upper bound on the ℓ²-norm of the inverse of the
ball matrix `A` using the rigorous enclosure for the singular values
implemented in svd/svd.jl
"""
function svd_bound_L2_opnorm_inverse(A::BallMatrix)
    σ = svdbox(A)

    if in(0, σ[end])
        return +Inf
    end

    inv_inf = Ball(1.0) / σ[end]

    return @up inv_inf.c + inv_inf.r
end

using LinearAlgebra
"""
    svd_bound_L2_resolvent(A::BallMatrix, lam::Ball)

Returns a rigorous upper bound on the ℓ²-norm of the resolvent
of `A` at `λ`, i.e., ||(A-λ)^{-1}||_{ℓ²}
"""
svd_bound_L2_resolvent(A::BallMatrix, λ::Ball) = svd_bound_L2_opnorm_inverse(A - λ * I)
