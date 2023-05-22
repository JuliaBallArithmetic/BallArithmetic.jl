export collatz_upper_bound_L2_norm, upper_bound_L1_norm, upper_bound_L_inf_norm, upper_bound_L2_norm

function upper_abs(A::BallMatrix)
    return abs.(A.c) + A.r
end

# we use Perron theory here: if for two matrices with B positive 
# |A| < B we have ρ(A)<=ρ(B)
# Wielandt's theorem 
# https://mathworld.wolfram.com/WielandtsTheorem.html

function collatz_upper_bound_L2_norm(A::BallMatrix{T}; iterates=10) where {T}
    m, k = size(A)
    x_old = ones(m)
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
        lam = maximum(x_new ./ x_old)
    end
    return lam
end

using LinearAlgebra

function upper_bound_L1_norm(A::BallMatrix{T}) where {T}
    norm = setrounding(T, RoundUp) do
        return opnorm(A.c, 1)+opnorm(A.r, 1)        
    end
    return norm
end

function upper_bound_L_inf_norm(A::BallMatrix{T}) where {T}
    norm = setrounding(T, RoundUp) do
        return opnorm(A.c, Inf)+opnorm(A.r, Inf)        
    end
    return norm
end

function upper_bound_L2_norm(A::BallMatrix{T}) where {T}
    norm1 = upper_bound_L1_norm(A)
    norminf = upper_bound_L_inf_norm(A)
    norm_prod = @up norm1*norminf
    
    return min(collatz_upper_bound_L2_norm(A), sqrt_up(norm_prod))
end