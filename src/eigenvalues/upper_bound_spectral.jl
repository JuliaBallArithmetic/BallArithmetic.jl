function collatz_upper_bound(A::BallMatrix{T}; iterates=10) where {T}
    m, k = size(A)
    x_old = ones(m)
    x_new = x_old

    absA = upper_abs(A)
    #@info opnorm(absA, Inf)

    # using Collatz theorem
    lam = setrounding(T, RoundUp) do
        for _ in 1:iterates
            x_old = x_new
            x_new = absA * x_old
            #@info maximum(x_new ./ x_old)
        end
        lam = maximum(x_new ./ x_old)
    end
    return lam
end