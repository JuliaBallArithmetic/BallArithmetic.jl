
function upper_abs(A::BallMatrix)
    return abs.(A.c) + A.r
end

# we use Perron theory here: if for two matrices with B positive 
# |A| < B we have ρ(A)<=ρ(B)
# Wielandt's theorem 
# https://mathworld.wolfram.com/WielandtsTheorem.html

function upper_bound_norm(A::BallMatrix{T}; iterates=10) where {T}
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

function svdbox(A::BallMatrix{T}) where {T}
    svdA = svd(A.c)

    U = BallMatrix(svdA.U)
    Vt = BallMatrix(svdA.Vt)
    Σ = BallMatrix(Diagonal(svdA.S))
    V = BallMatrix(svdA.V)

    E = U * Σ * Vt - A
    normE = sqrt_up(upper_bound_norm(E))
    @debug "norm E" normE

    F = Vt * V - I
    normF = sqrt_up(upper_bound_norm(F))
    @debug "norm F" normF

    G = U' * U - I
    normG = sqrt_up(upper_bound_norm(G))
    @debug "norm G" normG

    @assert normF < 1 "It is not possible to verify the singular values with this precision"
    @assert normG < 1 "It is not possible to verify the singular values with this precision"

    den_down = @up (1.0 + normF) * (1.0 + normG)
    den_up = @down (1.0 - normF) * (1.0 - normG)

    svdbounds_down = setrounding(T, RoundDown) do
        [(σ - normE) / den_down for σ in svdA.S]
    end

    svdbounds_up = setrounding(T, RoundUp) do
        [(σ + normE) / den_up for σ in svdA.S]
    end

    midpoints = (svdbounds_down + svdbounds_up) / 2
    rad = setrounding(T, RoundUp) do
        [max(svdbounds_up[i] - midpoints[i], midpoints[i] - svdbounds_down[i]) for i in 1:length(midpoints)]
    end

    return [Ball(midpoints[i], rad[i]) for i in 1:length(midpoints)]
end