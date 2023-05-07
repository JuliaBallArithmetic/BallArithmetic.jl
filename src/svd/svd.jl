
function upper_abs(A::BallMatrix)
    return abs.(A.c)+A.r
end

# we use Perron theory here: if two positive matrices 
# A <= B we have ρ(A)<=ρ(B)

function upper_bound_norm(A::BallMatrix{T}; iterates = 10) where {T}
    m, k = size(A)
    x_old = ones(m)
    x_new = x_old

    absA = upper_abs(A)
    #@info opnorm(absA, Inf)

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
    svdA  = svd(A.c)
    U = BallMatrix(svdA.U, zeros(size(svdA.U)))
    Vt = BallMatrix(svdA.Vt, zeros(size(svdA.Vt)))
    Σ = BallMatrix(Diagonal(svdA.S), Diagonal(zeros(size(svdA.S))))
    V = BallMatrix(svdA.V, zeros(size(svdA.V)))

    E = U*Σ*Vt - A
    normE = sqrt(upper_bound_norm(E))
    @info "E" normE
    
    F = Vt*V-I
    normF = sqrt(upper_bound_norm(F))
    @info normF

    G = U'*U-I
    normG = sqrt(upper_bound_norm(G))
    @info normG

    @assert normF < 1 "It is not possible to verify the singular values with this precision"
    @assert normG < 1 "It is not possible to verify the singular values with this precision"

    @info T
    svdbounds_up = setrounding(T, RoundUp) do
        return [(σ+normE)/(1-normF)*(1-normG) for σ in svdA.S]
    end
    
    svdbounds_down = setrounding(T, RoundDown) do
        return [(σ-normE)/(1+normF)*(1+normG) for σ in svdA.S]
    end
    
    midpoints = (svdbounds_down+svdbounds_up)/2
    rad = svdbounds_up-midpoints

    return [Ball(midpoints[i], rad[i]) for i in 1:length(midpoints)]
end