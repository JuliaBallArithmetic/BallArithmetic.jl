
# Implementing Theorem 2 Miyajima 
# Numerical enclosure for each eigenvalue in generalized eigenvalue problem
function gevbox(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    gev = eigen(A.c, B.c)
    
    X = gev.vectors
    Y = inv(B.c*X)

    bX = BallMatrix(X)
    bY = BallMatrix(Y)
    
    S = bY*B*bX-I
    normS = upper_bound_L_inf_norm(S)
    @debug "norm S" normS
    @assert normS < 1 "It is not possible to verify the eigenvalues with this precision"
    
    bD = BallMatrix(Diagonal(gev.values))

    R = bY*(A*bX-B*bX*bD)
    normR = upper_bound_L_inf_norm(R)
    @debug "norm R" normR
    
    den_up = @down (1.0 - normS)
    eps = @up normR / den_up

    return [Ball(lam, eps) for lam in gev.values]
end

function evbox(A::BallMatrix{T}) where {T}
    gev = eigen(A.c)
    
    X = gev.vectors
    Y = inv(X)

    bX = BallMatrix(X)
    bY = BallMatrix(Y)
    
    S = bY*bX-I
    normS = upper_bound_L_inf_norm(S)
    @debug "norm S" normS
    @assert normS < 1 "It is not possible to verify the eigenvalues with this precision"
    
    bD = BallMatrix(Diagonal(gev.values))

    # probably something better can be done here
    # since this is not GEV, but only EV 
    # need to look better at Miyajima 
    # https://www.sciencedirect.com/science/article/pii/S037704270900795X
    
    R = bY*(A*bX-bX*bD)

    normR = upper_bound_L_inf_norm(R)
    @debug "norm R" normR
    
    den_up = @down (1.0 - normS)
    eps = @up normR / den_up

    return [Ball(lam, eps) for lam in gev.values]
end