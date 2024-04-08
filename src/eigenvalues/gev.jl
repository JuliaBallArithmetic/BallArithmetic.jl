
# Implementing Theorem 2 Miyajima
# Numerical enclosure for each eigenvalue in generalized eigenvalue problem
"""
    Compute rigorous enclosure of each eigenvalue in generalized eigenvalue problem
    following Ref. [Miyajima2012](@cite)

    # References
    * [Miyajima2012](@cite) Miyajima, JCAM 246, 9 (2012)
"""
function gevbox(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    gev = eigen(A.c, B.c)
    return _certify_gev(A, B, gev)
end

function _certify_gev(A::BallMatrix{T}, B::BallMatrix{T}, gev::GeneralizedEigen) where {T}
    X = gev.vectors
    Y = inv(B.c * X)

    bX = BallMatrix(X)
    bY = BallMatrix(Y)

    S = bY * B * bX - I
    normS = upper_bound_L_inf_opnorm(S)
    @debug "norm S" normS
    @assert normS<1 "It is not possible to verify the eigenvalues with this precision"

    bD = BallMatrix(Diagonal(gev.values))

    R = bY * (A * bX - B * bX * bD)
    normR = upper_bound_L_inf_opnorm(R)
    @debug "norm R" normR

    den_up = @down (1.0 - normS)
    eps = @up normR / den_up

    return [Ball(lam, eps) for lam in gev.values]
end

function evbox(A::BallMatrix{T}) where {T}
    gev = eigen(A.c)
    return _certify_evbox(A, gev)
end

function _certify_evbox(A::BallMatrix{T}, gev::Eigen) where {T}
    X = gev.vectors
    Y = inv(X)

    bX = BallMatrix(X)
    bY = BallMatrix(Y)

    S = bY * bX - I
    normS = upper_bound_L_inf_opnorm(S)
    @debug "norm S" normS
    @assert normS<1 "It is not possible to verify the eigenvalues with this precision",
    normS,
    norm(X, 2),
    norm(Y, 2)

    bD = BallMatrix(Diagonal(gev.values))

    # probably something better can be done here
    # since this is not GEV, but only EV
    # need to look better at Miyajima
    # https://www.sciencedirect.com/science/article/pii/S037704270900795X

    R = bY * (A * bX - bX * bD)

    normR = upper_bound_L_inf_opnorm(R)
    @debug "norm R" normR

    den_up = @down (1.0 - normS)
    eps = @up normR / den_up

    return [Ball(lam, eps) for lam in gev.values]
end
