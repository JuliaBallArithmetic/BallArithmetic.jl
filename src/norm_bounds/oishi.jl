
"""
We implement the estimate of RumpOishi2024 in the special case of a triangular matrix

TODO: Understand why the collatz_upper_bound_L2_opnorm gives 1 on the 4x4 minor
in the corner

 T  = [3.0 0.1 0.1 0.2;
        0.0 2.0 0.1 4.0;
        0.0 0.0 0.3 0.1;
        0.0 0.0 0.0 0.1]

The problem is that multiplication by a Ball destroys the triangular structure...


"""

# α = √((1 + 1/√(1 + 4/μ²))/2), evaluated as √((1 + μ/√(μ² + 4))/2): the two are equal for μ > 0,
# and the second has no division by μ, so it is defined at μ = 0 (α = √(1/2), ψ = 1) and does not
# underflow for a tiny μ, where 4/μ² is not representable.
function α_bound(μ)
    bμ = Ball(μ)
    return sqrt(0.5 * (1 + bμ / sqrt(bμ * bμ + 4)))
end

function psi_bound(N)
    μ = collatz_upper_bound_L2_opnorm(N)
    α = α_bound(μ)
    return sqrt(1 + 2 * α * μ * sqrt(1 - α * α) + α * μ * α * μ)
end

function rump_oishi_triangular(T::BallMatrix, k)
    @assert istriu(T.c) && istriu(T.r)

    A = T[1:k, 1:k]
    B = T[1:k, (k + 1):end]
    # C is zero, since it is triangular
    D = T[(k + 1):end, (k + 1):end]

    Dd = Diagonal(diag(D))
    Df = D - Dd

    E = backward_substitution(A, B)

    Ddinv = Diagonal([1 / Dd[i, i] for i in 1:size(Dd)[1]])
    #@info Ddinv

    normDinv = maximum([add_up(abs(Ddinv[i, i]).c, abs(Ddinv[i, i]).r)
                        for i in 1:size(Dd)[1]])
    Ftemp = Ddinv * Df
    #@info Df
    #@info Ddinv

    # these checks use informations from the structure of the matrix
    F = _triangularize(BallMatrix(mid.(Ftemp), rad.(Ftemp)))

    for i in 1:size(Dd)[1]
        F.c[i, i] = 0.0
        F.r[i, i] = 0.0
    end

    #@info F

    psiE = psi_bound(E)
    #@info psiE

    normF = Ball(collatz_upper_bound_L2_opnorm(F))
    #@info normF

    @assert (@up normF.c + normF.r) < 1

    normAinv = Ball(svd_bound_L2_opnorm_inverse(A))

    α = @up normAinv.c + normAinv.r
    β_temp = Ball(normDinv) / (1 - normF)
    β = @up β_temp.c + β_temp.r

    est = max(α, β) * psiE

    return @up est.c + est.r
end
