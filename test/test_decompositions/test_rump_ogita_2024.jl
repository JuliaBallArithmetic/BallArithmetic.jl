using BallArithmetic
using LinearAlgebra
using Random
using Test

# Rump and Ogita (2024), Verified Error Bounds for Matrix Decompositions.

# LU decomposition without pivoting, in the arithmetic of the input: L is m × p unit lower
# trapezoidal and U is p × n upper trapezoidal, p = min(m, n)
function _lu_nopivot(A::AbstractMatrix{S}) where {S}
    m, n = size(A)
    p = min(m, n)
    U = copy(A)
    L = zeros(S, m, p)
    for k in 1:p
        L[k, k] = one(S)
        for i in (k + 1):m
            L[i, k] = U[i, k] / U[k, k]
            for j in k:n
                U[i, j] -= L[i, k] * U[k, j]
            end
        end
    end
    return L, triu(U[1:p, :])
end

_inside(X, B) = all(abs.(X - mid(B)) .<= rad(B))
_eye(S, m, n) = Matrix{S}(I, m, n)

@testset "Rump-Ogita 2024, Algorithm 3.2: the LU factors of a perturbed identity" begin
    rng = MersenneTwister(20261011)
    for (m, n) in ((6, 6), (9, 5), (5, 9), (1, 1), (1, 4), (4, 1)),
        S in (Float64, ComplexF64), ε in (1e-10, 1e-4, 0.05), ρ in (0.0, 1e-6)

        Em = ε * randn(rng, S, m, n) / max(m, n)
        E = BallMatrix(Em, fill(ρ * ε, m, n))
        r = BallArithmetic._rumpogita2024_alg3_2(E)
        @test r !== nothing
        p = min(m, n)
        @test size(r.LE) == (m, p) && size(r.UE) == (p, n)
        @test (r.LinvE === nothing) == (m > n)
        @test (r.UinvE === nothing) == (m < n)
        @test istril(mid(r.LE), -1) && iszero(triu(rad(r.LE)))
        @test istriu(mid(r.UE)) && iszero(tril(rad(r.UE), -1))
        for trial in 1:(ρ == 0 ? 1 : 6)
            Ep = ρ == 0 ? Em : Em + ρ * ε * (2 * rand(rng, m, n) .- 1)
            L, U = setprecision(BigFloat, 512) do
                B = S <: Complex ? Complex{BigFloat} : BigFloat
                _lu_nopivot(_eye(B, m, n) + B.(Ep))
            end
            @test _inside(L - _eye(eltype(L), m, p), r.LE)
            @test _inside(U - _eye(eltype(U), p, n), r.UE)
            r.LinvE === nothing || @test _inside(inv(L) - I, r.LinvE)
            r.UinvE === nothing || @test _inside(inv(U) - I, r.UinvE)
        end
        # the radius beyond that of E is of second order in E, as (3.1) and (3.5) say
        if ρ == 0
            @test maximum(rad(r.LE); init = 0.0) <= 4 * max(m, n) * ε^2
            @test maximum(rad(r.UE); init = 0.0) <= 4 * max(m, n) * ε^2
        end
    end
    # a perturbation of norm one or more is declined
    @test BallArithmetic._rumpogita2024_alg3_2(BallMatrix(fill(0.6, 3, 3))) === nothing
    @test BallArithmetic._rumpogita2024_alg3_2(BallMatrix(zeros(3, 3), fill(0.4, 3, 3))) === nothing
    # the exact identity: the factors are the identity with radius zero
    r = BallArithmetic._rumpogita2024_alg3_2(BallMatrix(zeros(4, 4)))
    @test all(iszero, mid(r.LE)) && all(iszero, rad(r.LE)) && all(iszero, rad(r.UinvE))
end
