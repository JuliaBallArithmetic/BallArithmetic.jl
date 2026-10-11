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

# a matrix with singular values from 1 to 10^-k, as in the paper (gallery/randsvd)
function _randsvd(rng, n, k; S = Float64)
    σ = exp10.(range(0, -k; length = n))[randperm(rng, n)]
    return Matrix(qr(randn(rng, S, n, n)).Q) * Diagonal(σ) * Matrix(qr(randn(rng, S, n, n)).Q)
end

_relerr(B) = (m = abs.(mid(B)); r = rad(B); [r[i] / m[i] for i in eachindex(m) if m[i] > 0])

@testset "Rump-Ogita 2024, Section 3.2: the LU decomposition of a square matrix" begin
    rng = MersenneTwister(20261011)
    med(v) = sort(v)[(length(v) + 1) ÷ 2]
    @testset "n = $n, cond = 1e$k, $S" for (n, k) in ((8, 1), (30, 2), (30, 8), (30, 12)),
        S in (Float64, ComplexF64)

        A = _randsvd(rng, n, k; S)
        r = BallArithmetic._rumpogita2024_sec3_2(BallMatrix(A))
        @test r !== nothing
        @test sort(r.p) == 1:n
        @test istril(mid(r.L)) && all(isone, diag(mid(r.L))) && iszero(triu(rad(r.L)))
        @test istriu(mid(r.U)) && iszero(tril(rad(r.U), -1))
        L, U = setprecision(BigFloat, 1024) do
            B = S <: Complex ? Complex{BigFloat} : BigFloat
            _lu_nopivot(B.(A[r.p, :]))
        end
        @test _inside(L, r.L)
        @test _inside(U, r.U)
        # the paper's Figure 3: with the products in two-fold precision the bounds are close to
        # the rounding unit for every condition number
        @test med(_relerr(r.U)) < 1e-13
        @test med(_relerr(r.L)) < 1e-12
    end
    @testset "a ball of matrices" begin
        A = _randsvd(rng, 10, 2)
        ρ = 1e-10
        r = BallArithmetic._rumpogita2024_sec3_2(BallMatrix(A, fill(ρ, 10, 10)))
        @test r !== nothing
        for _ in 1:10
            Ap = A + ρ * (2 * rand(rng, 10, 10) .- 1)
            L, U = setprecision(() -> _lu_nopivot(big.(Ap[r.p, :])), BigFloat, 1024)
            @test _inside(L, r.L) && _inside(U, r.U)
        end
    end
    @testset "declined" begin
        @test BallArithmetic._rumpogita2024_sec3_2(BallMatrix([1.0 2.0; 2.0 4.0])) === nothing
        @test BallArithmetic._rumpogita2024_sec3_2(BallMatrix([1.0 0.0; 0.0 1.0], fill(0.6, 2, 2))) === nothing
        @test_throws ArgumentError BallArithmetic._rumpogita2024_sec3_2(BallMatrix(randn(rng, 3, 4)))
    end
    @testset "the two-term product" begin
        for S in (Float64, ComplexF64)
            A, B = randn(rng, S, 7, 9) .* exp10.(8 .* randn(rng, 7, 9)), randn(rng, S, 9, 5)
            P1, P2, R = BallArithmetic._two_term_product_sum(((A, B, 1.0),))
            X = setprecision(BigFloat, 1024) do
                BT = S <: Complex ? Complex{BigFloat} : BigFloat
                BT.(A) * BT.(B) - BT.(P1) - BT.(P2)
            end
            @test all(abs.(X) .<= R)
            C = BallArithmetic._accurate_product_sum(((A, B, 1.0), (Matrix{S}(I, 7, 7), randn(rng, S, 7, 5), -1.0)))
            @test size(C) == (7, 5)
        end
    end
end
