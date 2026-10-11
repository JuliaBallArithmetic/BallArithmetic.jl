# The splitting scheme of Lange and Rump, "Accurate floating-point matrix residuals":
# `_lange_rump_two_terms` and the two routines that call it.
using Test, LinearAlgebra, Random
using BallArithmetic
const BA = BallArithmetic

# Σ_t sgn_t A_t B_t at 2048 bits, exact for Float32 and Float64 data of these sizes
function _exact_sum(terms)
    setprecision(BigFloat, 2048) do
        sum(sgn .* (Complex{BigFloat}.(A) * Complex{BigFloat}.(B)) for (A, B, sgn) in terms)
    end
end

# entries where the modulus of exact − P1 − P2 exceeds R
function _outside(terms, P1, P2, R)
    setprecision(BigFloat, 2048) do
        D = abs.(_exact_sum(terms) .- Complex{BigFloat}.(P1) .- Complex{BigFloat}.(P2))
        count(D .> R), maximum(Float64.(D))
    end
end

@testset "Lange-Rump splitting products" begin
    rng = MersenneTwister(20261011)

    @testset "Split: exact, a multiple of the point, remainder at most half" begin
        for T in (Float32, Float64), S in (T, Complex{T})
            A = randn(rng, S, 7, 9) .* T(10) .^ rand(rng, -5:5, 7, 9)
            for byrow in (true, false)
                parts, rest, μ, τ = BA._lr_split_k(A, byrow, 3, sqrt(T(2)^precision(T)), 2 * 9)
                @test sum(Complex{BigFloat}.(P) for P in parts) + Complex{BigFloat}.(rest) ==
                      Complex{BigFloat}.(A)
                for l in 1:3, j in 1:9, i in 1:7
                    pt = (byrow ? μ[i] : μ[j]) * τ^(l - 1)
                    x = parts[l][i, j]
                    @test isinteger(real(x) / pt) && isinteger(imag(x) / pt)
                    y = l < 3 ? parts[l + 1][i, j] : rest[i, j]
                    @test abs(real(y)) <= pt / 2 && abs(imag(y)) <= pt / 2
                end
            end
        end
    end

    @testset "the leading products are exact" begin
        for T in (Float32, Float64), S in (T, Complex{T}), n in (3, 50, 300)
            A, B = randn(rng, S, 6, n), randn(rng, S, n, 5)
            p = precision(T)
            N = (S <: Complex ? 2 : 1) * n
            qa = sqrt(T(2)^p)
            qb = ((one(T) - eps(T) / 2) * T(2)^p) / qa
            Ap, = BA._lr_split_k(A, true, 2, qa, N)
            Bp, = BA._lr_split_k(B, false, 2, qb, N)
            for X in Ap, Y in Bp
                @test Complex{BigFloat}.(X * Y) == Complex{BigFloat}.(X) * Complex{BigFloat}.(Y)
            end
        end
    end

    @testset "containment, $S, k = $k" for S in (Float64, ComplexF64, Float32, ComplexF32),
        k in (1, 2, 3)

        T = real(S)
        for (m, n, q) in ((1, 1, 1), (4, 9, 3), (30, 30, 30), (12, 80, 7))
            A, B = randn(rng, S, m, n), randn(rng, S, n, q)
            terms = ((A, B, one(T)),)
            P1, P2, R = BA._lange_rump_two_terms(terms; k = k)
            @test _outside(terms, P1, P2, R)[1] == 0
        end
        # a residual: R A − I with an ill-conditioned A, the identity as a diagonal summand
        n = 40
        U, V = Matrix(qr(randn(rng, S, n, n)).Q), Matrix(qr(randn(rng, S, n, n)).Q)
        A = U * Diagonal(T(10) .^ range(T(0), T <: Float32 ? T(-4) : T(-10), length = n)) * V'
        Rinv = inv(A)
        Id = Matrix{S}(I, n, n)
        terms = ((Rinv, A, one(T)), (Id, Id, -one(T)))
        P1, P2, R = BA._lange_rump_two_terms(terms; k = k)
        nout, err = _outside(terms, P1, P2, R)
        @test nout == 0
        # rows and columns of very different sizes
        D1 = Diagonal(T(2) .^ rand(rng, -30:30, n))
        D2 = Diagonal(T(2) .^ rand(rng, -30:30, n))
        terms = ((D1 * randn(rng, S, n, n), randn(rng, S, n, n) * D2, -one(T)),
            (randn(rng, S, n, 3), randn(rng, S, 3, n), one(T)))
        P1, P2, R = BA._lange_rump_two_terms(terms; k = k)
        @test _outside(terms, P1, P2, R)[1] == 0
    end

    @testset "accuracy for k = 2 in Float64" begin
        n = 100
        U, V = Matrix(qr(randn(rng, n, n)).Q), Matrix(qr(randn(rng, n, n)).Q)
        A = U * Diagonal(10.0 .^ range(0, -12, length = n)) * V'
        Rinv = inv(A)
        Id = Matrix{Float64}(I, n, n)
        terms = ((Rinv, A, 1.0), (Id, Id, -1.0))
        P1, P2, R = BA._lange_rump_two_terms(terms)
        nout, err = _outside(terms, P1, P2, R)
        @test nout == 0
        float_err = setprecision(BigFloat, 2048) do
            maximum(Float64.(abs.(_exact_sum(terms) .- (Rinv * A - Id))))
        end
        @test err < 1e-10 * float_err
        E = BA._accurate_product_sum(terms)
        @test setprecision(BigFloat, 2048) do
            all(abs.(_exact_sum(terms) .- mid(E)) .<= rad(E))
        end
    end

    @testset "diagonal summands and zero rows" begin
        for S in (Float64, ComplexF64)
            T = real(S)
            A = randn(rng, S, 9, 9)
            A[3, :] .= 0
            B = randn(rng, S, 9, 9)
            B[:, 5] .= 0
            d = Matrix(Diagonal(randn(rng, S, 9)))
            terms = ((A, B, one(T)), (A, d, -one(T)), (d, B, one(T)))
            P1, P2, R = BA._lange_rump_two_terms(terms)
            @test _outside(terms, P1, P2, R)[1] == 0
        end
    end

    @testset "declines, and the callers fall back" begin
        A, B = randn(rng, 5, 5), randn(rng, 5, 5)
        @test BA._lange_rump_two_terms(((A, B, 2.0),)) === nothing
        @test BA._lange_rump_two_terms(((big.(A), big.(B), big(1.0)),)) === nothing
        tiny = 1e-170 .* A
        @test BA._lange_rump_two_terms(((tiny, 1e-170 .* B, 1.0),)) === nothing
        huge = 1e300 .* A
        @test BA._lange_rump_two_terms(((huge, 1e10 .* B, 1.0),)) === nothing   # overflow
        for terms in (((A, B, 2.0),), ((tiny, 1e-170 .* B, 1.0),))
            P1, P2, R = BA._two_term_product_sum(terms)
            @test _outside(terms, P1, P2, R)[1] == 0
        end
        @test_throws ArgumentError BA._lange_rump_two_terms(((A, B, 1.0),); k = 0)
    end
end
