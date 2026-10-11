# The floating-point algorithms of polar_iterations.jl against the polar factor from a singular
# value decomposition at 512 bits.
using Test, LinearAlgebra, Random
using BallArithmetic

@testset "Polar iterations and CholeskyQR2" begin
    rng = MersenneTwister(20261017)
    function reference_polar(A)
        setprecision(BigFloat, 512) do
            F = svd(Complex{BigFloat}.(A))
            F.U * F.Vt
        end
    end
    function with_condition(S, m, n, κ)
        P, Q = Matrix(qr(randn(rng, S, m, n)).Q)[:, 1:n], Matrix(qr(randn(rng, S, n, n)).Q)
        P * Diagonal(exp10.(range(0, -log10(κ), length = n))) * Q'
    end
    err(U, A) = Float64(maximum(abs, Complex{BigFloat}.(U) .- reference_polar(A)))

    @testset "$name, $S, $(m)×$(n), cond $κ" for (name, f) in (("Algorithm Polar", polar_newton),
            ("QDWH", polar_qdwh)), S in (Float64, ComplexF64), (m, n) in ((6, 6), (9, 5)),
        κ in (1e1, 1e6, 1e12)

        A = with_condition(S, m, n, κ)
        r = f(A)
        @test r.converged
        @test err(r.U, A) <= 1e-13 * κ
        @test opnorm(r.U' * r.U - I) <= 1e-13
        @test r.H == r.H'
        @test opnorm(r.U * r.H - A) <= 1e-13 * opnorm(A)
        name == "QDWH" && @test r.iterations <= 6        # the bound of the paper
    end

    @testset "Newton-Schulz" begin
        for S in (Float64, ComplexF64), (m, n) in ((6, 6), (9, 5))
            A = with_condition(S, m, n, 10.0)
            r = polar_newton_schulz(A)
            @test r.converged && err(r.U, A) <= 1e-12
            @test opnorm(r.U * r.H - A) <= 1e-13 * opnorm(A)
        end
        @test_throws ArgumentError polar_newton_schulz(2.0 .* Matrix(1.0I, 3, 3))
    end

    @testset "CholeskyQR2" begin
        for S in (Float64, ComplexF64), κ in (1e1, 1e6)
            X = with_condition(S, 30, 8, κ)
            Z, U = cholesky_qr2(X)
            @test istriu(U)
            @test opnorm(Z' * Z - I) <= 1e-13
            @test opnorm(Z * U - X) <= 1e-13 * opnorm(X)
        end
        # beyond the assumption (2.1) the Cholesky factorisation breaks down
        @test_throws Exception cholesky_qr2(with_condition(Float64, 30, 8, 1e17))
        @test_throws DimensionMismatch cholesky_qr2(randn(rng, 3, 5))
    end

    @testset "BigFloat" begin
        setprecision(BigFloat, 256) do
            A = BigFloat.(with_condition(Float64, 5, 5, 1e3))
            for f in (polar_newton, polar_qdwh)
                r = f(A)
                @test r.converged && opnorm(Float64.(r.U' * r.U - I)) <= 1e-60
            end
            Z, U = cholesky_qr2(BigFloat.(with_condition(Float64, 8, 4, 1e3)))
            @test maximum(abs, Z' * Z - I) <= 1e-60
        end
    end
end
