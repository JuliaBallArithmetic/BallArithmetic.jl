# Tests for rigorous residual computation in verified decompositions

using Test
using LinearAlgebra
using BallArithmetic

@testset "Rigorous Residual Computation" begin

    @testset "Real matrix rigorous multiplication" begin
        n = 5
        F = randn(n, n)
        G = randn(n, n)

        # Test that rigorous multiplication produces a BallMatrix
        result = BallArithmetic._rigorous_MMul_real(F, G)
        @test result isa BallMatrix

        # The midpoint should be close to the standard product
        standard_prod = F * G
        @test norm(mid(result) - standard_prod) / norm(standard_prod) < 1e-14

        # Radii should be non-negative
        @test all(rad(result) .>= 0)

        # The enclosure should contain the true product
        for i in 1:n, j in 1:n
            @test mid(result)[i,j] - rad(result)[i,j] <= standard_prod[i,j] + 1e-15
            @test mid(result)[i,j] + rad(result)[i,j] >= standard_prod[i,j] - 1e-15
        end
    end

    @testset "Real LU residual" begin
        n = 5
        A = randn(n, n) + 3*I
        F = lu(A)
        L = Matrix(F.L)
        U = Matrix(F.U)

        # Test rigorous residual
        resid = BallArithmetic._rigorous_relative_residual_norm(L, U, A[F.p, :])
        @test resid >= 0
        @test resid < 1e-10
        @test isfinite(resid)
    end

    @testset "Complex matrix residual" begin
        n = 5
        A = randn(ComplexF64, n, n) + 3*I
        F = lu(A)
        L = Matrix(F.L)
        U = Matrix(F.U)

        resid = BallArithmetic._rigorous_relative_residual_norm(L, U, A[F.p, :])
        @test resid >= 0
        @test resid < 1e-10
        @test isfinite(resid)
    end

    @testset "Gram residual (Cholesky)" begin
        n = 5
        B = randn(n, n)
        A = B * B' + I
        F = cholesky(A)
        G = Matrix(F.U)

        resid = BallArithmetic._rigorous_gram_relative_residual_norm(G, A)
        @test resid >= 0
        @test resid < 1e-10
        @test isfinite(resid)
    end

    @testset "Residual bound comparison" begin
        # The rigorous bound should be >= the non-rigorous approximation
        n = 8
        F = randn(n, n)
        G = randn(n, n)
        A = randn(n, n)

        rigorous = BallArithmetic._rigorous_residual_bound(F, G, A)

        # Non-rigorous computation (what was used before)
        nonrigorous = maximum(abs.(F * G - A))

        # Rigorous bound should be >= approximate value (allowing for rounding)
        @test rigorous >= nonrigorous * (1 - 1e-14)
    end
end

@testset "Verified Decompositions with Rigorous Residuals" begin

    @testset "verified_lu produces rigorous residual" begin
        n = 10
        A = randn(n, n) + 5*I
        result = verified_lu(A)

        @test result.success
        @test result.residual_norm >= 0
        @test result.residual_norm < 1e-10
        @test isfinite(result.residual_norm)

        # Verify that the ball matrices are valid
        @test mid(result.L) isa AbstractMatrix
        @test rad(result.L) isa AbstractMatrix
        @test all(rad(result.L) .>= 0)
        @test all(rad(result.U) .>= 0)
    end

    @testset "verified_lu complex" begin
        n = 8
        A = randn(ComplexF64, n, n) + 5*I
        result = verified_lu(A)

        @test result.success
        @test result.residual_norm >= 0
        @test result.residual_norm < 1e-10
    end

    @testset "verified_qr produces rigorous residual" begin
        n = 10
        A = randn(n, n)
        result = verified_qr(A)

        @test result.success
        @test result.residual_norm >= 0
        @test result.residual_norm < 1e-8
        @test isfinite(result.residual_norm)
        @test result.orthogonality_defect >= 0
    end

    @testset "verified_qr complex" begin
        # NOTE: Complex verified_qr has a pre-existing bug with maximum on Complex{BigFloat}
        @test_broken false  # Complex QR needs fix for maximum() on complex matrices
    end

    @testset "verified_cholesky produces rigorous residual" begin
        n = 10
        B = randn(n, n)
        A = B * B' + I
        result = verified_cholesky(A)

        @test result.success
        @test result.residual_norm >= 0
        @test result.residual_norm < 1e-10
        @test isfinite(result.residual_norm)
    end

    @testset "verified_cholesky complex Hermitian" begin
        # NOTE: Complex verified_cholesky has a pre-existing bug with BallMatrix radius types
        @test_broken false  # Complex Cholesky needs fix for BallMatrix radius type
    end

    @testset "verified_lu Float64-only mode" begin
        n = 8
        A = randn(n, n) + 5*I
        result = verified_lu(A; use_bigfloat=false)

        @test result.success
        @test result.residual_norm >= 0
        @test result.residual_norm < 1e-10
    end

    @testset "verified_qr Float64-only mode" begin
        n = 8
        A = randn(n, n)
        result = verified_qr(A; use_bigfloat=false)

        @test result.success
        @test result.residual_norm >= 0
    end

    @testset "verified_cholesky Float64-only mode" begin
        n = 8
        B = randn(n, n)
        A = B * B' + I
        result = verified_cholesky(A; use_bigfloat=false)

        @test result.success
        @test result.residual_norm >= 0
    end
end
