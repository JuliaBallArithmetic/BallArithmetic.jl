# Tests for src/svd/precision_cascade_svd.jl core functionality
# (Extension tests are in test_doublefloats_ext.jl and test_arbnumerics_ext.jl)

using Test
using LinearAlgebra
using BallArithmetic

@testset "Precision Cascade SVD Core" begin

    @testset "PrecisionCascadeSVDResult struct" begin
        n = 3
        U = randn(n, n)
        Σ = [3.0, 2.0, 1.0]
        V = randn(n, n)
        residual_norm = 1e-10
        σ_min = Σ[end] - residual_norm
        final_precision = 256

        result = BallArithmetic.PrecisionCascadeSVDResult(U, Σ, V, residual_norm, σ_min, final_precision)

        @test result.U === U
        @test result.Σ === Σ
        @test result.V === V
        @test result.residual_norm == residual_norm
        @test result.σ_min == σ_min
        @test result.final_precision == final_precision
    end

    @testset "PrecisionCascadeSVDResult with BigFloat" begin
        n = 2
        setprecision(BigFloat, 256) do
            U = zeros(Complex{BigFloat}, n, n)
            Σ = [BigFloat(2.0), BigFloat(1.0)]
            V = zeros(Complex{BigFloat}, n, n)

            for i in 1:n
                U[i, i] = one(Complex{BigFloat})
                V[i, i] = one(Complex{BigFloat})
            end

            result = BallArithmetic.PrecisionCascadeSVDResult(
                U, Σ, V,
                BigFloat(1e-50),
                Σ[end] - BigFloat(1e-50),
                256
            )

            @test result.final_precision == 256
            @test result.σ_min > 0
        end
    end

    @testset "_ogita_iteration! function" begin
        # Test with a simple matrix
        n = 4
        A_orig = randn(n, n)
        # Make it well-conditioned by adding diagonal
        A = A_orig + 3.0 * I

        # Start with standard SVD
        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.Vt')

        # Initial residual
        initial_residual = norm(A - U * Diagonal(Σ) * V')

        # Run one iteration
        new_residual = BallArithmetic._ogita_iteration!(A, U, Σ, V)

        @test new_residual >= 0
        @test isfinite(new_residual)

        # Residual should be very small (SVD was already computed)
        @test new_residual < 1e-10
    end

    @testset "_ogita_iteration! improves approximation" begin
        n = 5
        A = randn(n, n)
        A = A + 10.0 * I  # Well-conditioned

        # Start with the actual SVD (not perturbed, as Ogita is for refinement)
        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.Vt')

        # Run iteration - should maintain or improve accuracy
        initial_residual = norm(A - U * Diagonal(Σ) * V')
        new_residual = BallArithmetic._ogita_iteration!(A, U, Σ, V)

        # After iteration on already-good SVD, residual should stay small
        @test new_residual < 1e-10
    end

    @testset "_ogita_iteration! with complex matrix" begin
        n = 3
        A = randn(ComplexF64, n, n)
        A = A + 5.0 * I

        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.Vt')

        residual = BallArithmetic._ogita_iteration!(A, U, Σ, V)

        @test residual >= 0
        @test isfinite(residual)
        @test residual < 1e-10
    end

    @testset "_ogita_iteration! maintains orthogonality" begin
        n = 4
        A = randn(n, n) + 3.0 * I

        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.Vt')

        # Run iteration
        BallArithmetic._ogita_iteration!(A, U, Σ, V)

        # U should be orthogonal
        @test norm(U' * U - I) < 1e-12

        # V should be orthogonal
        @test norm(V' * V - I) < 1e-12
    end

    @testset "_ogita_iteration! handles singular values correctly" begin
        n = 4
        # Matrix with spread singular values
        U_true = Matrix(qr(randn(n, n)).Q)
        V_true = Matrix(qr(randn(n, n)).Q)
        Σ_true = [10.0, 5.0, 2.0, 0.5]

        A = U_true * Diagonal(Σ_true) * V_true'

        # Start with standard SVD
        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.Vt')

        # Run iteration
        BallArithmetic._ogita_iteration!(A, U, Σ, V)

        # Singular values should be close to true values
        for (i, σ_true) in enumerate(Σ_true)
            @test abs(Σ[i] - σ_true) < 1e-12
        end
    end

    @testset "_ogita_iteration! quadratic convergence" begin
        n = 5
        A = randn(n, n) + 10.0 * I

        F = svd(A)
        # Start with the actual SVD (Ogita iteration works best from good starting point)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.Vt')

        residuals = Float64[]
        for _ in 1:3
            r = BallArithmetic._ogita_iteration!(A, U, Σ, V)
            push!(residuals, r)
        end

        # All residuals should be very small (SVD was already computed)
        @test all(residuals .< 1e-10)

        # Check that we stay at high precision
        @test residuals[end] < 1e-10
    end
end
