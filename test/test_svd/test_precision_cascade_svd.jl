"""
Test suite for precision cascade SVD refinement.

Tests the cascade through Float64 → Double64 → Float64x3 → Float64x4 → BigFloat
for efficient high-precision SVD certification.

Based on Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value
decomposition based on matrix multiplication", J. Comput. Appl. Math. 369, 112512.
"""

using Test
using BallArithmetic
using MultiFloats  # Required to trigger MultiFloatsExt which implements ogita_svd_cascade
using LinearAlgebra
using Random

@testset "Precision Cascade SVD" begin

    @testset "Basic functionality" begin
        # Small matrix to test correctness
        Random.seed!(42)
        n = 10
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade(T_bf, z_bf)

        # Check result structure
        @test result isa PrecisionCascadeSVDResult
        @test size(result.U) == (n, n)
        @test length(result.Σ) == n
        @test size(result.V) == (n, n)
        @test result.residual_norm >= 0
        @test result.final_precision == 256

        # σ_min should be positive (matrix is shifted away from spectrum)
        @test result.σ_min > 0

        # Residual should be small (convergence check)
        @test Float64(result.residual_norm) < 1e-10
    end

    @testset "Result consistency" begin
        # Check that cascade result is consistent with direct computation
        Random.seed!(123)
        n = 20
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade(T_bf, z_bf)

        # Reconstruct A and check residual
        A_bf = T_bf - z_bf * I
        reconstruction = result.U * Diagonal(result.Σ) * result.V'
        residual = A_bf - reconstruction
        computed_residual = sqrt(real(sum(abs2, residual)))

        # Computed residual should match reported residual
        @test Float64(computed_residual) ≈ Float64(result.residual_norm) rtol=0.1
    end

    @testset "Singular value ordering" begin
        Random.seed!(456)
        n = 15
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade(T_bf, z_bf)

        # Singular values should be in decreasing order
        for i in 1:n-1
            @test result.Σ[i] >= result.Σ[i+1]
        end
    end

    @testset "Custom iteration counts" begin
        Random.seed!(789)
        n = 10
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        # Test with more BigFloat iterations for higher accuracy
        result_more_bf = ogita_svd_cascade(T_bf, z_bf; bf_iters=4)

        # Test with fewer iterations
        result_fewer = ogita_svd_cascade(T_bf, z_bf;
            f64_iters=0, d64_iters=1, mf3_iters=1, mf4_iters=1, bf_iters=2)

        # Both should give valid results
        @test result_more_bf.σ_min > 0
        @test result_fewer.σ_min > 0

        # More iterations should give smaller or equal residual
        @test result_more_bf.residual_norm <= result_fewer.residual_norm * 1.1
    end

    @testset "Orthogonality of U and V" begin
        Random.seed!(111)
        n = 15
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade(T_bf, z_bf)

        # U should be approximately unitary
        UU = result.U' * result.U
        I_n = Matrix{Complex{BigFloat}}(I, n, n)
        U_orthog_error = maximum(abs.(UU - I_n))
        @test Float64(U_orthog_error) < 1e-10

        # V should be approximately unitary
        VV = result.V' * result.V
        V_orthog_error = maximum(abs.(VV - I_n))
        @test Float64(V_orthog_error) < 1e-10
    end

    @testset "Certified σ_min bound" begin
        Random.seed!(222)
        n = 20
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade(T_bf, z_bf)

        # The certified σ_min should be Σ[end] - residual_norm
        expected_σ_min = result.Σ[end] - result.residual_norm
        @test result.σ_min == expected_σ_min

        # σ_min should be a valid lower bound
        @test result.σ_min <= result.Σ[end]
    end

    @testset "Moderate size matrix (50×50)" begin
        Random.seed!(333)
        n = 50
        T = randn(n, n) + 5.0 * I

        setprecision(BigFloat, 256)
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade(T_bf, z_bf)

        @test result.σ_min > 0
        @test Float64(result.residual_norm) < 1e-8
        @test size(result.U) == (n, n)
    end

end

println("All precision cascade SVD tests passed!")
