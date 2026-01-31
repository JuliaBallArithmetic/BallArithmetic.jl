"""
Test suite for precision cascade SVD refinement.

Tests the cascade through Float64 → Double64 → Float64x3 → Float64x4 → BigFloat
for efficient high-precision SVD certification.

Based on Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value
decomposition based on matrix multiplication", J. Comput. Appl. Math. 369, 112512.

NOTE: This test requires MultiFloats.jl to be installed:
    ] add MultiFloats
If not installed, the test is skipped (marked as broken in test summary).
"""

using Test
using BallArithmetic
using LinearAlgebra
using Random

# Tests for core functionality that doesn't require extensions
@testset "Ogita SVD Core Iteration" begin
    @testset "PrecisionCascadeSVDResult struct" begin
        # Create a dummy result
        n = 3
        U = randn(n, n)
        Σ = [3.0, 2.0, 1.0]
        V = randn(n, n)
        residual = 1e-10
        σ_min = Σ[end] - residual

        result = PrecisionCascadeSVDResult(U, Σ, V, residual, σ_min, 256)

        @test result.U === U
        @test result.Σ === Σ
        @test result.V === V
        @test result.residual_norm == residual
        @test result.σ_min == σ_min
        @test result.final_precision == 256
    end

    @testset "_ogita_iteration! basic functionality" begin
        Random.seed!(12345)
        n = 5
        A = randn(n, n) + 5.0 * I  # Well-conditioned matrix

        # Get initial SVD approximation
        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.V)

        # The initial SVD should be quite accurate for Float64
        initial_residual = norm(A - U * Diagonal(Σ) * V')

        # Run one Ogita iteration (should improve or maintain accuracy)
        new_residual = BallArithmetic._ogita_iteration!(A, U, Σ, V)

        @test new_residual >= 0  # Residual should be non-negative
        @test isfinite(new_residual)

        # After iteration, U and V should still be approximately orthogonal
        @test norm(U' * U - I) < 1e-10
        @test norm(V' * V - I) < 1e-10

        # The singular values should remain positive
        @test all(Σ .> 0)
    end

    @testset "_ogita_iteration! convergence" begin
        Random.seed!(54321)
        n = 8
        A = randn(n, n) + 4.0 * I

        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.V)

        # Run multiple iterations
        residuals = Float64[]
        for _ in 1:3
            res = BallArithmetic._ogita_iteration!(A, U, Σ, V)
            push!(residuals, res)
        end

        # All residuals should be small and finite
        @test all(r -> r >= 0 && isfinite(r), residuals)

        # Final reconstruction should be accurate
        reconstruction = U * Diagonal(Σ) * V'
        @test norm(A - reconstruction) < 1e-10
    end

    @testset "_ogita_iteration! with clustered singular values" begin
        # Matrix with clustered singular values
        n = 4
        # Create matrix with singular values [4.0, 4.0001, 4.0002, 1.0]
        Σ_target = [4.0, 4.0001, 4.0002, 1.0]
        U_random = qr(randn(n, n)).Q
        V_random = qr(randn(n, n)).Q
        A = Matrix(U_random) * Diagonal(Σ_target) * Matrix(V_random)'

        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.V)

        # Run iteration - should handle clustered singular values
        residual = BallArithmetic._ogita_iteration!(A, U, Σ, V)

        @test residual >= 0
        @test isfinite(residual)

        # Singular values should remain in decreasing order
        for i in 1:n-1
            @test Σ[i] >= Σ[i+1] - 1e-10  # Allow small tolerance
        end
    end

    @testset "_ogita_iteration! with complex matrices" begin
        Random.seed!(98765)
        n = 5
        A = randn(ComplexF64, n, n) + 5.0 * I

        F = svd(A)
        U = copy(F.U)
        Σ = copy(F.S)
        V = copy(F.V)

        residual = BallArithmetic._ogita_iteration!(A, U, Σ, V)

        @test residual >= 0
        @test isfinite(residual)

        # Check orthogonality for complex case
        @test norm(U' * U - I) < 1e-10
        @test norm(V' * V - I) < 1e-10
    end
end

# Check if MultiFloats extension is available
const HAS_CASCADE = hasmethod(ogita_svd_cascade, Tuple{Matrix{Complex{BigFloat}}, Complex{BigFloat}})

@testset "Precision Cascade SVD" begin
    if !HAS_CASCADE
        # OPTIONAL DEPENDENCY: MultiFloats.jl provides Float64x2, Float64x3, Float64x4
        # types for extended precision without the overhead of BigFloat.
        # Install with: ] add MultiFloats
        @warn "ogita_svd_cascade not available (MultiFloats extension not loaded)"
        @test_skip true
        return
    end

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
