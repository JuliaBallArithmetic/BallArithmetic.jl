"""
Test suite for adaptive Ogita SVD refinement.

Tests the iterative refinement algorithm from:
Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value
decomposition based on matrix multiplication", J. Comput. Appl. Math. 369, 112512.
"""

using Test
using BallArithmetic
using LinearAlgebra

@testset "Adaptive Ogita SVD" begin

    @testset "Basic 3×3 Example" begin
        # Create a simple test matrix with well-separated singular values
        A_mid = Diagonal([10.0, 5.0, 1.0])
        A_rad = zeros(3, 3)
        A = BallMatrix(A_mid, A_rad)

        # Test with tight tolerance
        result = adaptive_ogita_svd(A; tolerance=1e-12, max_refinement_iterations=3)

        @test result.tolerance_achieved
        @test maximum(rad.(result.rigorous_result.singular_values)) < 1e-12
        @test result.final_precision >= 53  # At least Float64

        # Check that singular values are close to expected
        σ = result.rigorous_result.singular_values
        @test mid(σ[1]) ≈ 10.0 atol=1e-10
        @test mid(σ[2]) ≈ 5.0 atol=1e-10
        @test mid(σ[3]) ≈ 1.0 atol=1e-10
    end

    @testset "Matrix with Uncertainty" begin
        # Matrix with some off-diagonal uncertainty
        A_mid = [3.0 0.0; 0.0 2.0]
        A_rad = [1e-10 1e-12; 1e-12 1e-10]
        A = BallMatrix(A_mid, A_rad)

        result = adaptive_ogita_svd(A; tolerance=1e-10, max_refinement_iterations=2)

        # Should achieve tolerance or get close
        @test result.final_precision > 53
        @test !isempty(result.precision_levels)
        @test length(result.radii_history) == length(result.precision_levels)

        # Radii should be decreasing
        for i in 2:length(result.radii_history)
            @test result.radii_history[i] <= result.radii_history[i-1]
        end
    end

    @testset "Precision Doubling" begin
        # Test that precision actually doubles
        A_mid = Diagonal([5.0, 3.0])
        A = BallMatrix(A_mid, zeros(2, 2))

        result = adaptive_ogita_svd(A; tolerance=1e-50, max_refinement_iterations=4)

        # Check precision progression
        @test result.precision_levels[1] == 53  # Float64
        if length(result.precision_levels) > 1
            for i in 2:length(result.precision_levels)
                # Each level should roughly double
                @test result.precision_levels[i] >= result.precision_levels[i-1] * 1.5
            end
        end
    end

    @testset "Early Termination" begin
        # Matrix that already has very small uncertainties
        A_mid = [2.0 0.0; 0.0 1.0]
        A_rad = fill(1e-15, 2, 2)
        A = BallMatrix(A_mid, A_rad)

        result = adaptive_ogita_svd(A; tolerance=1e-10)

        # Should converge quickly (possibly in one iteration or immediately)
        @test result.tolerance_achieved
        @test length(result.precision_levels) <= 2
    end

    @testset "Different SVD Methods" begin
        A_mid = Diagonal([8.0, 4.0, 2.0])
        A = BallMatrix(A_mid, fill(1e-12, 3, 3))

        # Test with MiyajimaM1
        result_m1 = adaptive_ogita_svd(A; method=MiyajimaM1(), tolerance=1e-15)
        @test result_m1.tolerance_achieved || maximum(rad.(result_m1.rigorous_result.singular_values)) < 1e-14

        # Test with RumpOriginal
        result_rump = adaptive_ogita_svd(A; method=RumpOriginal(), tolerance=1e-15)
        @test result_rump.tolerance_achieved || maximum(rad.(result_rump.rigorous_result.singular_values)) < 1e-14
    end

    @testset "VBD Integration" begin
        # Test with and without VBD
        A_mid = Diagonal([10.0, 9.9, 5.0])  # Two close values
        A = BallMatrix(A_mid, fill(1e-10, 3, 3))

        result_with_vbd = adaptive_ogita_svd(A; apply_vbd=true, tolerance=1e-12)
        result_without_vbd = adaptive_ogita_svd(A; apply_vbd=false, tolerance=1e-12)

        # Both should work
        @test !isnothing(result_with_vbd.rigorous_result)
        @test !isnothing(result_without_vbd.rigorous_result)

        # With VBD should have cluster information
        if result_with_vbd.rigorous_result.block_diagonalisation !== nothing
            @test !isempty(result_with_vbd.rigorous_result.block_diagonalisation.clusters)
        end
    end

    @testset "Rectangular Matrices" begin
        # Test m > n case
        A_mid = [3.0 0.0; 2.0 0.0; 0.0 1.0]
        A = BallMatrix(A_mid, fill(1e-12, 3, 2))

        result = adaptive_ogita_svd(A; tolerance=1e-10)

        # Should have 2 singular values (min dimension)
        @test length(result.rigorous_result.singular_values) == 2
    end

    @testset "Maximum Precision Limit" begin
        A_mid = Diagonal([3.0, 2.0])
        A = BallMatrix(A_mid, fill(1e-10, 2, 2))

        # Set very tight tolerance and low max precision
        result = adaptive_ogita_svd(A; tolerance=1e-100, max_precision_bits=128, max_refinement_iterations=3)

        # Should stop at max precision even if tolerance not met
        @test result.final_precision <= 128
        @test length(result.precision_levels) <= 4  # Initial + 3 refinements
    end

    @testset "Comparison with Standard rigorous_svd" begin
        # Compare results with standard method
        A_mid = [4.0 1.0; 1.0 3.0]
        A_rad = fill(1e-14, 2, 2)
        A = BallMatrix(A_mid, A_rad)

        # Standard method
        result_standard = rigorous_svd(A; method=MiyajimaM1())

        # Adaptive method (should refine)
        result_adaptive = adaptive_ogita_svd(A; tolerance=1e-18, max_refinement_iterations=2)

        # Adaptive should have tighter or equal bounds
        for i in 1:length(result_standard.singular_values)
            @test rad(result_adaptive.rigorous_result.singular_values[i]) <=
                  rad(result_standard.singular_values[i]) * 1.1  # Allow small overhead
        end
    end

end

println("All adaptive Ogita SVD tests passed!")
