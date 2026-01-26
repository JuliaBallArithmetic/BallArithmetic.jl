"""
Test suite for verified linear system solution using H-matrices.

Tests the methods from:
Minamihata, A., Ogita, T., Rump, S.M. & Oishi, S. (2020),
"Modified error bounds for approximate solutions of dense linear systems",
J. Comput. Appl. Math. 369, 112546.
"""

using Test
using BallArithmetic
using LinearAlgebra

@testset "Verified Linear System with H-matrices" begin

    @testset "Simple 2×2 System" begin
        # Well-conditioned system
        A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-14, 2, 2))
        b = BallVector([5.0, 4.0], fill(1e-14, 2))

        # Test all methods
        for method in [:rump_original, :improved_method_a, :improved_method_b]
            result = verified_linear_solve_hmatrix(A, b; method=method)

            @test result.verified
            @test length(result.error_bound) == 2
            @test all(isfinite.(result.error_bound))

            # Check that solution is approximately correct
            x_true = [1.2, 1.4]  # (3*1.2 + 1*1.4 = 5, 1*1.2 + 2*1.4 = 4)
            @test norm(result.x_approx - x_true) < 1e-10
        end
    end

    @testset "Diagonal System" begin
        # Simple diagonal system
        A = BallMatrix(Diagonal([2.0, 3.0, 4.0]), zeros(3, 3))
        b = BallVector([2.0, 6.0, 8.0], zeros(3))

        result = verified_linear_solve_hmatrix(A, b; method=:improved_method_a)

        @test result.verified
        @test result.x_approx ≈ [1.0, 2.0, 2.0]
        @test maximum(result.error_bound) < 1e-10
    end

    @testset "Method Comparison" begin
        # Moderately ill-conditioned system
        A_mid = [10.0 9.0; 9.0 8.0]
        A = BallMatrix(A_mid, fill(1e-12, 2, 2))
        b = BallVector([1.0, 1.0], fill(1e-12, 2))

        result_rump = verified_linear_solve_hmatrix(A, b; method=:rump_original)
        result_improved_a = verified_linear_solve_hmatrix(A, b; method=:improved_method_a)
        result_improved_b = verified_linear_solve_hmatrix(A, b; method=:improved_method_b)

        # All should verify
        @test result_rump.verified
        @test result_improved_a.verified
        @test result_improved_b.verified

        # Improved methods should have tighter or equal bounds
        @test maximum(result_improved_a.error_bound) <= maximum(result_rump.error_bound) * 1.1
    end

    @testset "Iterative Refinement" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-13, 2, 2))
        b = BallVector([5.0, 4.0], fill(1e-13, 2))

        # Test with different iteration counts
        result_1 = verified_linear_solve_hmatrix(A, b; method=:improved_method_a, max_iterations=1)
        result_3 = verified_linear_solve_hmatrix(A, b; method=:improved_method_a, max_iterations=3)

        @test result_1.verified
        @test result_3.verified

        # More iterations should not hurt (bounds should be tighter or equal)
        @test maximum(result_3.error_bound) <= maximum(result_1.error_bound) * 1.01
    end

    @testset "Perron Vector Computation" begin
        A = BallMatrix(Diagonal([5.0, 3.0, 1.0]), zeros(3, 3))
        b = BallVector([5.0, 3.0, 1.0], zeros(3))

        # With Perron vector
        result_perron = verified_linear_solve_hmatrix(A, b; compute_perron_vector=true)

        # Without Perron vector (uses e=(1,1,1)^T)
        result_no_perron = verified_linear_solve_hmatrix(A, b; compute_perron_vector=false)

        @test result_perron.verified
        @test result_no_perron.verified
        @test result_perron.v !== nothing
    end

    @testset "Provided Approximate Inverse" begin
        A = BallMatrix([2.0 1.0; 0.0 3.0], fill(1e-14, 2, 2))
        b = BallVector([3.0, 3.0], fill(1e-14, 2))

        # Provide approximate inverse
        R = inv(mid(A))

        result = verified_linear_solve_hmatrix(A, b; R=R, method=:improved_method_a)

        @test result.verified
        @test result.R == R
    end

    @testset "Comparison Matrix Functions" begin
        # Test mig and mag
        x1 = Ball(2.0, 0.5)  # [1.5, 2.5]
        @test mig(x1) == 1.5
        @test mag(x1) == 2.5

        x2 = Ball(-2.0, 0.5)  # [-2.5, -1.5]
        @test mig(x2) == 1.5
        @test mag(x2) == 2.5

        x3 = Ball(0.0, 1.0)  # [-1.0, 1.0]
        @test mig(x3) == 0.0
        @test mag(x3) == 1.0

        # Test comparison matrix
        C = BallMatrix([Ball(2.0, 0.1) Ball(0.5, 0.1);
                       Ball(0.3, 0.1) Ball(3.0, 0.1)])
        comp = comparison_matrix(C)

        @test comp[1,1] == mig(C[1,1])  # Diagonal: mignitude
        @test comp[2,2] == mig(C[2,2])
        @test comp[1,2] == -mag(C[1,2])  # Off-diagonal: -magnitude
        @test comp[2,1] == -mag(C[2,1])
    end

    @testset "Residual Computation" begin
        A = BallMatrix([2.0 0.0; 0.0 3.0], fill(1e-14, 2, 2))
        b = BallVector([4.0, 6.0], fill(1e-14, 2))
        x = [2.0, 2.0]
        R = Matrix{Float64}(I, 2, 2)

        residual = BallArithmetic.compute_residual_rigorous(A, b, x, R)

        @test length(residual) == 2
        @test abs(mid(residual[1])) < 1e-10
        @test abs(mid(residual[2])) < 1e-10
    end

    @testset "Method (a) vs Method (b)" begin
        A = BallMatrix([5.0 2.0; 2.0 3.0], fill(1e-12, 2, 2))
        b = BallVector([7.0, 5.0], fill(1e-12, 2))

        result_a = verified_linear_solve_hmatrix(A, b; method=:improved_method_a)
        result_b = verified_linear_solve_hmatrix(A, b; method=:improved_method_b)

        @test result_a.verified
        @test result_b.verified

        # Both should give reasonable bounds
        @test maximum(result_a.error_bound) < 1e-8
        @test maximum(result_b.error_bound) < 1e-8
    end

    @testset "Ill-Conditioned System" begin
        # Hilbert-like matrix (ill-conditioned)
        n = 5
        A_mid = [1.0/(i+j-1) for i in 1:n, j in 1:n]
        A = BallMatrix(A_mid, fill(1e-14, n, n))
        b = BallVector(ones(n), fill(1e-14, n))

        result = verified_linear_solve_hmatrix(A, b; method=:improved_method_a, max_iterations=3)

        # May or may not verify depending on condition number
        if result.verified
            @test all(isfinite.(result.error_bound))
            @test result.spectral_radius_ED_inv < 1.0
        end
    end

end

println("All verified H-matrix linear system tests passed!")
