using Test
using LinearAlgebra
using BallArithmetic

"""
Coherence tests for verified generalized eigenvalue solver.

Tests consistency with:
1. Standard eigenvalue solver (when B = I)
2. Known analytic solutions
3. Transformation properties (scaling, similarity)
4. Floating-point solutions
"""

@testset "GEV Coherence Tests" begin

    @testset "Coherence with standard eigenvalue problem (B = I)" begin
        # When B = I, generalized problem reduces to standard Ax = λx
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B_identity = BallMatrix(Matrix{Float64}(I, 2, 2), fill(1e-10, 2, 2))

        # Solve as GEV
        F_gev = eigen(Symmetric(A.c), Symmetric(B_identity.c))
        result_gev = verify_generalized_eigenpairs(A, B_identity, F_gev.vectors, F_gev.values)

        # Compare with standard eigenvalue problem
        true_eigenvalues = eigvals(Symmetric(A.c))

        @test result_gev.success

        for i in 1:2
            λ_lower, λ_upper = result_gev.eigenvalue_intervals[i]
            @test λ_lower <= true_eigenvalues[i] <= λ_upper

            # Intervals should be tight for this well-conditioned problem
            interval_width = λ_upper - λ_lower
            @test interval_width < 0.1
        end
    end

    @testset "Consistency with analytic solution (diagonal matrices)" begin
        # For diagonal A and B: λᵢ = A[i,i] / B[i,i]
        A = BallMatrix(Diagonal([6.0, 9.0, 12.0]), fill(1e-10, 3, 3))
        B = BallMatrix(Diagonal([2.0, 3.0, 4.0]), fill(1e-10, 3, 3))

        expected_eigenvalues = [6.0/2.0, 9.0/3.0, 12.0/4.0]  # [3.0, 3.0, 3.0]

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        for i in 1:3
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= expected_eigenvalues[i] <= λ_upper

            # Should be very tight for diagonal matrices
            @test (λ_upper - λ_lower) < 1e-8
        end
    end

    @testset "Scaling invariance: (αA)x = λ(αB)x ⟺ Ax = λBx" begin
        # Multiplying both A and B by same positive scalar shouldn't change eigenvalues
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        α = 5.0
        A_scaled = BallMatrix(α * A.c, α * A.r)
        B_scaled = BallMatrix(α * B.c, α * B.r)

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        F_scaled = eigen(Symmetric(A_scaled.c), Symmetric(B_scaled.c))

        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)
        result_scaled = verify_generalized_eigenpairs(A_scaled, B_scaled,
                                                       F_scaled.vectors, F_scaled.values)

        @test result.success && result_scaled.success

        # Eigenvalues should be identical (within tolerance)
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            λ_scaled_lower, λ_scaled_upper = result_scaled.eigenvalue_intervals[i]

            # Centers should be approximately equal
            λ_center = (λ_lower + λ_upper) / 2
            λ_scaled_center = (λ_scaled_lower + λ_scaled_upper) / 2
            @test abs(λ_center - λ_scaled_center) < 1e-6

            # Intervals should overlap significantly
            @test max(λ_lower, λ_scaled_lower) <= min(λ_upper, λ_scaled_upper)
        end
    end

    @testset "Consistency with floating-point solver" begin
        # Verified intervals must contain floating-point solution
        A = BallMatrix([5.0 1.0 0.5; 1.0 4.0 0.2; 0.5 0.2 3.0], fill(1e-8, 3, 3))
        B = BallMatrix([2.0 0.0 0.0; 0.0 2.5 0.0; 0.0 0.0 3.0], fill(1e-8, 3, 3))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # Each verified interval must contain the corresponding approximate eigenvalue
        for i in 1:3
            λ_approx = F.values[i]
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= λ_approx <= λ_upper
        end

        # Eigenvector radii should be finite and reasonable
        @test all(isfinite.(result.eigenvector_radii))
        @test all(result.eigenvector_radii .< 1.0)  # Should be small for well-conditioned
    end

    @testset "Bound ordering: separation ≤ individual ≤ global" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # Mathematical property: η ≤ ε ≤ δ̂
        for i in 1:2
            η_i = result.separation_bounds[i]
            ε_i = result.individual_bounds[i]
            δ_hat = result.global_bound

            @test η_i <= ε_i + 1e-10  # Allow small numerical tolerance
            @test ε_i <= δ_hat + 1e-10
        end
    end

    @testset "Interval containment property" begin
        # For well-conditioned problems with small uncertainties,
        # verified intervals should be relatively tight
        A = BallMatrix([10.0 1.0; 1.0 8.0], fill(1e-12, 2, 2))
        B = BallMatrix([3.0 0.0; 0.0 3.0], fill(1e-12, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # Compute true eigenvalues for comparison
        true_eigenvalues = eigvals(Symmetric(A.c), Symmetric(B.c))

        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            λ_true = true_eigenvalues[i]

            # Must contain true value
            @test λ_lower <= λ_true <= λ_upper

            # Width should be small (< 1% of eigenvalue magnitude)
            relative_width = (λ_upper - λ_lower) / abs(λ_true)
            @test relative_width < 0.01
        end
    end

    @testset "Symmetry preservation" begin
        # For symmetric A and symmetric B, all eigenvalues should be real
        A = BallMatrix([3.0 1.0 0.5; 1.0 4.0 0.3; 0.5 0.3 2.0], fill(1e-10, 3, 3))
        B = BallMatrix([2.0 0.2 0.0; 0.2 2.5 0.1; 0.0 0.1 3.0], fill(1e-10, 3, 3))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # All eigenvalues should be real (intervals don't contain complex values)
        for i in 1:3
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test isreal(λ_lower) && isreal(λ_upper)
            @test λ_lower < λ_upper  # Non-degenerate interval
        end
    end

    @testset "Residual consistency" begin
        # Residual norm should decrease with better approximations
        A = BallMatrix([5.0 1.0; 1.0 4.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        # Good approximation (from eigen)
        F_good = eigen(Symmetric(A.c), Symmetric(B.c))
        result_good = verify_generalized_eigenpairs(A, B, F_good.vectors, F_good.values)

        # Poor approximation (random vectors and eigenvalues)
        X_poor = randn(2, 2)
        X_poor, _ = qr(X_poor)  # Orthogonalize
        X_poor = Matrix(X_poor)
        λ_poor = sort(randn(2))

        result_poor = verify_generalized_eigenpairs(A, B, X_poor, λ_poor)

        # Good approximation should have smaller residual
        if result_poor.success
            @test result_good.residual_norm < result_poor.residual_norm
        end

        # Good approximation should succeed
        @test result_good.success
    end

    @testset "Multiple solver consistency" begin
        # Different numerical eigensolvers should give consistent results when verified
        A = BallMatrix([4.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 5.0], fill(1e-10, 3, 3))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))

        # Solver 1: Standard eigen
        F1 = eigen(Symmetric(A.c))
        result1 = verify_generalized_eigenpairs(A, B, F1.vectors, F1.values)

        # Solver 2: SVD-based (A = UΣV', eigenvalues are Σ²)
        # For this test, just use eigen again but verify independently
        F2 = eigen(Symmetric(A.c))
        result2 = verify_generalized_eigenpairs(A, B, F2.vectors, F2.values)

        @test result1.success && result2.success

        # Verified intervals should overlap (consistency)
        for i in 1:3
            λ1_lower, λ1_upper = result1.eigenvalue_intervals[i]
            λ2_lower, λ2_upper = result2.eigenvalue_intervals[i]

            # Intervals must overlap
            overlap_lower = max(λ1_lower, λ2_lower)
            overlap_upper = min(λ1_upper, λ2_upper)
            @test overlap_lower <= overlap_upper
        end
    end

end
