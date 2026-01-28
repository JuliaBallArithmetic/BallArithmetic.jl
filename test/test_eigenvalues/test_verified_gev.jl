using Test
using BallArithmetic
using LinearAlgebra

@testset "Verified Generalized Eigenvalue Problems" begin

    @testset "Beta Bound Computation (Theorem 10)" begin
        # Simple diagonal SPD matrix
        B = BallMatrix([2.0 0.0; 0.0 3.0], fill(1e-10, 2, 2))
        β = compute_beta_bound(B)

        @test β > 0
        @test isfinite(β)

        # β is a rigorous upper bound on ‖B^{-1/2}‖₂, computed via Cholesky
        # with error accounting. For diagonal B = diag(2,3), the ideal value
        # is 1/√λ_min = 1/√2 ≈ 0.707, but the rigorous bound is larger due to
        # rounding error accounting. The bound should be within a reasonable
        # factor of the ideal value.
        @test β >= 1/sqrt(2.0)  # Must be at least the ideal value
        @test β < 2.0  # Should not be excessively large

        # Non-diagonal SPD matrix
        B2 = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        β2 = compute_beta_bound(B2)
        @test β2 > 0
        @test isfinite(β2)
    end

    @testset "Small 2×2 System" begin
        # Simple generalized eigenvalue problem with known solution
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        # Get approximate solution
        F = eigen(Symmetric(A.c), Symmetric(B.c))
        X̃ = F.vectors
        λ̃ = F.values

        # Verify
        result = verify_generalized_eigenpairs(A, B, X̃, λ̃)

        @test result.success
        @test length(result.eigenvalue_intervals) == 2
        @test length(result.eigenvector_radii) == 2
        @test all(isfinite.(result.eigenvector_radii))

        # Check that approximate eigenvalues are inside intervals
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= λ̃[i] <= λ_upper
        end

        # Check diagnostic information
        @test result.beta > 0
        @test result.global_bound > 0
        @test all(result.individual_bounds .> 0)
        @test all(result.separation_bounds .> 0)
    end

    @testset "Diagonal Matrices (Easy Case)" begin
        # When both A and B are diagonal, eigenvalues are A[i,i]/B[i,i]
        A = BallMatrix(Diagonal([4.0, 9.0, 16.0]), fill(1e-10, 3, 3))
        B = BallMatrix(Diagonal([2.0, 3.0, 4.0]), fill(1e-10, 3, 3))

        # Expected eigenvalues: [2.0, 3.0, 4.0]
        expected = [4.0/2.0, 9.0/3.0, 16.0/4.0]

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # Check eigenvalues are in intervals
        for i in 1:3
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= expected[i] <= λ_upper
            # Intervals should be tight for this easy problem
            @test λ_upper - λ_lower < 0.1
        end
    end

    @testset "3×3 System with Well-Separated Eigenvalues" begin
        A = BallMatrix([10.0 1.0 0.5;
                        1.0  5.0 0.2;
                        0.5  0.2 2.0], fill(1e-8, 3, 3))

        B = BallMatrix([2.0 0.0 0.0;
                        0.0 2.0 0.0;
                        0.0 0.0 2.0], fill(1e-8, 3, 3))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success
        @test length(result.eigenvalue_intervals) == 3

        # Well-separated eigenvalues should have good bounds
        for i in 1:3
            @test result.separation_bounds[i] > 0
            @test result.eigenvector_radii[i] < 1.0  # Should be small
        end

        # Intervals should be non-overlapping
        for i in 1:2
            λ_i_upper = result.eigenvalue_intervals[i][2]
            λ_ip1_lower = result.eigenvalue_intervals[i+1][1]
            @test λ_i_upper < λ_ip1_lower
        end
    end

    @testset "Residual Matrix Computation" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(0.01, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(0.01, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        X̃ = F.vectors
        λ̃ = F.values

        Rg = BallArithmetic.compute_residual_matrix(A, B, X̃, λ̃)

        @test size(Rg) == (2, 2)

        # Residual should be small for good approximate eigenpairs
        norm_Rg = maximum([abs(Rg[i, j].c) + Rg[i, j].r for i in 1:2, j in 1:2])
        @test norm_Rg < 0.1
    end

    @testset "Gram Matrix Computation" begin
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        # Use orthonormal vectors (should give identity)
        X̃ = Matrix{Float64}(I, 2, 2)

        Gg = BallArithmetic.compute_gram_matrix(B, X̃)

        @test size(Gg) == (2, 2)

        # X̃ᵀBX̃ should equal B for identity matrix
        # Check diagonal elements
        for i in 1:2
            if isa(Gg[i, i], Ball)
                @test abs(Gg[i, i].c - B.c[i, i]) < 0.01
            else
                @test abs(Gg[i, i] - B.c[i, i]) < 0.01
            end
        end
    end

    @testset "Global vs Individual Bounds" begin
        A = BallMatrix([5.0 1.0; 1.0 4.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.0; 0.0 2.0], fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # Individual bounds should be tighter than or equal to global bound
        for i in 1:2
            @test result.individual_bounds[i] <= result.global_bound + 1e-10
        end

        # Separation bounds should be at most the individual bounds
        for i in 1:2
            @test result.separation_bounds[i] <= result.individual_bounds[i] + 1e-10
        end
    end

    @testset "Eigenvalue Separation (Lemma 2)" begin
        # Clustered eigenvalues
        λ̃ = [1.0, 1.1, 1.2, 2.0, 3.0]
        δ̂ = 0.5
        ε = [0.3, 0.3, 0.3, 0.4, 0.5]

        η = BallArithmetic.compute_eigenvalue_separation(λ̃, δ̂, ε)

        @test length(η) == 5
        @test all(η .<= δ̂)
        @test all(η .<= ε)

        # Check that intervals don't overlap
        for i in 1:4
            @test λ̃[i] + η[i] <= λ̃[i+1] - η[i+1] + 1e-10
        end

        # Well-separated eigenvalues should keep their bounds
        @test η[5] ≈ min(δ̂, ε[5]) atol=1e-10
    end

    @testset "B = Identity (Standard Eigenvalue Problem)" begin
        # When B = I, this reduces to standard eigenvalue problem
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix(Matrix{Float64}(I, 2, 2), fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c))  # Standard eigenvalue problem
        X̃ = F.vectors
        λ̃ = F.values

        result = verify_generalized_eigenpairs(A, B, X̃, λ̃)

        @test result.success

        # Compare with true eigenvalues of A
        true_λ = eigvals(Symmetric(A.c))
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= true_λ[i] <= λ_upper
        end
    end

    @testset "Larger Matrices (4×4)" begin
        n = 4
        # Create random SPD matrices
        A_center = randn(n, n)
        A_center = A_center * A_center' + 5.0 * I  # Make SPD
        A = BallMatrix(A_center, fill(1e-8, n, n))

        B_center = randn(n, n)
        B_center = B_center * B_center' + 2.0 * I  # Make SPD
        B = BallMatrix(B_center, fill(1e-8, n, n))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success
        @test length(result.eigenvalue_intervals) == n
        @test length(result.eigenvector_radii) == n

        # All bounds should be finite
        @test all(isfinite.(result.separation_bounds))
        @test all(isfinite.(result.eigenvector_radii))
    end

    @testset "Interval Matrices with Larger Uncertainties" begin
        # Test with significant uncertainties
        A = BallMatrix([10.0 2.0; 2.0 8.0], fill(0.1, 2, 2))  # 10% uncertainty
        B = BallMatrix([3.0 0.5; 0.5 3.0], fill(0.05, 2, 2))  # 5% uncertainty

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # Bounds should be larger due to uncertainties
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            interval_width = λ_upper - λ_lower
            @test interval_width > 0.01  # Should have non-trivial width
        end
    end

    @testset "Error Handling: Non-Square Matrices" begin
        A = BallMatrix(randn(3, 2), fill(1e-10, 3, 2))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))
        X̃ = randn(3, 3)
        λ̃ = randn(3)

        result = verify_generalized_eigenpairs(A, B, X̃, λ̃)
        @test !result.success
        @test contains(result.message, "square")
    end

    @testset "Error Handling: Dimension Mismatch" begin
        A = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))
        X̃ = randn(3, 2)  # Wrong size
        λ̃ = randn(3)

        result = verify_generalized_eigenpairs(A, B, X̃, λ̃)
        @test !result.success
    end

    @testset "Poor Approximate Solution" begin
        # Use random vectors instead of eigenvectors
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        X̃ = randn(2, 2)
        λ̃ = randn(2)
        λ̃ = sort(λ̃)  # Sort for separation algorithm

        result = verify_generalized_eigenpairs(A, B, X̃, λ̃)

        # Should fail or give very large bounds
        if result.success
            @test any(result.separation_bounds .> 10.0) || any(result.eigenvector_radii .> 10.0)
        else
            @test !result.success
        end
    end

    @testset "Nearly Clustered Eigenvalues" begin
        # Create matrix with two close eigenvalues
        A = BallMatrix([5.0 0.0 0.0;
                        0.0 5.01 0.0;
                        0.0 0.0 10.0], fill(1e-10, 3, 3))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        @test result.success

        # First two eigenvalues are very close
        # Separation bounds should be small for them
        @test result.separation_bounds[1] < 0.01
        @test result.separation_bounds[2] < 0.01

        # Third eigenvalue is well-separated
        @test result.separation_bounds[3] > result.separation_bounds[1]
    end

    @testset "Verification Diagnostics" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

        # Check all diagnostic fields are present and reasonable
        @test result.beta > 0
        @test isfinite(result.beta)

        @test result.global_bound > 0
        @test isfinite(result.global_bound)

        @test length(result.individual_bounds) == 2
        @test all(result.individual_bounds .> 0)

        @test length(result.separation_bounds) == 2
        @test all(result.separation_bounds .> 0)

        @test result.residual_norm >= 0
        @test isfinite(result.residual_norm)

        @test !isempty(result.message)
    end

end
