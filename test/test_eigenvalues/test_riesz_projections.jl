using Test
using LinearAlgebra
using BallArithmetic

@testset "Riesz Projection Interfaces" begin

    @testset "Simple eigenspace projection - Hermitian case" begin
        # Symmetric matrix with well-separated eigenvalues
        A = [4.0 1.0; 1.0 3.0]
        F = eigen(Symmetric(A))

        v = [1.0, 2.0]

        # Project onto first eigenvalue's eigenspace
        v_proj = project_onto_eigenspace(v, F.vectors, 1:1; hermitian=true)

        # Check projection is idempotent: P(Pv) = Pv
        v_proj2 = project_onto_eigenspace(v_proj, F.vectors, 1:1; hermitian=true)
        @test norm(v_proj2 - v_proj) < 1e-10

        # Check that projected vector is in span of first eigenvector
        # Should be proportional to F.vectors[:, 1]
        e1 = F.vectors[:, 1]
        coeff = dot(e1, v_proj) / dot(e1, e1)
        @test norm(v_proj - coeff * e1) < 1e-10
    end

    @testset "Schur subspace projection" begin
        # Matrix with distinct eigenvalues
        A = [1.0 0.5; 0.0 3.0]  # Upper triangular
        F = schur(A)

        v = [1.0, 1.0]

        # Project onto first Schur vector
        v_proj = project_onto_schur_subspace(v, F.Z, 1:1)

        # Verify idempotency
        v_proj2 = project_onto_schur_subspace(v_proj, F.Z, 1:1)
        @test norm(v_proj2 - v_proj) < 1e-10

        # Check orthogonality to second Schur vector
        q2 = F.Z[:, 2]
        @test abs(dot(q2, v_proj)) < 1e-10
    end

    @testset "Eigenspace projector matrix - Hermitian" begin
        A = Diagonal([1.0, 2.0, 3.0])
        F = eigen(A)

        # Compute projector onto first two eigenvalues
        P = compute_eigenspace_projector(F.vectors, 1:2; hermitian=true)

        # Check idempotency: P² = P
        @test norm(P * P - P) < 1e-10

        # Check range: should project onto span of first two eigenvectors
        v = [1.0, 2.0, 3.0]
        v_proj = P * v

        # Projected vector should have zero component in third direction
        e3 = F.vectors[:, 3]
        @test abs(dot(e3, v_proj)) < 1e-10
    end

    @testset "Schur projector matrix" begin
        A = [2.0 1.0 0.0; 0.0 3.0 1.0; 0.0 0.0 4.0]
        F = schur(A)

        # Projector onto first two Schur vectors
        P = compute_schur_projector(F.Z, 1:2)

        # Check idempotency
        @test norm(P * P - P) < 1e-10

        # Check that P projects onto 2D subspace
        @test rank(P) == 2

        # Check orthogonality to third Schur vector
        q3 = F.Z[:, 3]
        @test norm(P * q3) < 1e-10
    end

    @testset "Spectral projector from Schur - Hermitian matrix" begin
        # Use Hermitian version (simpler, no Sylvester equation)
        A = BallMatrix([4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 5.0], fill(1e-10, 3, 3))

        # Compute projector onto first two eigenvalues
        result = compute_spectral_projector_hermitian(A, 1:2)

        @test result.idempotency_defect < 1e-8
        @test result.eigenvalue_separation > 0
        @test isfinite(result.projector_norm)

        # Verify properties
        @test verify_spectral_projector_properties(result, A; tol=1e-8)
    end

    @testset "Spectral projector from Schur - Upper triangular" begin
        # Simple upper triangular matrix (easier than general case)
        A_mat = [1.0 0.5 0.0; 0.0 2.0 0.3; 0.0 0.0 5.0]
        A = BallMatrix(A_mat, fill(1e-10, 3, 3))

        # Compute projector onto first eigenvalue (λ = 1.0)
        result = compute_spectral_projector_schur(A, 1:1)

        # Check that projector was computed
        @test size(result.projector) == (3, 3)
        @test result.idempotency_defect < 1e-6
        @test result.eigenvalue_separation > 0.5  # Gap between 1.0 and 2.0

        # Verify properties
        @test verify_spectral_projector_properties(result, A; tol=1e-6)

        # Project a vector
        v = BallVector([1.0, 2.0, 3.0], fill(1e-10, 3))
        v_proj = project_vector_spectral(v, result)

        @test length(v_proj) == 3
        @test all(isfinite.(v_proj.c))
    end

    @testset "Vector projection with BallVector" begin
        # Hermitian case with intervals
        A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-10, 2, 2))
        result = compute_spectral_projector_hermitian(A, 1:1)

        # Project interval vector
        v = BallVector([1.0, 1.0], [1e-10, 1e-10])
        v_proj = project_vector_spectral(v, result)

        @test length(v_proj.c) == 2
        @test all(isfinite.(v_proj.c))
        @test all(v_proj.r .>= 0)  # Radii should be non-negative
    end

    @testset "Eigenvalue separation diagnostics" begin
        # Matrix with one isolated and one pair of close eigenvalues
        A = BallMatrix(Diagonal([1.0, 5.0, 5.1]), fill(1e-10, 3, 3))

        # Projector for first eigenvalue (well-separated)
        result1 = compute_spectral_projector_hermitian(A, 1:1)
        @test result1.eigenvalue_separation >= 3.9  # Gap to 5.0

        # Projector for last two eigenvalues (close together)
        result2 = compute_spectral_projector_hermitian(A, 2:3)
        @test result2.eigenvalue_separation >= 3.9  # Gap to 1.0

        # Both should have small idempotency defect
        @test result1.idempotency_defect < 1e-8
        @test result2.idempotency_defect < 1e-8
    end

    @testset "Error handling" begin
        A = [1.0 0.5; 0.0 2.0]
        F = eigen(A)
        v = [1.0, 2.0]

        # Out of bounds indices
        @test_throws AssertionError project_onto_eigenspace(v, F.vectors, 1:3; hermitian=true)

        # Dimension mismatch
        v_wrong = [1.0, 2.0, 3.0]
        @test_throws AssertionError project_onto_eigenspace(v_wrong, F.vectors, 1:1; hermitian=true)
    end

end
