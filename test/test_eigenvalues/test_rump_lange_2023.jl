using Test
using LinearAlgebra
using BallArithmetic

@testset "RumpLange2023 Cluster Bounds" begin
    @testset "Isolated eigenvalues (no clusters)" begin
        A = BallMatrix(Diagonal([1.0, 5.0, 10.0]))
        result = rump_lange_2023_cluster_bounds(A; hermitian=true)

        @test length(result) == 3
        @test result.num_clusters == 3  # All isolated
        @test result.verified

        # Each eigenvalue should be in its own cluster
        @test all(==(1), result.cluster_sizes)

        # True eigenvalues should be contained
        λ_true = [1.0, 5.0, 10.0]
        for i in 1:3
            @test λ_true[i] ∈ result.eigenvalues[i]
        end
    end

    @testset "Two clusters" begin
        # Create matrix with two clusters: {1.0, 1.1} and {10.0}
        A_mid = Matrix(Diagonal([1.0, 1.1, 10.0]))
        A_rad = zeros(size(A_mid))
        A_rad[1, 2] = A_rad[2, 1] = 0.2  # Couple first two
        A = BallMatrix(A_mid, A_rad)

        result = rump_lange_2023_cluster_bounds(A; hermitian=true, cluster_tol=0.5)

        @test result.num_clusters <= 2  # Should identify at most 2 clusters

        # Find the cluster with size 2 (or 1 if not clustered)
        two_element_cluster = findfirst(==(2), result.cluster_sizes)
        if two_element_cluster !== nothing
            # The close eigenvalues should be in same cluster
            cluster1_indices = findall(==(two_element_cluster), result.cluster_assignments)
            @test length(cluster1_indices) == 2
        end
    end

    @testset "Multiple clusters" begin
        # Three clusters: {1.0, 1.1}, {5.0, 5.2}, {10.0}
        A_mid = Matrix(Diagonal([1.0, 1.1, 5.0, 5.2, 10.0]))
        A_rad = zeros(size(A_mid))
        A_rad[1, 2] = A_rad[2, 1] = 0.15  # Cluster 1
        A_rad[3, 4] = A_rad[4, 3] = 0.25  # Cluster 2
        A = BallMatrix(A_mid, A_rad)

        result = rump_lange_2023_cluster_bounds(A; hermitian=true, cluster_tol=0.5)

        @test 1 <= result.num_clusters <= 3

        # All eigenvalues should be assigned to some cluster
        @test all(c -> 1 <= c <= result.num_clusters, result.cluster_assignments)

        # Cluster sizes should sum to n
        @test sum(result.cluster_sizes) == 5
    end

    @testset "Fast vs rigorous mode" begin
        A = BallMatrix(Diagonal([1.0, 1.1, 5.0]))

        result_fast = rump_lange_2023_cluster_bounds(A; hermitian=true, fast=true)
        result_rigorous = rump_lange_2023_cluster_bounds(A; hermitian=true, fast=false)

        # Both should verify
        @test result_fast.verified
        @test result_rigorous.verified

        # Both should identify same cluster structure
        @test result_fast.num_clusters == result_rigorous.num_clusters
        @test result_fast.cluster_assignments == result_rigorous.cluster_assignments

        # True eigenvalues should be contained in both
        λ_true = eigvals(mid(A))
        for i in 1:3
            @test λ_true[i] ∈ result_fast.eigenvalues[i]
            @test λ_true[i] ∈ result_rigorous.eigenvalues[i]
        end
    end

    @testset "Cluster separations" begin
        A = BallMatrix(Diagonal([1.0, 1.1, 10.0]))

        result = rump_lange_2023_cluster_bounds(A; hermitian=true, cluster_tol=0.3)

        # Cluster separations should be positive
        @test all(s -> s > 0, result.cluster_separations)

        # If we have 2 clusters ({1.0,1.1} and {10.0}), separation should be ~8-9
        if result.num_clusters == 2
            # The well-separated cluster should have large separation
            large_sep = maximum(result.cluster_separations)
            @test large_sep > 8.0
        end
    end

    @testset "Cluster residuals" begin
        A = BallMatrix(Diagonal([1.0, 2.0, 3.0]))
        result = rump_lange_2023_cluster_bounds(A; hermitian=true)

        # All residuals should be small for diagonal matrix
        @test all(r -> r < 1e-12, result.cluster_residuals)
        @test all(r -> r >= 0, result.cluster_residuals)
    end

    @testset "Cluster bounds vs individual bounds" begin
        A_mid = Matrix(Diagonal([1.0, 1.05, 1.1]))
        A_rad = zeros(size(A_mid))
        A_rad[1, 2] = A_rad[2, 1] = 0.1
        A_rad[2, 3] = A_rad[3, 2] = 0.1
        A = BallMatrix(A_mid, A_rad)

        result = rump_lange_2023_cluster_bounds(A; hermitian=true, cluster_tol=0.3)

        # If all three form one cluster
        if result.num_clusters == 1
            # All individual bounds should be contained in cluster bound
            for i in 1:3
                cluster_idx = result.cluster_assignments[i]
                cluster_bound = result.cluster_bounds[cluster_idx]

                # Individual bound should intersect with cluster bound
                @test mid(result.eigenvalues[i]) ∈ cluster_bound ||
                      mid(cluster_bound) ∈ result.eigenvalues[i]
            end
        end
    end

    @testset "Refine cluster bounds" begin
        A = BallMatrix(Diagonal([1.0, 1.1, 5.0]))
        result_initial = rump_lange_2023_cluster_bounds(A; hermitian=true)

        # Refine with 2 iterations
        result_refined = refine_cluster_bounds(result_initial, A; iterations=2)

        @test length(result_refined) == 3
        @test result_refined.num_clusters == result_initial.num_clusters

        # Refined bounds should be equal or tighter
        for i in 1:3
            @test rad(result_refined.eigenvalues[i]) <=
                  rad(result_initial.eigenvalues[i]) + 1e-14
        end

        # True eigenvalues should still be contained
        λ_true = eigvals(mid(A))
        for i in 1:3
            @test λ_true[i] ∈ result_refined.eigenvalues[i]
        end
    end

    @testset "Non-Hermitian matrix" begin
        A = BallMatrix([2.0 1.0; 0.5 3.0])
        result = rump_lange_2023_cluster_bounds(A; hermitian=false)

        @test length(result) == 2
        @test 1 <= result.num_clusters <= 2

        # Check containment (may have complex eigenvalues)
        λ_true = eigvals(mid(A))
        for i in 1:2
            ball_i = result.eigenvalues[i]
            # For real matrices with real eigenvalues
            @test abs(real(λ_true[i]) - mid(ball_i)) <= rad(ball_i)
        end
    end

    @testset "Gershgorin disc computation" begin
        # Matrix where Gershgorin gives good bounds
        A = BallMatrix([10.0 1.0 0.5;
                        1.0  5.0 0.3;
                        0.5  0.3 2.0])

        result = rump_lange_2023_cluster_bounds(A; hermitian=true)

        # All eigenvalues should be contained in some ball
        # (eigenvalue ordering may differ between true and computed)
        λ_true = eigvals(Hermitian(mid(A)))
        for λ in λ_true
            contained = any(λ ∈ ball for ball in result.eigenvalues)
            @test contained
        end
    end

    @testset "Tolerance parameter effect" begin
        A = BallMatrix(Diagonal([1.0, 1.05, 1.1]))

        result_tight = rump_lange_2023_cluster_bounds(A; hermitian=true, cluster_tol=0.01)
        result_loose = rump_lange_2023_cluster_bounds(A; hermitian=true, cluster_tol=0.5)

        # Tighter tolerance should give more clusters
        @test result_tight.num_clusters >= result_loose.num_clusters
    end
end
