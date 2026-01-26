using Test
using LinearAlgebra
using BallArithmetic

@testset "RumpOishi2024 Triangular Matrix Bounds" begin
    @testset "Simple 3×3 upper triangular matrix" begin
        T_mid = [3.0 0.1 0.05;
                 0.0 2.0 0.1;
                 0.0 0.0 1.0]
        T = BallMatrix(UpperTriangular(T_mid))

        # Test all three methods
        bound_psi = rump_oishi_2024_triangular_bound(T, 2; method=:psi)
        bound_back = rump_oishi_2024_triangular_bound(T, 2; method=:backward)
        bound_hybrid = rump_oishi_2024_triangular_bound(T, 2; method=:hybrid)

        # All bounds should be positive and finite
        @test bound_psi > 0 && isfinite(bound_psi)
        @test bound_back > 0 && isfinite(bound_back)
        @test bound_hybrid > 0 && isfinite(bound_hybrid)

        # Hybrid should be best of both
        @test bound_hybrid <= min(bound_psi, bound_back) + 1e-14
    end

    @testset "Diagonal triangular matrix (easy case)" begin
        T = BallMatrix(UpperTriangular(Diagonal([3.0, 2.0, 1.0])))

        bound = rump_oishi_2024_triangular_bound(T, 2; method=:hybrid)

        # For diagonal matrix, inverse bound should be approximately 1/min(diag)
        # For first 2×2 block, that's 1/min(3, 2) = 0.5
        @test bound < 1.0  # Should be relatively small
        @test bound > 0.3  # But not too small
    end

    @testset "Backward singular value bound" begin
        T_mid = [3.0 0.5 0.2;
                 0.0 2.0 0.3;
                 0.0 0.0 1.0]
        T = BallMatrix(UpperTriangular(T_mid))

        σ_bounds = backward_singular_value_bound(T)

        @test length(σ_bounds) == 3
        @test all(σ -> sup(σ) > 0, σ_bounds)
        @test all(σ -> isfinite(sup(σ)), σ_bounds)

        # Bounds should be ordered (roughly)
        # σ_1 ≥ σ_2 ≥ σ_3 for upper triangular
        @test sup(σ_bounds[1]) >= sup(σ_bounds[2]) - 1e-10
        @test sup(σ_bounds[2]) >= sup(σ_bounds[3]) - 1e-10
    end

    @testset "Full matrix (k = n) special case" begin
        T_mid = [3.0 0.1 0.05;
                 0.0 2.0 0.1;
                 0.0 0.0 1.0]
        T = BallMatrix(UpperTriangular(T_mid))

        # When k = n, should use direct SVD bound
        bound = rump_oishi_2024_triangular_bound(T, 3; method=:hybrid)

        @test bound > 0 && isfinite(bound)

        # Should be approximately 1/σ_min where σ_min is smallest singular value
        true_svd = svd(T_mid)
        expected_bound_approx = 1.0 / minimum(true_svd.S)

        # Rigorous bound should be larger than true value
        @test bound >= expected_bound_approx - 1e-10
    end

    @testset "Well-conditioned vs ill-conditioned" begin
        # Well-conditioned: large diagonal entries
        T_good = BallMatrix(UpperTriangular([10.0 0.1; 0.0 5.0]))

        # Ill-conditioned: small diagonal entry
        T_bad = BallMatrix(UpperTriangular([10.0 0.1; 0.0 0.1]))

        bound_good = rump_oishi_2024_triangular_bound(T_good, 2; method=:hybrid)
        bound_bad = rump_oishi_2024_triangular_bound(T_bad, 2; method=:hybrid)

        # Ill-conditioned should have larger bound
        @test bound_bad > bound_good
    end

    @testset "Different block sizes" begin
        T_mid = [4.0 0.2 0.1 0.05;
                 0.0 3.0 0.15 0.08;
                 0.0 0.0 2.0 0.1;
                 0.0 0.0 0.0 1.0]
        T = BallMatrix(UpperTriangular(T_mid))

        # Try different k values
        bound_k1 = rump_oishi_2024_triangular_bound(T, 1; method=:hybrid)
        bound_k2 = rump_oishi_2024_triangular_bound(T, 2; method=:hybrid)
        bound_k3 = rump_oishi_2024_triangular_bound(T, 3; method=:hybrid)

        # All should be positive
        @test bound_k1 > 0 && isfinite(bound_k1)
        @test bound_k2 > 0 && isfinite(bound_k2)
        @test bound_k3 > 0 && isfinite(bound_k3)
    end

    @testset "Backward substitution method" begin
        T_mid = [3.0 0.5; 0.0 2.0]
        T = BallMatrix(UpperTriangular(T_mid))

        bound = rump_oishi_2024_triangular_bound(T, 2; method=:backward)

        # Should give reasonable bound
        @test bound > 0 && isfinite(bound)

        # Compare with true inverse norm
        T_inv = inv(T_mid)
        true_norm_inv = opnorm(T_inv, 2)

        # Rigorous bound should be ≥ true value
        @test bound >= true_norm_inv - 1e-10
    end

    @testset "Ψ-bound method" begin
        T_mid = [3.0 0.5; 0.0 2.0]
        T = BallMatrix(UpperTriangular(T_mid))

        bound = rump_oishi_2024_triangular_bound(T, 2; method=:psi)

        # Should give reasonable bound
        @test bound > 0 && isfinite(bound)

        # Compare with true inverse norm
        T_inv = inv(T_mid)
        true_norm_inv = opnorm(T_inv, 2)

        # Rigorous bound should be ≥ true value
        @test bound >= true_norm_inv - 1e-10
    end

    @testset "Backward singular value bound accuracy" begin
        # For diagonal triangular matrix, bounds should be exact
        T_diag = BallMatrix(UpperTriangular(Diagonal([3.0, 2.0, 1.0])))
        σ_bounds = backward_singular_value_bound(T_diag)

        # For diagonal matrix, singular values = |diagonal entries|
        @test abs(sup(σ_bounds[1]) - 3.0) < 1e-14
        @test abs(sup(σ_bounds[2]) - 2.0) < 1e-14
        @test abs(sup(σ_bounds[3]) - 1.0) < 1e-14
    end

    @testset "Matrix with intervals" begin
        T_mid = [3.0 0.1; 0.0 2.0]
        T_rad = [0.01 0.01; 0.0 0.01]
        T = BallMatrix(UpperTriangular(T_mid), UpperTriangular(T_rad))

        bound = rump_oishi_2024_triangular_bound(T, 2; method=:hybrid)

        # Should handle uncertainties
        @test bound > 0 && isfinite(bound)

        # Bound should be larger than for exact matrix (due to uncertainties)
        T_exact = BallMatrix(UpperTriangular(T_mid))
        bound_exact = rump_oishi_2024_triangular_bound(T_exact, 2; method=:hybrid)

        @test bound >= bound_exact - 1e-12
    end
end
