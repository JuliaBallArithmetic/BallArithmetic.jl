using Test
using LinearAlgebra
using BallArithmetic

# Import internal functions for testing
const α_bound = BallArithmetic.α_bound
const psi_bound = BallArithmetic.psi_bound
const rump_oishi_triangular = BallArithmetic.rump_oishi_triangular

@testset "Oishi Triangular Methods" begin
    @testset "alpha bound function" begin
        # α_bound computes α = sqrt(0.5 * (1 + 1/sqrt(1 + 4/μ²)))
        # For large μ, α → sqrt(1) = 1
        # For small μ, α → sqrt(0.5)

        # Test with small μ
        α_small = α_bound(0.5)
        @test α_small isa Ball
        @test mid(α_small) > 0
        @test mid(α_small) < 1

        # Test with moderate μ
        α_mod = α_bound(2.0)
        @test mid(α_mod) > 0
        @test mid(α_mod) < 1

        # Test with large μ
        α_large = α_bound(100.0)
        @test mid(α_large) > 0
        # For large μ, α approaches 1
        @test mid(α_large) < 1
        @test mid(α_large) > 0.9  # Should be close to 1 for large μ
    end

    @testset "psi bound function" begin
        # Test with simple triangular matrix
        # psi_bound computes sqrt(1 + 2αμ√(1-α²) + (αμ)²)
        N = BallMatrix([3.0 0.1; 0.0 2.0], fill(1e-10, 2, 2))
        ψ = psi_bound(N)

        @test ψ isa Ball
        @test mid(ψ) > 0
        @test mid(ψ) >= 1.0  # psi should be at least 1
    end

    @testset "Rump-Oishi triangular inverse norm bound" begin
        # Create strictly upper triangular matrix with positive diagonal
        # Both center and radius must be upper triangular
        T_mid = [3.0 0.1 0.05;
                 0.0 2.0 0.1;
                 0.0 0.0 4.0]
        T_rad = [1e-10 1e-10 1e-10;
                 0.0   1e-10 1e-10;
                 0.0   0.0   1e-10]

        T_ball = BallMatrix(T_mid, T_rad)

        # Test with k=1 (1x1 block in upper left)
        bound_k1 = rump_oishi_triangular(T_ball, 1)
        @test bound_k1 > 0
        @test isfinite(bound_k1)

        # Test with k=2 (2x2 block in upper left)
        bound_k2 = rump_oishi_triangular(T_ball, 2)
        @test bound_k2 > 0
        @test isfinite(bound_k2)
    end

    @testset "Triangular with well-separated diagonal" begin
        # Triangular matrix with well-separated diagonal entries
        T_mid = [5.0 0.1 0.1;
                 0.0 4.0 0.2;
                 0.0 0.0 3.0]
        T_rad = [1e-10 1e-10 1e-10;
                 0.0   1e-10 1e-10;
                 0.0   0.0   1e-10]

        T_ball = BallMatrix(T_mid, T_rad)

        bound = rump_oishi_triangular(T_ball, 2)
        @test bound > 0

        # The bound should be roughly comparable to 1/min(diagonal)
        min_diag = 3.0
        @test bound < 10.0 / min_diag  # Sanity check
    end

    @testset "Single element block" begin
        # 2x2 upper triangular matrix
        T_mid = [4.0 0.1;
                 0.0 2.0]
        T_rad = [1e-10 1e-10;
                 0.0   1e-10]

        T_ball = BallMatrix(T_mid, T_rad)

        # k=1 means 1x1 block A and 1x1 block D
        bound = rump_oishi_triangular(T_ball, 1)
        @test bound > 0
        @test isfinite(bound)
    end

    @testset "Diagonal dominant case" begin
        # Diagonal matrix (special case of triangular)
        T_mid = [10.0 0.0 0.0;
                 0.0 5.0 0.0;
                 0.0 0.0 2.0]
        T_rad = [1e-10 0.0 0.0;
                 0.0   1e-10 0.0;
                 0.0   0.0   1e-10]

        T_ball = BallMatrix(T_mid, T_rad)

        # For diagonal matrix, the inverse norm should be bounded by 1/min(|diagonal|)
        bound = rump_oishi_triangular(T_ball, 2)
        @test bound > 0
        @test bound >= 1.0 / 10.0  # At least this large
    end
end
