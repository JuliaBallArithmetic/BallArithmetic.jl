# Tests for src/norm_bounds/oishi_triangular.jl

using Test
using LinearAlgebra
using BallArithmetic

@testset "Oishi-Rump Triangular Bounds" begin

    @testset "oishi_rump_bound function" begin
        # Test with simple upper triangular matrix
        T_mid = [4.0 0.1 0.05;
                 0.0 3.0 0.1;
                 0.0 0.0 2.0]
        T_rad = [1e-12 1e-12 1e-12;
                 0.0   1e-12 1e-12;
                 0.0   0.0   1e-12]
        T_ball = BallMatrix(T_mid, T_rad)

        # Test with k=1
        bound_k1 = BallArithmetic.oishi_rump_bound(T_ball, 1)
        @test bound_k1 > 0
        @test isfinite(bound_k1)

        # Test with k=2
        bound_k2 = BallArithmetic.oishi_rump_bound(T_ball, 2)
        @test bound_k2 > 0
        @test isfinite(bound_k2)

        # Test with k=n (full matrix)
        bound_full = BallArithmetic.oishi_rump_bound(T_ball, 3)
        @test bound_full > 0
        @test isfinite(bound_full)
    end

    @testset "oishi_rump_bound with diagonal matrices" begin
        # Diagonal matrix is a special case of triangular
        D_mid = [5.0 0.0 0.0;
                 0.0 3.0 0.0;
                 0.0 0.0 1.0]
        D_rad = [1e-12 0.0 0.0;
                 0.0   1e-12 0.0;
                 0.0   0.0   1e-12]
        D_ball = BallMatrix(D_mid, D_rad)

        # For diagonal matrix, inverse is also diagonal
        bound = BallArithmetic.oishi_rump_bound(D_ball, 2)
        @test bound > 0

        # The bound should be related to 1/min(diag)
        min_diag = 1.0
        @test bound >= 1.0 / 5.0 - 0.1  # At least roughly related to largest singular value of inverse
    end

    @testset "oishi_rump_bound with 2x2 matrix" begin
        T_mid = [3.0 0.5;
                 0.0 2.0]
        T_rad = [1e-12 1e-12;
                 0.0   1e-12]
        T_ball = BallMatrix(T_mid, T_rad)

        # Test k=1
        bound1 = BallArithmetic.oishi_rump_bound(T_ball, 1)
        @test bound1 > 0
        @test isfinite(bound1)

        # Test k=2 (full matrix)
        bound2 = BallArithmetic.oishi_rump_bound(T_ball, 2)
        @test bound2 > 0
        @test isfinite(bound2)
    end

    @testset "oishi_rump_bound validates triangular structure" begin
        # Non-triangular matrix should fail
        non_triu_mid = [1.0 2.0;
                        3.0 4.0]
        non_triu_rad = [1e-12 1e-12;
                        1e-12 1e-12]
        non_triu = BallMatrix(non_triu_mid, non_triu_rad)

        @test_throws AssertionError BallArithmetic.oishi_rump_bound(non_triu, 1)
    end

    @testset "oishi_rump_bound with well-conditioned matrix" begin
        # Well-conditioned upper triangular
        T_mid = [10.0 0.1 0.1 0.1;
                  0.0 9.0 0.1 0.1;
                  0.0 0.0 8.0 0.1;
                  0.0 0.0 0.0 7.0]
        T_rad = [1e-12 1e-12 1e-12 1e-12;
                  0.0  1e-12 1e-12 1e-12;
                  0.0   0.0  1e-12 1e-12;
                  0.0   0.0   0.0  1e-12]
        T_ball = BallMatrix(T_mid, T_rad)

        for k in 1:3
            bound = BallArithmetic.oishi_rump_bound(T_ball, k)
            @test bound > 0
            @test isfinite(bound)
        end
    end
end
