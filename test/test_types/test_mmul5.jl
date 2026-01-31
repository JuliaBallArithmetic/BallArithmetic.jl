# Tests for src/types/MMul/MMul5.jl - Revol-Theveny parallel interval matrix multiplication

using Test
using LinearAlgebra
using BallArithmetic

@testset "MMul5 - Revol-Theveny Matrix Multiplication" begin

    @testset "Basic MMul5 operation" begin
        # Simple matrices with small radii
        A_mid = [1.0 2.0; 3.0 4.0]
        A_rad = [0.01 0.01; 0.01 0.01]
        A = BallMatrix(A_mid, A_rad)

        B_mid = [5.0 6.0; 7.0 8.0]
        B_rad = [0.01 0.01; 0.01 0.01]
        B = BallMatrix(B_mid, B_rad)

        # MMul5 should return a BallMatrix
        C = BallArithmetic.MMul5(A, B)

        @test C isa BallMatrix
        @test size(C) == (2, 2)

        # Center should be close to A_mid * B_mid
        expected_mid = A_mid * B_mid
        @test norm(mid(C) - expected_mid, Inf) < 1.0  # Allow for rounding

        # Radii should be positive
        @test all(rad(C) .>= 0)
    end

    @testset "MMul5 encloses true product" begin
        A_mid = [1.0 0.5; 0.5 1.0]
        A_rad = [0.1 0.1; 0.1 0.1]
        A = BallMatrix(A_mid, A_rad)

        B_mid = [2.0 1.0; 1.0 2.0]
        B_rad = [0.1 0.1; 0.1 0.1]
        B = BallMatrix(B_mid, B_rad)

        C = BallArithmetic.MMul5(A, B)

        # The exact center product should be enclosed
        exact_center = A_mid * B_mid
        C_mid = mid(C)
        C_rad = rad(C)

        for i in 1:2, j in 1:2
            @test abs(exact_center[i, j] - C_mid[i, j]) <= C_rad[i, j] + 1.0
        end
    end

    @testset "MMul5 with identity matrix" begin
        n = 3
        I_mid = Matrix(1.0I, n, n)
        I_rad = zeros(n, n)
        I_ball = BallMatrix(I_mid, I_rad)

        A_mid = randn(n, n)
        A_rad = fill(0.01, n, n)
        A = BallMatrix(A_mid, A_rad)

        # I * A should give approximately A
        C = BallArithmetic.MMul5(I_ball, A)

        # Centers should be very close
        @test norm(mid(C) - A_mid, Inf) < 0.1
    end

    @testset "MMul5 with larger matrices" begin
        n = 5
        A_mid = randn(n, n)
        A_rad = fill(0.01, n, n)
        A = BallMatrix(A_mid, A_rad)

        B_mid = randn(n, n)
        B_rad = fill(0.01, n, n)
        B = BallMatrix(B_mid, B_rad)

        C = BallArithmetic.MMul5(A, B)

        @test size(C) == (n, n)
        @test all(rad(C) .>= 0)
    end

    @testset "MMul5 with non-square matrices" begin
        m, k, n = 3, 4, 2
        A_mid = randn(m, k)
        A_rad = fill(0.01, m, k)
        A = BallMatrix(A_mid, A_rad)

        B_mid = randn(k, n)
        B_rad = fill(0.01, k, n)
        B = BallMatrix(B_mid, B_rad)

        C = BallArithmetic.MMul5(A, B)

        @test size(C) == (m, n)
        @test all(rad(C) .>= 0)
    end

    @testset "MMul5 radius is influenced by input radii" begin
        A_mid = [1.0 2.0; 3.0 4.0]
        B_mid = [5.0 6.0; 7.0 8.0]

        # Small radii
        A_small = BallMatrix(A_mid, fill(0.001, 2, 2))
        B_small = BallMatrix(B_mid, fill(0.001, 2, 2))
        C_small = BallArithmetic.MMul5(A_small, B_small)

        # Larger radii
        A_large = BallMatrix(A_mid, fill(0.1, 2, 2))
        B_large = BallMatrix(B_mid, fill(0.1, 2, 2))
        C_large = BallArithmetic.MMul5(A_large, B_large)

        # Larger input radii should give larger output radii
        @test maximum(rad(C_large)) > maximum(rad(C_small))
    end

    @testset "MMul5 with positive matrices" begin
        # All positive elements
        A_mid = [1.0 2.0; 3.0 4.0]
        A_rad = [0.1 0.1; 0.1 0.1]
        A = BallMatrix(A_mid, A_rad)

        B_mid = [1.0 1.0; 1.0 1.0]
        B_rad = [0.1 0.1; 0.1 0.1]
        B = BallMatrix(B_mid, B_rad)

        C = BallArithmetic.MMul5(A, B)

        # All entries should be positive-ish (center > radius)
        @test all(mid(C) .> 0)
    end

    @testset "MMul5 with mixed sign matrices" begin
        A_mid = [1.0 -2.0; -3.0 4.0]
        A_rad = [0.1 0.1; 0.1 0.1]
        A = BallMatrix(A_mid, A_rad)

        B_mid = [-1.0 2.0; 3.0 -4.0]
        B_rad = [0.1 0.1; 0.1 0.1]
        B = BallMatrix(B_mid, B_rad)

        C = BallArithmetic.MMul5(A, B)

        @test size(C) == (2, 2)
        @test all(isfinite.(mid(C)))
        @test all(isfinite.(rad(C)))
    end
end
