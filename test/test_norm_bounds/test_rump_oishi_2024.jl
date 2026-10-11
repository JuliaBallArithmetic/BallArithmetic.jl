using Test
using LinearAlgebra
using Random
using BallArithmetic

@testset "Rump-Oishi bound for a triangular matrix" begin
    rng = MersenneTwister(20261016)

    @testset "an upper bound of ‖T⁻¹‖₂ for the members of the ball, $S" for S in (Float64,
        ComplexF64)

        finite = 0
        for n in (2, 3, 6), _ in 1:40
            Tc = Matrix(UpperTriangular(randn(rng, S, n, n) .* 0.3)) +
                 Diagonal(S.(rand(rng, n) .+ 1.0) .* rand(rng, (-1, 1), n))
            Tr = Matrix(UpperTriangular(rand(rng, n, n))) .* 10.0^rand(rng, -8:-2)
            T = BallMatrix(Tc, Tr)
            for k in 1:n
                bound = rump_oishi_2024_triangular_bound(T, k)
                @test bound > 0
                isfinite(bound) || continue
                finite += 1
                for _ in 1:4
                    member = Tc .+ Tr .* (2 .* rand(rng, n, n) .- 1)
                    @test opnorm(inv(member), 2) <= bound * (1 + 1e-10)
                end
            end
        end
        @test finite > 0
    end

    @testset "the audit's example" begin
        # ‖T⁻¹‖₂ = 2.288 for this matrix; the removed :backward method returned 2.000
        T = BallMatrix([1.0 0.5; 0.0 0.5])
        @test rump_oishi_2024_triangular_bound(T, 1) >= opnorm(inv(mid(T)), 2)
        @test rump_oishi_2024_triangular_bound(T, 2) >= opnorm(inv(mid(T)), 2)
        # the 3 × 3 matrix of the removed oishi_rump_bound, ‖T⁻¹‖₂ = 1.4519...
        T3 = BallMatrix([2.0 0.15 0.05; 0.0 1.3 0.2; 0.0 0.0 0.7])
        for k in 1:3
            @test rump_oishi_2024_triangular_bound(T3, k) >= 1.451948
        end
    end

    @testset "a diagonal matrix: the bound is the largest reciprocal" begin
        T = BallMatrix(Matrix(Diagonal([3.0, 2.0, 1.0])))
        for k in 1:3
            b = rump_oishi_2024_triangular_bound(T, k)
            @test 1.0 <= b <= 1.0 + 1e-10
        end
    end

    @testset "where the hypothesis fails, and bad arguments" begin
        # ‖D_d⁻¹ D_f‖ ≥ 1
        T = BallMatrix([1.0 0.0 0.0; 0.0 1.0 5.0; 0.0 0.0 1.0])
        @test rump_oishi_2024_triangular_bound(T, 1) == Inf
        # a diagonal entry of D that may vanish
        T0 = BallMatrix([1.0 0.1; 0.0 0.01], [0.0 0.0; 0.0 0.02])
        @test rump_oishi_2024_triangular_bound(T0, 1) == Inf
        @test_throws ArgumentError rump_oishi_2024_triangular_bound(BallMatrix([1.0 0.0; 1.0 1.0]), 1)
        @test_throws ArgumentError rump_oishi_2024_triangular_bound(BallMatrix([1.0 0.0; 0.0 1.0]), 3)
    end
end
