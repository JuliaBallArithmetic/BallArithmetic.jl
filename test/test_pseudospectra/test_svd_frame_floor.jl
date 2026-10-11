using BallArithmetic
using LinearAlgebra
using Random
using Test

@testset "SVD-frame floor" begin
    rng = MersenneTwister(20261011)
    σtrue(A, z) = minimum(svdvals(ComplexF64.(A) - z * I))
    grid(c, h, m) = [c + x + im * y for x in range(-h, h; length = m), y in range(-h, h; length = m)]
    Q = Matrix(qr(randn(rng, 8, 8)).Q)
    gallery = [
        ("normal", Q * Diagonal([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 6.0]) * Q'),
        ("symmetric", (S = randn(rng, 7, 7); S + S')),
        ("nearly normal", Q * Diagonal(collect(1.0:8.0)) * Q' + 1e-3 * randn(rng, 8, 8)),
        ("random", randn(rng, 7, 7)),
        ("complex", randn(rng, ComplexF64, 6, 6)),
        ("chain", Matrix(Bidiagonal(Float64.(1:9), fill(0.3, 8), :U)))
    ]
    @testset "$name" for (name, A) in gallery
        f = svd_frame_floor(BallMatrix(A))
        @test 1 ≤ f.gram_norm ≤ 1 + 1e-12
        positive = 0
        for z in grid(sum(diag(A)) / size(A, 1) + 0im, 12.0, 13)
            truth = σtrue(A, z)
            s = sigma_min_floor(f, z)
            @test 0 ≤ s ≤ truth * (1 + 1e-12)
            positive += s > 0
            rb = resolvent_bound(f, z)
            @test rb ≥ (1 / truth) * (1 - 1e-12)
            @test (s > 0) == isfinite(rb)
        end
        @test positive > 0
    end

    @testset "at z = 0 the bound is the smallest singular value" begin
        A = randn(rng, 9, 9)
        f = svd_frame_floor(BallMatrix(A))
        s = sigma_min_floor(f, 0)
        @test s ≤ minimum(svdvals(A)) * (1 + 1e-12)
        @test s ≥ minimum(svdvals(A)) * (1 - 1e-8)
    end

    @testset "a Hermitian positive definite matrix: the distance to the spectrum on the left" begin
        S = randn(rng, 6, 6)
        A = S' * S + I
        f = svd_frame_floor(BallMatrix(A))
        for x in (-0.5, -3.0, -40.0)
            @test sigma_min_floor(f, x) ≥ (minimum(eigvals(Symmetric(A))) - x) * (1 - 1e-8)
        end
    end

    @testset "a ball of matrices" begin
        A = randn(rng, 6, 6)
        f = svd_frame_floor(BallMatrix(A, fill(1e-4, 6, 6)))
        for z in grid(0.0 + 0im, 10.0, 7)
            s = sigma_min_floor(f, z)
            for _ in 1:5
                B = A + 1e-4 * (2 * rand(rng, 6, 6) .- 1)
                @test s ≤ σtrue(B, z) * (1 + 1e-12)
            end
        end
    end

    @testset "BigFloat and argument checks" begin
        A = BigFloat.(randn(rng, 4, 4))
        f = svd_frame_floor(BallMatrix(A))
        s = sigma_min_floor(f, 9 + 2im)
        @test s isa BigFloat
        @test 0 < s ≤ minimum(svdvals(Complex{BigFloat}.(A) - (9 + 2im) * I))
        @test_throws ArgumentError svd_frame_floor(BallMatrix(randn(rng, 3, 4)))
    end
end
