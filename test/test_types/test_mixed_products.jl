# Products of a plain real or complex matrix, dense or sparse, with a ball matrix of the other
# kind: the ball contains the exact product of the plain matrix with every sampled member.
using LinearAlgebra, SparseArrays, Random

@testset "plain × ball products across real and complex" begin
    rng = MersenneTwister(3)
    contains(P, X) = all(abs.(X .- mid(P)) .<= rad(P) .* (1 + 1e-12) .+ 1e-300)
    for (A, B) in ((randn(rng, 6, 4), BallMatrix(randn(rng, ComplexF64, 4, 3), fill(1e-8, 4, 3))),
                   (sprandn(rng, 6, 4, 0.5), BallMatrix(randn(rng, ComplexF64, 4, 3), fill(1e-8, 4, 3))),
                   (randn(rng, ComplexF64, 6, 4), BallMatrix(randn(rng, 4, 3), fill(1e-8, 4, 3))))
        P = A * B
        @test size(P) == (6, 3)
        @test eltype(mid(P)) <: Complex
        for _ in 1:10
            X = mid(B) .+ rad(B) .* (2 .* rand(rng, 4, 3) .- 1)
            @test contains(P, Matrix(A) * X)
        end
    end
end
