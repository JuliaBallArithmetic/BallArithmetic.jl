using BallArithmetic
using Test
using Random
using LinearAlgebra

# The defining property of an error-free transformation: the exact result is the sum of the two
# returned floats, with nothing lost to rounding.

@testset "two_product is exact" begin
    rng = MersenneTwister(1)
    for _ in 1:2000
        a, b = randn(rng), randn(rng)
        x, y = two_product(a, b)
        @test x == a * b                      # x is the rounded product
        # x + y is the exact product: check in higher precision
        @test BigFloat(x) + BigFloat(y) == BigFloat(a) * BigFloat(b)
    end
end

@testset "two_sum is exact" begin
    rng = MersenneTwister(2)
    for _ in 1:2000
        a, b = randn(rng), randn(rng) * 1e8   # wildly different magnitudes too
        x, y = two_sum(a, b)
        @test x == a + b
        @test BigFloat(x) + BigFloat(y) == BigFloat(a) + BigFloat(b)
    end
end

@testset "split_veltkamp is exact and halves the significand" begin
    rng = MersenneTwister(3)
    for _ in 1:2000
        a = randn(rng)
        hi, lo = split_veltkamp(a)
        @test hi + lo == a
        @test BigFloat(hi) + BigFloat(lo) == BigFloat(a)
        # products of halves are exact, which is the point of the splitting
        b = randn(rng)
        bhi, blo = split_veltkamp(b)
        @test BigFloat(hi) * BigFloat(bhi) == BigFloat(hi * bhi)
    end
end

@testset "compensated_terms recovers a residual when the cancellation is internal" begin
    rng = MersenneTwister(4)
    n = 60
    A = randn(rng, n, n)
    B = randn(rng, n, n)
    C = A * B                                  # the rounded product
    Id = Matrix{Float64}(I, n, n)

    # The quantity of interest is  sum_k A[i,k] B[k,j]  -  C[i,j], the rounding the product
    # committed. Both terms must sit INSIDE the accumulation: the function returns s + e as one
    # Float64, so a cancellation performed afterwards is lost to that final rounding. This is
    # exactly how the residual B W - W X is formed in Rump's transformation.
    comp = [compensated_terms(((A, B, 1.0), (C, Id, -1.0)), i, j) for i in 1:n, j in 1:n]
    exact = BigFloat.(A) * BigFloat.(B) - BigFloat.(C)
    plain = A * B - C

    @test all(iszero, plain)                   # the plain route sees nothing
    @test maximum(abs, exact) > 1e-16          # there is something to see
    @test maximum(abs, BigFloat.(comp) .- exact) < 1e-28
    # and within the Ogita-Rump-Oishi bound for the accumulation
    u = eps(Float64) / 2
    γ = gamma_bound(2n, Float64)
    absprod = abs.(A) * abs.(B) .+ abs.(C)
    @test all(abs(BigFloat(comp[i, j]) - exact[i, j]) <=
              u * abs(exact[i, j]) + γ^2 * absprod[i, j] + 1e-300 for i in 1:n, j in 1:n)
end

@testset "gamma_bound is an upper bound" begin
    for N in (1, 10, 100, 1000)
        u = eps(Float64) / 2
        @test gamma_bound(N, Float64) >= (N * u) / (1 - N * u)
    end
end
