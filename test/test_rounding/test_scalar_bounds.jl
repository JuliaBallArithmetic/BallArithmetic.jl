# The scalar bounds built on the directed operations (moved from RigorousPseudospectra), checked
# against BigFloat references.
using LinearAlgebra, Random

@testset "scalar bounds: sum, modulus, powers, roots, gamma" begin
    rng = MersenneTwister(7)
    big256(x) = setprecision(() -> BigFloat(x), 256)

    @testset "sum_up" begin
        for _ in 1:50
            xs = randn(rng, 40) .* 10.0 .^ rand(rng, -5:5, 40)
            exact = setprecision(() -> sum(BigFloat.(xs)), 512)
            @test sum_up(xs) >= exact
        end
        @test sum_up(Float64[]) == 0.0
    end

    @testset "abs_up and abs_down bracket the complex modulus" begin
        for _ in 1:200
            z = complex(randn(rng), randn(rng)) * 10.0^rand(rng, -100:100)
            exact = setprecision(() -> hypot(BigFloat(real(z)), BigFloat(imag(z))), 256)
            @test abs_down(z) <= exact <= abs_up(z)
        end
        @test abs_up(-3.5) == 3.5 && abs_down(-3.5) == 3.5
    end

    @testset "pow_up, pow_down, root_up" begin
        for _ in 1:100
            x = rand(rng) * 10.0^rand(rng, -3:3)
            e = rand(rng, 0:12)
            exact = setprecision(() -> BigFloat(x)^e, 512)
            @test pow_down(x, e) <= exact <= pow_up(x, e)
            r = root_up(x, max(e, 1))
            @test setprecision(() -> BigFloat(r)^max(e, 1), 512) >= big256(x)
        end
        @test root_up(0.0, 3) == 0.0
        @test root_up(Inf, 3) == Inf
    end

    @testset "gamma_bound: real, complex, and the validity limit" begin
        u = eps(Float64) / 2
        for N in (1, 10, 1000, 10^6)
            exact = setprecision(() -> (N * BigFloat(u)) / (1 - N * BigFloat(u)), 256)
            @test gamma_bound(N, Float64) >= exact
            @test gamma_bound(N, ComplexF64) == gamma_bound(N + 2, Float64)
        end
        @test_throws ArgumentError gamma_bound(2^52, Float64)
    end
end
