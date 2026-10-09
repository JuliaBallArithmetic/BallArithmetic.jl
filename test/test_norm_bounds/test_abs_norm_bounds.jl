# The norm bounds through |C| + R (moved from RigorousPseudospectra): each bounds the norm of
# matrices sampled from the ball, and none is larger than the norm of C plus the norm of R.
using LinearAlgebra, Random

@testset "norm bounds through |C| + R" begin
    rng = MersenneTwister(11)
    sample(B) = mid(B) .+ rad(B) .* (2 .* rand(rng, size(mid(B))...) .- 1) .*
                         (eltype(mid(B)) <: Complex ? cis.(2π .* rand(rng, size(mid(B))...)) : 1)

    for T in (Float64, ComplexF64), (m, n) in ((7, 7), (12, 5), (5, 12))
        C = randn(rng, T, m, n)
        R = abs.(randn(rng, m, n)) .* 1e-3
        B = BallMatrix(C, R)
        A = upper_abs(B)
        @test all(A .>= abs.(C) .+ R .- 4eps() .* (abs.(C) .+ R))
        n1, ninf = upper_bound_L1_opnorm(B), upper_bound_L_inf_opnorm(B)
        nf = upper_bound_frobenius(B)
        n2 = upper_bound_L2_opnorm(B)
        n2s = upper_bound_L2_opnorm(B; svd = true)
        @test n2s <= n2
        for _ in 1:20
            X = sample(B)
            @test opnorm(X, 1) <= n1
            @test opnorm(X, Inf) <= ninf
            @test norm(X) <= nf
            @test opnorm(X) <= n2s
        end
        # never looser than the bounds on C and R separately
        @test n1 <= nextfloat(opnorm(C, 1) + opnorm(R, 1), 4)
        @test ninf <= nextfloat(opnorm(C, Inf) + opnorm(R, Inf), 4)
        @test nf <= nextfloat(norm(C) + norm(R), 4)
    end

    @testset "svd_bounds and the smallest singular value" begin
        for T in (Float64, ComplexF64)
            C = randn(rng, T, 9, 9)
            B = BallMatrix(C, fill(1e-10, 9, 9))
            lo, hi = svd_bounds(B; method = :miyajima2014_thm10_weyl)
            s = svdvals(C)
            @test all(lo .<= s .+ 1e-9) && all(s .- 1e-9 .<= hi)
            @test svd_lower_bound_sigma_min(B) == lo[end]
            @test svd_bound_L2_opnorm(B) >= opnorm(C) - 1e-9
        end
        lo, hi = svd_bounds(BallMatrix([NaN 1.0; 0.0 1.0]))
        @test lo == [0.0, 0.0] && hi == [Inf, Inf]
        @test svd_bounds(BallMatrix(zeros(0, 3))) == (Float64[], Float64[])
    end
end
