using Test
using BallArithmetic
using LinearAlgebra
using Random

BA = BallArithmetic

# every entry of the 256-bit product of the midpoints must lie in the returned ball
function _encloses_midproduct(A::BallMatrix, B::BallMatrix, C::BallMatrix)
    ref = Complex{BigFloat}.(mid(A)) * Complex{BigFloat}.(mid(B))
    for i in eachindex(ref)
        abs(ref[i] - Complex{BigFloat}(mid(C)[i])) <= BigFloat(rad(C)[i]) || return false
    end
    return true
end

# every entry of X*Y must lie in the ball, for X in A and Y in B
function _encloses_members(A::BallMatrix, B::BallMatrix, C::BallMatrix; trials = 8,
        rng = MersenneTwister(11))
    for _ in 1:trials
        X = mid(A) .+ rad(A) .* (2 .* rand(rng, size(mid(A))...) .- 1)
        Y = mid(B) .+ rad(B) .* (2 .* rand(rng, size(mid(B))...) .- 1)
        P = Complex{BigFloat}.(X) * Complex{BigFloat}.(Y)
        for i in eachindex(P)
            abs(P[i] - Complex{BigFloat}(mid(C)[i])) <= BigFloat(rad(C)[i]) || return false
        end
    end
    return true
end

@testset "mmul_ogita_rump_oishi_2005: the accurate ball product" begin
    setprecision(256)
    rng = MersenneTwister(2026)

    @testset "encloses, real and complex, exact input" begin
        for k in (5, 17, 60)
            for A in (BallMatrix(randn(rng, k, k)), BallMatrix(randn(rng, ComplexF64, k, k)))
                B = A isa BallMatrix{Float64, Float64} ? BallMatrix(randn(rng, k, k)) :
                    BallMatrix(randn(rng, ComplexF64, k, k))
                C = mmul_ogita_rump_oishi_2005(A, B)
                @test size(C) == (k, k)
                @test all(isfinite, mid(C))
                @test all(>=(0), rad(C))
                @test _encloses_midproduct(A, B, C)
            end
        end
    end

    @testset "encloses every member when the input carries a radius" begin
        k = 12
        for r in (1e-14, 1e-8)
            A = BallMatrix(randn(rng, ComplexF64, k, k), fill(r, k, k))
            B = BallMatrix(randn(rng, ComplexF64, k, k), fill(r, k, k))
            C = mmul_ogita_rump_oishi_2005(A, B)
            @test _encloses_members(A, B, C)
        end
    end

    @testset "tighter than the rounding-mode product on exact input" begin
        # The accumulation is charged at γ_N² rather than γ_N, so the gain grows with the inner
        # dimension. Measured maxima at 16 threads on complex input: ratio 0.0075 at k = 200,
        # 0.0044 at 400, 0.0037 at 800, 0.0029 at 1600. The thresholds below are deliberately
        # loose, since the comparison is against whichever kernel `*` dispatches to.
        for (k, bound) in ((20, 1.0), (60, 0.5), (200, 0.1))
            A = BallMatrix(randn(rng, ComplexF64, k, k))
            B = BallMatrix(randn(rng, ComplexF64, k, k))
            C = mmul_ogita_rump_oishi_2005(A, B)
            @test maximum(rad(C)) <= bound * maximum(rad(A * B))
        end
    end

    @testset "the interval part is unchanged, so a wide input converges to the same ball" begin
        # With a radius far above the rounding, both routes are dominated by the same expression,
        # rad(A)(|mid(B)|+rad(B)) + |mid(A)|rad(B), and must agree to within it. Tested on a REAL
        # input so that `*` dispatches to MMul4 and the comparison is against that expression; for
        # a complex input `*` takes a specialised path which sums the radii of the real and
        # imaginary parts and is looser in the interval part as well, so the two need not agree
        # there and only the one-sided bound is asserted.
        k = 15
        A = BallMatrix(randn(rng, k, k), fill(1e-3, k, k))
        B = BallMatrix(randn(rng, k, k), fill(1e-3, k, k))
        Co = mmul_ogita_rump_oishi_2005(A, B)
        Cb = A * B
        @test maximum(rad(Co)) <= maximum(rad(Cb))
        @test maximum(rad(Co)) >= 0.99 * maximum(rad(Cb))

        Az = BallMatrix(randn(rng, ComplexF64, k, k), fill(1e-3, k, k))
        Bz = BallMatrix(randn(rng, ComplexF64, k, k), fill(1e-3, k, k))
        @test maximum(rad(mmul_ogita_rump_oishi_2005(Az, Bz))) <= maximum(rad(Az * Bz))
    end

    @testset "the residual case the error-free transformations exist for" begin
        # B W − W X nearly cancels for an approximate eigendecomposition; a plain ball product
        # returns a radius the size of the residual and carries no information
        k = 40
        W = randn(rng, ComplexF64, k, k)
        M = randn(rng, ComplexF64, k, k)
        X = W \ (M * W)
        Co = mmul_ogita_rump_oishi_2005(BallMatrix(W), BallMatrix(X))
        @test _encloses_midproduct(BallMatrix(W), BallMatrix(X), Co)
        @test maximum(rad(Co)) < maximum(rad(BallMatrix(W) * BallMatrix(X)))
    end

    @testset "rectangular and degenerate shapes" begin
        A = BallMatrix(randn(rng, ComplexF64, 7, 13))
        B = BallMatrix(randn(rng, ComplexF64, 13, 5))
        C = mmul_ogita_rump_oishi_2005(A, B)
        @test size(C) == (7, 5)
        @test _encloses_midproduct(A, B, C)
        @test_throws DimensionMismatch mmul_ogita_rump_oishi_2005(A, A)
        # a single column and a single row
        u = BallMatrix(randn(rng, ComplexF64, 9, 1))
        v = BallMatrix(randn(rng, ComplexF64, 1, 9))
        @test size(mmul_ogita_rump_oishi_2005(u, v)) == (9, 9)
        @test size(mmul_ogita_rump_oishi_2005(v, u)) == (1, 1)
    end

    @testset "no directed rounding is needed for the midpoint" begin
        # the midpoint must be reproducible whatever the ambient rounding mode, since it uses only
        # FMA and Knuth's two_sum; the radius is accumulated under RoundUp internally
        k = 11
        A = BallMatrix(randn(rng, ComplexF64, k, k))
        B = BallMatrix(randn(rng, ComplexF64, k, k))
        C0 = mmul_ogita_rump_oishi_2005(A, B)
        C1 = setrounding(Float64, RoundUp) do
            mmul_ogita_rump_oishi_2005(A, B)
        end
        @test mid(C0) == mid(C1)
    end
end
