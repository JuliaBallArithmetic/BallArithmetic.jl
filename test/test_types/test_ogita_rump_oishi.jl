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

# the exact product, in rationals
_exact_product(A, B) = Rational{BigInt}.(A) * Rational{BigInt}.(B)

function _bounds_hold(A, B)
    down, up = BA._product_down_up_dot2err(A, B)
    ref = _exact_product(A, B)
    return all(i -> (isinf(down[i]) || Rational{BigInt}(down[i]) <= ref[i]) &&
                    (isinf(up[i]) || ref[i] <= Rational{BigInt}(up[i])), eachindex(ref))
end

@testset "Dot2Err (Ogita, Rump and Oishi 2005, Algorithm 5.8): product bounds without a rounding mode" begin
    rng = MersenneTwister(58)

    @testset "random factors, $T" for T in (Float64, Float32)
        for (m, k, n) in ((1, 1, 1), (7, 13, 5), (30, 20, 25))
            A, B = randn(rng, T, m, k), randn(rng, T, k, n)
            @test _bounds_hold(A, B)
            down, up = BA._product_down_up_dot2err(A, B)
            @test all(down .<= up)
            # the bound β is far below one unit in the last place here
            @test all(up .<= nextfloat.(down, 2))
        end
    end

    @testset "cancellation" begin
        for T in (Float64, Float32)
            k = 40
            x = randn(rng, T, k)
            A = vcat(x', -x', (x .* T(3))')
            B = hcat(x, reverse(x), T.(exp10.(range(-6, 6, length = k))) .* randn(rng, T, k))
            A[1, 1] = nextfloat(A[1, 1])
            @test _bounds_hold(A, B)
            @test _bounds_hold(hcat(A, -A), vcat(B, B))        # the exact product is zero
        end
    end

    @testset "underflow" begin
        A = randn(rng, 6, 9) .* 1e-170
        B = randn(rng, 9, 4) .* 1e-160
        @test _bounds_hold(A, B)
        down, up = BA._product_down_up_dot2err(A, B)
        @test all(isfinite, down) && all(isfinite, up)
        @test _bounds_hold(A .* 1e16, B)                       # products near the smallest normal number
        A32 = randn(rng, Float32, 6, 9) .* 1.0f-25
        B32 = randn(rng, Float32, 9, 4) .* 1.0f-22
        @test _bounds_hold(A32, B32)
    end

    @testset "overflow gives (-Inf, Inf)" begin
        down, up = BA._product_down_up_dot2err(fill(1e200, 2, 3), fill(1e200, 3, 2))
        @test all(==(-Inf), down) && all(==(Inf), up)
    end

    @testset "vectors and sizes" begin
        A, x = randn(rng, 5, 8), randn(rng, 8)
        down, up = BA._product_down_up_dot2err(A, x)
        @test down isa Vector{Float64} && length(down) == 5
        ref = _exact_product(A, reshape(x, :, 1))
        @test all(i -> Rational{BigInt}(down[i]) <= ref[i] <= Rational{BigInt}(up[i]), 1:5)
        @test_throws DimensionMismatch BA._product_down_up_dot2err(randn(3, 4), randn(5, 2))
    end

    @testset "res and err of Corollary 5.9" begin
        for T in (Float64, Float32), k in (1, 2, 50, 1000)
            x, y = randn(rng, T, k), randn(rng, T, k)
            k > 2 && (y[end] = -sum(x[1:(end - 1)] .* y[1:(end - 1)]) / x[end])   # a small dot product
            res, err = BA._dot2err_ogita_rump_oishi_2005(x, y)
            ref = sum(Rational{BigInt}.(x) .* Rational{BigInt}.(y))
            @test Rational{BigInt}(res) - Rational{BigInt}(err) <= ref <=
                  Rational{BigInt}(res) + Rational{BigInt}(err)
            # err is eps|res| and a term δe, with e at most about k eps Σ|xᵢyᵢ| by (5.2) of the
            # paper and δ about k eps; the last entry makes res small
            @test err <= eps(T) * abs(res) + (k * eps(T))^2 * sum(abs.(x .* y)) + 4 * floatmin(T)
        end
        @test_throws DimensionMismatch BA._dot2err_ogita_rump_oishi_2005(randn(3), randn(4))
    end
end
