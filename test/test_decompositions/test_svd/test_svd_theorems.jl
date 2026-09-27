using Test
using BallArithmetic
using LinearAlgebra
using Random

BA = BallArithmetic

const SVD_METHODS = (:miyajima2014_thm7, :miyajima2014_thm4, :miyajima2014_thm11,
    :rump2011_thm3_1)

# every singular value of the 256-bit reference must lie in the corresponding ball
function _svd_encloses(M, method)
    sv = svdbox(BallMatrix(M); method)
    strue = svdvals(BigFloat.(M))
    length(sv) == min(size(M)...) || return false
    for i in eachindex(sv)
        lo = BigFloat(mid(sv[i])) - BigFloat(rad(sv[i]))
        hi = BigFloat(mid(sv[i])) + BigFloat(rad(sv[i]))
        (lo <= strue[i] <= hi) || return false
    end
    return true
end

@testset "svdbox: the singular-value enclosures of Miyajima (2014) and Rump (2011)" begin
    setprecision(256)

    @testset "the caller" begin
        A = BallMatrix(randn(MersenneTwister(1), 6, 6))
        # the default is Miyajima's Theorem 7, which is what `svdbox` is called in his paper
        @test svdbox(A) == svdbox(A; method = :miyajima2014_thm7)
        @test_throws ArgumentError svdbox(A; method = :nonsense)
        @test length(svdbox(A)) == 6
        # Rump's Theorem 3.1 is stated for square frames only
        @test_throws ArgumentError svdbox(BallMatrix(randn(8, 4)); method = :rump2011_thm3_1)
    end

    @testset "every method encloses the exact singular values" begin
        rng = MersenneTwister(3)
        for (label, M) in (("square", randn(rng, 8, 8)),
            ("larger square", randn(rng, 20, 20)),
            ("thin", randn(rng, 12, 6)),
            ("wide", randn(rng, 6, 12)),
            ("graded, cond 1e8",
                Matrix(Diagonal(10.0 .^ range(0, -8, length = 10))) *
                Matrix(qr(randn(rng, 10, 10)).Q)))
            for method in SVD_METHODS
                if method === :rump2011_thm3_1 && size(M, 1) != size(M, 2)
                    continue
                end
                @test _svd_encloses(M, method)
            end
        end
    end

    @testset "clustered and repeated singular values" begin
        # the case Theorem 11's cluster branch exists for: when the Gershgorin intervals of
        # (AV)'AV overlap, an individual eigenvalue need not lie in its OWN interval, so the
        # bound must come from the union over the cluster
        rng = MersenneTwister(4)
        n = 12
        for (k, gap) in ((2, 1e-8), (3, 1e-10), (4, 0.0))
            s = Float64[]
            while length(s) < n
                b = 1 + rand(rng)
                for j in 1:k
                    push!(s, b * (1 + (j - 1) * gap))
                end
            end
            s = sort(s[1:n]; rev = true)
            U = Matrix(qr(randn(rng, n, n)).Q)
            V = Matrix(qr(randn(rng, n, n)).Q)
            M = U * Diagonal(s) * V'
            for method in SVD_METHODS
                @test _svd_encloses(M, method)
            end
        end
    end

    @testset "Theorem 8: Theorem 7 is at least as tight as Theorem 4" begin
        # Theorem 8 compares the BOUNDS. Converting each pair of bounds to a midpoint-radius ball
        # rounds the midpoint, which costs about one ulp of the singular value: at sigma = 5.8 that
        # is 8.9e-16 against a radius of 4.2e-14, so comparing radii can invert a 2% difference
        # even though both intervals are valid and one contains the other. The allowance below is
        # that representation cost, not slack in the theorem.
        rng = MersenneTwister(7)
        for n in (5, 12)
            A = BallMatrix(randn(rng, n, n))
            s7 = svdbox(A; method = :miyajima2014_thm7)
            s4 = svdbox(A; method = :miyajima2014_thm4)
            for i in 1:n
                slack = 4 * eps(mid(s4[i]))
                @test mid(s4[i]) - rad(s4[i]) - slack <= mid(s7[i]) - rad(s7[i])
                @test mid(s7[i]) + rad(s7[i]) <= mid(s4[i]) + rad(s4[i]) + slack
            end
        end
    end

    @testset "Remark 3: the economy residual is measured against A" begin
        # Miyajima's counterexample to the economy form of Rump's split: m = 2n, A = [I; 0],
        # U the block swap, V = I. Every truncated residual vanishes while every singular value
        # is 1, so a bound built on that split would report 0. Theorem 7 must report 1.
        n = 3
        M = vcat(Matrix(1.0I, n, n), zeros(n, n))
        sv = svdbox(BallMatrix(M); method = :miyajima2014_thm7)
        @test length(sv) == n
        for i in 1:n
            @test abs(mid(sv[i]) - 1) <= rad(sv[i]) + 1e-12
        end
    end

    @testset "a singular matrix is handled, not crashed on" begin
        Z = BallMatrix(zeros(4, 4))
        for method in SVD_METHODS
            sv = svdbox(Z; method)
            @test length(sv) == 4
            # zero is a singular value of the zero matrix, and must be enclosed
            @test all(mid(s) - rad(s) <= 0 <= mid(s) + rad(s) for s in sv)
        end
    end

    @testset "MiyajimaM4 is Theorem 11, and no longer Theorem 7 behind a warning" begin
        rng = MersenneTwister(13)
        A = BallMatrix(randn(rng, 10, 10))
        r4 = rigorous_svd(A; method = MiyajimaM4(), apply_vbd = false)
        r1 = rigorous_svd(A; method = MiyajimaM1(), apply_vbd = false)
        thm11 = svdbox(A; method = :miyajima2014_thm11)
        # the M4 route returns Theorem 11's values, not Theorem 7's
        @test mid.(r4.singular_values) == mid.(thm11)
        @test rad.(r4.singular_values) == rad.(thm11)
        @test mid.(r1.singular_values) != mid.(thm11)
        # and it still encloses the truth
        strue = svdvals(BigFloat.(mid(A)))
        for i in eachindex(r4.singular_values)
            s = r4.singular_values[i]
            @test BigFloat(mid(s)) - BigFloat(rad(s)) <= strue[i] <=
                  BigFloat(mid(s)) + BigFloat(rad(s))
        end
        # the path that cannot form (AV)'AV refuses rather than substituting another theorem
        @test_throws ArgumentError BA._compute_svd_bounds(MiyajimaM4(), [1.0], 0.0, 0.0, 0.0,
            Float64)
    end

    @testset "BigFloat" begin
        setprecision(128) do
            M = BigFloat.(randn(MersenneTwister(5), 6, 6))
            sv = svdbox(BallMatrix(M); method = :miyajima2014_thm7)
            strue = svdvals(M)
            for i in eachindex(sv)
                @test mid(sv[i]) - rad(sv[i]) <= strue[i] <= mid(sv[i]) + rad(sv[i])
            end
            @test maximum(rad.(sv)) < 1e-25
        end
    end
end
