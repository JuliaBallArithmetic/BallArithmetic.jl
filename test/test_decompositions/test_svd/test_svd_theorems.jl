using Test
using BallArithmetic
using LinearAlgebra
using Random

BA = BallArithmetic

const SVD_METHODS = (:miyajima2014_thm10, :miyajima2014_thm7, :miyajima2014_thm4,
    :miyajima2014_thm11, :rump2011_thm3_1)

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
        # the default is :auto, which for an exact input picks Theorem 10, the paper's M3
        @test svdbox(A) == svdbox(A; method = :auto)
        @test svdbox(A) == svdbox(A; method = :miyajima2014_thm10)
        @test svdbox(A) != svdbox(A; method = :miyajima2014_thm7)
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

    @testset "Theorem 10 uses the economy frames and the one-sided residual" begin
        # No theorem in the paper orders Theorem 10 against Theorem 7, so nothing is asserted
        # about which is tighter; what is asserted is that both enclose. Measured here, Theorem 10
        # is tighter on every case tried: 2.2e-14 against 2.5e-14 on a random 8x8, 9.9e-14 against
        # 1.2e-13 at 20x20, 2e-14 against 2.6e-14 on a 6x12, matching Tables 2 and 3.
        rng = MersenneTwister(23)
        for M in (randn(rng, 9, 9), randn(rng, 14, 5), randn(rng, 5, 14))
            @test _svd_encloses(M, :miyajima2014_thm10)
            @test _svd_encloses(M, :miyajima2014_thm7)
        end
        # the m < n branch exchanges the roles of Fhat and Ghat, since the square frame is the one
        # carrying the inverse; both orientations must enclose
        @test _svd_encloses(randn(rng, 4, 11), :miyajima2014_thm10)
        @test _svd_encloses(randn(rng, 11, 4), :miyajima2014_thm10)
    end

    @testset "Theorem 10 against Theorem 7 depends on whether the input has a radius" begin
        # Theorem 10's residual is A Vhat - Uhat Sigmahat, so an interval input's radius passes
        # through a ball product; Theorem 7's is Uhat Sigmahat Vhat' - A, whose product is formed
        # from floats alone, so the input radius enters once and unpropagated. Measured maxima:
        #
        #   rad(A)   n    Theorem 10   Theorem 7    ratio
        #   0        6    1.954e-14    2.220e-14    0.88
        #   1e-12    6    1.204e-11    6.022e-12    2.00
        #   0        20   8.882e-14    1.066e-13    0.83
        #   1e-12    20   7.330e-11    2.011e-11    3.65
        #
        # so Theorem 10 wins on point input, which is what Miyajima's tables measure, and loses on
        # interval input by a factor that grows with n. Both enclose; only the widths differ.
        rng = MersenneTwister(31)
        n = 12
        M = randn(rng, n, n)
        point = BallMatrix(M)
        wide = BallMatrix(M, fill(1e-12, n, n))
        @test maximum(rad.(svdbox(point; method = :miyajima2014_thm10))) <=
              maximum(rad.(svdbox(point; method = :miyajima2014_thm7)))
        @test maximum(rad.(svdbox(wide; method = :miyajima2014_thm10))) >
              maximum(rad.(svdbox(wide; method = :miyajima2014_thm7)))
        # and both still enclose the midpoint matrix's singular values on the interval input
        strue = svdvals(BigFloat.(M))
        for method in (:miyajima2014_thm10, :miyajima2014_thm7)
            sv = svdbox(wide; method)
            for i in 1:n
                @test BigFloat(mid(sv[i])) - BigFloat(rad(sv[i])) <= strue[i] <=
                      BigFloat(mid(sv[i])) + BigFloat(rad(sv[i]))
            end
        end
    end

    @testset ":auto picks by exactness, and both entry points agree" begin
        rng = MersenneTwister(41)
        n = 12
        M = randn(rng, n, n)
        point = BallMatrix(M)
        wide = BallMatrix(M, fill(1e-12, n, n))

        # exact input: Theorem 10, which is tighter there
        @test svdbox(point) == svdbox(point; method = :miyajima2014_thm10)
        @test maximum(rad.(svdbox(point))) <
              maximum(rad.(svdbox(point; method = :miyajima2014_thm7)))

        # input with a radius: Theorem 7, avoiding the factor Theorem 10 pays there
        @test svdbox(wide) == svdbox(wide; method = :miyajima2014_thm7)
        @test maximum(rad.(svdbox(wide))) <
              maximum(rad.(svdbox(wide; method = :miyajima2014_thm10)))

        # svdbox and rigorous_svd resolve the rule the same way, so their values agree
        for A in (point, wide)
            @test svdbox(A) == rigorous_svd(A; apply_vbd = false).singular_values
        end

        # MiyajimaAuto resolves to a theorem, so nothing downstream sees the rule itself
        @test BA._resolve_svd_method(MiyajimaAuto(), point) isa MiyajimaM3
        @test BA._resolve_svd_method(MiyajimaAuto(), wide) isa MiyajimaM1
        @test BA._resolve_svd_method(MiyajimaM4(), point) isa MiyajimaM4

        # an explicit method overrides the rule in both directions
        @test svdbox(wide; method = :miyajima2014_thm10) !=
              svdbox(wide; method = :miyajima2014_thm7)
        @test rigorous_svd(wide; method = MiyajimaM3(),
            apply_vbd = false).singular_values ==
              svdbox(wide; method = :miyajima2014_thm10)

        # and the rule never costs soundness: both branches enclose
        strue = svdvals(BigFloat.(M))
        for A in (point, wide)
            sv = svdbox(A)
            for i in 1:n
                @test BigFloat(mid(sv[i])) - BigFloat(rad(sv[i])) <= strue[i] <=
                      BigFloat(mid(sv[i])) + BigFloat(rad(sv[i]))
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
