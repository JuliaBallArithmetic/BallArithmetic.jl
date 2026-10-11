using Test
using BallArithmetic
using LinearAlgebra
using Random

BA = BallArithmetic

# every eigenvalue of the reference lies in some returned disc
function _all_enclosed(r, lams)
    return all(any(abs(l - r.centers[i]) <= r.radii[i] for i in eachindex(r.clusters))
    for l in lams)
end

@testset "Miyajima (2014a): Algorithms 1 and 2" begin
    @testset "Lemma 2.3, the functional <f>_t" begin
        t = [0.5, 0.25]
        f = [1.0, 3.0]
        # max_i |f_i| / (1 - t_i) = max(1/0.5, 3/0.75) = 4
        @test BA._miyajima2014a_lem2_3(f, t) ≈ 4.0
        # rounded upward, so never below the exact value
        @test BA._miyajima2014a_lem2_3([1.0], [1 / 3]) >= 1 / (1 - 1 / 3)
        # t_i >= 1 has no bound to give
        @test BA._miyajima2014a_lem2_3([1.0], [1.0]) == Inf
        # Corollary 2.4 is the same functional on each column
        F = [1.0 3.0; 3.0 1.0]
        w = BA._miyajima2014a_cor2_4(F, t)
        @test w ≈ [BA._miyajima2014a_lem2_3(F[:, 1], t), BA._miyajima2014a_lem2_3(F[:, 2], t)]
    end

    @testset "Corollary 3.2 covers the spectrum of a triangular matrix" begin
        # the eigenvalues of a triangular matrix are its diagonal, exactly
        rng = MersenneTwister(5)
        for n in (8, 20)
            d = randn(rng, n)
            M = triu(randn(rng, n, n), 1) + Diagonal(d)
            r = verifyeigall(BallMatrix(M); method = :miyajima2014a)
            @test r.spectrum_covered
            @test r.transform_defect < 1          # ||t||_inf < 1 is what proves the cover
            @test _all_enclosed(r, ComplexF64.(d))
            @test sum(length, r.clusters) == n    # the clusters partition 1:n
        end
    end

    @testset "against a 256-bit reference" begin
        setprecision(256) do
            rng = MersenneTwister(9)
            n = 16
            M = randn(rng, n, n)
            r = verifyeigall(BallMatrix(M); method = :miyajima2014a)
            @test r.spectrum_covered
            lams = eigvals(Complex{BigFloat}.(M))
            for l in lams
                @test any(abs(l - Complex{BigFloat}(r.centers[i])) <= BigFloat(r.radii[i])
                for i in eachindex(r.clusters))
            end
        end
    end

    @testset "Algorithm 2 on a genuine cluster" begin
        # repeated eigenvalues, so the discs overlap and the singleton route cannot apply
        rng = MersenneTwister(4)
        n = 12
        for k in (2, 3)
            d = Float64[]
            while length(d) < n
                l = randn(rng)
                for _ in 1:k
                    push!(d, l)
                end
            end
            d = d[1:n]
            M = triu(randn(rng, n, n) .* 1e-6, 1) + Diagonal(d)
            Q = Matrix(qr(randn(rng, n, n)).Q)
            r = verifyeigall(BallMatrix(Q * M * Q'); method = :miyajima2014a)
            @test r.spectrum_covered
            @test any(length(c) > 1 for c in r.clusters)   # Algorithm 2 really ran
            @test _all_enclosed(r, ComplexF64.(d))
        end
    end

    @testset "Theorem 3.11: the subspace returned is an invariant subspace" begin
        rng = MersenneTwister(21)
        n = 12
        M = BallMatrix(randn(rng, n, n))
        r = verifyeigall(M; method = :miyajima2014a)
        tested = 0
        for i in eachindex(r.clusters)
            r.certified[i] || continue
            Y, Blk = r.subspaces[i], r.blocks[i]
            # A Y = Y M for the enclosed block, so the residual must be at the rounding level
            res = M * Y - Y * Blk
            @test BA.upper_bound_L2_opnorm(res) / BA.upper_bound_L2_opnorm(Y) < 1e-8
            tested += 1
        end
        @test tested > 0
    end

    @testset "the pencil A x = lambda B x" begin
        rng = MersenneTwister(17)
        n = 8
        Am, Bm = randn(rng, n, n), randn(rng, n, n) + 3I
        r = verifyeigall(BallMatrix(Am), BallMatrix(Bm))
        @test r.spectrum_covered
        # B is nonsingular here, so B \ A has the pencil's eigenvalues
        lams = eigvals(Complex{BigFloat}.(Bm) \ Complex{BigFloat}.(Am))
        for l in lams
            @test any(abs(l - Complex{BigFloat}(r.centers[i])) <= BigFloat(r.radii[i])
            for i in eachindex(r.clusters))
        end
        # nonsingularity of B is proved, not assumed
        @test r.transform_defect < 1
    end

    @testset "the caller" begin
        A = BallMatrix(randn(MersenneTwister(1), 5, 5))
        B = BallMatrix(Matrix(3.0I, 5, 5))
        # only Miyajima's method takes a pencil; Rump's is stated for a single matrix
        @test_throws ArgumentError verifyeigall(A, B; method = :rump2022aneumann)
        @test_throws ArgumentError verifyeigall(A, B; method = :nonsense)
        @test_throws DimensionMismatch verifyeigall(A, BallMatrix(randn(4, 4)))
        @test_throws ArgumentError verifyeigall(BallMatrix(randn(2, 3)), BallMatrix(randn(2, 3)))
        # one argument means B = I, and must agree with passing I explicitly
        r1 = verifyeigall(A; method = :miyajima2014a)
        r2 = verifyeigall(A, BallMatrix(Matrix(1.0I, 5, 5)))
        @test r1.radii == r2.radii
    end

    @testset "a singular B declines rather than asserting" begin
        # rank-deficient B: the pencil has no finite spectrum to speak of, and ||t||_inf < 1
        # cannot hold, so the method must refuse
        A = BallMatrix(Matrix(1.0I, 4, 4))
        B = BallMatrix(zeros(4, 4))
        r = verifyeigall(A, B)
        @test !r.spectrum_covered
        @test isempty(r.clusters)
    end

    @testset "BigFloat" begin
        # the candidate GED falls back to eigen(mid(B) \ mid(A)) where LAPACK has no method,
        # and the radii then follow the working precision
        setprecision(128) do
            rng = MersenneTwister(4)
            M = BallMatrix(BigFloat.(randn(rng, 6, 6)))
            r = verifyeigall(M; method = :miyajima2014a)
            @test r.spectrum_covered
            @test maximum(filter(isfinite, r.radii)) < 1e-30
        end
    end
end
