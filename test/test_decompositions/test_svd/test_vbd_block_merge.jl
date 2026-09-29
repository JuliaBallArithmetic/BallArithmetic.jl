using BallArithmetic
using Test
using LinearAlgebra
using Random

# Regression test for the guard on the Sylvester solution in `_vbd_solve(mode = :block)`.
#
# The block mode eliminates the strip above each cluster by `V = sylvester(...)` and applies
# `W <- W[I V; 0 I]`. It used to apply whatever came back, checking only `all(isfinite, V)`.
# The conditioning of that transform is `kappa ~ ||V||_2^2`, so two near-coincident clusters
# produce a basis the downstream Neumann test `||R2|| < 1` rejects, and the candidate is lost
# with "no VBD candidate could be certified" rather than being made usable.
#
# A matrix unitarily similar to a single Jordan block is the extreme case: every eigenvalue
# coincides, the Gershgorin clustering still returns singletons because the computed eigenvalues
# are spread by u^(1/n), and the eliminations blow up. Before the guard this built
# cond(W) = 1.8e15 and failed at ||R2||_inf = 1.46; with it the blocks merge and W stays unitary.

@testset "VBD block mode: an unbounded Sylvester solution merges instead of failing" begin
    rng = MersenneTwister(20260927)
    n = 24
    J = diagm(0 => fill(0.5 + 0im, n), 1 => fill(1.0 + 0im, n - 1))
    U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    M = U * J * U'

    W, cl = BallArithmetic._vbd_solve(Matrix{ComplexF64}(M); mode = :block)
    # the clusters merged rather than each being separated by a huge V
    @test length(cl) < n
    # and the basis is well conditioned, which is what the downstream Neumann test needs
    @test cond(W) < 1e3

    # the whole routine now certifies instead of throwing
    res = miyajima2014a_schurnewton(BallMatrix(M); refine = :block)
    @test res.nrmR2 < 1
    d = block_enclosure(res)
    @test !isempty(d)
    # every eigenvalue of the midpoint lies in some disc
    for λ in eigvals(M)
        @test any(abs(λ - x.center) <= x.radius * (1 + 1e-12) for x in d)
    end
end

@testset "VBD block mode: separated clusters are unaffected" begin
    rng = MersenneTwister(4242)
    n = 24
    D = Diagonal(vcat(fill(1.0 + 0im, n ÷ 2), fill(-2.0 + 0im, n - n ÷ 2)))
    S = I + 0.3 * triu(randn(rng, ComplexF64, n, n), 1)
    M = Matrix(S * D * inv(S))
    res = miyajima2014a_schurnewton(BallMatrix(M); refine = :block)
    @test res.nrmR2 < 1
    d = block_enclosure(res)
    for λ in eigvals(M)
        @test any(abs(λ - x.center) <= x.radius * (1 + 1e-12) for x in d)
    end
end

@testset "clusters are the connected components of the discs returned with them" begin
    # Gershgorin's counting theorem licenses "this cluster holds exactly |C| eigenvalues" only
    # when the clusters ARE the connected components of the discs being returned, and
    # `block_enclosure` reports that count as `mult`. Reorthogonalising a merged block changes the
    # frame and hence the discs, so it can change the components; doing it after the clustering
    # loop and certifying once more left `clusters` describing the previous discs. Measured over
    # 14 matrices, 6 came back inconsistent, a defective triangular at n = 12 returning
    # [6,1,1,1,1,1,1] whose own discs merge into [9,1,1,1].
    function consistent(M)
        res = BallArithmetic.miyajima2014a_schurnewton(BallMatrix(M))
        implied, ord = BallArithmetic._interval_clusters(res.cluster_intervals)
        return implied == res.clusters && ord == collect(1:size(M, 1))
    end

    n = 12
    d = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    rng = MersenneTwister(7)
    # the case that used to fail: repeated diagonal with an O(1) strict upper part, so each
    # repeated pair is a genuine 2x2 Jordan block
    @test consistent(triu(randn(rng, n, n), 1) + Diagonal(d))
    # and a nearly normal version, which used to pass
    Q = Matrix(qr(randn(rng, n, n)).Q)
    @test consistent(Q * (triu(randn(rng, n, n) .* 1e-8, 1) + Diagonal(d)) * Q')
    @test consistent(randn(rng, 12, 12))
    @test consistent([(j == i - 1) ? -1.0 : (0 <= j - i <= 3 ? 1.0 : 0.0) for i in 1:20, j in 1:20])
    Qj = Matrix(qr(randn(rng, 20, 20)).Q)
    @test consistent(Qj * diagm(0 => fill(0.7, 20), 1 => ones(19)) * Qj')
    # the six seeds that produced the inconsistencies
    for s in 1:6
        rr = MersenneTwister(100 + s)
        nn = 14
        dd = Float64[]
        while length(dd) < nn
            l = randn(rr)
            for _ in 1:2
                push!(dd, l)
            end
        end
        @test consistent(triu(randn(rr, nn, nn) .* 0.3, 1) + Diagonal(dd[1:nn]))
    end
end
