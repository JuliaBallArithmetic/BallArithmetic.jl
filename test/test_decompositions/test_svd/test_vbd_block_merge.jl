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
    res = schur_newton_vbd(BallMatrix(M); refine = :block)
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
    res = schur_newton_vbd(BallMatrix(M); refine = :block)
    @test res.nrmR2 < 1
    d = block_enclosure(res)
    for λ in eigvals(M)
        @test any(abs(λ - x.center) <= x.radius * (1 + 1e-12) for x in d)
    end
end
