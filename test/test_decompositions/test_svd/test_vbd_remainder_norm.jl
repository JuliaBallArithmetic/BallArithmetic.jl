using BallArithmetic
using Test
using LinearAlgebra
using Random
using BallArithmetic: mid, rad

# Regression test for the rigor of `remainder_norm` in the verified block
# diagonalisations.  Historically `remainder_norm` was computed as
# `min(collatz_bound, r2_infty_bound_by_blocks(...))`, which mixed the genuine
# ‖remainder‖₂ bound with the (smaller, unrelated) Sylvester/invariant-subspace
# separation bound and therefore UNDERESTIMATED ‖remainder‖₂ for well-separated
# clusters.  The bound must dominate the off-block norm of every matrix in the
# ball; we certify this by Monte-Carlo sampling.

# off-block mask (zeros inside the within-cluster diagonal blocks)
function offblock_mask(n, clusters)
    M = ones(n, n)
    for cl in clusters
        M[cl, cl] .= 0
    end
    return M
end

# uniform sample of a midpoint matrix strictly inside the ball matrix A
sample_in_ball(rng, A) = mid(A) .+ (2 .* rand(rng, size(mid(A))...) .- 1) .* rad(A)

@testset "VBD remainder_norm is a rigorous ‖remainder‖₂ bound" begin

    @testset "Hermitian path (miyajima_vbd)" begin
        rng = MersenneTwister(20260525)
        # clustered spectrum {10, 10.05} {1, 1.02} {-5}; sizeable off-diagonal radii
        for ro in (0.005, 0.01, 0.02)
            D = Diagonal([10.0, 10.05, 1.0, 1.02, -5.0])
            Q = Matrix(qr(randn(rng, 5, 5)).Q)
            M = Symmetric(Q * D * Q') |> Matrix
            R = fill(ro, 5, 5); R[diagind(R)] .= ro / 2
            A = BallMatrix(M, R)

            vbd = miyajima_vbd(A; hermitian = true)
            U = vbd.basis
            mask = offblock_mask(5, vbd.clusters)

            @test isfinite(vbd.remainder_norm)
            # must dominate the off-block norm of the remainder midpoint
            @test vbd.remainder_norm + 1e-12 >= opnorm(mid(vbd.remainder) .* mask, 2)
            # must dominate every sampled matrix in the ball
            worst = 0.0
            for _ in 1:400
                T = U' * sample_in_ball(rng, A) * U
                worst = max(worst, opnorm(T .* mask, 2))
            end
            @test vbd.remainder_norm + 1e-12 >= worst
        end
    end

    @testset "Schur+Newton path (schur_newton_vbd)" begin
        rng = MersenneTwister(7)
        # well-separated spectrum -> Schur+Newton produces a (block-orthonormal,
        # non-unitary) basis whose off-block remainder is nonzero, exercising the bound.
        for ro in (0.002, 0.005)
            D = Diagonal([12.0, 12.04, 2.0, 2.03, -7.0])
            Q = Matrix(qr(randn(rng, 5, 5)).Q)
            M = Symmetric(Q * D * Q') |> Matrix
            R = fill(ro, 5, 5); R[diagind(R)] .= ro / 2
            A = BallMatrix(M, R)

            r = schur_newton_vbd(A)
            W = r.basis
            Winv = inv(W)
            mask = offblock_mask(5, r.clusters)

            @test isfinite(r.remainder_norm)
            @test r.remainder_norm + 1e-10 >= opnorm(mid(r.remainder) .* mask, 2)
            worst = 0.0
            for _ in 1:400
                T = Winv * sample_in_ball(rng, A) * W
                worst = max(worst, opnorm(T .* mask, 2))
            end
            @test r.remainder_norm + 1e-10 >= worst
        end
    end
end
