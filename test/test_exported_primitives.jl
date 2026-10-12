# Tests of small exported functions that had none: norm bounds, the Horner evaluation in balls,
# the two-term compensated product, the inclusion discs, and the SVD cache.

using Test
using BallArithmetic
using LinearAlgebra
using Random
using GenericLinearAlgebra

@testset "exported primitives" begin
    setprecision(BigFloat, 512) do
        rng = MersenneTwister(4242)

        @testset "upper_bound_norm" begin
            for T in (Float64, ComplexF64), n in (1, 5, 40), ρ in (0.0, 1e-9, 0.3)
                c = randn(rng, T, n)
                r = ρ .* rand(rng, n)
                v = BallVector(c, r)
                exact(p) = p == 2 ? sqrt(sum((abs.(big.(c)) .+ big.(r)) .^ 2)) :
                           p == 1 ? sum(abs.(big.(c)) .+ big.(r)) : maximum(abs.(big.(c)) .+ big.(r))
                for p in (1, 2, Inf)
                    b = upper_bound_norm(v, p)
                    @test b isa Float64
                    @test big(b) >= exact(p)
                    @test big(b) <= exact(p) * (1 + 1e-12)
                    # points of the ball: the centre, and the centre moved outwards by the radius
                    x = c .+ r .* sign.(c)
                    @test big(b) >= norm(big.(x), p)
                end
            end
            M = BallMatrix(randn(rng, 4, 6), fill(1e-3, 4, 6))
            @test upper_bound_norm(M) >= norm(mid(M))                  # the Frobenius norm
            @test upper_bound_norm(M, 2) == upper_bound_frobenius(M)
            @test upper_bound_norm(BallVector(Float64[])) == 0.0
            @test_throws ArgumentError upper_bound_norm(M, 3)
            vb = BallVector(BigFloat[3, 4])
            @test upper_bound_norm(vb) isa BigFloat
            @test upper_bound_norm(vb) >= 5
        end

        @testset "pi_norm" begin
            for T in (Float64, ComplexF64)
                Mc = randn(rng, T, 6, 6)
                M = BallMatrix(Mc, fill(1e-6, 6, 6))
                @test pi_norm(M) >= opnorm(Mc, 2)
                @test big(pi_norm(M)) >= sqrt(big(opnorm(Mc, 1)) * big(opnorm(Mc, Inf)))
            end
        end

        @testset "ball_horner" begin
            coeffs = [1.0, -3.0, 0.5, 2.0, -0.25]            # 1 − 3t + t²/2 + 2t³ − t⁴/4
            p(t) = sum(big(coeffs[k]) * t^(k - 1) for k in eachindex(coeffs))
            for (c, r) in ((0.3, 0.0), (0.3, 1e-10), (-1.7, 0.05), (2.0, 0.5))
                b = ball_horner(coeffs, Ball(c, r))
                for t in range(big(c) - big(r), big(c) + big(r); length = 21)
                    @test abs(p(t) - big(mid(b))) <= big(rad(b))
                end
            end
            # a complex ball
            z = Ball(0.2 + 0.7im, 1e-8)
            b = ball_horner(coeffs, z)
            for θ in 0:0.5:6
                t = big(0.2 + 0.7im) + big(1e-8) * cis(big(θ))
                @test abs(p(t) - big(mid(b))) <= big(rad(b))
            end
            @test mid(ball_horner([2.0], Ball(5.0, 1.0))) == 2.0
        end

        @testset "compensated_terms2" begin
            A = randn(rng, 3, 50) .* exp10.(rand(rng, -8:8, 3, 50))
            B = randn(rng, 50, 2)
            C = randn(rng, 3, 7)
            D = randn(rng, 7, 2)
            pairs = ((A, B, 1.0), (C, D, -1.0))
            N = 57
            for i in 1:3, j in 1:2
                s, e = compensated_terms2(pairs, i, j)
                exact = sum(big(A[i, k]) * big(B[k, j]) for k in 1:50) -
                        sum(big(C[i, k]) * big(D[k, j]) for k in 1:7)
                Σ = sum(abs(big(A[i, k]) * big(B[k, j])) for k in 1:50) +
                    sum(abs(big(C[i, k]) * big(D[k, j])) for k in 1:7)
                γ = big(gamma_bound(N + 1, Float64))
                # the bound of the docstring
                @test abs(big(s) + big(e) - exact) <= γ^2 * Σ
                @test compensated_terms(pairs, i, j) == s + e
            end
        end

        @testset "inclusion_discs and eigencount_outside" begin
            D = Diagonal([3.0, 2.0, 0.5, -0.1])
            Q = Matrix(qr(randn(rng, 4, 4)).Q)
            M = Q * D * Q'
            M = (M + M') / 2
            r = verifyeigall(BallMatrix(M))
            ctr, rad_, cnt, ok = inclusion_discs(r)
            @test ok
            @test sum(cnt) == 4
            # the eigenvalues of the matrix as stored, which are those of D up to rounding
            for λ in eigvals(Hermitian(big.(M)))
                @test any(abs(λ - big(ctr[i])) <= big(rad_[i]) for i in eachindex(ctr))
            end
            @test eigencount_outside(r, 1.0) == (2, true)
            @test eigencount_outside(r, 0.3) == (3, true)
            @test eigencount_outside(r, 10.0) == (0, true)
            # a circle through an eigenvalue cannot be decided
            @test eigencount_outside(r, 2.0) == (0, false)
        end

        @testset "det_cofactor and schur_sylvester_midpoint" begin
            M = randn(rng, 3, 3)
            d = det_cofactor(BallMatrix(M, fill(1e-10, 3, 3))).determinant
            @test abs(det(big.(M)) - big(mid(d))) <= big(rad(d))
            @test det_cofactor(BallMatrix(reshape([2.5], 1, 1))).determinant == Ball(2.5, 0.0)
            A = randn(rng, 4, 4) + 4I
            B = randn(rng, 3, 3) + 4I
            C = randn(rng, 4, 3)
            X = schur_sylvester_midpoint(A, B, C)
            @test norm(A * X + X * B - C) <= 1e-11 * norm(C)
            Xr = schur_sylvester_midpoint(A, B, C; prefer_complex_schur = false)
            @test norm(A * Xr + Xr * B - C) <= 1e-11 * norm(C)
        end

        @testset "the SVD cache" begin
            clear_svd_cache!()
            @test svd_cache_stats() == (hits = 0, misses = 0)
            set_svd_cache!(ones(2, 2), [1.0, 1.0], ones(2, 2), UInt64(7))
            @test BallArithmetic._svd_cache_A_hash[] == UInt64(7)
            clear_svd_cache!()
            @test BallArithmetic._svd_cache_U[] === nothing
            @test BallArithmetic._svd_cache_A_hash[] == 0
            @test ogita_iterations_for_precision(53) == 0
            @test ogita_iterations_for_precision(256) == 3
            @test ogita_iterations_for_precision(1024) == 5
        end

        @testset "the routines for matrices with exact entries refuse a ball matrix" begin
            for S in (Float64, ComplexF64), A in (BallMatrix(S[2 1; 1 3]),
                BallMatrix(S[2 1; 1 3], fill(1e-10, 2, 2)))
                for f in (polar_newton, polar_newton_schulz, polar_qdwh, cholesky_qr2,
                    triangular_eigenvectors, verified_lu, verified_qr, verified_cholesky,
                    verified_polar, verified_takagi, verified_lu_gla, verified_qr_gla,
                    verified_cholesky_gla, verified_polar_gla, verified_svd_gla,
                    verified_lu_double64, verified_qr_double64, verified_cholesky_double64,
                    verified_polar_double64, verified_takagi_double64, verified_lu_multifloat,
                    verified_qr_multifloat, verified_cholesky_multifloat,
                    verified_polar_multifloat, verified_takagi_multifloat)
                    @test_throws ArgumentError f(A)
                end
                @test_throws ArgumentError verified_lu(A; use_bigfloat = false)
            end
            # and still take the matrix itself
            @test verified_lu([2.0 1.0; 1.0 3.0]).certified
            @test polar_qdwh([2.0 1.0; 1.0 3.0]) isa Tuple
        end
    end
end
