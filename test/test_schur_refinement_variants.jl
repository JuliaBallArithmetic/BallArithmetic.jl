# The refinements of a Schur decomposition, of a symmetric eigendecomposition and of a singular
# value decomposition in Double64 and in a MultiFloat: the defects of what they return, measured
# here in BigFloat, independently of the diagnostics they report.

using Test
using BallArithmetic
using LinearAlgebra
using Random
using DoubleFloats
using MultiFloats

@testset "refinements in Double64 and in a MultiFloat" begin
    setprecision(BigFloat, 512) do
        rng = MersenneTwister(808)
        defects(A, Q, T) = (Float64(norm(big.(A) - big.(Q) * big.(T) * big.(Q)') / norm(big.(A))),
            Float64(norm(big.(Q)' * big.(Q) - I)))
        # (name, routine, bound for the two defects)
        variants = (
            ("double64", (A, Q, T) -> refine_schur_double64(A, Q, T), 1e-28),
            ("multifloat x2", (A, Q, T) -> refine_schur_multifloat(A, Q, T), 1e-28),
            ("multifloat x4", (A, Q, T) -> refine_schur_multifloat(A, Q, T; precision = :x4), 1e-58),
            ("hybrid", (A, Q, T) -> refine_schur_hybrid(A, Q, T), 1e-70),
            ("bigfloat", (A, Q, T) -> refine_schur_decomposition(A, Q, T), 1e-70))

        @testset "Schur: $label, n = $n" for n in (8, 20), label in ("real, real eigenvalues",
            "real, complex eigenvalues, complex Schur form", "complex")
            if label == "real, real eigenvalues"
                Q0 = Matrix(qr(randn(rng, n, n)).Q)
                A = Q0 * (triu(randn(rng, n, n), 1) + Diagonal(collect(1.0:n))) * Q0'
                F = schur(A)
                @test istriu(F.T)
            elseif label == "complex"
                A = randn(rng, ComplexF64, n, n)
                F = schur(A)
            else
                A = complex(randn(rng, n, n))
                F = schur(A)
            end
            start = defects(A, F.Z, F.T)[1]
            for (name, f, bound) in variants
                r = f(A, F.Z, F.T)
                res, orth = defects(A, r.Q, r.T)
                @test r isa SchurRefinementResult
                @test istriu(r.T)
                @test res < bound
                @test orth < bound
                @test res < start
                @test r.converged
            end
        end

        @testset "a real Schur form with 2 × 2 blocks is refused" begin
            A = [0.0 -1.0 0.3; 1.0 0.0 0.2; 0.0 0.0 2.0]
            F = schur(A)
            @test !istriu(F.T)
            for (name, f, _) in variants
                @test_throws ArgumentError f(A, F.Z, F.T)
            end
        end

        @testset "symmetric eigendecomposition and singular values in a MultiFloat" begin
            for n in (6, 15), (prec, bound) in ((:x2, 1e-28), (:x4, 1e-58))
                M = randn(rng, n, n)
                S = M + M'
                E = eigen(Symmetric(S))
                r = refine_symmetric_eigen_multifloat(S, E.vectors, E.values; precision = prec,
                    max_iterations = 4)
                Q, λ = big.(r.Q), big.(r.λ)
                @test Float64(norm(big.(S) - Q * Diagonal(λ) * Q') / norm(big.(S))) < bound
                @test Float64(norm(Q' * Q - I)) < bound
                G = svd(M)
                s = ogita_svd_refine_multifloat(M, G.U, G.S, G.V; precision = prec,
                    max_iterations = 4)
                U, Σ, V = big.(s.U), big.(s.Σ), big.(s.V)
                @test Float64(norm(big.(M) - U * Diagonal(Σ) * V') / norm(big.(M))) < bound
                @test Float64(norm(U' * U - I)) < bound
                @test Float64(norm(V' * V - I)) < bound
            end
        end
    end
end
