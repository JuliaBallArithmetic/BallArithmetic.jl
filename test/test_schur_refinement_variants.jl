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

        @testset "the three algorithms: $method" for method in (:bujanovicetal2023,
            :bujanovicetal2023truncated, :bujanovicetal2023unordered)
            for n in (8, 20), T in (Float64, ComplexF64)
                A = randn(rng, T, n, n)
                F = schur(complex(A))
                r = refine_schur_decomposition(A, F.Z, F.T; method, rng = MersenneTwister(5))
                res, orth = defects(A, r.Q, r.T)
                @test r.converged
                @test istriu(r.T)
                @test res < 1e-70
                @test orth < 1e-70
                # the eigenvalues are those of the input, as a set
                d0, d1 = diag(F.T), ComplexF64.(diag(r.T))
                @test all(minimum(abs.(z .- d1)) < 1e-8 for z in d0)
                @test all(minimum(abs.(z .- d0)) < 1e-8 for z in d1)
                if method === :bujanovicetal2023unordered
                    # the order of the diagonal is kept
                    @test maximum(abs.(ComplexF64.(diag(r.T)) .- diag(F.T))) < 1e-8
                end
                rd = refine_schur_double64(A, F.Z, F.T; method, rng = MersenneTwister(5))
                res, orth = defects(A, rd.Q, rd.T)
                @test res < 1e-28 && orth < 1e-28
                rm = refine_schur_multifloat(A, F.Z, F.T; method, precision = :x4,
                    rng = MersenneTwister(5))
                res, orth = defects(A, rm.Q, rm.T)
                @test res < 1e-58 && orth < 1e-58
                rh = refine_schur_hybrid(A, F.Z, F.T; method, rng = MersenneTwister(5))
                res, orth = defects(A, rh.Q, rh.T)
                @test res < 1e-70 && orth < 1e-70
            end
        end

        @testset "line 1 of Algorithm 4: the reordering" begin
            for T in (Float64, ComplexF64), n in (5, 12)
                # a real matrix with real eigenvalues, so that its real Schur form is triangular
                A = T === Float64 ? triu(randn(rng, n, n), 1) + Diagonal(randn(rng, n)) :
                    randn(rng, T, n, n)
                if T === Float64
                    Qo = Matrix(qr(randn(rng, n, n)).Q)
                    A = Qo * A * Qo'
                end
                F = schur(A)
                istriu(F.T) || continue
                M = F.Z * F.T * F.Z'
                g = MersenneTwister(77)
                Q, Tr = BallArithmetic._bujanovicetal2023_order(F.Z, F.T, g)
                @test istriu(Tr)
                @test norm(Q * Tr * Q' - M) <= 1e-12 * norm(M)
                @test norm(Q' * Q - I) <= 1e-12
                # the same random line, and the diagonal in the order of its projections
                θ = 2 * pi * rand(MersenneTwister(77))
                proj = real.(diag(Tr) .* cis(-θ))
                @test all(diff(proj) .>= -1e-10)
                # the same eigenvalues
                d0, d1 = complex(diag(F.T)), complex(diag(Tr))
                @test all(minimum(abs.(z .- d1)) < 1e-10 for z in d0)
                @test all(minimum(abs.(z .- d0)) < 1e-10 for z in d1)
            end
            @test_throws ArgumentError BallArithmetic._bujanovicetal2023_order(
                big.([1.0 0.0; 0.0 1.0]), big.([1.0 2.0; 0.0 3.0]), nothing)
        end

        @testset "the truncation of Example 4, and the method keyword" begin
            # stril(T L − L T) = −E with two close eigenvalues: L₂₁ = −E₂₁/(t₂₂ − t₁₁) is large
            T2 = [1.0 2.0; 0.0 1.0 + 1e-9]
            E2 = [0.0 0.0; 1e-12 0.0]
            L = solve_triangular_matrix_equation(T2, E2; use_recursive = false, max_entry = Inf)
            @test abs(L[2, 1]) > 1e-4
            L = solve_triangular_matrix_equation(T2, E2; use_recursive = false, max_entry = 1e-5)
            @test L[2, 1] == 0
            A = randn(rng, 6, 6)
            F = schur(complex(A))
            @test_throws ArgumentError refine_schur_decomposition(A, F.Z, F.T; method = :other)
            @test_throws ArgumentError refine_schur_double64(A, F.Z, F.T; method = :other)
            # a threshold of zero removes every correction: nothing is refined
            r = refine_schur_decomposition(A, F.Z, F.T; method = :bujanovicetal2023truncated,
                threshold = 0.0, max_iterations = 3)
            @test !r.converged
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
