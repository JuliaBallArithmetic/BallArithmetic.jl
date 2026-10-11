# Tests of verified_linear_solve_hmatrix against A. Minamihata, T. Ogita, S. M. Rump and
# S. Oishi, "Modified error bounds for approximate solutions of dense linear systems",
# J. Comput. Appl. Math. 369 (2020), 112546. Ground truth: systems solved in rational
# arithmetic at the midpoint and at vertices of the data.

using Test
using BallArithmetic
using LinearAlgebra
using Random

const HMATRIX_METHODS = (:theorem_3_1, :theorem_3_2, :theorem_3_3, :corollary_3_2)

# the solutions of the midpoint system and of `nvert` vertex systems, in Rational{BigInt}
function _hmatrix_exact_solutions(A, b, rng; nvert = 6)
    Am, Ar, bm, br = mid(A), rad(A), mid(b), rad(b)
    sols = Vector{Vector{Rational{BigInt}}}()
    for t in 0:nvert
        SA = t == 0 ? zeros(Int, size(Am)) : rand(rng, (-1, 1), size(Am))
        sb = t == 0 ? zeros(Int, length(bm)) : rand(rng, (-1, 1), length(bm))
        Aq = Rational{BigInt}.(Am) .+ SA .* Rational{BigInt}.(Ar)
        bq = Rational{BigInt}.(bm) .+ sb .* Rational{BigInt}.(br)
        push!(sols, Aq \ bq)
    end
    return sols
end

_hmatrix_contains(res, x) = all(abs.(x .- Rational{BigInt}.(res.x_approx)) .<=
                                Rational{BigInt}.(res.error_bound))

@testset "Verified Linear System with H-matrices" begin
    @testset "mig, mag and the comparison matrix" begin
        @test mig(Ball(2.0, 0.5)) == 1.5
        @test mag(Ball(2.0, 0.5)) == 2.5
        @test mig(Ball(-2.0, 0.5)) == 1.5
        @test mag(Ball(-2.0, 0.5)) == 2.5
        @test mig(Ball(0.5, 1.0)) == 0.0
        @test mag(Ball(0.5, 1.0)) == 1.5
        # rounding: mig from below, mag from above
        x = Ball(1.0, 1e-17)
        @test mig(x) < 1.0 < mag(x)
        C = BallMatrix([3.0 -1.0; 2.0 4.0], fill(0.25, 2, 2))
        @test comparison_matrix(C) == [2.75 -1.25; -2.25 3.75]
        @test_throws DimensionMismatch comparison_matrix(BallMatrix(zeros(2, 3)))
    end

    @testset "containment, random systems with radii" begin
        rng = MersenneTwister(20261011)
        for n in (2, 5, 12), ρ in (0.0, 1e-13, 1e-8), trial in 1:4
            Am = randn(rng, n, n) + n * I
            A = BallMatrix(Am, ρ .* abs.(Am))
            bm = randn(rng, n)
            b = BallVector(bm, ρ .* abs.(bm))
            sols = _hmatrix_exact_solutions(A, b, rng)
            for method in HMATRIX_METHODS, m in (1, 3)
                res = verified_linear_solve_hmatrix(A, b; method = method, m = m)
                @test res.certified
                res.certified || continue
                @test res.method === method
                @test all(isfinite, res.error_bound)
                @test mid(res.solution) == res.x_approx
                @test rad(res.solution) == res.error_bound
                for x in sols
                    @test _hmatrix_contains(res, x)
                end
            end
        end
    end

    @testset "containment, ill-conditioned" begin
        rng = MersenneTwister(7)
        for n in (4, 6, 8)
            Am = [1.0 / (i + j - 1) for i in 1:n, j in 1:n]
            A = BallMatrix(Am)
            b = BallVector(Am * ones(n))
            sols = _hmatrix_exact_solutions(A, b, rng; nvert = 0)
            for method in HMATRIX_METHODS
                res = verified_linear_solve_hmatrix(A, b; method = method)
                res.certified || continue
                @test _hmatrix_contains(res, sols[1])
            end
        end
    end

    @testset "the bounds compared" begin
        rng = MersenneTwister(3)
        for trial in 1:10
            n = 8
            Am = randn(rng, n, n) + 3 * I
            A = BallMatrix(Am, fill(1e-12, n, n))
            b = BallVector(randn(rng, n), fill(1e-12, n))
            r1 = verified_linear_solve_hmatrix(A, b; method = :theorem_3_1)
            r1.certified || continue
            r3 = verified_linear_solve_hmatrix(A, b; method = :theorem_3_3)
            r3m = verified_linear_solve_hmatrix(A, b; method = :theorem_3_3, m = 4)
            # Corollary 3.1: Theorem 3.3 with m = 1 is never above Theorem 3.1
            @test all(r3.error_bound .<= r1.error_bound .* (1 + 1e-10))
            # the minimum over more terms is never larger
            @test all(r3m.error_bound .<= r3.error_bound)
        end
    end

    @testset "given R and x̃" begin
        A = BallMatrix([2.0 1.0; 0.0 3.0], fill(1e-14, 2, 2))
        b = BallVector([3.0, 3.0], fill(1e-14, 2))
        R = inv(mid(A))
        res = verified_linear_solve_hmatrix(A, b; R = R, x_approx = [1.0, 1.0])
        @test res.certified
        @test res.R == R
        @test res.x_approx == [1.0, 1.0]
        @test all(res.error_bound .< 1e-12)
        # a poor x̃ is still enclosed, with a bound of its size
        res = verified_linear_solve_hmatrix(A, b; x_approx = [1.5, 0.5])
        @test res.certified
        @test all(abs.([1.0, 1.0] .- res.x_approx) .<= res.error_bound)
        # a poor R that leaves RA an H-matrix
        res = verified_linear_solve_hmatrix(A, b; R = [0.5 0.0; 0.0 0.3])
        @test res.certified
        @test all(abs.([1.0, 1.0] .- res.x_approx) .<= res.error_bound)
    end

    @testset "failure is reported" begin
        # a singular matrix in the ball: no proof is possible
        A = BallMatrix([1.0 1.0; 1.0 1.0 + 1e-10], fill(1e-6, 2, 2))
        b = BallVector([1.0, 2.0])
        for method in HMATRIX_METHODS
            res = verified_linear_solve_hmatrix(A, b; method = method)
            @test !res.certified
            @test all(isinf, res.error_bound)
        end
        res = verified_linear_solve_hmatrix(BallMatrix([1.0 2.0; 2.0 4.0]), b)
        @test !res.certified
        # R far from an inverse
        res = verified_linear_solve_hmatrix(BallMatrix([1.0 2.0; 3.0 4.0]), b;
            R = [1.0 0.0; 0.0 1.0])
        @test !res.certified
        @test_throws ArgumentError verified_linear_solve_hmatrix(
            BallMatrix([2.0 0.0; 0.0 2.0]), b; method = :improved_method_a)
        @test_throws ArgumentError verified_linear_solve_hmatrix(
            BallMatrix([2.0 0.0; 0.0 2.0]), b; m = 0)
        @test_throws DimensionMismatch verified_linear_solve_hmatrix(
            BallMatrix(zeros(2, 3)), b)
    end
end

println("All verified H-matrix linear system tests passed!")
