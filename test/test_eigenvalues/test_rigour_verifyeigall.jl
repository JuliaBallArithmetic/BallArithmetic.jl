using Test
using BallArithmetic
using LinearAlgebra
using Random

# Regression tests for the findings of the rigour audit of 2026-10-10 in verifyeigall and the
# bounds it calls (docs/audit/rigour_audit_2026-10-10.md, items P3 to P7). References are
# BigFloat computations on the exact floating-point inputs.

@testset "rigour of verifyeigall and its bounds (audit 2026-10-10)" begin
    @testset "collatz_upper_bound never returns NaN and bounds the spectral radius" begin
        # a nilpotent matrix: the iterate has a zero entry, and the old quotient was 0/0
        ρ0 = BallArithmetic.collatz_upper_bound(BallMatrix([0.0 1.0; 0.0 0.0]))
        @test !isnan(ρ0) && ρ0 >= 0
        @test !isnan(BallArithmetic.collatz_upper_bound(BallMatrix(zeros(3, 3))))
        @test_throws DimensionMismatch BallArithmetic.collatz_upper_bound(BallMatrix(ones(2, 3)))
        rng = MersenneTwister(1)
        for n in (2, 5, 12), _ in 1:20
            A = randn(rng, ComplexF64, n, n)
            ρ = BallArithmetic.collatz_upper_bound(BallMatrix(A))
            ref = setprecision(256) do
                maximum(abs, eigvals(Complex{BigFloat}.(abs.(Complex{BigFloat}.(A)))))
            end
            @test ρ >= ref * (1 - big(2.0)^-200)
            # and with a radius: the spectral radius of a member
            R = 1e-3 .* rand(rng, n, n)
            ρr = BallArithmetic.collatz_upper_bound(BallMatrix(A, R))
            member = A .+ R .* cis.(2π .* rand(rng, n, n)) .* (1 - 1e-12)
            refm = setprecision(256) do
                maximum(abs, eigvals(Complex{BigFloat}.(member)))
            end
            @test ρr >= refm * (1 - big(2.0)^-200)
        end
    end

    @testset "upper_bound_L2_opnorm: no iterations, huge entries, random" begin
        I3 = BallMatrix(Matrix(3.0I, 3, 3))
        @test BallArithmetic.collatz_upper_bound_L2_opnorm(I3; iterates = 0) >= 3.0
        big2 = BallMatrix(fill(1e40, 2, 2))
        v = upper_bound_L2_opnorm(big2)
        @test !isnan(v) && v >= 2e40
        c = BallArithmetic.collatz_upper_bound_L2_opnorm(big2)
        @test !isnan(c) && c >= 2e40
        # entries large enough that the squared norm overflows: Inf is acceptable, NaN is not
        @test !isnan(BallArithmetic.collatz_upper_bound_L2_opnorm(BallMatrix(fill(1e200, 2, 2))))
        rng = MersenneTwister(2)
        for (m, n) in ((3, 3), (6, 4), (4, 7)), _ in 1:15
            A = randn(rng, ComplexF64, m, n)
            ref = setprecision(256) do
                sqrt(maximum(real, eigvals(Hermitian(Complex{BigFloat}.(A)' * Complex{BigFloat}.(A)))))
            end
            @test upper_bound_L2_opnorm(BallMatrix(A)) >= ref * (1 - big(2.0)^-200)
            @test BallArithmetic.collatz_upper_bound_L2_opnorm(BallMatrix(A)) >= ref * (1 - big(2.0)^-200)
        end
    end

    @testset "the prodK residual encloses B W − W X" begin
        rng = MersenneTwister(3)
        for (n, k) in ((4, 4), (9, 9), (7, 3))
            B = randn(rng, ComplexF64, n, n)
            W = randn(rng, ComplexF64, n, k)
            X = randn(rng, ComplexF64, k, k)
            Res = BallArithmetic._rump2022a_prodK(BallMatrix(B), W, X)
            exact = setprecision(1024) do
                Complex{BigFloat}.(B) * Complex{BigFloat}.(W) - Complex{BigFloat}.(W) * Complex{BigFloat}.(X)
            end
            @test all(eachindex(exact)) do i
                abs(exact[i] - mid(Res)[i]) <= rad(Res)[i]
            end
        end
        # a residual that nearly cancels, as in the transformation: W the eigenvectors, X the
        # eigenvalues of B
        B = randn(rng, 8, 8)
        F = eigen(B)
        W, X = Matrix{ComplexF64}(F.vectors), Matrix{ComplexF64}(Diagonal(F.values))
        Res = BallArithmetic._rump2022a_prodK(BallMatrix(B), W, X)
        exact = setprecision(1024) do
            Complex{BigFloat}.(B) * Complex{BigFloat}.(W) - Complex{BigFloat}.(W) * Complex{BigFloat}.(X)
        end
        @test all(i -> abs(exact[i] - mid(Res)[i]) <= rad(Res)[i], eachindex(exact))
        @test maximum(rad(Res)) < 1e-13            # still a compensated radius
    end

    @testset "the block of a certified cluster contains the eigenvalue" begin
        rng = MersenneTwister(4)
        for n in (6, 15), shift in (0.0, 1000.0)
            A = randn(rng, n, n) + shift * I
            λ = setprecision(1024) do
                eigvals(Complex{BigFloat}.(A))
            end
            for method in (:rump2022a, :rump2022aneumann)
                r = verifyeigall(BallMatrix(A); method)
                @test r.spectrum_covered
                for i in eachindex(r.clusters)
                    (r.certified[i] && length(r.clusters[i]) == 1) || continue
                    blk = r.blocks[i]
                    @test any(l -> abs(l - mid(blk)[1, 1]) <= rad(blk)[1, 1], λ)
                    @test any(l -> abs(l - r.centers[i]) <= r.radii[i], λ)
                end
            end
        end
    end
end
