using BallArithmetic
using LinearAlgebra
using Random
using Test

@testset "Pseudospectrum cells" begin
    rng = MersenneTwister(20261011)
    σtrue(A, z) = minimum(svdvals(ComplexF64.(A) - z * I))
    # points of a cell: the centre, the corners, and random ones
    function points(c, rng)
        pts = [c.centre]
        for sx in (-1, 1), sy in (-1, 1)
            push!(pts, c.centre + complex(sx * c.hx, sy * c.hy))
        end
        for _ in 1:4
            push!(pts, c.centre + complex((2rand(rng) - 1) * c.hx, (2rand(rng) - 1) * c.hy))
        end
        return pts
    end
    area(cs) = sum((4 * c.hx * c.hy for c in cs); init = 0.0)

    @testset "sigma_min_upper" begin
        for A in (randn(rng, 7, 7), randn(rng, ComplexF64, 5, 5))
            bA = BallMatrix(A)
            for z in (0.3 + 0.2im, -2.0 + 1.0im, 5.0 + 0im)
                u = sigma_min_upper(bA, z)
                @test u ≥ σtrue(A, z) * (1 - 1e-12)
                @test u ≤ σtrue(A, z) * (1 + 1e-6)
            end
        end
        A = randn(rng, 5, 5)
        u = sigma_min_upper(BallMatrix(A, fill(1e-3, 5, 5)), 0.4 + 0.1im)
        for _ in 1:20
            @test u ≥ σtrue(A + 1e-3 * (2 * rand(rng, 5, 5) .- 1), 0.4 + 0.1im)
        end
    end

    @testset "$name" for (name, A, ε) in (("normal", (Q = Matrix(qr(randn(rng, 6, 6)).Q);
            Q * Diagonal([1.0, 2.0, 3.0, -1.0, -2.0, 0.5]) * Q'), 0.2),
        ("random", randn(rng, 6, 6), 0.1),
        ("chain", Matrix(Bidiagonal(Float64.(1:6), fill(0.3, 5), :U)), 0.15))

        bA = BallMatrix(A)
        fb = block_resolvent_floor(miyajima2014a_schurnewton(bA))
        fs = svd_frame_floor(bA)
        lower = z -> max(sigma_min_floor(fb, z; near = true), sigma_min_floor(fs, z))
        upper = z -> sigma_min_upper(bA, z)
        lo, hi = -5.0 - 4.0im, 8.0 + 4.0im
        r = pseudospectrum_cells(lower, upper, lo, hi; ε, min_halfdiag = 0.05)
        @test !isempty(r.excluded) && !isempty(r.included)
        for c in r.excluded, z in points(c, rng)
            @test σtrue(A, z) > ε
        end
        for c in r.included, z in points(c, rng)
            @test σtrue(A, z) < ε
        end
        # the cells tile the rectangle
        @test area(vcat(r.excluded, r.included, r.undecided)) ≈ 13.0 * 8.0 rtol = 1e-10
        # every eigenvalue is in an included or undecided cell
        for λ in eigvals(A)
            @test any(c -> abs(real(λ - c.centre)) ≤ c.hx && abs(imag(λ - c.centre)) ≤ c.hy,
                vcat(r.included, r.undecided))
        end
        @test r.evaluations ≥ length(r.excluded) + length(r.included) + length(r.undecided)
    end

    @testset "bounds from a singular value computation: the undecided cells follow the level curve" begin
        A = randn(rng, 5, 5)
        bA = BallMatrix(A)
        lower = z -> (σ = svdbox(BallArithmetic._shifted_ball(ComplexF64(z), bA));
        max(0.0, minimum(mid(x) - rad(x) for x in σ)))
        upper = z -> sigma_min_upper(bA, z)
        ε = 0.2
        r1 = pseudospectrum_cells(lower, upper, -4 - 4im, 4 + 4im; ε, min_halfdiag = 0.1)
        r2 = pseudospectrum_cells(lower, upper, -4 - 4im, 4 + 4im; ε, min_halfdiag = 0.025)
        @test area(r2.undecided) < 0.5 * area(r1.undecided)
        for c in r2.excluded, z in points(c, rng)
            @test σtrue(A, z) > ε
        end
        for c in r2.included, z in points(c, rng)
            @test σtrue(A, z) < ε
        end
    end

    @testset "limits and arguments" begin
        A = BallMatrix(randn(rng, 4, 4))
        f = svd_frame_floor(A)
        lower, upper = z -> sigma_min_floor(f, z), z -> sigma_min_upper(A, z)
        r = pseudospectrum_cells(lower, upper, -3 - 3im, 3 + 3im; ε = 0.1, min_halfdiag = 0.01,
            max_evaluations = 50)
        @test r.evaluations == 50
        @test sum(4 * c.hx * c.hy for c in vcat(r.excluded, r.included, r.undecided)) ≈ 36.0 rtol = 1e-10
        # bounds that say nothing: everything is undecided
        r0 = pseudospectrum_cells(z -> 0.0, z -> Inf, -1 - 1im, 1 + 1im; ε = 0.1, min_halfdiag = 0.3)
        @test isempty(r0.excluded) && isempty(r0.included) && !isempty(r0.undecided)
        @test_throws ArgumentError pseudospectrum_cells(lower, upper, 0, 1 + 1im; ε = -1, min_halfdiag = 0.1)
        @test_throws ArgumentError pseudospectrum_cells(lower, upper, 0, 1; ε = 0.1, min_halfdiag = 0.1)
    end
end
