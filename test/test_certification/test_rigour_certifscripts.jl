using Test
using LinearAlgebra
using BallArithmetic

# Regression tests for the contour certification after the rigour audit of 2026-10-10
# (docs/audit/rigour_audit_2026-10-10.md, item C1). The certified contour is the polygon inscribed
# in the circle; these tests check the bound against exact resolvent norms of diagonal matrices,
# where ‖(zI − A)⁻¹‖₂ = 1 / min_i |z − a_i|.

const _RC = BallArithmetic.CertifScripts

# the largest resolvent norm of diag(d) over points of the polygon with N vertices inscribed in
# the circle (c, r), `per` points on each side
function _rc_polygon_max(d, c, r, N; per = 400)
    worst = 0.0
    for j in 0:(N - 1)
        za = c + r * cis(2π * j / N)
        zb = c + r * cis(2π * (j + 1) / N)
        for t in range(0, 1; length = per)
            z = (1 - t) * za + t * zb
            worst = max(worst, 1 / minimum(abs.(z .- d)))
        end
    end
    return worst
end
_rc_circle_max(d, c, r; n = 200_000) =
    maximum(1 / minimum(abs.(c + r * cis(2π * j / n) .- d)) for j in 0:(n - 1))

@testset "contour certification (audit 2026-10-10)" begin
    @testset "the polygon bound holds on the polygon" begin
        for δ in (1e-2, 1e-4, 1e-6), N in (64, 256)
            # an eigenvalue just outside the circle, at the angle halfway between two vertices
            p = (1 + δ) * cis(π / N)
            d = ComplexF64[p, 0]
            circle = _RC.CertificationCircle(0.0, 1.0; samples = N)
            res = _RC.run_certification(BallMatrix(Matrix(Diagonal(d))), circle;
                η = 0.5, log_io = devnull)
            @test res.resolvent_schur >= _rc_polygon_max(d, 0.0, 1.0, N) * (1 - 1e-9)
            # and the transfer to the circle is either a valid bound or declines
            M = _RC.circle_resolvent_bound(res.resolvent_schur, circle)
            @test M >= _rc_circle_max(d, 0.0, 1.0) * (1 - 1e-6)
        end
    end

    @testset "circle_resolvent_bound: finite where the polygon is fine enough" begin
        d = ComplexF64[3, 0, -2im]
        circle = _RC.CertificationCircle(0.0, 1.0; samples = 128)
        res = _RC.run_certification(BallMatrix(Matrix(Diagonal(d))), circle; η = 0.5, log_io = devnull)
        M = _RC.circle_resolvent_bound(res.resolvent_schur, circle)
        @test isfinite(M)
        @test M >= _rc_circle_max(d, 0.0, 1.0) * (1 - 1e-9)        # the exact maximum is 1
        @test M <= 1.01 * res.resolvent_schur                      # the sagitta costs little here
        @test _RC.circle_resolvent_bound(Inf, circle) == Inf
        @test _RC.circle_resolvent_bound(1e9, circle) == Inf       # M h ≥ 1: nothing follows
    end

    @testset "polygon_sagitta bounds r(1 − cos(π/N))" begin
        for N in (3, 7, 64, 1000), r in (1.0, 0.3, 1e-3, 250.0)
            circle = _RC.CertificationCircle(0.2 + 0.1im, r; samples = N)
            exact = setprecision(256) do
                big(r) * (1 - cos(big(π) / N))
            end
            @test _RC.polygon_sagitta(circle) >= exact
        end
    end

    @testset "a degenerate contour is refused" begin
        for N in (1, 2)
            # the keyword constructor refuses it, and so does the polygon itself when the circle
            # was built positionally
            @test_throws ArgumentError _RC.CertificationCircle(0.0, 1.0; samples = N)
            @test_throws ArgumentError _RC._initial_arcs(_RC.CertificationCircle(0.0 + 0.0im, 1.0, N))
            @test_throws ArgumentError _RC.polygon_sagitta(_RC.CertificationCircle(0.0 + 0.0im, 1.0, N))
        end
    end

    @testset "a contour through the spectrum stops, with an error" begin
        # the vertex θ = 0 of the unit circle is the eigenvalue 1: no bisection can repair that,
        # and the driver must say so rather than loop or return a bound
        circle = _RC.CertificationCircle(0.0, 1.0; samples = 8)
        @test_throws ErrorException _RC.run_certification(BallMatrix([1.0 0.0; 0.0 3.0]), circle;
            log_io = devnull)
    end

    @testset "bound_resolvent_schur" begin
        @test _RC.bound_resolvent_schur(2.0, 0.5) >= 4.0
        @test _RC.bound_resolvent_schur(2.0, 1.0) == Inf
        @test _RC.bound_resolvent_schur(2.0, NaN) == Inf
        # rounded outward: 1/(1 − 0.1) is not representable
        @test _RC.bound_resolvent_schur(1.0, 0.1) >= setprecision(() -> 1 / (1 - big(0.1)), 256)
    end
end
