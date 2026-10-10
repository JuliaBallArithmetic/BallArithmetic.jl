using Test
using BallArithmetic
using LinearAlgebra
using Random

# Regression tests for the foundation findings of the rigour audit of 2026-10-10
# (docs/audit/rigour_audit_2026-10-10.md, items P1, P2, P8 and section 3). Each case is the probe
# that exposed the defect; the reference is exact arithmetic on the floating-point inputs,
# evaluated in BigFloat at 1024 bits.

const _RF_BITS = 1024
_rf_big(x::Real) = BigFloat(x; precision = _RF_BITS)
_rf_big(z::Complex) = Complex(_rf_big(real(z)), _rf_big(imag(z)))
# is the exact value `v` (BigFloat or Complex{BigFloat}) in the ball?
_rf_holds(b::Ball, v) = setprecision(_RF_BITS) do
    abs(v - _rf_big(mid(b))) <= _rf_big(rad(b))
end

@testset "rigour foundations (audit 2026-10-10)" begin
    @testset "machine_epsilon is eps for every type" begin
        @test machine_epsilon(Float64) == eps(Float64)
        @test machine_epsilon(Float32) == eps(Float32)
        @test setprecision(() -> machine_epsilon(BigFloat) == eps(BigFloat), 128)
    end

    @testset "complex Ball product contains the exact product" begin
        # the pair found by the audit: the radius was 2.2211e-16 for an error of 2.4660e-16
        x = 0.7072206895038807 + 0.7071808071502466im
        y = 0.7072143182887239 + 0.7072079628735991im
        exact = setprecision(() -> _rf_big(x) * _rf_big(y), _RF_BITS)
        @test _rf_holds(Ball(x) * Ball(y), exact)
        rng = MersenneTwister(1)
        @test (all(1:20000) do _
            a, b = randn(rng, ComplexF64), randn(rng, ComplexF64)
            _rf_holds(Ball(a) * Ball(b), setprecision(() -> _rf_big(a) * _rf_big(b), _RF_BITS))
        end)
        # the same with radii, at a member on the boundary of each factor
        @test (all(1:5000) do _
            a, b = randn(rng, ComplexF64), randn(rng, ComplexF64)
            ra, rb = 1e-3 * rand(rng), 1e-3 * rand(rng)
            ea, eb = cis(2π * rand(rng)), cis(2π * rand(rng))
            v = setprecision(_RF_BITS) do
                (_rf_big(a) + _rf_big(ra) * _rf_big(ea) * (1 - big(2.0)^-40)) *
                (_rf_big(b) + _rf_big(rb) * _rf_big(eb) * (1 - big(2.0)^-40))
            end
            _rf_holds(Ball(a, ra) * Ball(b, rb), v)
        end)
    end

    @testset "complex BigFloat Ball product at 128 bits" begin
        rng = MersenneTwister(2)
        setprecision(128) do
            @test (all(1:5000) do _
                a = Complex(BigFloat(randn(rng)) / 3, BigFloat(randn(rng)) / 7)
                b = Complex(BigFloat(randn(rng)) / 11, BigFloat(randn(rng)) / 13)
                p = Ball(a) * Ball(b)
                exact = setprecision(() -> _rf_big(a) * _rf_big(b), _RF_BITS)
                _rf_holds(p, exact)
            end)
        end
    end

    @testset "scalar times ball matrix and vector, complex" begin
        x = 0.7072206895038807 + 0.7071808071502466im
        y = 0.7072143182887239 + 0.7072079628735991im
        exact = setprecision(() -> _rf_big(x) * _rf_big(y), _RF_BITS)
        P = x * BallMatrix(reshape([y], 1, 1))
        @test _rf_holds(Ball(mid(P)[1, 1], rad(P)[1, 1]), exact)
        v = x * BallVector([y])
        @test _rf_holds(Ball(mid(v)[1], rad(v)[1]), exact)
        # a scalar that is not exactly representable is enclosed, not rounded
        Q = (1 // 3) * BallMatrix(reshape([3.0], 1, 1))
        @test _rf_holds(Ball(mid(Q)[1, 1], rad(Q)[1, 1]), _rf_big(1))
    end

    @testset "constructors and conversions enclose what they are given" begin
        third = setprecision(() -> _rf_big(1) / 3, _RF_BITS)
        @test _rf_holds(Ball(1 // 3), third)
        @test _rf_holds(convert(Ball{Float64, Float64}, 1 // 3), third)
        @test _rf_holds(Ball{Float64, Float64}(π), setprecision(() -> BigFloat(π), _RF_BITS))
        @test rad(Ball(1.5)) == 0.0                       # exactly representable: no radius
        @test rad(Ball(3)) == 0.0
        @test rad(convert(Ball{Float64, Float64}, 0.25)) == 0.0
        # BigFloat ball to Float64: the midpoint moves, the radius must cover it
        b = setprecision(256) do
            convert(Ball{Float64, Float64}, Ball(big(1) / 3, big(0.0)))
        end
        @test _rf_holds(b, setprecision(() -> _rf_big(setprecision(() -> big(1) / 3, 256)), _RF_BITS))
        b2 = setprecision(256) do
            convert(Ball{Float64, Float64}, Ball(big(1) / 3, big(2.0)^-80))
        end
        @test rad(b2) >= 2.0^-80
        @test _rf_holds(b2, setprecision(() -> _rf_big(setprecision(() -> big(1) / 3, 256)), _RF_BITS))
        # widening is exact
        @test rad(convert(Ball{BigFloat, BigFloat}, Ball(0.1, 0.0))) == 0
        # the two-argument constructor rounds the radius up
        @test rad(Ball(1.0, big(1) / 3)) >= big(1) / 3
    end

    @testset "abs of a complex ball contains the modulus" begin
        @test _rf_holds(abs(Ball(1.0 + 1.0im)), setprecision(() -> sqrt(_rf_big(2)), _RF_BITS))
        rng = MersenneTwister(3)
        @test (all(1:20000) do _
            z = randn(rng, ComplexF64) * 10.0^rand(rng, -8:8)
            _rf_holds(abs(Ball(z, 0.0)), setprecision(() -> abs(_rf_big(z)), _RF_BITS))
        end)
        # with a radius: the moduli of boundary members
        @test (all(1:5000) do _
            z = randn(rng, ComplexF64)
            r = rand(rng)
            w = setprecision(() -> _rf_big(z) + _rf_big(r) * _rf_big(cis(2π * rand(rng))) * (1 - big(2.0)^-40),
                _RF_BITS)
            _rf_holds(abs(Ball(z, r)), setprecision(() -> abs(w), _RF_BITS))
        end)
        # real balls: unchanged
        @test abs(Ball(-2.0, 0.5)) == Ball(2.0, 0.5)
        a = abs(Ball(0.25, 1.0))
        @test inf(a) <= 0 && sup(a) >= 1.25
    end

    @testset "inv of a complex ball" begin
        # the audit's case: the old code returned a negative radius
        y = Ball(1.0 + 0.0im, prevfloat(1.0))
        ok = try
            w = inv(y)
            rad(w) >= 0
        catch e
            e isa ArgumentError
        end
        @test ok
        @test_throws ArgumentError inv(Ball(1.0 + 0.0im, 1.0))
        rng = MersenneTwister(4)
        @test (all(1:5000) do _
            m = randn(rng, ComplexF64)
            r = 0.9 * abs(m) * rand(rng)
            w = inv(Ball(m, r))
            rad(w) >= 0 && all(1:4) do _
                z = setprecision(() -> _rf_big(m) + _rf_big(r) * _rf_big(cis(2π * rand(rng))) * (1 - big(2.0)^-40),
                    _RF_BITS)
                _rf_holds(w, setprecision(() -> 1 / z, _RF_BITS))
            end
        end)
        # the radius used for the reciprocal differences in verifyeigall: eps|c|
        @test (all(1:5000) do _
            m = randn(rng, ComplexF64)
            r = eps(abs(m))
            w = inv(Ball(m, r))
            all(1:4) do _
                z = setprecision(() -> _rf_big(m) + _rf_big(r) * _rf_big(cis(2π * rand(rng))) * (1 - big(2.0)^-40),
                    _RF_BITS)
                _rf_holds(w, setprecision(() -> 1 / z, _RF_BITS))
            end
        end)
    end

    @testset "in0 is false for a point outside the open ball" begin
        # the audit's example: |c1 − c2| − r2 = +2.58e-17 and in0 was true
        c1 = 0.0015265390109222654 + 0.02133161083110136im
        c2 = 0.5083786390013106 + 0.6316925394720126im
        r2 = 0.7933722420629942
        @test !in0(Ball(c1, 0.0), Ball(c2, r2))
        @test !in0(BallMatrix(reshape([c1], 1, 1)), BallMatrix(reshape([c2], 1, 1), reshape([r2], 1, 1)))
        rng = MersenneTwister(5)
        # r2 is the distance rounded down, so c1 is never in the open ball
        @test (all(1:50000) do _
            a, b = randn(rng, ComplexF64), randn(rng, ComplexF64)
            r = Float64(setprecision(() -> abs(_rf_big(a) - _rf_big(b)), _RF_BITS), RoundDown)
            !in0(Ball(a, 0.0), Ball(b, r)) &&
                !in0(BallMatrix(reshape([a], 1, 1)), BallMatrix(reshape([b], 1, 1), reshape([r], 1, 1)))
        end)
        # and it still accepts a ball well inside
        @test in0(Ball(0.1 + 0.1im, 0.1), Ball(0.0 + 0.0im, 1.0))
        @test in0(Ball(0.1, 0.1), Ball(0.0, 1.0))
    end

    @testset "in, for a number and for a ball" begin
        @test !in(-2.0^-60, Ball(1.0, 1.0))               # the distance is 1 + 2^-60
        @test in(0.0, Ball(1.0, 1.0))
        @test in(1 // 2, Ball(0.5, 0.0))
        @test in(Ball(0.5, 0.25), Ball(0.5, 0.5))
        @test in(Ball(0.5 + 0.5im, 0.25), Ball(0.5 + 0.5im, 0.5))
        @test !in(Ball(0.5 + 0.5im, 0.5), Ball(0.5 + 0.5im, 0.25))
    end

    @testset "ball_hull and intersect_ball contain what they should" begin
        a, b = Ball(1.0, 0.0), Ball(nextfloat(1.0), 0.0)
        h = ball_hull(a, b)
        @test inf(h) <= 1.0 && sup(h) >= nextfloat(1.0)
        # exact intersection [1, 1 + 2^-52]
        i = intersect_ball(Ball(2.0, 1.0), Ball(0.5 + 2.0^-53, 0.5 + 2.0^-53))
        @test i !== nothing && inf(i) <= 1.0 && sup(i) >= 1.0 + 2.0^-52
        rng = MersenneTwister(6)
        @test (all(1:20000) do _
            x, y = Ball(randn(rng), rand(rng)), Ball(randn(rng), rand(rng))
            hh = ball_hull(x, y)
            inf(hh) <= min(inf(x), inf(y)) && sup(hh) >= max(sup(x), sup(y))
        end)
        @test (all(1:20000) do _
            x, y = Ball(randn(rng, ComplexF64), rand(rng)), Ball(randn(rng, ComplexF64), rand(rng))
            hh = ball_hull(x, y)
            in(x, hh) && in(y, hh)
        end)
        @test intersect_ball(Ball(0.0, 1.0), Ball(3.0, 1.0)) === nothing
    end

    @testset "Oishi centre-radius conversion contains both ends" begin
        Fc, Fr = BallArithmetic._cr(reshape([1.0], 1, 1), reshape([nextfloat(1.0)], 1, 1), Float64)
        @test Fc[1, 1] - Fr[1, 1] <= 1.0 && Fc[1, 1] + Fr[1, 1] >= nextfloat(1.0)
        H = BallArithmetic._ccr(reshape([1.0], 1, 1), reshape([nextfloat(1.0)], 1, 1),
            reshape([0.0], 1, 1), reshape([0.0], 1, 1), Float64)
        @test _rf_holds(Ball(mid(H)[1, 1], rad(H)[1, 1]), _rf_big(nextfloat(1.0)))
        @test _rf_holds(Ball(mid(H)[1, 1], rad(H)[1, 1]), _rf_big(1.0))
    end

    @testset "error-free product under underflow" begin
        A = BallMatrix(fill(3.0e-165, 1, 100))
        B = BallMatrix(fill(7.0e-160, 100, 1))
        C = mmul_ogita_rump_oishi_2005(A, B)
        exact = setprecision(() -> 100 * _rf_big(3.0e-165) * _rf_big(7.0e-160), _RF_BITS)
        @test _rf_holds(Ball(mid(C)[1, 1], rad(C)[1, 1]), exact)
    end
end
