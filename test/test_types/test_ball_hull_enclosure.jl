using Random

# The enclosure property of `ball_hull`, checked against exact arithmetic (issue #73). The
# endpoints of the arguments and of the result are converted to `Rational{BigInt}` and compared
# there: the loss reported upstream was of one unit in the last place, and a comparison carried out
# in Float64 does not see it. The package's own `in(::Ball, ::Ball)` is a floating-point test built
# on `dist_up`, so it cannot serve here; it is part of what is being checked.
#
# For a real ball the exact set is [c − r, c + r] with c and r the stored floats. For a complex
# ball the containment of `b` in `h` reads |c_h − c_b| + r_b ≤ r_h; the modulus is not rational, so
# the test squares it, which is exact: r_h ≥ r_b together with |c_h − c_b|² ≤ (r_h − r_b)².

# exact endpoints of a real ball
_hull_ends(b) = (Rational{BigInt}(b.c) - Rational{BigInt}(b.r),
    Rational{BigInt}(b.c) + Rational{BigInt}(b.r))

# [b.c − b.r, b.c + b.r] ⊆ [h.c − h.r, h.c + h.r], exactly
function _hull_encloses(h, b)
    hlo, hhi = _hull_ends(h)
    blo, bhi = _hull_ends(b)
    return hlo <= blo && hhi >= bhi
end

# the disc of `b` inside the disc of `h`, exactly. A ball of infinite radius is the whole plane,
# so it contains `b`; the comparison below is rational and cannot be made with `Inf`.
function _hull_encloses_complex(h, b)
    R = Rational{BigInt}
    isinf(h.r) && return isfinite(real(h.c)) && isfinite(imag(h.c))
    isfinite(b.r) || return false
    R(h.r) >= R(b.r) || return false
    dre = R(real(h.c)) - R(real(b.c))
    dim = R(imag(h.c)) - R(imag(b.c))
    slack = R(h.r) - R(b.r)
    return dre * dre + dim * dim <= slack * slack
end

@testset "ball_hull encloses its arguments" begin
    R = Rational{BigInt}

    @testset "the pair reported in issue #73" begin
        left = Ball(-1.0, 0.5)     # [-1.5, -0.5]
        right = Ball(-0.6, 0.75)   # [-1.35, 0.15000000000000002]
        h = ball_hull(left, right)

        @test _hull_encloses(h, left)
        @test _hull_encloses(h, right)
        # the two assertions of the issue, in floating point
        @test inf(h) <= -1.5
        @test sup(h) >= 0.15
        # and the exact one: the upper end of the hull is not below the upper end of `right`
        @test R(h.c) + R(h.r) >= R(right.c) + R(right.r)
        @test ball_hull(right, left) == h
    end

    @testset "structured real pairs" begin
        tiny = nextfloat(0.0)                      # 5.0e-324, subnormal
        pool = [Ball(0.0, 0.0), Ball(-0.0, 0.0), Ball(1.0, 0.0), Ball(-1.0, 0.0),
            Ball(tiny, 0.0), Ball(-tiny, 0.0), Ball(tiny, tiny), Ball(-tiny, tiny),
            Ball(1.0e-320, 1.0e-321), Ball(0.1, 0.1), Ball(-0.6, 0.75), Ball(-1.0, 0.5),
            Ball(0.15, 0.0), Ball(1.0e300, 1.0e290), Ball(-1.0e300, 1.0e290),
            Ball(1.0, nextfloat(0.0)), Ball(0.5, 2.0)]
        for a in pool, b in pool
            h = ball_hull(a, b)
            @test isfinite(h.c) && isfinite(h.r) && h.r >= 0
            @test _hull_encloses(h, a)
            @test _hull_encloses(h, b)
            @test ball_hull(a, b) == ball_hull(b, a)
        end
        # a ball is enclosed by its own hull, and one ball inside another
        outer = Ball(0.5, 2.0)
        for a in pool
            @test _hull_encloses(ball_hull(a, a), a)
            if _hull_encloses(outer, a)
                @test _hull_encloses(ball_hull(a, outer), a)
                @test _hull_encloses(ball_hull(a, outer), outer)
            end
        end
    end

    @testset "randomised real pairs" begin
        rng = MersenneTwister(0x73)
        radii = [0.0, nextfloat(0.0), 1.0e-320, 1.0e-16, 1.0e-3]
        nbad = 0
        for _ in 1:4000
            ca = (rand(rng) - 0.5) * 20.0
            cb = (rand(rng) - 0.5) * 20.0
            ra = rand(rng, Bool) ? rand(rng) * 3.0 : rand(rng, radii)
            rb = rand(rng, Bool) ? rand(rng) * 3.0 : rand(rng, radii)
            a, b = Ball(ca, ra), Ball(cb, rb)
            h = ball_hull(a, b)
            (isfinite(h.c) && isfinite(h.r) && h.r >= 0 &&
             _hull_encloses(h, a) && _hull_encloses(h, b)) || (nbad += 1)
        end
        # `b` drawn inside `a`
        for _ in 1:2000
            ca = (rand(rng) - 0.5) * 20.0
            ra = rand(rng) * 3.0
            b = Ball(ca + (rand(rng) - 0.5) * ra, rand(rng) * (ra / 4))
            a = Ball(ca, ra)
            h = ball_hull(a, b)
            (_hull_encloses(h, a) && _hull_encloses(h, b)) || (nbad += 1)
        end
        # endpoints one unit in the last place apart
        for _ in 1:2000
            c, r = (rand(rng) - 0.5) * 20.0, rand(rng) * 3.0
            a, b = Ball(c, r), Ball(nextfloat(c), nextfloat(r))
            h = ball_hull(a, b)
            (_hull_encloses(h, a) && _hull_encloses(h, b)) || (nbad += 1)
        end
        @test nbad == 0
    end

    @testset "structured and randomised complex pairs" begin
        tiny = nextfloat(0.0)
        pool = [Ball(0.0 + 0.0im, 0.0), Ball(1.0 + 0.0im, 0.0), Ball(-1.0 + 2.0im, 0.5),
            Ball(0.25 - 0.75im, 0.2), Ball(tiny + tiny * im, tiny), Ball(-0.6 + 0.3im, 0.75),
            Ball(1.0e300 - 1.0e300im, 1.0e290), Ball(0.1 + 0.1im, 0.1)]
        for a in pool, b in pool
            h = ball_hull(a, b)
            @test isfinite(real(h.c)) && isfinite(imag(h.c)) && h.r >= 0
            @test _hull_encloses_complex(h, a)
            @test _hull_encloses_complex(h, b)
        end
        rng = MersenneTwister(0x73c)
        nbad = 0
        for _ in 1:4000
            ca = complex((rand(rng) - 0.5) * 20.0, (rand(rng) - 0.5) * 20.0)
            cb = complex((rand(rng) - 0.5) * 20.0, (rand(rng) - 0.5) * 20.0)
            ra = rand(rng, Bool) ? rand(rng) * 3.0 : 0.0
            rb = rand(rng, Bool) ? rand(rng) * 3.0 : nextfloat(0.0)
            a, b = Ball(ca, ra), Ball(cb, rb)
            h = ball_hull(a, b)
            (_hull_encloses_complex(h, a) && _hull_encloses_complex(h, b)) || (nbad += 1)
        end
        for _ in 1:2000
            ca = complex((rand(rng) - 0.5) * 20.0, (rand(rng) - 0.5) * 20.0)
            ra = rand(rng) * 3.0
            b = Ball(ca + complex((rand(rng) - 0.5) * ra / 2, (rand(rng) - 0.5) * ra / 2),
                rand(rng) * (ra / 8))
            a = Ball(ca, ra)
            h = ball_hull(a, b)
            (_hull_encloses_complex(h, a) && _hull_encloses_complex(h, b)) || (nbad += 1)
        end
        @test nbad == 0
    end

    @testset "complex midpoints far apart" begin
        # `ball_hull` for complex midpoints measures their distance with `dist_up`, whose
        # `abs_up(::Complex)` is `sqrt_up(a² + b²)`: the square overflows once a component of the
        # difference passes about 1.34e154, and the radius comes back as `Inf`. The enclosure
        # holds, a ball of infinite radius being the whole plane, and nothing below asserts that
        # the radius is infinite, so a scaled evaluation of the modulus would keep these passing.
        a = Ball(1.0e300 - 1.0e300im, 1.0e290)
        b = Ball(0.1 + 0.1im, 0.1)
        h = ball_hull(a, b)
        @test _hull_encloses_complex(h, a)
        @test _hull_encloses_complex(h, b)
        @test h.r >= 1.0e300            # at least the distance between the two midpoints
        # below the overflow the radius is finite and the enclosure is tight to the scale
        c = Ball(1.0e150 - 1.0e150im, 1.0e140)
        hc = ball_hull(c, b)
        @test isfinite(hc.r)
        @test _hull_encloses_complex(hc, c)
        @test _hull_encloses_complex(hc, b)
    end

    @testset "endpoints that are not finite" begin
        # An unbounded interval is not a ball with a finite midpoint; the only enclosure the
        # representation admits is the whole line, radius `Inf`. The hull returns that rather than
        # a finite ball.
        for (a, b) in ((Ball(Inf, 0.0), Ball(1.0, 0.0)), (Ball(-Inf, 0.0), Ball(1.0, 0.0)),
            (Ball(0.0, Inf), Ball(1.0, 0.0)), (Ball(Inf, 0.0), Ball(-Inf, 0.0)))
            h = ball_hull(a, b)
            @test isinf(h.r)
            @test 0.0 ∈ h && 1.0 ∈ h && -1.0e300 ∈ h
        end
        # A NaN endpoint encloses nothing; `ball_hull` raises instead of returning a ball whose
        # midpoint or radius is NaN.
        @test_throws ArgumentError ball_hull(Ball(NaN, 0.0), Ball(1.0, 0.0))
        @test_throws ArgumentError ball_hull(Ball(1.0, 0.0), Ball(0.0, NaN))
    end

    @testset "the hull of two endpoints does not overflow" begin
        # the width of the interval is above `floatmax`, so it cannot be formed as `hi - lo`
        h = ball_hull(Ball(-floatmax(Float64), 0.0), Ball(floatmax(Float64), 0.0))
        @test isfinite(h.c) && isfinite(h.r)
        @test R(h.c) - R(h.r) <= R(-floatmax(Float64))
        @test R(h.c) + R(h.r) >= R(floatmax(Float64))
    end

    @testset "intersect_ball encloses the intersection" begin
        pool = [Ball(0.0, 1.0), Ball(0.5, 1.0), Ball(-0.6, 0.75), Ball(-1.0, 0.5),
            Ball(0.1, 0.1), Ball(0.0, 0.0), Ball(nextfloat(0.0), nextfloat(0.0))]
        for a in pool, b in pool
            s = intersect_ball(a, b)
            s === nothing && continue
            slo, shi = _hull_ends(s)
            alo, ahi = _hull_ends(a)
            blo, bhi = _hull_ends(b)
            # the result contains [max(lower ends), min(upper ends)], the exact intersection
            @test slo <= max(alo, blo)
            @test shi >= min(ahi, bhi)
        end
    end
end
