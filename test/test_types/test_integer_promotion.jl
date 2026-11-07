# Promotion and arithmetic against integers of every width (issue #71). The rules used to name
# `Int64`, so an `Int32` argument found no rule at all; on a 32-bit platform an integer literal is
# an `Int32`, which is why `x + 1` failed there.
#
# Each case asserts the type of the result and, in exact arithmetic, that the image of the ball's
# set under the operation is contained in the returned ball. For a real ball that image is an
# interval with rational endpoints; for a complex ball it is a disc, whose centre and radius are
# rational in every case here, inversion included, and the containment is squared so that no
# modulus is taken approximately.

# the exact image of [lo, hi] under the operation with the exact rational `n`
function _intprom_real_image(op, lo, hi, n, ball_on_left)
    if ball_on_left
        op === (+) && return (lo + n, hi + n)
        op === (-) && return (lo - n, hi - n)
        op === (*) && return minmax(lo * n, hi * n)
        op === (/) && return minmax(lo / n, hi / n)
    else
        op === (+) && return (n + lo, n + hi)
        op === (-) && return (n - hi, n - lo)
        op === (*) && return minmax(n * lo, n * hi)
        op === (/) && return minmax(n / lo, n / hi)   # 0 ∉ [lo, hi]
    end
    error("unsupported operation $op")
end

# the exact image of the disc of centre `c` and radius `r` under the operation with the exact
# rational `n`, returned as (centre, radius). For `n / disc` the image of a disc under inversion
# is the disc of centre conj(c)/D and radius r/D with D = |c|² − r² > 0.
function _intprom_cplx_image(op, c, r, n, ball_on_left)
    if ball_on_left
        op === (+) && return (c + n, r)
        op === (-) && return (c - n, r)
        op === (*) && return (c * n, r * abs(n))
        op === (/) && return (c / n, r / abs(n))
    else
        op === (+) && return (n + c, r)
        op === (-) && return (n - c, r)
        op === (*) && return (n * c, r * abs(n))
        if op === (/)
            D = real(c) * real(c) + imag(c) * imag(c) - r * r
            return (n * conj(c) / D, abs(n) * r / D)
        end
    end
    error("unsupported operation $op")
end

# [lo, hi] ⊆ the set of the real ball `res`, exactly
function _intprom_contains(res::Ball{T, T}, image) where {T}
    R = Rational{BigInt}
    lo, hi = image
    return R(res.c) - R(res.r) <= lo && R(res.c) + R(res.r) >= hi
end

# the disc of centre `c` and radius `r` inside the disc of `res`, exactly. The condition
# |c_res − c| + r ≤ r_res is tested as r ≤ r_res and |c_res − c|² ≤ (r_res − r)².
function _intprom_contains(res::Ball{T, Complex{T}}, image) where {T}
    R = Rational{BigInt}
    c, r = image
    rres = R(res.r)
    rres >= r || return false
    dre = R(real(res.c)) - real(c)
    dim = R(imag(res.c)) - imag(c)
    slack = rres - r
    return dre * dre + dim * dim <= slack * slack
end

@testset "integer promotion of every width (issue #71)" begin
    R = Rational{BigInt}
    inttypes = (Int8, Int16, Int32, Int64, Int128, UInt8, UInt32, BigInt, Bool)
    balltypes = (Ball{Float64, Float64}, Ball{Float64, Complex{Float64}},
        Ball{BigFloat, BigFloat})
    balls = (Ball(1.0, 0.1), Ball(1.0 + 2.0im, 0.1), Ball(BigFloat(1), BigFloat(1) / 10))

    @testset "promote_type and convert" begin
        for I in inttypes
            for B in (balltypes..., Ball{BigFloat, Complex{BigFloat}})
                @test promote_type(B, I) == B
                @test promote_type(I, B) == B
            end
            # The matching conversions exist, and are exact for an integer a float can hold.
            # `mid` and `rad` are compared rather than the balls: `Ball` defines no `==`, so the
            # comparison falls back on Base's `===` for equal types, which is false for two
            # BigFloats of equal value.
            for B in (Ball{Float64, Float64}, Ball{Float64, Complex{Float64}},
                Ball{BigFloat, BigFloat}, Ball{BigFloat, Complex{BigFloat}})
                b = convert(B, I(1))
                @test b isa B
                @test mid(b) == 1 && rad(b) == 0
            end
        end
    end

    @testset "arithmetic in both argument orders" begin
        for (b, B) in zip(balls, balltypes), I in inttypes
            values = I === Bool ? (true,) : (I(1), I(3))
            for v in values, op in (+, -, *, /)
                n = R(v)
                for ball_on_left in (true, false)
                    res = ball_on_left ? op(b, v) : op(v, b)
                    @test res isa B
                    image = if b isa Ball{<:AbstractFloat, <:AbstractFloat}
                        lo, hi = R(b.c) - R(b.r), R(b.c) + R(b.r)
                        _intprom_real_image(op, lo, hi, n, ball_on_left)
                    else
                        c = complex(R(real(b.c)), R(imag(b.c)))
                        _intprom_cplx_image(op, c, R(b.r), n, ball_on_left)
                    end
                    @test _intprom_contains(res, image)
                end
            end
        end
    end

    @testset "a literal of the platform's integer width" begin
        # `1` is an `Int32` on a 32-bit platform and an `Int64` on a 64-bit one; both must work
        x = Ball(1.0, 0.1)
        @test typeof(x + 1) == Ball{Float64, Float64}
        @test typeof(x + Int32(1)) == Ball{Float64, Float64}
        @test typeof(x + Int64(1)) == Ball{Float64, Float64}
        y = Ball(1.0 + im, 0.1)
        @test typeof(y + 1) == Ball{Float64, Complex{Float64}}
        @test typeof(y + Int32(1)) == Ball{Float64, Complex{Float64}}
        z = Ball(BigFloat(2), BigFloat(1) / 10)
        @test typeof(z + Int32(3)) == Ball{BigFloat, BigFloat}
        @test mid(z + Int32(3)) == BigFloat(5)
    end

    @testset "integers that Float64 cannot hold exactly" begin
        b = Ball(1.0, 0.0)
        for v in (Int64(2)^60 + 1, Int128(2)^80 + 1, BigInt(2)^80 + 1)
            res = b + v
            @test res isa Ball{Float64, Float64}
            # the exact sum lies in the returned ball: the rounding of the integer's conversion
            # is charged to the radius
            exact = R(1) + R(v)
            @test R(res.c) - R(res.r) <= exact <= R(res.c) + R(res.r)
        end
    end
end
