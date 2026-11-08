# API - Arithmetic and Rounding

The directed rounding, the error-free transformations and the transforms the package is built on.

```@meta
CurrentModule = BallArithmetic
```

## The rounding units

Three constants of a floating-point format are used throughout, and they are not interchangeable:
the unit roundoff `u`, which bounds the relative error of one correctly rounded operation; the
machine epsilon `2u`, with which the radii are inflated; and the spacing of the subnormals, with
which a radius is kept strictly positive where an operation underflows.

```@docs
unit_roundoff
machine_epsilon
subnormal_min
ϵp
η
```

## The directed rounding operations

Every function in this section returns a **bound**, not an approximation: `*_up` is at least the
exact value and `*_down` at most, and neither is the correctly rounded result. They are built on
the correctly rounded operations of `RoundingEmulator` for `Float32` and `Float64` and on MPFR for
`BigFloat`, and none of them changes the hardware rounding mode, which is what makes them safe
inside `Threads.@threads`.

```@docs
abs_down
sum_up
pow_up
pow_down
root_up
dist_up
dist_down
```

The two macros rewrite a whole expression into these operations.

```@docs
@up
@down
```

## The extended-precision directed operations

`RoundingEmulator` supplies the directed operations for `Float32` and `Float64`; the methods
below extend the same names to `BigFloat` and, where the operation is defined for them, to
`Complex{BigFloat}`, by setting the MPFR rounding mode for the one operation. The downward
complex pairs are componentwise, each part bounded on its own, and say nothing about the
modulus of the result. The three downward complex methods were missing until 2026-11-08, so a
complex lower bound in extended precision raised a `MethodError`.

```@docs
RoundingEmulator.add_up(::BigFloat, ::BigFloat)
RoundingEmulator.add_down(::BigFloat, ::BigFloat)
RoundingEmulator.sub_up(::BigFloat, ::BigFloat)
RoundingEmulator.sub_down(::BigFloat, ::BigFloat)
RoundingEmulator.mul_up(::BigFloat, ::BigFloat)
RoundingEmulator.mul_down(::BigFloat, ::BigFloat)
RoundingEmulator.div_up(::BigFloat, ::BigFloat)
RoundingEmulator.div_down(::BigFloat, ::BigFloat)
RoundingEmulator.sqrt_up(::BigFloat)
RoundingEmulator.sqrt_down(::BigFloat)
RoundingEmulator.add_up(::Complex{BigFloat}, ::Complex{BigFloat})
RoundingEmulator.add_down(::Complex{BigFloat}, ::Complex{BigFloat})
RoundingEmulator.sub_up(::Complex{BigFloat}, ::Complex{BigFloat})
RoundingEmulator.sub_down(::Complex{BigFloat}, ::Complex{BigFloat})
RoundingEmulator.mul_up(::Complex{BigFloat}, ::Complex{BigFloat})
RoundingEmulator.mul_down(::Complex{BigFloat}, ::Complex{BigFloat})
```

## The error-free transformations

A floating-point operation is *error-free* when the rounding it commits is itself representable,
so that the exact result is the sum of two floats. These are the primitives behind compensated
arithmetic and behind the accurate matrix products below; each is one numbered algorithm of
[OgitaEtAl2005](@cite), with its own condition for exactness and its own behaviour under
underflow. None of them needs a directed rounding mode.

```@docs
two_sum
two_product
split_veltkamp
gamma_bound
```

Accumulating a sum of products with them gives an entry to about twice the working precision,
Algorithm 5.3 (Dot2) of the same paper.

```@docs
compensated_terms
compensated_terms2
```

## The doubled-precision products

Products of point matrices computed to about twice the working precision, and ball products whose
radius charges the accuracy bound of the compensated accumulation rather than the rounding of a
BLAS call. The gain is `γ_N²` in place of `γ_N`; at inner dimension `5000` in `Float64` that is
`1.2e-28` against `1.1e-12`.

```@docs
mmul_ogita_rump_oishi_2005
BallArithmetic._dot2err_ogita_rump_oishi_2005
BallArithmetic._product_down_up_dot2err
BallArithmetic._two_term_product_sum
BallArithmetic._accurate_product_sum
BallArithmetic._lange_rump_two_terms
```

## The rounding-mode controlled complex products

The complex matrix product bounded from below and from above with directed rounding, after
[Oishi2001](@cite), and the conversions between rectangular and midpoint-radius bounds that
[Miyajima2010](@cite) builds on it. `oishi_MMul` is the user-facing one; the rest are the
numbered algorithms of that paper, reached from the verification routines.

```@docs
oishi_MMul
BallArithmetic._oishi_MMul_up_lo
BallArithmetic._cprod
BallArithmetic._cr
BallArithmetic._ccr
BallArithmetic._ccrprod
BallArithmetic._ccrprod_prime
BallArithmetic._iprod
BallArithmetic._iprod_right
BallArithmetic._ciprod
BallArithmetic._ciprod_prime
```

## The transforms

The discrete Fourier transform of a ball vector or matrix, with the a priori bound of
[BrisebarreEtAl2023](@cite). The bound is proved for one algorithm, the radix-2 transform
of that paper's Fig. 1 with the roots of unity rounded to nearest, and that algorithm is the
default; the FFTW method carries the same radius as a hypothesis, which its name says.

```@docs
ball_fft
plan_ball_fft
BallFFTPlan
BallArithmetic._fftw_transform
```

## The polynomial ranges

A certified enclosure of the range of a real polynomial on an interval, by two algorithms:
interval branch-and-bound, and the critical-point method through certified roots of the
derivative.

```@docs
PolyRange
ball_horner
range_bisect
range_critical
```
