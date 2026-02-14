module ArbNumericsExt

using BallArithmetic
import ArbNumerics
using SetRounding: setrounding

# ============================================================================
# Float64 conversion helpers — truncation error absorbed into radius
# ============================================================================

"""
    _arb_to_float_with_radius(x::ArbNumerics.ArbReal)

Return a tuple `(mid, rad)` where `mid` is the `Float64` midpoint of `x`
and `rad` is an upper bound on the uncertainty after rounding `x` to
`Float64` precision.  The bound combines the original Arb radius and the
rounding error introduced by the midpoint conversion.
"""
function _arb_to_float_with_radius(x::ArbNumerics.ArbReal{P}) where {P}
    mid = ArbNumerics.midpoint(x)
    rad = ArbNumerics.radius(x)

    mid_f = Float64(mid)

    total = Ref(rad)
    rad_f = setrounding(Float64, RoundUp) do
        round_err = abs(mid - ArbNumerics.ArbReal(mid_f))
        total[] = rad + round_err
        Float64(ArbNumerics.midpoint(total[])) + Float64(ArbNumerics.radius(total[]))
    end

    if rad_f == 0.0 && !iszero(total[])
        rad_f = nextfloat(0.0)
    end

    return mid_f, rad_f
end

function _arbcomplex_to_float_with_radius(x::ArbNumerics.ArbComplex{P}) where {P}
    real_part = real(x)
    imag_part = imag(x)

    mid_real = ArbNumerics.midpoint(real_part)
    mid_imag = ArbNumerics.midpoint(imag_part)

    mid = ComplexF64(Float64(mid_real), Float64(mid_imag))

    rad_real = ArbNumerics.radius(real_part)
    rad_imag = ArbNumerics.radius(imag_part)

    total_real = Ref(rad_real)
    total_imag = Ref(rad_imag)

    rad = setrounding(Float64, RoundUp) do
        err_real = abs(mid_real - ArbNumerics.ArbReal(real(mid)))
        err_imag = abs(mid_imag - ArbNumerics.ArbReal(imag(mid)))

        total_real[] = rad_real + err_real
        total_imag[] = rad_imag + err_imag

        real_hi = Float64(ArbNumerics.midpoint(total_real[])) + Float64(ArbNumerics.radius(total_real[]))
        imag_hi = Float64(ArbNumerics.midpoint(total_imag[])) + Float64(ArbNumerics.radius(total_imag[]))
        sqrt(real_hi^2 + imag_hi^2)
    end

    if rad == 0.0 && (!iszero(total_real[]) || !iszero(total_imag[]))
        rad = nextfloat(0.0)
    end

    return mid, rad
end

# ============================================================================
# BigFloat conversion helpers — no Float64 truncation
# ============================================================================
#
# When BigFloat precision ≥ Arb precision, BigFloat(ArbReal_midpoint) is exact
# (both use MPFR internally), so no conversion error needs to be tracked.
# The BallMatrix radius is just the Arb ball radius, converted to BigFloat.

"""
    _arb_to_bigfloat_with_radius(x::ArbNumerics.ArbReal)

Return `(mid, rad)` where `mid` is the `BigFloat` midpoint of `x` and
`rad` is a rigorous `BigFloat` upper bound on the Arb ball radius.

When `precision(BigFloat) ≥ P` (the Arb working precision), the midpoint
conversion is exact and the radius equals the Arb radius.  No conversion
error is introduced (unlike the `Float64` variant).
"""
function _arb_to_bigfloat_with_radius(x::ArbNumerics.ArbReal{P}) where {P}
    mid_bf = BigFloat(ArbNumerics.midpoint(x))
    rad_bf = BigFloat(ArbNumerics.radius(x))
    return mid_bf, rad_bf
end

function _arbcomplex_to_bigfloat_with_radius(x::ArbNumerics.ArbComplex{P}) where {P}
    real_part = real(x)
    imag_part = imag(x)

    mid_real = BigFloat(ArbNumerics.midpoint(real_part))
    mid_imag = BigFloat(ArbNumerics.midpoint(imag_part))
    rad_real = BigFloat(ArbNumerics.radius(real_part))
    rad_imag = BigFloat(ArbNumerics.radius(imag_part))

    mid_bf = Complex{BigFloat}(mid_real, mid_imag)

    # Rigorous combined radius: √(rad_real² + rad_imag²) rounded up
    rad_bf = setrounding(BigFloat, RoundUp) do
        sqrt(rad_real * rad_real + rad_imag * rad_imag)
    end

    return mid_bf, rad_bf
end

# ============================================================================
# Default constructors — preserve precision (BigFloat)
# ============================================================================

"""
    BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbReal})

Convert a matrix of `ArbReal` numbers into a `BallMatrix{BigFloat}`,
preserving the full Arb precision.  Requires `precision(BigFloat) ≥` Arb
precision for exact midpoint conversion; otherwise the midpoint is rounded
to BigFloat precision (still much better than Float64).

To explicitly truncate to Float64 (with the conversion error absorbed into
the radius), use `BallMatrix(Float64, A)`.
"""
function BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbReal{P}}) where {P}
    return BallArithmetic.BallMatrix(BigFloat, A)
end

"""
    BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbComplex})

Convert a matrix of `ArbComplex` numbers into a `BallMatrix{Complex{BigFloat}}`,
preserving the full Arb precision.  Requires `precision(BigFloat) ≥` Arb
precision for exact midpoint conversion.

To explicitly truncate to Float64 (with the conversion error absorbed into
the radius), use `BallMatrix(Float64, A)`.
"""
function BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbComplex{P}}) where {P}
    return BallArithmetic.BallMatrix(BigFloat, A)
end

# ============================================================================
# Explicit BigFloat constructors
# ============================================================================

"""
    BallArithmetic.BallMatrix(::Type{BigFloat}, A::AbstractMatrix{ArbNumerics.ArbReal})

Convert a matrix of `ArbReal` into a `BallMatrix{BigFloat}` without
Float64 truncation.  Requires `precision(BigFloat) ≥` Arb precision for
exact midpoint conversion.
"""
function BallArithmetic.BallMatrix(::Type{BigFloat}, A::AbstractMatrix{ArbNumerics.ArbReal{P}}) where {P}
    dims = size(A)
    mid = Array{BigFloat}(undef, dims)
    rad = Array{BigFloat}(undef, dims)

    for idx in eachindex(A)
        mid[idx], rad[idx] = _arb_to_bigfloat_with_radius(A[idx])
    end

    return BallMatrix(mid, rad)
end

"""
    BallArithmetic.BallMatrix(::Type{BigFloat}, A::AbstractMatrix{ArbNumerics.ArbComplex})

Convert a matrix of `ArbComplex` into a `BallMatrix{Complex{BigFloat}}`
without Float64 truncation.  Requires `precision(BigFloat) ≥` Arb
precision for exact midpoint conversion.
"""
function BallArithmetic.BallMatrix(::Type{BigFloat}, A::AbstractMatrix{ArbNumerics.ArbComplex{P}}) where {P}
    dims = size(A)
    mid = Array{Complex{BigFloat}}(undef, dims)
    rad = Array{BigFloat}(undef, dims)

    for idx in eachindex(A)
        mid[idx], rad[idx] = _arbcomplex_to_bigfloat_with_radius(A[idx])
    end

    return BallMatrix(mid, rad)
end

# ============================================================================
# Explicit Float64 constructors — truncation with rigorous error tracking
# ============================================================================

"""
    BallArithmetic.BallMatrix(::Type{Float64}, A::AbstractMatrix{ArbNumerics.ArbReal})

Convert a matrix of `ArbReal` numbers into a `BallMatrix{Float64}`.  The
midpoint matrix stores the `Float64` midpoints of the Arb entries, and the
radius matrix contains an upper bound for the combined Arb and rounding
uncertainty.  This is useful when downstream code requires Float64 matrices
and the precision loss is acceptable.
"""
function BallArithmetic.BallMatrix(::Type{Float64}, A::AbstractMatrix{ArbNumerics.ArbReal{P}}) where {P}
    dims = size(A)
    mid = Array{Float64}(undef, dims)
    rad = Array{Float64}(undef, dims)

    for idx in eachindex(A)
        mid[idx], rad[idx] = _arb_to_float_with_radius(A[idx])
    end

    return BallMatrix(mid, rad)
end

"""
    BallArithmetic.BallMatrix(::Type{Float64}, A::AbstractMatrix{ArbNumerics.ArbComplex})

Convert a matrix of `ArbComplex` numbers into a `BallMatrix{ComplexF64}`.
The midpoint matrix stores the `ComplexF64` midpoints of the Arb entries,
and the radius matrix bounds the uncertainty stemming from both the Arb
radius and the rounding of the midpoint to `ComplexF64`.
"""
function BallArithmetic.BallMatrix(::Type{Float64}, A::AbstractMatrix{ArbNumerics.ArbComplex{P}}) where {P}
    dims = size(A)
    mid = Array{ComplexF64}(undef, dims)
    rad = Array{Float64}(undef, dims)

    for idx in eachindex(A)
        mid[idx], rad[idx] = _arbcomplex_to_float_with_radius(A[idx])
    end

    return BallMatrix(mid, rad)
end

end
