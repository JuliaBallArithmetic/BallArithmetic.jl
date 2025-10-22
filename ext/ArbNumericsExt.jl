module ArbNumericsExt

using BallArithmetic
import ArbNumerics
using SetRounding: setrounding

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
    round_err = abs(mid - ArbNumerics.ArbReal(mid_f))

    rad_f = setrounding(Float64, RoundUp) do
        total = rad + round_err
        Float64(ArbNumerics.midpoint(total)) + Float64(ArbNumerics.radius(total))
    end

    return mid_f, rad_f
end

function _arbcomplex_to_float_with_radius(x::ArbNumerics.ArbComplex{P}) where {P}
    real_part = real(x)
    imag_part = imag(x)

    mid_real = ArbNumerics.midpoint(real_part)
    mid_imag = ArbNumerics.midpoint(imag_part)

    mid = ComplexF64(Float64(mid_real), Float64(mid_imag))

    err_real = abs(mid_real - ArbNumerics.ArbReal(real(mid)))
    err_imag = abs(mid_imag - ArbNumerics.ArbReal(imag(mid)))

    rad = setrounding(Float64, RoundUp) do
        total_real = ArbNumerics.radius(real_part) + err_real
        total_imag = ArbNumerics.radius(imag_part) + err_imag
        real_hi = Float64(ArbNumerics.midpoint(total_real)) + Float64(ArbNumerics.radius(total_real))
        imag_hi = Float64(ArbNumerics.midpoint(total_imag)) + Float64(ArbNumerics.radius(total_imag))
        sqrt(real_hi^2 + imag_hi^2)
    end

    return mid, rad
end

"""
    BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbReal})

Convert a matrix of `ArbReal` numbers into a `BallMatrix{Float64}`.  The
midpoint matrix stores the `Float64` midpoints of the Arb entries, and the
radius matrix contains an upper bound for the combined Arb and rounding
uncertainty.
"""
function BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbReal{P}}) where {P}
    dims = size(A)
    mid = Array{Float64}(undef, dims)
    rad = Array{Float64}(undef, dims)

    for idx in eachindex(A)
        mid[idx], rad[idx] = _arb_to_float_with_radius(A[idx])
    end

    return BallMatrix(mid, rad)
end

"""
    BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbComplex})

Convert a matrix of `ArbComplex` numbers into a `BallMatrix{ComplexF64}`.
The midpoint matrix stores the `ComplexF64` midpoints of the Arb entries,
and the radius matrix bounds the uncertainty stemming from both the Arb
radius and the rounding of the midpoint to `ComplexF64`.
"""
function BallArithmetic.BallMatrix(A::AbstractMatrix{ArbNumerics.ArbComplex{P}}) where {P}
    dims = size(A)
    mid = Array{ComplexF64}(undef, dims)
    rad = Array{Float64}(undef, dims)

    for idx in eachindex(A)
        mid[idx], rad[idx] = _arbcomplex_to_float_with_radius(A[idx])
    end

    return BallMatrix(mid, rad)
end

end
