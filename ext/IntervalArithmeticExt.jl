module IntervalArithmeticExt

using BallArithmetic
import IntervalArithmetic
import BallArithmetic: add_up, mul_up, sqrt_up, add_down, sub_down

# Radius of a disk containing the rectangle [c_re ± r_re] × [c_im ± r_im]:
# √(r_re² + r_im²), rounded up. Uses the directed-rounding helpers rather than
# `setrounding` so the result is correct for any float type, BigFloat included.
function _disk_radius(r_re::T, i_re::T) where {T}
    sqrt_up(add_up(mul_up(r_re, r_re), mul_up(i_re, i_re)))
end

"""
Convert an Interval from IntervalArithmetic to a Ball
"""
function BallArithmetic.Ball(x::IntervalArithmetic.Interval{T}) where {T}
    c, r = IntervalArithmetic.mid(x), IntervalArithmetic.radius(x)
    return Ball(c, r)
end

"""
Convert an Complex{Interval} from IntervalArithmetic to a Ball
"""
function BallArithmetic.Ball(x::Complex{IntervalArithmetic.Interval{T}}) where {T}
    r_mid, r_rad = IntervalArithmetic.mid(real(x)), IntervalArithmetic.radius(real(x))
    i_mid, i_rad = IntervalArithmetic.mid(imag(x)), IntervalArithmetic.radius(imag(x))
    return Ball(r_mid + im * i_mid, _disk_radius(r_rad, i_rad))
end

"""
Construct a BallMatrix from a matrix of Interval{T}
"""
function BallArithmetic.BallMatrix(
        x::AbstractMatrix{IntervalArithmetic.Interval{T}}) where {T}
    C, R = IntervalArithmetic.mid.(x), IntervalArithmetic.radius.(x)
    return BallMatrix(C, R)
end

"""
Construct a BallMatrix from a matrix of Complex{Interval{T}}, remark
that the radius may be bigger, to ensure mathematical consistency, i.e.,
we need to find a ball that contains a rectangle
"""
function BallArithmetic.BallMatrix(
        x::AbstractMatrix{Complex{IntervalArithmetic.Interval{T}}}) where {T}
    R_mid, R_rad = IntervalArithmetic.mid.(real.(x)), IntervalArithmetic.radius.(real.(x))
    I_mid, I_rad = IntervalArithmetic.mid.(imag.(x)), IntervalArithmetic.radius.(imag.(x))
    return BallMatrix(R_mid + im * I_mid, _disk_radius.(R_rad, I_rad))
end

"""
Construct a BallVector from a vector of Interval{T}

The generic `BallVector(::AbstractVector)` in `src/types/vector.jl` goes through
`rad`, which has no method for a vector of intervals, so without these the
vector case failed with a `MethodError` while the matrix case worked.
"""
function BallArithmetic.BallVector(
        x::AbstractVector{IntervalArithmetic.Interval{T}}) where {T}
    C, R = IntervalArithmetic.mid.(x), IntervalArithmetic.radius.(x)
    return BallVector(C, R)
end

"""
Construct a BallVector from a vector of Complex{Interval{T}}; as for the matrix
case the radius is inflated to a disk containing the rectangle.
"""
function BallArithmetic.BallVector(
        x::AbstractVector{Complex{IntervalArithmetic.Interval{T}}}) where {T}
    R_mid, R_rad = IntervalArithmetic.mid.(real.(x)), IntervalArithmetic.radius.(real.(x))
    I_mid, I_rad = IntervalArithmetic.mid.(imag.(x)), IntervalArithmetic.radius.(imag.(x))
    return BallVector(R_mid + im * I_mid, _disk_radius.(R_rad, I_rad))
end

function IntervalArithmetic.interval(x::Ball{T, T}) where {T}
    # RoundDown on the lower bound / RoundUp on the upper is what keeps this a
    # valid enclosure.
    return IntervalArithmetic.interval(sub_down(x.c, x.r), add_up(x.c, x.r))
end

end
