module IntervalArithmeticExt

using BallArithmetic
import IntervalArithmetic

"""
Convert an Interval from IntervalArithmetic to a Ball
"""
function BallArithmetic.Ball(x::IntervalArithmetic.Interval{Float64})
    c, r = IntervalArithmetic.mid(x), IntervalArithmetic.radius(x)
    return Ball(c, r)
end

"""
Convert an Complex{Interval} from IntervalArithmetic to a Ball
"""
function BallArithmetic.Ball(x::Complex{IntervalArithmetic.Interval{Float64}})
    r_mid, r_rad = IntervalArithmetic.mid(real(x)), IntervalArithmetic.radius(real(x))
    i_mid, i_rad = IntervalArithmetic.mid(imag(x)), IntervalArithmetic.radius(imag(x))
    rad = setrounding(Float64, RoundUp) do
        return sqrt(r_rad^2 + i_rad^2)
    end
    return Ball(r_mid + im * i_mid, rad)
end

"""
Construct a BallMatrix from a matrix of Interval{Float64}
"""
function BallArithmetic.BallMatrix(x::Matrix{IntervalArithmetic.Interval{Float64}})
    C, R = IntervalArithmetic.mid.(x), IntervalArithmetic.radius.(x)
    return BallMatrix(C, R)
end

#TODO: add 

"""
Construct a BallMatrix from a matrix of Complex{Interval{Float64}}, remark
that the radius may be bigger, to ensure mathematical consistency, i.e.,
we need to find a ball that contains a rectangle
"""
function BallArithmetic.BallMatrix(x::Matrix{Complex{IntervalArithmetic.Interval{Float64}}})
    R_mid, R_rad = IntervalArithmetic.mid.(real.(x)), IntervalArithmetic.radius.(real.(x))
    I_mid, I_rad = IntervalArithmetic.mid.(imag.(x)), IntervalArithmetic.radius.(imag.(x))
    Rad = setrounding(Float64, RoundUp) do
        return sqrt.(R_rad .^ 2 + I_rad .^ 2)
    end
    return BallMatrix(R_mid + im * I_mid, Rad)
end

function IntervalArithmetic.interval(x::Ball{Float64, Float64})
    up = setrounding(Float64, RoundUp) do
        return x.c + x.r
    end
    # CRITICAL FIX: Must use RoundDown for lower bound to ensure valid enclosure
    down = setrounding(Float64, RoundDown) do
        return x.c - x.r
    end
    return IntervalArithmetic.interval(down, up)
end

end
