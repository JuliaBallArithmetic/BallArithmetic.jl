module IntervalArithmeticExt

using IntervalArithmetic, BallArithmetic

function BallArithmetic.Ball(x::Interval{Float64})
    mid, rad = IntervalArithmetic.midpointradius(x)
    return Ball(mid, rad)
end

end
