module BallArithmetic

using RoundingEmulator, MacroTools, SetRounding

export Ball, BallMatrix, ±

include("rounding.jl")
include("ball.jl")
include("matrix.jl")

end
