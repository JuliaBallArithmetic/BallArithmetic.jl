x = Ball(1.0)
@test typeof(x + 1) == Ball{Float64, Float64}
@test typeof(x + 1.0) == Ball{Float64, Float64}
@test typeof(x + im) == Ball{Float64, Complex{Float64}}
@test typeof(x + 1.0 * im) == Ball{Float64, Complex{Float64}}

x = Ball(1.0 + im)
@test typeof(x + 1) == Ball{Float64, Complex{Float64}}
@test typeof(x + 1.0) == Ball{Float64, Complex{Float64}}
@test typeof(x + im) == Ball{Float64, Complex{Float64}}
@test typeof(x + 1.0 * im) == Ball{Float64, Complex{Float64}}
@test typeof(x + Ball(1.0)) == Ball{Float64, Complex{Float64}}
