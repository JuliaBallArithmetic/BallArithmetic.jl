"""
    ϵp

Unit roundoff for `Float64` values. The constant represents the maximum
relative error introduced by a single rounded floating-point operation
and is used throughout the package when inflating radii after
computations.
"""
const ϵp = 2.0^-52

"""
    η

Smallest positive subnormal `Float64` value. Adding `η` to a computed
radius guarantees that results remain strictly positive even in edge
cases where an operation underflows to zero.
"""
const η = 2.0^-1074

const op_up = Dict(:+ => :add_up, :- => :sub_up, :* => :mul_up, :/ => :div_up)

"""
    @up expr

Rewrite arithmetic in `expr` so that every operation is evaluated with
outward rounding toward `+∞`. The macro replaces the standard `+`, `-`,
`*`, and `/` operators with the corresponding helpers from
`RoundingEmulator`, making it convenient to derive guaranteed upper
bounds for composite expressions.
"""
macro up(ex)
    esc(MacroTools.postwalk(x -> get(op_up, x, x), ex))
end

const op_down = Dict(:+ => :add_down, :- => :sub_down, :* => :mul_down, :/ => :div_down)

"""
    @down expr

Mirror of [`@up`](@ref) that rewrites arithmetic so that each operation
rounds toward `-∞`. The transformation is useful for computing lower
bounds while sharing the same algebraic expression as the optimistic
estimate.
"""
macro down(ex)
    esc(MacroTools.postwalk(x -> get(op_down, x, x), ex))
end
