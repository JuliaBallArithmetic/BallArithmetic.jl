#############################
# MACHINE EPSILON FUNCTIONS #
#############################

"""
    machine_epsilon(::Type{T}) where T <: AbstractFloat

Return the machine epsilon (unit roundoff) for floating-point type `T`.
This is the maximum relative error introduced by a single rounded floating-point
operation and is used throughout the package when inflating radii.

For Float64, this equals 2^-52 ≈ 2.22e-16.
For Float32, this equals 2^-23 ≈ 1.19e-7.
For BigFloat, this depends on the current precision setting.

# Examples
```julia
machine_epsilon(Float64)  # 2.220446049250313e-16
machine_epsilon(Float32)  # 1.1920929f-7
setprecision(256); machine_epsilon(BigFloat)  # ≈ 8.6e-78
```
"""
machine_epsilon(::Type{Float64}) = 2.0^-52
machine_epsilon(::Type{Float32}) = Float32(2.0^-23)
machine_epsilon(::Type{BigFloat}) = BigFloat(2)^(-precision(BigFloat))
machine_epsilon(::Type{Complex{T}}) where T = machine_epsilon(T)

"""
    subnormal_min(::Type{T}) where T <: AbstractFloat

Return the smallest positive subnormal value for floating-point type `T`.
Adding this to a computed radius guarantees that results remain strictly
positive even in edge cases where an operation underflows to zero.

For Float64, this equals 2^-1074 ≈ 4.94e-324.
For Float32, this equals 2^-149 ≈ 1.40e-45.
For BigFloat, this depends on the current precision and exponent settings.

# Examples
```julia
subnormal_min(Float64)  # 4.9406564584124654e-324
subnormal_min(Float32)  # 1.0f-45
```
"""
subnormal_min(::Type{Float64}) = 2.0^-1074
subnormal_min(::Type{Float32}) = Float32(2.0^-149)
# For BigFloat, use the smallest representable positive number
function subnormal_min(::Type{BigFloat})
    # BigFloat doesn't have subnormals in the IEEE sense, but we need
    # a small positive value. Use 2^(emin - p + 1) where emin is the
    # minimum exponent and p is the precision.
    emin = BigFloat(2)^(Base.MPFR.get_emin() + 1)
    return emin
end
subnormal_min(::Type{Complex{T}}) where T = subnormal_min(T)

"""
    ϵp

Unit roundoff for `Float64` values. This is a convenience constant
equal to `machine_epsilon(Float64)` for backwards compatibility.
For type-generic code, prefer `machine_epsilon(T)`.
"""
const ϵp = machine_epsilon(Float64)

"""
    η

Smallest positive subnormal `Float64` value. This is a convenience constant
equal to `subnormal_min(Float64)` for backwards compatibility.
For type-generic code, prefer `subnormal_min(T)`.
"""
const η = subnormal_min(Float64)


################################
# BIGFLOAT ROUNDING OPERATIONS #
################################

"""
    add_up(x::BigFloat, y::BigFloat)

Add two BigFloat values with rounding toward +∞.
"""
function add_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x + y
    end
end

"""
    add_down(x::BigFloat, y::BigFloat)

Add two BigFloat values with rounding toward -∞.
"""
function add_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x + y
    end
end

"""
    sub_up(x::BigFloat, y::BigFloat)

Subtract two BigFloat values with rounding toward +∞.
"""
function sub_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x - y
    end
end

"""
    sub_down(x::BigFloat, y::BigFloat)

Subtract two BigFloat values with rounding toward -∞.
"""
function sub_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x - y
    end
end

"""
    mul_up(x::BigFloat, y::BigFloat)

Multiply two BigFloat values with rounding toward +∞.
"""
function mul_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x * y
    end
end

"""
    mul_down(x::BigFloat, y::BigFloat)

Multiply two BigFloat values with rounding toward -∞.
"""
function mul_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x * y
    end
end

"""
    div_up(x::BigFloat, y::BigFloat)

Divide two BigFloat values with rounding toward +∞.
"""
function div_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x / y
    end
end

"""
    div_down(x::BigFloat, y::BigFloat)

Divide two BigFloat values with rounding toward -∞.
"""
function div_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x / y
    end
end

"""
    sqrt_up(x::BigFloat)

Compute the square root of a BigFloat value with rounding toward +∞.
"""
function sqrt_up(x::BigFloat)
    setrounding(BigFloat, RoundUp) do
        sqrt(x)
    end
end

"""
    sqrt_down(x::BigFloat)

Compute the square root of a BigFloat value with rounding toward -∞.
"""
function sqrt_down(x::BigFloat)
    setrounding(BigFloat, RoundDown) do
        sqrt(x)
    end
end

"""
    abs_up(x::BigFloat)

Compute the absolute value of a BigFloat with guaranteed upper bound.
For positive x, this is just x. For negative x, this rounds -x upward.
"""
function abs_up(x::BigFloat)
    if x >= 0
        return x
    else
        setrounding(BigFloat, RoundUp) do
            -x
        end
    end
end

"""
    abs_down(x::BigFloat)

Compute the absolute value of a BigFloat with guaranteed lower bound.
"""
function abs_down(x::BigFloat)
    if x >= 0
        return x
    else
        setrounding(BigFloat, RoundDown) do
            -x
        end
    end
end


#######################################
# COMPLEX BIGFLOAT ROUNDING HELPERS   #
#######################################

"""
    add_up(x::Complex{BigFloat}, y::Complex{BigFloat})

Add two Complex{BigFloat} values with outward rounding for real and imaginary parts.
"""
function add_up(x::Complex{BigFloat}, y::Complex{BigFloat})
    Complex(add_up(real(x), real(y)), add_up(imag(x), imag(y)))
end

"""
    sub_up(x::Complex{BigFloat}, y::Complex{BigFloat})

Subtract two Complex{BigFloat} values with outward rounding.
"""
function sub_up(x::Complex{BigFloat}, y::Complex{BigFloat})
    Complex(sub_up(real(x), real(y)), sub_up(imag(x), imag(y)))
end

"""
    mul_up(x::Complex{BigFloat}, y::Complex{BigFloat})

Multiply two Complex{BigFloat} values with outward rounding.
Uses the formula (a+bi)(c+di) = (ac-bd) + (ad+bc)i with careful rounding.
"""
function mul_up(x::Complex{BigFloat}, y::Complex{BigFloat})
    a, b = real(x), imag(x)
    c, d = real(y), imag(y)
    # Real part: ac - bd (need upper bound)
    # For upper bound of ac - bd, we need upper(ac) - lower(bd) if ac > bd, etc.
    # Conservative approach: use interval arithmetic
    ac_up = mul_up(a, c)
    bd_down = mul_down(b, d)
    real_part = sub_up(ac_up, bd_down)

    # Imaginary part: ad + bc (need upper bound)
    ad_up = mul_up(a, d)
    bc_up = mul_up(b, c)
    imag_part = add_up(ad_up, bc_up)

    Complex(real_part, imag_part)
end


##########################
# MACROS FOR ROUNDING    #
##########################

const op_up = Dict(:+ => :add_up, :- => :sub_up, :* => :mul_up, :/ => :div_up)

"""
    @up expr

Rewrite arithmetic in `expr` so that every operation is evaluated with
outward rounding toward `+∞`. The macro replaces the standard `+`, `-`,
`*`, and `/` operators with the corresponding helpers from
`RoundingEmulator`, making it convenient to derive guaranteed upper
bounds for composite expressions.

Note: This macro works with Float64 operations from RoundingEmulator.
For BigFloat, use the explicit `add_up`, `mul_up`, etc. functions.
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

Note: This macro works with Float64 operations from RoundingEmulator.
For BigFloat, use the explicit `add_down`, `mul_down`, etc. functions.
"""
macro down(ex)
    esc(MacroTools.postwalk(x -> get(op_down, x, x), ex))
end
