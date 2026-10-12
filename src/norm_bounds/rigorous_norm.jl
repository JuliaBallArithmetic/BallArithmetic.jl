import LinearAlgebra

# An upper bound of the p-norm of every array x with |x_i − c_i| ≤ r_i, for p = 1, 2, Inf: the
# norm of the nonnegative array |c| + r, since |x_i| ≤ |c_i| + r_i, with every operation rounded
# upwards (the modulus of a complex number included, by `abs_up`).
function _upper_bound_norm(center, radius, p::Real = 2)
    T = eltype(radius)
    if p == 2
        s = zero(T)
        @inbounds for i in eachindex(center, radius)
            a = add_up(abs_up(center[i]), radius[i])
            s = add_up(s, mul_up(a, a))
        end
        return sqrt_up(s)::T
    elseif p == 1
        s = zero(T)
        @inbounds for i in eachindex(center, radius)
            s = add_up(s, add_up(abs_up(center[i]), radius[i]))
        end
        return s
    elseif p == Inf
        s = zero(T)
        @inbounds for i in eachindex(center, radius)
            s = max(s, add_up(abs_up(center[i]), radius[i]))
        end
        return s
    end
    throw(ArgumentError("upper_bound_norm: p must be 1, 2 or Inf"))
end

"""
    upper_bound_norm(A::BallMatrix, p::Real = 2)

An upper bound of the entrywise `p`-norm of every matrix in the ball `A`, for `p = 1`, `2` or
`Inf`: the norm of `|mid(A)| + rad(A)`, every operation rounded upwards. `p = 2` is the Frobenius
norm. It is not an operator norm; for those see [`upper_bound_L2_opnorm`](@ref),
[`upper_bound_L1_opnorm`](@ref) and [`upper_bound_L_inf_opnorm`](@ref).
"""
function upper_bound_norm(A::BallMatrix, p::Real = 2)
    return _upper_bound_norm(A.c, A.r, p)
end

"""
    upper_bound_norm(v::BallVector, p::Real = 2)

An upper bound of the `p`-norm of every vector in the ball `v`, for `p = 1`, `2` or `Inf`: the
norm of `|mid(v)| + rad(v)`, every operation rounded upwards.
"""
function upper_bound_norm(v::BallVector, p::Real = 2)
    return _upper_bound_norm(v.c, v.r, p)
end
