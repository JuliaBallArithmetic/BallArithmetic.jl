import LinearAlgebra

"""
    _upper_bound_norm(center, radius, p)

Internal helper that computes a rigorous upper bound on the p-norm of an
array represented by separate midpoint `center` and radius `radius`
arrays. The function evaluates the norm of both parts with rounding
directed upward and adds the results to preserve an enclosure.
"""
function _upper_bound_norm(center, radius, p::Real = 2)
    T = eltype(radius)
    norm = setrounding(T, RoundUp) do
        return LinearAlgebra.norm(center, p) + LinearAlgebra.norm(radius, p)
    end
    return norm
end

"""
    upper_bound_norm(A::BallMatrix, p::Real = 2)

Compute a rigorous upper bound for the p-norm of a `BallMatrix` by
computing the norm of the midpoint and radius arrays with upward rounding.
The default `p = 2` corresponds to the Frobenius norm.
"""
function upper_bound_norm(A::BallMatrix, p::Real = 2)
    return _upper_bound_norm(A.c, A.r, p)
end

"""
    upper_bound_norm(v::BallVector, p::Real = 2)

Compute a rigorous upper bound for the p-norm of a `BallVector`.
"""
function upper_bound_norm(v::BallVector, p::Real = 2)
    return _upper_bound_norm(v.c, v.r, p)
end
