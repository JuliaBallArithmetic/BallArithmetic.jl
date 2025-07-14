import LinearAlgebra

function _upper_bound_norm(center, radius, p::Real = 2)
    T = eltype(radius)
    norm = setrounding(T, RoundUp) do
        return LinearAlgebra.norm(center, p) + LinearAlgebra.norm(radius, p)
    end
    return norm
end

"""
    upper_bound_norm(A::BallMatrix, p::Real = 2)

Compute a rigorous upper bound for the Frobenius p-norm of a BallMatrix
"""
function upper_bound_norm(A::BallMatrix, p::Real = 2)
    return _upper_bound_norm(A.c, A.r, p)
end

"""
    upper_bound_norm(v::BallVector, p::Real = 2)

Compute a rigorous upper bound for the p-norm of a BallVector
"""
function upper_bound_norm(v::BallVector, p::Real = 2)
    return _upper_bound_norm(v.c, v.r, p)
end
