import LinearAlgebra

function upper_bound_norm(center, radius, p::Real = 2)
    T = eltype(center)
    norm = setrounding(T, RoundUp) do
        return LinearAlgebra.norm(center, p) + LinearAlgebra.norm(radius, p)
    end
    return norm
end

function upper_bound_norm(A::BallMatrix, p::Real = 2)
    return upper_bound_norm(A.c, A.r, p)
end

function upper_bound_norm(v::BallVector, p::Real = 2)
    return upper_bound_norm(v.c, v.r, p)
end
