import LinearAlgebra

function upper_bound_norm(center, radius, p::Real = 2)
    T = eltype(center)
    norm = setrounding(T, RoundUp) do
        return LinearAlgebra.norm(center, p) + LinearAlgebra.norm(radius, p)
    end
    return norm
end


function upper_bound_norm(A::BallMatrix{T}, p::Real = 2) where {T}
    return upper_bound_norm(A.c, A.r)
end