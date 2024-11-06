using LinearAlgebra

function krawczyk(A::BallMatrix{T},
        v::BallVector{T}) where {T <: Union{AbstractFloat}}
    C = inv(A.c)
    v0 = C * (v.c)
end
