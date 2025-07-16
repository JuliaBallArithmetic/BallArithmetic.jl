using LinearAlgebra

"""
    krawczyk(A::BallMatrix{T}, b::BallVector{T}; iter_max = 20) where {T <: AbstractFloat}

Return a `BallVector` enclosing the solution of ``Ax = b`` using the Krawczyk
method.  The algorithm follows the classical interval Krawczyk iteration and
stops once the enclosure is proved to be invariant or the maximum number of
iterations is reached.
"""
function krawczyk(A::BallMatrix{T},
        b::BallVector{T}; iter_max = 20) where {T <: AbstractFloat}
    C = inv(A.c)                    # approximate inverse of the midpoint matrix
    xs = C * b.c                    # approximate solution using midpoints

    Y = I - C * A                   # iteration matrix
    z = C * (b - A * BallVector(xs))

    x = z
    for _ in 1:iter_max
        x_new = z + Y * x
        if all(in.(x_new, x))
            return BallVector(xs) + x_new
        end
        x = x_new
    end
    return BallVector(xs) + x
end
