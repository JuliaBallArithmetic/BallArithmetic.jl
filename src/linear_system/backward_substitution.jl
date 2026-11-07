using LinearAlgebra

#TODO: Vectorize?

"""
    backward_substitution(A::BallMatrix, v::BallVector)
    backward_substitution(A::BallMatrix, B::BallMatrix)

Enclose the solution of `A x = v`, or of `A X = B` column by column, for an upper triangular
`A`: the rows are eliminated from the last to the first, each step a ball subtraction and one
ball division by the diagonal entry, so the result encloses the solution of every real or
complex system whose matrix and right hand side lie in `A` and in `v`.

Both the midpoints and the radii of `A` are required to be upper triangular, since a nonzero
radius below the diagonal would describe a matrix that is not; an `ArgumentError` is raised
otherwise, and a `DimensionMismatch` when `A` is not square. A diagonal ball containing zero
raises an `ArgumentError` from the division rather than returning an unbounded enclosure, so a
matrix whose enclosure admits a singular member is rejected and not reported as solved.
"""
function backward_substitution(A::BallMatrix{T, NT}, v::BallVector{T, NS}) where {T, NT, NS}
    istriu(A.c) && istriu(A.r) ||
        throw(ArgumentError("the matrix must be upper triangular, midpoints and radii"))

    m, k = size(A)
    m == k || throw(DimensionMismatch("the matrix must be square, got $m × $k"))

    RT = promote_type(NT, NS)
    sol = Vector{Ball{T, RT}}(undef, m)

    for i in m:-1:1
        rhs = v[i]
        for j in (i + 1):m
            rhs -= A[i, j] * sol[j]
        end
        sol[i] = rhs / A[i, i]
    end
    return sol
end

function backward_substitution(A::BallMatrix{T, NT}, B::BallMatrix{T, NS}) where {T, NT, NS}
    istriu(A.c) && istriu(A.r) ||
        throw(ArgumentError("the matrix must be upper triangular, midpoints and radii"))

    m, k = size(A)
    m == k || throw(DimensionMismatch("the matrix must be square, got $m × $k"))

    RT = promote_type(NT, NS)
    sol = fill(Ball{T, RT}(0.0, 0.0), size(B))

    for i in m:-1:1
        sol[i, :] = B[i, :]
        for j in m:-1:(i + 1)
            sol[i, :] -= A[i, j] * sol[j, :]
        end
        sol[i, :] /= Ball(A[i, i])
    end
    return BallMatrix(mid.(sol), rad.(sol))
end

"""
    forward_substitution(A::BallMatrix, v::BallVector)
    forward_substitution(A::BallMatrix, B::BallMatrix)

The same as [`backward_substitution`](@ref) for a lower triangular `A`, eliminating the rows
from the first to the last; midpoints and radii are both required to be lower triangular.
"""
function forward_substitution(A::BallMatrix{T, NT}, v::BallVector{T, NS}) where {T, NT, NS}
    istril(A.c) && istril(A.r) ||
        throw(ArgumentError("the matrix must be lower triangular, midpoints and radii"))

    m, k = size(A)
    m == k || throw(DimensionMismatch("the matrix must be square, got $m × $k"))

    RT = promote_type(NT, NS)
    sol = Vector{Ball{T, RT}}(undef, m)

    for i in 1:m
        rhs = v[i]
        for j in 1:(i - 1)
            rhs -= A[i, j] * sol[j]
        end
        sol[i] = rhs / A[i, i]
    end
    return sol
end

function forward_substitution(A::BallMatrix{T, NT}, B::BallMatrix{T, NS}) where {T, NT, NS}
    istril(A.c) && istril(A.r) ||
        throw(ArgumentError("the matrix must be lower triangular, midpoints and radii"))

    m, k = size(A)
    m == k || throw(DimensionMismatch("the matrix must be square, got $m × $k"))

    RT = promote_type(NT, NS)
    sol = fill(Ball{T, RT}(0.0, 0.0), size(B))

    for i in 1:m
        sol[i, :] = B[i, :]
        for j in 1:(i - 1)
            sol[i, :] -= A[i, j] * sol[j, :]
        end
        sol[i, :] /= Ball(A[i, i])
    end
    return BallMatrix(mid.(sol), rad.(sol))
end
