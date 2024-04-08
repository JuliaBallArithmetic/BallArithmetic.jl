"""
Returns a matrix containing the off-diagonal elements
"""
function off_diagonal_abs(A::BallMatrix)
    B = deepcopy(upper_abs(A))
    for i in diagind(B)
        B[i] = 0.0
    end
    return BallMatrix(B)
end

"""
Computes a vector containing lower bounds for the diagonal elements of |A|
"""
function diagonal_abs_lower_bound(A::BallMatrix{T}) where {T}
    v = setrounding(T, RoundDown) do
        abs.(diag(A.c)) - abs.(diag(A.r))
    end
    return v
end

using LinearAlgebra: diag

"""
Rigorous computer assisted proof of the fact that a matrix is an
[M-matrix](https://en.wikipedia.org/wiki/M-matrix)
"""
function is_M_matrix(A::BallMatrix)
    B = off_diagonal_abs(A)
    v = diagonal_abs_lower_bound(A)

    if all(v .> 0.0) && iszero(B.c)
        return true
    end

    ρ = collatz_upper_bound(BallMatrix(B))
    if all(v .> ρ)
        return true
    end
    return false
end
