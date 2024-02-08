function off_diagonal_abs(A::BallMatrix)
    B = deepcopy(upper_abs(A))
    for i in diagind(B)
        B[i] = 0.0
    end
    return BallMatrix(B)
end

function diagonal_abs_lower_bound(A::BallMatrix{T}) where {T}
    v = setrounding(T, RoundDown) do
        abs.(diag(A.c)) - abs.(diag(A.r))
    end
    return v
end

using LinearAlgebra: diag

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
