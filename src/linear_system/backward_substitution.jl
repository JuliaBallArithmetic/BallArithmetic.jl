using LinearAlgebra

function backward_substitution(A::BallMatrix{T, NT}, v::BallVector{T, NS}) where {T, NT, NS}
    @assert istriu(A.c) && istriu(A.r)

    m, k = size(A)
    @assert m == k

    sol = fill(Ball{T, promote_rule(NT, NS)}(0.0, 0.0), m)

    for i in m:-1:1
        rhs = v[i]
        if i < m
            rhs -= dot(A[i, (i + 1):end], sol[(i + 1):end])
        end
        sol[i] = rhs / A[i, i]
    end
    return sol
end

function backward_substitution(A::BallMatrix{T, NT}, B::BallMatrix{T, NS}) where {T, NT, NS}
    @assert istriu(A.c) && istriu(A.r)

    m, k = size(A)
    @assert m == k

    sol = fill(Ball{T, promote_rule(NT, NS)}(0.0, 0.0), size(B))

    for i in m:-1:1
        sol[i, :] = B[i, :]
        for j in m:-1:(i + 1)
            sol[i, :] -= A[i, j] * sol[j, :]
        end
        sol[i, :] /= Ball(A[i, i])
    end
    return BallMatrix(mid.(sol), rad.(sol))
end
