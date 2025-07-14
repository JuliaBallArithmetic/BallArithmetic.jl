using LinearAlgebra

#TODO: Vectorize?

function backward_substitution(A::BallMatrix{T, NT}, v::BallVector{T, NS}) where {T, NT, NS}
    @assert istriu(A.c) && istriu(A.r)

    m, k = size(A)
    @assert m == k

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
    @assert istriu(A.c) && istriu(A.r)

    m, k = size(A)
    @assert m == k

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
