using LinearAlgebra

function backward_substitution(A::BallMatrix, v::BallVector)
    @assert istriu(A.c) && istriu(A.r)

    m, k = size(A)
    @assert m == k

    sol = fill(Ball(0.0), m)

    for i in m:-1:1
        sol[i] = v[i]
        for j in m:-1:(i + 1)
            sol[i] -= A[i, j] * sol[j]
        end
        sol[i] /= Ball(A[i, i])
    end
    return sol
end

function backward_substitution(A::BallMatrix, B::BallMatrix)
    @assert istriu(A.c) && istriu(A.r)

    m, k = size(A)
    @assert m == k

    sol = fill(Ball(0.0), size(B))

    for i in m:-1:1
        sol[i, :] = B[i, :]
        for j in m:-1:(i + 1)
            sol[i, :] -= A[i, j] * sol[j, :]
        end
        sol[i, :] /= Ball(A[i, i])
    end
    return BallMatrix(mid.(sol), rad.(sol))
end
