using LinearAlgebra

function _triangularize(A::BallMatrix)
    return BallMatrix(UpperTriangular(A.c), UpperTriangular(A.r))
end
