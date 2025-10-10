using LinearAlgebra

"""
    _triangularize(A::BallMatrix)

Return an upper-triangular `BallMatrix` enclosure by wrapping the stored
midpoint and radius matrices in `UpperTriangular`. This helper is used by
algorithms that operate on triangular factors while keeping the original
enclosure guarantees.
"""
function _triangularize(A::BallMatrix)
    return BallMatrix(UpperTriangular(A.c), UpperTriangular(A.r))
end
