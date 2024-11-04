module NumericalTest

export rounding_test

function _test_matrix(k)
    A = zeros(Float64, (k, k))
    A[:, end] = fill(2^(-53), k)
    for i in 1:(k - 1)
        A[i, i] = 1.0
    end
    return A
end

using LinearAlgebra
"""
    rounding_test(n, k)

Let `u=fill(2^(-53), k-1)` and let A be the matrix
[I u;
0 2^(-53)]

This test checks the result of A*A' in different rounding modes,
running BLAS on `n` threads
"""
function rounding_test(n, k)
    BLAS.set_num_threads(n)
    A = _test_matrix(k)

    test_up = false
    B = setrounding(Float64, RoundUp) do
        BLAS.gemm('N', 'T', 1.0, A, A)
    end
    test_up = all([B[i, i] == nextfloat(1.0) for i in 1:(k - 1)])

    test_down = false
    B = setrounding(Float64, RoundDown) do
        BLAS.gemm('N', 'T', 1.0, A, A)
    end
    test_down = all([B[i, i] == 1.0 for i in 1:(k - 1)])

    return test_up && test_down
end

end
