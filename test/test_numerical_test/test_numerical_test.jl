@testset "Test Numerical Test" begin
    A = BallArithmetic.NumericalTest._test_matrix(4)
    @test A == [1.0 0 0 2^(-53);
                0 1.0 0 2^(-53);
                0 0 1.0 2^(-53);
                0 0 0 2^(-53)]

    # we test the singlethread version, so that CI points
    # out if setting rounding modes is broken
    @test BallArithmetic.NumericalTest.rounding_test(1, 2) == true
end

@testset "array operations under a rounding mode against the directed operations" begin
    using LinearAlgebra
    NT = BallArithmetic.NumericalTest
    # the check that runs when the package is loaded
    for T in (Float64, Float32), n in (1, 7, 48, 200)
        @test isempty(NT.array_rounding_test(T; n))
    end
    @test NT.ARRAY_ROUNDING_OK[]
    @test isempty(NT.check_array_rounding())

    # the fallback gives the same radii as the rounding-mode form
    A = BallMatrix(randn(9, 9), rand(9, 9) .* 1e-9)
    B = BallMatrix(randn(9, 9), rand(9, 9) .* 1e-9)
    H = BallMatrix(Hermitian(randn(9, 9)), Symmetric(rand(9, 9) .* 1e-9))
    X, P, Q = rand(9, 9), rand(9, 9), rand(9, 9)
    ref = (rad(A + B), rad(A - B), rad(A + mid(B)), rad(H + H), BallArithmetic._sum_product_up(X, P, Q))
    # the products: every routine, ball and point factors, real and complex
    Ac = BallMatrix(randn(ComplexF64, 9, 9), rand(9, 9) .* 1e-9)
    Bc = BallMatrix(randn(ComplexF64, 9, 9), rand(9, 9) .* 1e-9)
    A32 = BallMatrix(randn(Float32, 9, 9), rand(Float32, 9, 9) .* 1.0f-5)
    U = BallMatrix(UpperTriangular(randn(9, 9)), UpperTriangular(rand(9, 9) .* 1e-9))
    products = (() -> BallArithmetic.MMul4(A, B), () -> BallArithmetic.MMul4(mid(A), B),
        () -> BallArithmetic.MMul4(A, mid(B)), () -> BallArithmetic.MMul4(mid(A), mid(B)),
        () -> BallArithmetic.MMul3(A, B), () -> BallArithmetic.MMul3(mid(A), B),
        () -> BallArithmetic.MMul3(A, mid(B)), () -> BallArithmetic.MMul3(mid(A), mid(B)),
        () -> BallArithmetic.MMul5(A, B), () -> oishi_MMul(mid(Ac), mid(Bc)),
        () -> A * B, () -> Ac * Bc, () -> Ac * B, () -> A * mid(Bc), () -> A32 * A32,
        () -> U * U, () -> U * A)
    refprod = map(f -> f(), products)
    NT.ARRAY_ROUNDING_OK[] = false
    try
        for (f, C) in zip(products, refprod)
            D = f()
            @test mid(D) == mid(C) && rad(D) == rad(C)
        end
        @test rad(A + B) == ref[1]
        @test rad(A - B) == ref[2]
        @test rad(A + mid(B)) == ref[3]
        @test rad(H + H) isa Symmetric && Matrix(rad(H + H)) == Matrix(ref[4])
        @test BallArithmetic._sum_product_up(X, P, Q) == ref[5]
    finally
        NT.ARRAY_ROUNDING_OK[] = true
    end
end
