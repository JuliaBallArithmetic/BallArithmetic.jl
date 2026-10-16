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
    Dg = BallMatrix(Diagonal(randn(9)), Diagonal(rand(9) .* 1e-9))
    products = (() -> BallArithmetic.MMul4(A, B), () -> BallArithmetic.MMul4(mid(A), B),
        () -> BallArithmetic.MMul4(A, mid(B)), () -> BallArithmetic.MMul4(mid(A), mid(B)),
        () -> BallArithmetic.MMul3(A, B), () -> BallArithmetic.MMul3(mid(A), B),
        () -> BallArithmetic.MMul3(A, mid(B)), () -> BallArithmetic.MMul3(mid(A), mid(B)),
        () -> BallArithmetic.MMul5(A, B), () -> oishi_MMul(mid(Ac), mid(Bc)),
        () -> A * B, () -> Ac * Bc, () -> Ac * B, () -> A * mid(Bc), () -> A32 * A32,
        () -> U * U, () -> U * A, () -> Dg * A, () -> A * Dg, () -> Dg * Dg)
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

    # where the BLAS does not follow the rounding mode the products are bounded without it
    @test NT.BLAS_ROUNDING_OK[]
    refnorm = (upper_bound_L2_opnorm(A), BallArithmetic.collatz_upper_bound(A))
    NT.BLAS_ROUNDING_OK[] = false
    try
        setprecision(BigFloat, 256) do
            # bounds of a point product against the product in BigFloat
            for (P1, Q1) in ((P, Q), (Float32.(P), Float32.(Q)), (P, Q[:, 1]))
                exact = big.(P1) * big.(Q1)
                down, up = BallArithmetic._product_down_up_prodk(P1, Q1)
                @test all(down .<= exact .<= up)
                @test all(BallArithmetic._product_down(P1, Q1) .<= exact)
                @test all(exact .<= BallArithmetic._product_up(P1, Q1))
                @test maximum(up .- down) <= 64 * eps(eltype(P1)) * maximum(abs, exact)
            end
            @test all(BallArithmetic._sum_product_up(X, P, Q) .>= big.(X) .+ big.(P) * big.(Q))
            # every product routine: the result contains the product of members of the factors
            # (for a point factor the member is the factor)
            sgn(F) = F isa BallMatrix ?
                     Complex{BigFloat}.(mid(F)) .+ big.(rad(F)) .* 0.999 .* rand((-1, 1), size(F)...) :
                     Complex{BigFloat}.(F)
            cases = ((BallArithmetic.MMul4, A, B), (BallArithmetic.MMul4, mid(A), B),
                (BallArithmetic.MMul4, A, mid(B)), (BallArithmetic.MMul4, mid(A), mid(B)),
                (BallArithmetic.MMul3, A, B), (BallArithmetic.MMul5, A, B),
                (oishi_MMul, mid(Ac), mid(Bc)), (*, A, B), (*, Ac, Bc), (*, Ac, B),
                (*, A, mid(Bc)), (*, A32, A32), (*, U, U), (*, U, A))
            for (f, F, G) in cases
                D = f(F, G)
                outside = 0
                for _ in 1:4
                    Z = sgn(F) * sgn(G)
                    outside += count(abs.(Z .- Complex{BigFloat}.(Matrix(mid(D)))) .>
                                     big.(Matrix(rad(D))))
                end
                @test outside == 0
            end
        end
        # the norm bounds that take a product upward
        @test upper_bound_L2_opnorm(A) >= opnorm(mid(A))
        @test upper_bound_L2_opnorm(A) <= 1.01 * refnorm[1]
        @test BallArithmetic.collatz_upper_bound(A) <= 1.01 * refnorm[2]
    finally
        NT.BLAS_ROUNDING_OK[] = true
    end
end

@testset "products with a Diagonal ball factor" begin
    using LinearAlgebra
    n = 9
    Db = BallMatrix(Diagonal(randn(n)), Diagonal(rand(n) .* 1e-9))
    Dd = BallMatrix(Matrix(mid(Db)), Matrix(rad(Db)))
    A = BallMatrix(randn(n, n), rand(n, n) .* 1e-9)
    # the same ball as the dense storage of the factor (the dense product adds exact zeros)
    for (P, Q) in ((Db * A, Dd * A), (A * Db, A * Dd), (Db * Db, Dd * Dd))
        @test Matrix(mid(P)) == mid(Q) && Matrix(rad(P)) == rad(Q)
    end
    @test mid(Db * Db) isa Diagonal && rad(Db * Db) isa Diagonal
end
