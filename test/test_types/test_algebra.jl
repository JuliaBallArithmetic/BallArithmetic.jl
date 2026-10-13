@testset "Test Matrix Algebra" begin
    import IntervalArithmetic

    function check_included(bA, IA)
        @assert size(bA) == size(IA)

        lower = [IntervalArithmetic.inf(x) for x in IA]
        higher = [IntervalArithmetic.sup(x) for x in IA]

        test_lo = [lower[i, j] >= BallArithmetic.sub_down(bA[i, j].c, bA[i, j].r)
                   for (i, j) in Iterators.product(1:size(bA)[1], 1:size(bA)[2])]
        test_hi = [higher[i, j] <= BallArithmetic.add_up(bA[i, j].c, bA[i, j].r)
                   for (i, j) in Iterators.product(1:size(bA)[1], 1:size(bA)[2])]

        if all(test_lo) == false
            @info "lo", test_lo
        end

        if all(test_hi) == false
            @info "hi", test_hi
        end

        return all(test_lo) && all(test_hi)
    end

    A = rand(4, 4)
    err = rand(4, 4)
    ierr = IntervalArithmetic.interval(-2^-16, 2^-16) * err
    rA = 2^-16 * err

    iA = IntervalArithmetic.interval.(A) .+ ierr
    bA = BallMatrix(A, rA)

    Iλ = 1 + 2^(-10) * IntervalArithmetic.interval(-1, 1)
    bλ = Ball(1, 2^(-10))

    IB = Iλ * iA
    bB = bλ * bA

    @test check_included(bB, IB)

    B = rand(4, 4)

    iB = IntervalArithmetic.interval.(B)
    bB = BallMatrix(B)

    isum = iA + iB
    bsum = bA + bB

    @test check_included(bsum, isum)

    B = rand(4, 4)

    iB = IntervalArithmetic.interval.(B)

    isum = iA + iB
    bsum = bA + B

    @test check_included(bsum, isum)

    bsum = B + bA

    @test check_included(bsum, isum)

    A = BallArithmetic.NumericalTest._test_matrix(16)

    bA = BallMatrix(A)
    bprod = bA * bA'
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = bA * A'
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = A * bA'
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = BallArithmetic.MMul4(A, A')
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = BallArithmetic.MMul3(bA, bA')
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = BallArithmetic.MMul3(bA', A)
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = BallArithmetic.MMul3(A, bA')
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    bprod = BallArithmetic.MMul3(A, A')
    @test all([nextfloat(1.0) in bprod[i, i] for i in 1:15])

    using LinearAlgebra

    A = zeros(Ball{Float64, Float64}, (16, 16))
    lam = Ball(1 / 8, 1 / 8)

    B = A - lam * I

    #TODO diag does not seem to work on BallMatrices
    @test all(-lam.c .== diag(B.c)) && all(lam.r .<= diag(B.r))

    B = lam * I - A
    @test all(lam.c .== diag(B.c)) && all(lam.r .<= diag(B.r))

    lam = Ball(im * 1 / 8, 1 / 8)
    B = A - lam * I
    @test all(-lam.c .== diag(B.c)) && all(lam.r .<= diag(B.r))

    B = lam * I - A
    @test all(lam.c .== diag(B.c)) && all(lam.r .<= diag(B.r))

    A = rand(4, 4)
    B = rand(4, 4)

    bC = BallArithmetic.MMul4(A, B)

    bC2 = BallMatrix(A) * BallMatrix(B)

    @test bC.c == bC2.c && bC.r == bC2.r
end

@testset "sum, difference and real multiple keep a Symmetric radius" begin
    using LinearAlgebra, Random
    rng = MersenneTwister(4)
    setprecision(BigFloat, 256) do
        for S in (Float64, ComplexF64), uplo in (:U, :L)
            n = 5
            herm(X) = Hermitian((X + X') / 2, uplo)
            M, N = herm(randn(rng, S, n, n)), herm(randn(rng, S, n, n))
            R, Q = Symmetric(rand(rng, n, n) .* 1e-3, uplo), Symmetric(rand(rng, n, n) .* 1e-3, uplo)
            A, B = BallMatrix(M, R), BallMatrix(N, Q)
            for (name, C, op) in (("A + B", A + B, +), ("A - B", A - B, -))
                @test mid(C) isa Hermitian && rad(C) isa Symmetric
                # Hermitian matrices of the two balls, at the ends of their tolerances
                for _ in 1:10
                    E = Hermitian(big.(Matrix(R)) .* rand(rng, (-1, 1), n, n), uplo)
                    F = Hermitian(big.(Matrix(Q)) .* rand(rng, (-1, 1), n, n), uplo)
                    X = op(big.(Matrix(M)) + Matrix(E), big.(Matrix(N)) + Matrix(F))
                    @test all(abs.(X .- big.(Matrix(mid(C)))) .<= big.(Matrix(rad(C))))
                end
            end
            for λ in (2.0, -0.3, 3)
                C = λ * A
                @test mid(C) isa Hermitian && rad(C) isa Symmetric
                E = Hermitian(big.(Matrix(R)) .* rand(rng, (-1, 1), n, n), uplo)
                X = big(λ) .* (big.(Matrix(M)) + Matrix(E))
                @test all(abs.(X .- big.(Matrix(mid(C)))) .<= big.(Matrix(rad(C))))
            end
            # the radius agrees with the one obtained on dense storage
            Ad, Bd = BallMatrix(Matrix(M), Matrix(R)), BallMatrix(Matrix(N), Matrix(Q))
            @test Matrix(rad(A + B)) == rad(Ad + Bd)
            @test Matrix(rad(2.0 * A)) == rad(2.0 * Ad)
            # a dense radius, a plain matrix or a complex factor: no Symmetric radius comes out
            @test rad(A + Bd) isa Matrix
            @test rad(BallMatrix(M, Matrix(R)) + B) isa Matrix
            @test !(mid((1.0 + 2.0im) * A) isa Hermitian)
            @test rad(A * B) isa Matrix && mid(A * B) isa Matrix
            # a point Hermitian matrix added to the ball keeps it
            @test rad(A + N) isa Symmetric && mid(A + N) isa Hermitian
        end
    end
end
