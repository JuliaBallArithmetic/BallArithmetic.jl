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

    # Structured matrix support
    diag_vals_A = rand(4)
    diag_vals_B = rand(4)
    rad_vals_A = rand(4)
    rad_vals_B = rand(4)

    Ad = BallMatrix(Diagonal(diag_vals_A), Diagonal(rad_vals_A))
    Bd = BallMatrix(Diagonal(diag_vals_B), Diagonal(rad_vals_B))

    Cd_struct = BallArithmetic.MMul4(Ad, Bd)
    Cd_dense = BallArithmetic.MMul4(
        BallMatrix(Matrix(Diagonal(diag_vals_A)), Matrix(Diagonal(rad_vals_A))),
        BallMatrix(Matrix(Diagonal(diag_vals_B)), Matrix(Diagonal(rad_vals_B))),
    )

    @test Cd_struct.c == Cd_dense.c && Cd_struct.r == Cd_dense.r
    Cd_mul = Ad * Bd
    @test Cd_mul.c == Cd_struct.c && Cd_mul.r == Cd_struct.r

    UA = UpperTriangular(rand(4, 4))
    UB = UpperTriangular(rand(4, 4))
    rUA = UpperTriangular(rand(4, 4))
    rUB = UpperTriangular(rand(4, 4))

    Atri = BallMatrix(UA, rUA)
    Btri = BallMatrix(UB, rUB)

    Ctri_struct = BallArithmetic.MMul4(Atri, Btri)
    Ctri_dense = BallArithmetic.MMul4(
        BallMatrix(Matrix(UA), Matrix(rUA)),
        BallMatrix(Matrix(UB), Matrix(rUB)),
    )

    @test Ctri_struct.c == Ctri_dense.c && Ctri_struct.r == Ctri_dense.r
    Ctri_mul = Atri * Btri
    @test Ctri_mul.c == Ctri_struct.c && Ctri_mul.r == Ctri_struct.r

    Cmix_left = BallArithmetic.MMul4(UA, Btri)
    Cmix_left_dense = BallArithmetic.MMul4(
        Matrix(UA),
        BallMatrix(Matrix(UB), Matrix(rUB)),
    )
    @test Cmix_left.c == Cmix_left_dense.c && Cmix_left.r == Cmix_left_dense.r

    Cmix_right = BallArithmetic.MMul4(Atri, UB)
    Cmix_right_dense = BallArithmetic.MMul4(
        BallMatrix(Matrix(UA), Matrix(rUA)),
        Matrix(UB),
    )

    @test Cmix_right.c == Cmix_right_dense.c && Cmix_right.r == Cmix_right_dense.r
end
