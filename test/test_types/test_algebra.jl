@testset "Test Matrix Algebra" begin
    import IntervalArithmetic

    A = rand(4, 4)
    err = rand(4, 4)
    ierr = IntervalArithmetic.interval(-2^-16, 2^-16) * err
    rA = 2^16 * err

    iA = IntervalArithmetic.interval.(A) .+ ierr
    bA = BallMatrix(A, rA)

    Iλ = 1 + 2^(-10) * IntervalArithmetic.interval(-1, 1)
    bλ = Ball(1, 2^(-10))

    IB = Iλ * iA
    bB = bλ * bA

    lower = [IntervalArithmetic.inf(x) for x in IB]
    higher = [IntervalArithmetic.sup(x) for x in IB]

    @test all(in.(lower, bB)) && all(in.(higher, bB))

    B = rand(4, 4)

    iB = IntervalArithmetic.interval.(B)
    bB = BallMatrix(B)

    isum = iA + iB
    lower = [IntervalArithmetic.inf(x) for x in isum]
    higher = [IntervalArithmetic.sup(x) for x in isum]

    bsum = bA + bB

    @test all(in.(lower, bsum))
    @test all(in.(higher, bsum))

    B = rand(4, 4)

    iB = IntervalArithmetic.interval.(B)

    isum = iA + iB
    lower = [IntervalArithmetic.inf(x) for x in isum]
    higher = [IntervalArithmetic.sup(x) for x in isum]

    bsum = bA + B

    @test all(in.(lower, bsum))
    @test all(in.(higher, bsum))

    bB = BallMatrix(B)

    iprod = iA * iB
    bprod = bA * bB

    lower = [IntervalArithmetic.inf(x) for x in iprod]
    higher = [IntervalArithmetic.sup(x) for x in iprod]

    @test all(in.(lower, bprod))
    @test all(in.(higher, bprod))

    iprod = A * iB

    bprod = A * bB

    lower = [IntervalArithmetic.inf(x) for x in iprod]
    higher = [IntervalArithmetic.sup(x) for x in iprod]

    @test all(in.(lower, bprod))
    @test all(in.(higher, bprod))

    iprod = iB * A
    bprod = bB * A

    lower = [IntervalArithmetic.inf(x) for x in iprod]
    higher = [IntervalArithmetic.sup(x) for x in iprod]

    @test all(in.(lower, bprod))
    @test all(in.(higher, bprod))

    using LinearAlgebra

    A = zeros(Ball{Float64, Float64}, (16, 16))
    lam = Ball(1 / 8, 1 / 8)

    B = A - lam * I

    #TODO diag does not seem to work on BallMatrices
    @test all(-lam.c .== diag(B.c)) && all(lam.r .<= diag(B.r))

    lam = Ball(im * 1 / 8, 1 / 8)
    B = A - lam * I
    @test all(-lam.c .== diag(B.c)) && all(lam.r .<= diag(B.r))

    A = rand(4, 4)
    B = rand(4, 4)

    bC = BallArithmetic.MMul3(A, B)

    bC2 = BallMatrix(A) * BallMatrix(B)

    @test bC.c == bC2.c && bC.r == bC2.r
end
