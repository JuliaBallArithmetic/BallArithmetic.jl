@testset "Test Matrix Algebra" begin

    import IntervalArithmetic

    A = rand(4, 4)
    err = rand(4, 4)
    ierr = IntervalArithmetic.Interval(-2^-16, 2^-16) * err
    rA = 2^16 * err

    iA = IntervalArithmetic.Interval.(A) .+ ierr
    bA = BallMatrix(A, rA)

    B = rand(4, 4)

    iB = IntervalArithmetic.Interval.(B)
    bB = BallMatrix(B)

    isum = iA + iB
    lower = [x.lo for x in isum]
    higher = [x.hi for x in isum]

    bsum = bA + bB

    @test all(in.(lower, bsum))
    @test all(in.(higher, bsum))

    iprod = iA * iB
    bprod = bA * bB

    lower = [x.lo for x in iprod]
    higher = [x.hi for x in iprod]

    @test all(in.(lower, bprod))
    @test all(in.(higher, bprod))

    iprod = A * iB

    bprod = A * bB

    lower = [x.lo for x in iprod]
    higher = [x.hi for x in iprod]

    @test all(in.(lower, bprod))
    @test all(in.(higher, bprod))

    iprod = iB * A
    bprod = bB * A

    lower = [x.lo for x in iprod]
    higher = [x.hi for x in iprod]

    @test all(in.(lower, bprod))
    @test all(in.(higher, bprod))

    using LinearAlgebra

    A = zeros(Ball{Float64,Float64}, (16, 16))
    lam = Ball(1 / 8, 1 / 8)

    B = A - lam * I

    #TODO diag does not seem to work on BallMatrices
    @test all(-lam.c .== diag(B.c))
    @test all(lam.r .<= diag(B.r))

    lam = Ball(im * 1 / 8, 1 / 8)
    B = A - lam * I
    @test all(-lam.c .== diag(B.c))
    @test all(lam.r .<= diag(B.r))


end