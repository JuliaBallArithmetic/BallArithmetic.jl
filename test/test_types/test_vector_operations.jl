@testset "Test Vector Operations" begin
    import IntervalArithmetic

    A = rand(4, 4)
    err = rand(4, 4)
    ierr = IntervalArithmetic.Interval(-2^-16, 2^-16) * err
    rA = 2^16 * err

    iA = IntervalArithmetic.Interval.(A) .+ ierr
    bA = BallMatrix(A, rA)

    v = ones(4)

    iProd = iA * v
    bProd = bA * v

    lower = [x.lo for x in iProd]
    higher = [x.hi for x in iProd]

    @test all(in.(lower, bProd))
    @test all(in.(higher, bProd))

    v = rand(4)
    err = rand(4)
    ierr = IntervalArithmetic.Interval(-2^-16, 2^-16) * err
    rv = 2^16 * err

    iv = IntervalArithmetic.Interval.(v) + ierr
    bv = BallVector(v, rv)

    iProd = iA * iv
    bProd = bA * bv

    lower = [x.lo for x in iProd]
    higher = [x.hi for x in iProd]

    @test all(in.(lower, bProd))
    @test all(in.(higher, bProd))

    iProd = A * iv
    bProd = A * bv

    lower = [x.lo for x in iProd]
    higher = [x.hi for x in iProd]

    @test all(in.(lower, bProd))
    @test all(in.(higher, bProd))

    vA = rand(4)
    vB = rand(4)

    iA = IntervalArithmetic.Interval.(vA)
    bA = BallVector(vA)

    iB = IntervalArithmetic.Interval.(vB)
    bB = BallVector(vB)

    isum = iA + iB
    lower = [x.lo for x in isum]
    higher = [x.hi for x in isum]

    bsum = bA + bB

    @test all(in.(lower, bsum))
    @test all(in.(higher, bsum))

    λ = 2 + 2^(-10)

    IB = λ * iA
    bB = λ * bA

    lower = [x.lo for x in IB]
    higher = [x.hi for x in IB]

    @test all(in.(lower, bB))
    @test all(in.(higher, bB))

    Iλ = 1 + 2^(-10) * IntervalArithmetic.Interval(-1, 1)
    bλ = Ball(1, 2^(-10))

    IB = Iλ * iA
    bB = bλ * bA

    lower = [x.lo for x in IB]
    higher = [x.hi for x in IB]

    @test all(in.(lower, bB))
    @test all(in.(higher, bB))

    # B = rand(4, 4)

    # iB = IntervalArithmetic.Interval.(B)

    # isum = iA + iB
    # lower = [x.lo for x in isum]
    # higher = [x.hi for x in isum]

    # bsum = bA + B

    # @test all(in.(lower, bsum))
    # @test all(in.(higher, bsum))

    # bB = BallMatrix(B)

    # iprod = iA * iB
    # bprod = bA * bB

    # lower = [x.lo for x in iprod]
    # higher = [x.hi for x in iprod]

    # @test all(in.(lower, bprod))
    # @test all(in.(higher, bprod))

    # iprod = A * iB

    # bprod = A * bB

    # lower = [x.lo for x in iprod]
    # higher = [x.hi for x in iprod]

    # @test all(in.(lower, bprod))
    # @test all(in.(higher, bprod))

    # iprod = iB * A
    # bprod = bB * A

    # lower = [x.lo for x in iprod]
    # higher = [x.hi for x in iprod]

    # @test all(in.(lower, bprod))
    # @test all(in.(higher, bprod))

    # using LinearAlgebra

    # A = zeros(Ball{Float64, Float64}, (16, 16))
    # lam = Ball(1 / 8, 1 / 8)

    # B = A - lam * I

    # #TODO diag does not seem to work on BallMatrices
    # @test all(-lam.c .== diag(B.c))
    # @test all(lam.r .<= diag(B.r))

    # lam = Ball(im * 1 / 8, 1 / 8)
    # B = A - lam * I
    # @test all(-lam.c .== diag(B.c))
    # @test all(lam.r .<= diag(B.r))
end
