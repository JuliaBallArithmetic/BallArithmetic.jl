@testset "Ball Arithmetic" begin
    x = Ball(0.0, 0.5)
    @test in(0, x) == true

    x = abs(x)
    @test x.c == 0.25 && x.r == 0.25

    x = 0.0 ± 1.0
    @test x.c == 0.0 && x.r == 1.0

    x = Ball(1.0 + im, 2.0)
    @test in(0, x) == true

    import IntervalArithmetic

    v = rand(Float64, 1024)
    iv = IntervalArithmetic.interval.(v)
    bv = Ball.(v)

    w = rand(1024)
    iw = IntervalArithmetic.interval.(w)
    bw = Ball.(w)

    isum = iv + iw
    lower = [IntervalArithmetic.inf(x) for x in isum]
    higher = [IntervalArithmetic.sup(x) for x in isum]

    bsum = bv + bw

    @test all(in.(lower, bsum)) && all(in.(higher, bsum))

    iprod = iv .* iw
    lower = [IntervalArithmetic.inf(x) for x in iprod]
    higher = [IntervalArithmetic.sup(x) for x in iprod]

    bprod = bv .* bw
    @test all(in.(lower, bprod)) && all(in.(higher, bprod))

    x = Ball(1.0, 1 / 4) #interval [3/4, 5/4]
    t = inv(x) # the inverse is [4/5, 4/3]
    @test 4 / 5 ∈ t
    @test 4 / 3 ∈ t

    x = Ball(rand() + im * rand())
    @test 1.0 ∈ x * inv(x)

    # test sqrt on positive ball and domain error on ball containing zero
    x = Ball(4.0, 0.1)
    y = sqrt(x)
    @test 2.0 ∈ y
    @test x ∈ (y * y)
    @test_throws DomainError sqrt(Ball(1.0, 1.0))

    # conjugation of complex balls
    cball = Ball(3.0 + 4.0im, 0.5)
    conjball = conj(cball)
    @test conjball.c == conj(cball.c) && conjball.r == cball.r

    # membership of balls
    small = Ball(1.0, 0.2)
    big = Ball(1.0, 0.5)
    @test small ∈ big
    @test !(big ∈ small)

    csmall = Ball(1.0 + im, 0.2)
    cbig = Ball(1.0 + im, 0.5)
    @test csmall ∈ cbig
    @test !(cbig ∈ csmall)

    # sup and inf
    rball = Ball(0.0, 1.0)
    @test sup(rball) >= 1.0
    @test inf(rball) <= -1.0

    # type utilities and conversions
    @test BallArithmetic.midtype(rball) == Float64
    @test BallArithmetic.radtype(rball) == Float64
    @test BallArithmetic.midtype(BallComplexF64) == ComplexF64
    @test BallArithmetic.radtype(BallComplexF64) == Float64
    @test BallArithmetic.midtype(Ball) == Float64
    @test BallArithmetic.radtype(Ball) == Float64

    cb = convert(Ball{Float64, ComplexF64}, Ball(1.0 + 2.0im, 0.3))
    @test cb.c == 1.0 + 2.0im && cb.r == 0.3
    rb = convert(Ball{Float64, Float64}, 1.25)
    @test rb.c == 1.25 && rb.r == 0.0

    # negative and subtraction operations
    nb = -rball
    @test nb.c == -rball.c && nb.r == rball.r

    a = Ball(3.0, 0.1)
    b = Ball(1.5, 0.1)
    diff = a - b
    @test 1.5 ∈ diff
end
