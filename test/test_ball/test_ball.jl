@testset "Ball Arithmetic" begin

    x = Ball(0.0, 0.5)
    @test in(0, x) == true

    x = Ball(1.0+im, 2.0)
    @test in(0, x) == true

    import IntervalArithmetic

    v = rand(Float64, 1024)
    iv = IntervalArithmetic.Interval.(v)
    bv = Ball.(v)

    w = rand(1024)
    iw = IntervalArithmetic.Interval.(w)
    bw = Ball.(w)

    isum = iv+iw
    lower = [x.lo for x in isum]
    higher = [x.hi for x in isum]

    bsum = bv+bw

    @test all(in.(lower, bsum))
    @test all(in.(higher, bsum))

end