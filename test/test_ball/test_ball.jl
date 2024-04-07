@testset "Ball Arithmetic" begin
    x = Ball(0.0, 0.5)
    @test in(0, x) == true

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
end
