@testset "Testing IntervalArithmetic external module" begin
    import IntervalArithmetic

    x = IntervalArithmetic.interval(-1.0, 1.0)

    b = Ball(x)

    @test b.c == 0.0 && b.r == 1.0

    y = x + im * (x + 1.0)
    b = Ball(y)

    @test b.c == im && b.r >= sqrt(2)

    A = fill(x, (2, 2))
    B = BallMatrix(A)

    @test all([x == 0.0 for x in B.c]) && all(x == 1.0 for x in B.r)

    A = A + im * (A .+ 1.0)
    B = BallMatrix(A)

    @test all([x == im for x in B.c]) && all(x >= sqrt(2) for x in B.r)

    b = Ball(0.0, 1.0)
    x = IntervalArithmetic.interval(b)
    @test IntervalArithmetic.inf(x) == -1.0 && IntervalArithmetic.sup(x) == 1.0

    # Test enclosure property: interval must contain [c-r, c+r]
    # This verifies the RoundDown fix for lower bound computation
    @testset "Ball to Interval enclosure property" begin
        for _ in 1:50
            c = randn()
            r = abs(randn())
            ball = Ball(c, r)
            intv = IntervalArithmetic.interval(ball)

            # The interval should properly contain the ball bounds
            @test IntervalArithmetic.inf(intv) <= c - r
            @test IntervalArithmetic.sup(intv) >= c + r
        end

        # Specific test: value where subtraction is not exact
        c = 1.0 + eps(1.0)
        r = eps(1.0) / 2
        ball = Ball(c, r)
        intv = IntervalArithmetic.interval(ball)
        @test IntervalArithmetic.inf(intv) <= c - r
        @test IntervalArithmetic.sup(intv) >= c + r
    end

    @testset "BallVector from intervals" begin
        # The generic BallVector(::AbstractVector) routes through `rad`, which
        # has no method for a vector of intervals, so these constructors were
        # missing while their BallMatrix counterparts worked.
        x = IntervalArithmetic.interval(-1.0, 1.0)

        v = fill(x, 3)
        bv = BallVector(v)
        @test bv.c == zeros(3) && bv.r == ones(3)

        cv = v + im * (v .+ 1.0)
        bcv = BallVector(cv)
        @test all(c == im for c in bcv.c) && all(r >= sqrt(2) for r in bcv.r)

        # The complex radius must give a disk containing the rectangle.
        z = complex(IntervalArithmetic.interval(0.0, 0.5),
            IntervalArithmetic.interval(0.0, 0.25))
        b1 = BallVector([z])
        @test b1.r[1] >= sqrt(0.25^2 + 0.125^2)

        # Arbitrary precision goes through the same path.
        setprecision(BigFloat, 128) do
            vb = fill(IntervalArithmetic.interval(BigFloat(-1), BigFloat(1)), 2)
            bb = BallVector(vb)
            @test eltype(bb.r) == BigFloat
            @test bb.c == zeros(BigFloat, 2) && bb.r == ones(BigFloat, 2)

            Mb = fill(IntervalArithmetic.interval(BigFloat(-1), BigFloat(1)), (2, 2))
            BB = BallMatrix(Mb)
            @test eltype(BB.r) == BigFloat
        end
    end
end
