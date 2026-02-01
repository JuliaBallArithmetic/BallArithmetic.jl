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
end
