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
