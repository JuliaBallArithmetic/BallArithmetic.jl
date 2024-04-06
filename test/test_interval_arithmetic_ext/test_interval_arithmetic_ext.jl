using IntervalArithmetic

x = interval(-1.0, 1.0)

b = Ball(x)

@test b.c == 0.0 && b.r == 1.0
