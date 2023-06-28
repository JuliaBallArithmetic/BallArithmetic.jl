A = BallMatrix(rand(256, 256))

v = BallArithmetic.gevbox(A, A)

@test all([abs(v[i].c-1.0)<v[i].r for i in 1:256])