function inflation(A::BallMatrix, b::BallVector; rescale = (0.1), ϵ = 10^{-20}, iter = 5)
    C = BallMatrix(inv(A.c))

    x = C * (b.c)
    epsvector = BallVector(zeros(length(b)), fill(ϵ, length(b)))

    for _ in 1:iter
        y = Ball(1, 0.1) * x + epsvector
        x = C * b + (I - C * A) * y
    end
    return x
end
