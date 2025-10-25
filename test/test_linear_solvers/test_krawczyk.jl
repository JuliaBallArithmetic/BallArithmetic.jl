A = [2.0 4.0; 0.0 4.0]

bA = BallMatrix(A)

b = BallVector(ones(2))

x = BallArithmetic.krawczyk(bA, b)

@test 0.0 ∈ x[1] && 0.25 ∈ x[2]
