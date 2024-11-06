A = [2.0 4.0; 0.0 4.0]

bA = BallMatrix(A)

b = BallVector(ones(2))

x = BallArithmetic.epsilon_inflation(bA, BallVector(ones(2)))
v = x[1]

@test 0.0 ∈ v[1] && 0.25 in v[2]

x = BallArithmetic.epsilon_inflation(bA, BallMatrix([1.0 0.0; 0.0 1.0]))

A = x[1]
@test -0.5 ∈ A[1, 2] && 0.0 in A[2, 1]
