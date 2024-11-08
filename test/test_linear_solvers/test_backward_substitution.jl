A = [2.0 4.0; 0.0 4.0]

bA = BallMatrix(A)

b = BallVector(ones(2))

x = BallArithmetic.backward_substitution(bA, BallVector(ones(2)))

@test 0.0 ∈ x[1] && 0.25 in x[2]

B = BallMatrix([1.0 0.0; 1.0 1.0])

C = BallArithmetic.backward_substitution(bA, B)

@test C[:, 1] == x
