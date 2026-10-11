@testset "Matrix classifier" begin
    using BallArithmetic
    using LinearAlgebra
    using Random

    bA = BallMatrix([1.0 0.0; 0.0 1.0])
    @test BallArithmetic.is_M_matrix(bA) == true
    @test BallArithmetic.is_H_matrix(bA) == true

    bA = BallMatrix([1.0 0.1; 0.1 1.0])

    bB = BallArithmetic.off_diagonal_abs(bA)

    @test bB.c == [0.0 0.1; 0.1 0.0]

    v = BallArithmetic.diagonal_abs_lower_bound(bA)

    @test v == [1.0; 1.0]

    ρ = BallArithmetic.collatz_upper_bound(bB)

    @test ρ >= 0.1

    # positive off-diagonal entries: the comparison matrix is a nonsingular M-matrix, the matrix
    # itself is not in Z^{n×n} (Varga 2004, (C.3) and Definition C.3)
    @test BallArithmetic.is_H_matrix(bA) == true
    @test BallArithmetic.is_M_matrix(bA) == false
    @test BallArithmetic.is_M_matrix(BallMatrix([1.0 -0.1; -0.1 1.0])) == true

    @testset "Definition C.3: μ > ρ(B), against the inverse" begin
        rng = MersenneTwister(20261011)
        isM(A) = all(A[i, j] <= 0 for i in axes(A, 1), j in axes(A, 2) if i != j) &&
                 (μ = maximum(diag(A)); maximum(abs, eigvals(μ * I - A)) < μ)
        agree = 0
        for _ in 1:200
            n = rand(rng, 2:6)
            A = -rand(rng, n, n) + Diagonal(rand(rng, n) .* n)
            truth = isM(A)
            got = BallArithmetic.is_M_matrix(BallMatrix(A))
            got && @test truth                               # never a false certificate
            got && @test all(inv(A) .>= -1e-12)              # Proposition C.4
            agree += got == truth
        end
        @test agree >= 190                                   # declined only near the boundary
        # a diagonal that is not constant: μ is its maximum, the test is not min a_ii > ρ(off)
        A = [10.0 -1.0; -1.0 0.2]
        @test isM(A) && BallArithmetic.is_M_matrix(BallMatrix(A))
        # the sign conditions
        @test !BallArithmetic.is_M_matrix(BallMatrix(-Matrix(1.0I, 3, 3)))
        @test !BallArithmetic.is_H_matrix(BallMatrix(zeros(2, 2)))
        @test BallArithmetic.is_H_matrix(BallMatrix(-Matrix(1.0I, 3, 3)))
        @test !BallArithmetic.is_M_matrix(BallMatrix([1.0 -2.0; -2.0 1.0]))
        @test !BallArithmetic.is_M_matrix(BallMatrix([1.0+0im -0.1; -0.1 1.0]))
        @test BallArithmetic.is_H_matrix(BallMatrix([1.0+1im -0.1im; 0.3 -2.0]))
    end

    @testset "a ball of matrices" begin
        rng = MersenneTwister(20261012)
        A = [4.0 -1.0 -0.5; -1.0 3.0 -1.0; -0.2 -1.0 2.5]
        @test BallArithmetic.is_M_matrix(BallMatrix(A, fill(0.1, 3, 3)))
        for _ in 1:20
            Ap = A + 0.1 * (2 * rand(rng, 3, 3) .- 1)
            @test all(inv(Ap) .>= 0)
        end
        # the radius lets an off-diagonal entry be positive: not proved
        @test !BallArithmetic.is_M_matrix(BallMatrix([1.0 -0.1; -0.1 1.0], fill(0.2, 2, 2)))
        @test BallArithmetic.is_H_matrix(BallMatrix([1.0 -0.1; -0.1 1.0], fill(0.2, 2, 2)))
        # a singular member in the ball: not proved
        @test !BallArithmetic.is_H_matrix(BallMatrix([1.0 1.0; 1.0 1.0], fill(0.01, 2, 2)))
        @test_throws DimensionMismatch BallArithmetic.is_M_matrix(BallMatrix(zeros(2, 3)))
    end
end
