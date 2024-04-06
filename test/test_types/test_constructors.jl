
@testset "Constructors" begin
    A = rand(1024, 1024)

    @test mid(A) == A
    @test rad(A) == zeros(1024, 1024)

    bA = BallMatrix(A)

    @test bA.c == A
    @test bA.r == zeros(1024, 1024)

    reduced = bA[2:end, 2:end]

    @test reduced.c == A[2:end, 2:end]

    bA = BallMatrix(A, A)
    @test bA.c == A
    @test bA.r == A
    @test Base.eltype(bA) == Ball{Float64, Float64}
    @test mid(bA[1, 2]) == A[1, 2]
    @test rad(bA[1, 2]) == A[1, 2]

    bA = BallMatrix(A + im * A, A)
    @test bA.c == A + im * A
    @test bA.r == A
    @test Base.eltype(bA) == Ball{Float64, Complex{Float64}}

    A = zeros(BallF64, (16, 8))
    @test A.c == zeros(Float64, (16, 8))
    @test A.r == zeros(Float64, (16, 8))

    B = ones(BallF64, (8, 4))

    @test B.c == ones(Float64, (8, 4))
    @test B.r == zeros(Float64, (8, 4))

    A[1:8, 1:4] = B
    @test A.c[1:8, 1:4] == ones((8, 4))
    @test A.r[1:8, 1:4] == zeros((8, 4))
end
