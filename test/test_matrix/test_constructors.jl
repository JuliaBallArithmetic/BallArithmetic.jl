
@testset "Constructors" begin
    A = rand(1024, 1024)

    @test mid(A) == A
    @test rad(A) == zeros(1024, 1024)

    bA = BallMatrix(A)

    @test bA.c == A
    @test bA.r == zeros(1024, 1024)

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
end
