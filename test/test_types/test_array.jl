@testset "Array interface" begin
    A = ones(16, 16, 16)

    bA = BallArray(A)

    @test mid(A) == A && rad(A) == zeros(16, 16, 16)
    @test mid(bA) == ones(16, 16, 16) && rad(bA) == zeros(16, 16, 16)
    @test size(bA) == size(A)
    @test length(bA) == length(A)

    bA[1, 1, 1] = Ball(2.0, 1.0)

    @test mid(bA[1, 1, 1]) == 2.0 && rad(bA[1, 1, 1]) == 1.0

    bA[1, 1:2, 1] = BallArray([3.0; 1.0], [4.0, 5.0])

    @test mid(bA[1, 1:2, 1]) == [3.0; 1.0] && rad(bA[1, 1:2, 1]) == [4.0, 5.0]

    # bA = ones(Ball, (2, 1, 2))

    # @test mid(bA) == ones(2, 1, 2) && rad(bA) == zeros(2, 1, 2)

    # bA = zeros(Ball, (2, 1, 2))
    # @test mid(bA) == zeros(2, 1, 2) && rad(bA) == zeros(2, 1, 2)

    # bA = fill(Ball(1.0, 2.0), 2)
    # @test mid(bA) == fill(1.0, 2) && rad(bA) == fill(2.0, 2)
end
