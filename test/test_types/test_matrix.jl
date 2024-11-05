@testset "Matrix type" begin
    A = BallMatrix(rand(4, 4))

    B = copy(A)
    @test A.c == B.c && A.r == B.r
end
