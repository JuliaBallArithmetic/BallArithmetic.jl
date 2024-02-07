@testset "pseudospectra" begin
    A = [1.0 0.0; 0.0 -1.0]
    bA = BallMatrix(A)

    using LinearAlgebra
    K = svd(A)

    @test BallArithmetic._follow_level_set(0.5 + im * 0, 0.01, K) == (0.5 - 0.01im, 1.0)

    enc = BallArithmetic.compute_enclosure(bA, 0.0, 2.0, 0.01)

    @test enc[1][1] == 1.0 + 0.0 * im
    @test enc[1][2] >= 100
    @test all(abs.(enc[1][3] .- 1.0) .<= 0.02)
end
