@testset "BallVectors" begin
    v = ones(128)
    @test mid(v) == v
    @test rad(v) == zeros(128)

    vr = 2^(-10) * ones(128)

    bv = BallVector(v, vr)
    @test mid(bv) == v
    @test rad(bv) == vr

    @test eltype(bv) == BallF64
    @test length(bv) == 128

    reduced = bv[1:5]

    @test mid(reduced) == v[1:5]
    @test rad(reduced) == rad(bv)[1:5]
end
