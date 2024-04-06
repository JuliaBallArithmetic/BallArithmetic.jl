@testset "BallVectors" begin
    v = ones(128)
    @test mid(v) == v
    @test rad(v) == zeros(128)

    bv = ones(BallF64, 128)
    @test mid(bv) == v
    @test rad(bv) == zeros(128)

    bv = zeros(BallF64, 128)
    @test mid(bv) == zeros(128)
    @test rad(bv) == zeros(128)

    vr = 2^(-10) * ones(128)

    bv = BallVector(v, vr)
    @test mid(bv) == v
    @test rad(bv) == vr

    @test eltype(bv) == BallF64
    @test length(bv) == 128

    reduced = bv[1:5]

    @test mid(reduced) == v[1:5]
    @test rad(reduced) == rad(bv)[1:5]

    w = rand(5)
    bv[1:5] = BallVector(w)
    @test bv.c[1:5] == w
end
