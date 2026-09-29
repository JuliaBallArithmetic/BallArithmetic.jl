using Test
using BallArithmetic
using LinearAlgebra
using Random

@testset "verifylss: Algorithm 10.7 of Rump (2010)" begin
    A = [2.0 4.0; 0.0 4.0]
    bA = BallMatrix(A)

    @testset "Vector right-hand side" begin
        (v, cert) = verifylss(bA, BallVector(ones(2)))
        @test cert
        @test 0.0 ∈ v[1] && 0.25 in v[2]
    end

    @testset "Matrix right-hand side" begin
        (A_result, cert) = verifylss(bA, BallMatrix([1.0 0.0; 0.0 1.0]))
        @test cert
        @test -0.5 ∈ A_result[1, 2] && 0.0 in A_result[2, 1]
    end

    @testset "the caller selects an algorithm and rejects the rest" begin
        b = BallVector(ones(2))
        @test mid(verifylss(bA, b).solution) == mid(verifylss(bA, b; method = :rump2010).solution)
        @test_throws ArgumentError verifylss(bA, b; method = :nonsense)
        # a square matrix and a conformable right-hand side are the caller's business
        @test_throws ArgumentError verifylss(BallMatrix(randn(2, 3)), b)
        @test_throws DimensionMismatch verifylss(bA, BallVector(ones(3)))
    end

    # The iteration matrix C must ENCLOSE I - RA, not merely carry I - R*mid(A) as its midpoint.
    # Forming it as BallMatrix(I - R*mid(A), abs.(R)*rad(A)) claims radius 0 on an exact input
    # while the true entries differ by about 1e-15, and every claim of the method rests on this
    # being an enclosure. This test is the regression for that.
    @testset "the iteration matrix encloses I - RA" begin
        rng = MersenneTwister(7)
        for n in (10, 50)
            Am = randn(rng, n, n)
            Ab = BallMatrix(Am)
            R = inv(Am)
            C = I - BallMatrix(R) * Ab            # the form Algorithm 10.7 prescribes
            Ctrue = I - BigFloat.(R) * BigFloat.(Am)
            @test all(abs.(BigFloat.(mid(C)) .- Ctrue) .<= BigFloat.(rad(C)))
            # and the discarded form does not enclose it
            Cfloat = BallMatrix(I - R * Am, abs.(R) * rad(Ab))
            @test !all(abs.(BigFloat.(mid(Cfloat)) .- Ctrue) .<= BigFloat.(rad(Cfloat)))
        end
    end

    @testset "the enclosure contains the exact solution" begin
        rng = MersenneTwister(11)
        for n in (5, 20)
            Am = randn(rng, n, n)
            bm = randn(rng, n)
            res = verifylss(BallMatrix(Am), BallVector(bm))
            @test res.certified
            xtrue = BigFloat.(Am) \ BigFloat.(bm)
            m, r = mid(res.solution), rad(res.solution)
            for i in 1:n
                @test abs(BigFloat(m[i]) - xtrue[i]) <= BigFloat(r[i])
            end
        end
    end

    @testset "complex input" begin
        # the eigenvector basis of a real matrix is complex, so verifyeigall needs this path;
        # before it was fixed, a complex right-hand side threw a MethodError on the ball product
        rng = MersenneTwister(13)
        W = randn(rng, ComplexF64, 8, 8)
        Rhs = BallMatrix(randn(rng, ComplexF64, 8, 8))
        res = verifylss(BallMatrix(W), Rhs)
        @test res.certified
        Xtrue = Complex{BigFloat}.(W) \ Complex{BigFloat}.(mid(Rhs))
        m, r = mid(res.solution), rad(res.solution)
        @test all(abs.(Complex{BigFloat}.(m) .- Xtrue) .<= BigFloat.(r))
    end

    @testset "a singular midpoint declines instead of throwing" begin
        res = verifylss(BallMatrix([1.0 1.0; 1.0 1.0]), BallVector(ones(2)))
        @test !res.certified
    end
end
