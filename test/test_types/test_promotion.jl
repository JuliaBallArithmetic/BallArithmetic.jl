# Tests for src/types/convertpromote.jl - Ball promotion rules

using Test
using BallArithmetic

@testset "Ball Type Promotion Rules" begin

    @testset "Ball{Float64,Float64} promotion with scalars" begin
        b = Ball(1.0, 0.1)

        # Promotion with Int64
        @test promote_type(typeof(b), Int64) == Ball{Float64, Float64}

        # Promotion with Float64
        @test promote_type(typeof(b), Float64) == Ball{Float64, Float64}

        # Promotion with Complex{Float64} should give complex ball
        @test promote_type(typeof(b), Complex{Float64}) == Ball{Float64, Complex{Float64}}

        # Promotion with Complex{Bool}
        @test promote_type(typeof(b), Complex{Bool}) == Ball{Float64, Complex{Float64}}
    end

    @testset "Ball{Float64,Complex{Float64}} promotion with scalars" begin
        b = Ball(1.0 + 2.0im, 0.1)

        # Promotion with Int64
        @test promote_type(typeof(b), Int64) == Ball{Float64, Complex{Float64}}

        # Promotion with Float64
        @test promote_type(typeof(b), Float64) == Ball{Float64, Complex{Float64}}

        # Promotion with Complex{Float64}
        @test promote_type(typeof(b), Complex{Float64}) == Ball{Float64, Complex{Float64}}

        # Promotion with Complex{Bool}
        @test promote_type(typeof(b), Complex{Bool}) == Ball{Float64, Complex{Float64}}
    end

    @testset "Real Ball + Complex Ball promotion" begin
        real_ball = Ball(1.0, 0.1)
        complex_ball = Ball(1.0 + 0.0im, 0.1)

        @test promote_type(typeof(real_ball), typeof(complex_ball)) == Ball{Float64, Complex{Float64}}
    end

    @testset "BigFloat Ball promotion rules" begin
        bf_ball = Ball(BigFloat(1.0), BigFloat(0.1))

        # BigFloat ball + BigFloat
        @test promote_type(typeof(bf_ball), BigFloat) == Ball{BigFloat, BigFloat}

        # BigFloat ball + Int64
        @test promote_type(typeof(bf_ball), Int64) == Ball{BigFloat, BigFloat}

        # BigFloat ball + Float64
        @test promote_type(typeof(bf_ball), Float64) == Ball{BigFloat, BigFloat}

        # BigFloat ball + BigInt
        @test promote_type(typeof(bf_ball), BigInt) == Ball{BigFloat, BigFloat}
    end

    @testset "Complex BigFloat Ball promotion rules" begin
        cbf_ball = Ball(Complex{BigFloat}(1.0, 2.0), BigFloat(0.1))

        # Complex BigFloat ball + BigFloat
        @test promote_type(typeof(cbf_ball), BigFloat) == Ball{BigFloat, Complex{BigFloat}}

        # Complex BigFloat ball + Int64
        @test promote_type(typeof(cbf_ball), Int64) == Ball{BigFloat, Complex{BigFloat}}

        # Complex BigFloat ball + Float64
        @test promote_type(typeof(cbf_ball), Float64) == Ball{BigFloat, Complex{BigFloat}}

        # Complex BigFloat ball + Complex{Float64}
        @test promote_type(typeof(cbf_ball), Complex{Float64}) == Ball{BigFloat, Complex{BigFloat}}

        # Complex BigFloat ball + Complex{BigFloat}
        @test promote_type(typeof(cbf_ball), Complex{BigFloat}) == Ball{BigFloat, Complex{BigFloat}}

        # Complex BigFloat ball + BigInt
        @test promote_type(typeof(cbf_ball), BigInt) == Ball{BigFloat, Complex{BigFloat}}
    end

    @testset "Float64 Ball + BigFloat Ball promotion" begin
        f64_ball = Ball(1.0, 0.1)
        bf_ball = Ball(BigFloat(2.0), BigFloat(0.2))

        @test promote_type(typeof(f64_ball), typeof(bf_ball)) == Ball{BigFloat, BigFloat}
    end

    @testset "Complex Float64 Ball + BigFloat Ball promotion" begin
        cf64_ball = Ball(1.0 + 2.0im, 0.1)
        bf_ball = Ball(BigFloat(2.0), BigFloat(0.2))

        @test promote_type(typeof(cf64_ball), typeof(bf_ball)) == Ball{BigFloat, Complex{BigFloat}}
    end

    @testset "Float64 Ball + Complex BigFloat Ball promotion" begin
        f64_ball = Ball(1.0, 0.1)
        cbf_ball = Ball(Complex{BigFloat}(1.0, 2.0), BigFloat(0.1))

        @test promote_type(typeof(f64_ball), typeof(cbf_ball)) == Ball{BigFloat, Complex{BigFloat}}
    end

    @testset "Complex Float64 Ball + Complex BigFloat Ball promotion" begin
        cf64_ball = Ball(1.0 + 2.0im, 0.1)
        cbf_ball = Ball(Complex{BigFloat}(1.0, 2.0), BigFloat(0.1))

        @test promote_type(typeof(cf64_ball), typeof(cbf_ball)) == Ball{BigFloat, Complex{BigFloat}}
    end

    @testset "Promotion enables arithmetic" begin
        # Real ball + Int
        b1 = Ball(2.0, 0.1)
        result = b1 + 3
        @test result isa Ball{Float64, Float64}
        @test mid(result) == 5.0

        # Real ball + Complex
        b2 = Ball(1.0, 0.1)
        result2 = b2 + (1.0 + 2.0im)
        @test result2 isa Ball{Float64, Complex{Float64}}
        @test mid(result2) == 2.0 + 2.0im
    end

    @testset "BigFloat promotion enables arithmetic" begin
        bf_ball = Ball(BigFloat(2.0), BigFloat(0.1))

        # BigFloat ball + Int
        result = bf_ball + 3
        @test result isa Ball{BigFloat, BigFloat}
        @test mid(result) == BigFloat(5.0)

        # BigFloat ball + Float64
        result2 = bf_ball * 2.5
        @test result2 isa Ball{BigFloat, BigFloat}
    end
end
