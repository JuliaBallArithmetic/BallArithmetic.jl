using Test
using BallArithmetic

@testset "Ball BigFloat Support" begin
    # Test Float64 operations
    @testset "Float64 operations" begin
        x = Ball(1.0, 0.1)
        y = Ball(2.0, 0.2)

        # Addition
        z = x + y
        @test mid(z) ≈ 3.0
        @test rad(z) >= 0.3

        # Subtraction
        z = x - y
        @test mid(z) ≈ -1.0
        @test rad(z) >= 0.3

        # Multiplication
        z = x * y
        @test mid(z) ≈ 2.0
        @test rad(z) >= 0.4

        # Division - midpoint may not be exactly 0.5 due to rigorous enclosure
        z = x / y
        @test mid(z) ≈ 0.5 atol=0.01
        @test rad(z) >= 0

        # Square root
        z = sqrt(Ball(4.0, 0.1))
        @test mid(z) ≈ 2.0 atol=0.1
    end

    # Test BigFloat operations
    @testset "BigFloat operations" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            x = Ball(BigFloat("1.0"), BigFloat("0.1"))
            y = Ball(BigFloat("2.0"), BigFloat("0.2"))

            # Addition
            z = x + y
            @test mid(z) ≈ BigFloat(3)
            @test rad(z) >= BigFloat("0.3")

            # Subtraction
            z = x - y
            @test mid(z) ≈ BigFloat(-1)
            @test rad(z) >= BigFloat("0.3")

            # Multiplication
            z = x * y
            @test mid(z) ≈ BigFloat(2)
            @test rad(z) >= BigFloat("0.4")

            # Division - midpoint may not be exactly 0.5 due to rigorous enclosure
            z = x / y
            @test mid(z) ≈ BigFloat(0.5) atol=0.01
            @test rad(z) >= 0

            # Square root
            z = sqrt(Ball(BigFloat("4.0"), BigFloat("0.1")))
            @test mid(z) ≈ BigFloat(2) atol=0.1
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    # Test complex Ball operations with BigFloat
    @testset "Complex BigFloat operations" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            x = Ball(Complex{BigFloat}(BigFloat("1.0"), BigFloat("2.0")), BigFloat("0.1"))
            y = Ball(Complex{BigFloat}(BigFloat("3.0"), BigFloat("4.0")), BigFloat("0.2"))

            # Addition
            z = x + y
            @test real(mid(z)) ≈ BigFloat(4)
            @test imag(mid(z)) ≈ BigFloat(6)
            @test rad(z) >= BigFloat("0.3")

            # Subtraction
            z = x - y
            @test real(mid(z)) ≈ BigFloat(-2)
            @test imag(mid(z)) ≈ BigFloat(-2)
            @test rad(z) >= BigFloat("0.3")
        finally
            setprecision(BigFloat, old_prec)
        end
    end
end
