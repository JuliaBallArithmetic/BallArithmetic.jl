using Test
using BallArithmetic

@testset "BigFloat Rounding Operations" begin

    @testset "machine_epsilon" begin
        # Float64 epsilon (unit roundoff = eps/2 in IEEE, but ϵp = 2^-52 = eps)
        @test BallArithmetic.machine_epsilon(Float64) == 2.0^-52
        @test BallArithmetic.machine_epsilon(Float64) == eps(Float64)

        # Float32 epsilon
        @test BallArithmetic.machine_epsilon(Float32) == Float32(2.0^-23)
        @test BallArithmetic.machine_epsilon(Float32) == eps(Float32)

        # BigFloat epsilon depends on precision
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)
            @test BallArithmetic.machine_epsilon(BigFloat) == BigFloat(2)^(-256)

            setprecision(BigFloat, 512)
            @test BallArithmetic.machine_epsilon(BigFloat) == BigFloat(2)^(-512)
        finally
            setprecision(BigFloat, old_prec)
        end

        # Complex types
        @test BallArithmetic.machine_epsilon(Complex{Float64}) == BallArithmetic.machine_epsilon(Float64)
        @test BallArithmetic.machine_epsilon(Complex{BigFloat}) == BallArithmetic.machine_epsilon(BigFloat)
    end

    @testset "subnormal_min" begin
        # Float64 subnormal minimum
        @test BallArithmetic.subnormal_min(Float64) == 2.0^-1074
        @test BallArithmetic.subnormal_min(Float64) > 0
        @test BallArithmetic.subnormal_min(Float64) < floatmin(Float64)

        # Float32 subnormal minimum
        @test BallArithmetic.subnormal_min(Float32) == Float32(2.0^-149)
        @test BallArithmetic.subnormal_min(Float32) > 0

        # BigFloat subnormal minimum
        @test BallArithmetic.subnormal_min(BigFloat) > 0
    end

    @testset "Backwards compatibility" begin
        # ϵp and η should still work
        @test BallArithmetic.ϵp == BallArithmetic.machine_epsilon(Float64)
        @test BallArithmetic.η == BallArithmetic.subnormal_min(Float64)
    end

    @testset "BigFloat add_up/add_down" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            a = BigFloat("1.0")
            b = BigFloat("1e-100")

            # add_up should round up
            result_up = BallArithmetic.add_up(a, b)
            result_down = BallArithmetic.add_down(a, b)

            @test result_up >= result_down
            @test result_up >= a + b  # Should be at least as large as exact
            @test result_down <= a + b  # Should be at most as large as exact

            # Test with negative numbers
            c = BigFloat("-1.0")
            @test BallArithmetic.add_up(c, b) >= BallArithmetic.add_down(c, b)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "BigFloat mul_up/mul_down" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            a = BigFloat("3.14159265358979323846")
            b = BigFloat("2.71828182845904523536")

            result_up = BallArithmetic.mul_up(a, b)
            result_down = BallArithmetic.mul_down(a, b)

            @test result_up >= result_down
            @test result_up >= a * b
            @test result_down <= a * b

            # Test multiplication sign behavior
            c = BigFloat("-2.0")
            # Negative * positive
            @test BallArithmetic.mul_up(c, a) >= BallArithmetic.mul_down(c, a)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "BigFloat div_up/div_down" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            a = BigFloat("1.0")
            b = BigFloat("3.0")

            result_up = BallArithmetic.div_up(a, b)
            result_down = BallArithmetic.div_down(a, b)

            @test result_up >= result_down
            @test result_up >= a / b
            @test result_down <= a / b
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "BigFloat sqrt_up/sqrt_down" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            a = BigFloat("2.0")

            result_up = BallArithmetic.sqrt_up(a)
            result_down = BallArithmetic.sqrt_down(a)

            @test result_up >= result_down
            @test result_up >= sqrt(a)
            @test result_down <= sqrt(a)

            # Check that [down, up] contains exact sqrt(2)
            @test result_down <= sqrt(BigFloat("2.0")) <= result_up
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "BigFloat sub_up/sub_down" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            a = BigFloat("1.0")
            b = BigFloat("1e-100")

            result_up = BallArithmetic.sub_up(a, b)
            result_down = BallArithmetic.sub_down(a, b)

            @test result_up >= result_down
            @test result_up >= a - b
            @test result_down <= a - b
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "BigFloat abs_up/abs_down" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Positive number
            a = BigFloat("3.14159")
            @test BallArithmetic.abs_up(a) == a
            @test BallArithmetic.abs_down(a) == a

            # Negative number
            b = BigFloat("-2.71828")
            @test BallArithmetic.abs_up(b) >= abs(b)
            @test BallArithmetic.abs_down(b) <= abs(b)
            @test BallArithmetic.abs_up(b) >= BallArithmetic.abs_down(b)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Complex BigFloat operations" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            z1 = Complex(BigFloat("1.0"), BigFloat("2.0"))
            z2 = Complex(BigFloat("3.0"), BigFloat("4.0"))

            # Addition
            result_add = BallArithmetic.add_up(z1, z2)
            @test real(result_add) >= real(z1 + z2)
            @test imag(result_add) >= imag(z1 + z2)

            # Subtraction
            result_sub = BallArithmetic.sub_up(z1, z2)
            @test real(result_sub) >= real(z1 - z2)
            @test imag(result_sub) >= imag(z1 - z2)

            # Multiplication (more complex)
            result_mul = BallArithmetic.mul_up(z1, z2)
            # The multiplication result should be a valid upper bound
            # (though conservative due to interval arithmetic)
            @test isa(result_mul, Complex{BigFloat})
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Rounding consistency across operations" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Test that up/down rounding creates enclosures
            x = BigFloat("1.234567890123456789")
            y = BigFloat("9.876543210987654321")

            # For any operation op, [op_down(x,y), op_up(x,y)] should contain exact result
            for (op, op_up, op_down) in [
                (+, BallArithmetic.add_up, BallArithmetic.add_down),
                (-, BallArithmetic.sub_up, BallArithmetic.sub_down),
                (*, BallArithmetic.mul_up, BallArithmetic.mul_down),
                (/, BallArithmetic.div_up, BallArithmetic.div_down),
            ]
                exact = op(x, y)
                up = op_up(x, y)
                down = op_down(x, y)

                @test down <= exact <= up
            end

            # sqrt
            @test BallArithmetic.sqrt_down(x) <= sqrt(x) <= BallArithmetic.sqrt_up(x)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

end
