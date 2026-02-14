@testset "Testing ArbNumerics external module" begin
    import ArbNumerics

    real_entry = ArbNumerics.setball(pi, 1e-30)
    complex_entry = ArbNumerics.ArbComplex(
        ArbNumerics.setball(exp(1), 2e-30),
        ArbNumerics.setball(-sqrt(2), 3e-30),
    )

    @testset "Default BallMatrix(A) preserves precision (BigFloat)" begin
        A = fill(real_entry, 1, 1)
        B = BallMatrix(A)

        # Default should produce BigFloat midpoints
        @test eltype(B.c) == BigFloat
        @test eltype(B.r) == BigFloat

        # Midpoint should be close to the Arb midpoint (within BigFloat precision)
        bf_mid = BigFloat(ArbNumerics.midpoint(real_entry))
        @test B.c[1] == bf_mid

        # Radius should be the Arb radius (no truncation error)
        bf_rad = BigFloat(ArbNumerics.radius(real_entry))
        @test B.r[1] == bf_rad

        # Complex case
        C = fill(complex_entry, 1, 1)
        D = BallMatrix(C)

        @test eltype(D.c) == Complex{BigFloat}
        @test eltype(D.r) == BigFloat

        expected_mid = Complex{BigFloat}(
            BigFloat(ArbNumerics.midpoint(real(complex_entry))),
            BigFloat(ArbNumerics.midpoint(imag(complex_entry))),
        )
        @test D.c[1] == expected_mid
    end

    @testset "BallMatrix(Float64, A) truncates with radius" begin
        A = fill(real_entry, 1, 1)
        B = BallMatrix(Float64, A)

        @test eltype(B.c) == Float64
        @test eltype(B.r) == Float64
        @test B.c[1] == Float64(ArbNumerics.midpoint(real_entry))

        mid_err = abs(ArbNumerics.midpoint(real_entry) - ArbNumerics.ArbReal(B.c[1]))
        total_real = ArbNumerics.radius(real_entry) + mid_err
        expected_rad = BallArithmetic.setrounding(Float64, RoundUp) do
            Float64(ArbNumerics.midpoint(total_real)) + Float64(ArbNumerics.radius(total_real))
        end

        @test B.r[1] == expected_rad

        C = fill(complex_entry, 1, 1)
        D = BallMatrix(Float64, C)

        @test eltype(D.c) == ComplexF64
        @test eltype(D.r) == Float64

        expected_mid = ComplexF64(
            Float64(ArbNumerics.midpoint(real(complex_entry))),
            Float64(ArbNumerics.midpoint(imag(complex_entry))),
        )

        @test D.c[1] == expected_mid

        err_real = abs(ArbNumerics.midpoint(real(complex_entry)) - ArbNumerics.ArbReal(real(expected_mid)))
        err_imag = abs(ArbNumerics.midpoint(imag(complex_entry)) - ArbNumerics.ArbReal(imag(expected_mid)))

        total_real = ArbNumerics.radius(real(complex_entry)) + err_real
        total_imag = ArbNumerics.radius(imag(complex_entry)) + err_imag

        expected_rad = BallArithmetic.setrounding(Float64, RoundUp) do
            real_hi = Float64(ArbNumerics.midpoint(total_real)) + Float64(ArbNumerics.radius(total_real))
            imag_hi = Float64(ArbNumerics.midpoint(total_imag)) + Float64(ArbNumerics.radius(total_imag))
            sqrt(real_hi^2 + imag_hi^2)
        end

        if expected_rad == 0.0 && (!iszero(total_real) || !iszero(total_imag))
            expected_rad = nextfloat(0.0)
        end

        @test D.r[1] == expected_rad
    end

    @testset "BigFloat preserves much more precision than Float64" begin
        # High-precision Arb entry (default 128-bit precision)
        x = ArbNumerics.setball(pi, 1e-30)
        A = fill(x, 1, 1)

        B_bf = BallMatrix(A)           # BigFloat (default)
        B_f64 = BallMatrix(Float64, A) # Float64

        # BigFloat radius should be MUCH smaller (no truncation error)
        @test B_bf.r[1] < B_f64.r[1]

        # The Float64 radius includes conversion error ~10⁻¹⁶
        # The BigFloat radius is just the Arb radius ~10⁻³⁰
        @test Float64(B_bf.r[1]) < 1e-29
        @test B_f64.r[1] > 1e-17
    end

    @testset "BallMatrix(BigFloat, A) matches default" begin
        A = fill(real_entry, 2, 2)
        B_default = BallMatrix(A)
        B_explicit = BallMatrix(BigFloat, A)

        @test B_default.c == B_explicit.c
        @test B_default.r == B_explicit.r

        C = fill(complex_entry, 2, 2)
        D_default = BallMatrix(C)
        D_explicit = BallMatrix(BigFloat, C)

        @test D_default.c == D_explicit.c
        @test D_default.r == D_explicit.r
    end

    @testset "Subnormal rounding (Float64 path)" begin
        tiny_mid = ArbNumerics.ArbReal("1e-4000")
        tiny_rad = ArbNumerics.ArbReal("1e-4010")
        tiny_entry = ArbNumerics.setball(tiny_mid, tiny_rad)

        tiny_ball = BallMatrix(Float64, fill(tiny_entry, 1, 1))

        @test tiny_ball.c[1] == 0.0

        mid_err = abs(ArbNumerics.midpoint(tiny_entry) - ArbNumerics.ArbReal(tiny_ball.c[1]))
        total = ArbNumerics.radius(tiny_entry) + mid_err

        expected_tiny_radius = BallArithmetic.setrounding(Float64, RoundUp) do
            Float64(ArbNumerics.midpoint(total)) + Float64(ArbNumerics.radius(total))
        end

        if expected_tiny_radius == 0.0 && !iszero(total)
            expected_tiny_radius = nextfloat(0.0)
        end

        @test tiny_ball.r[1] == expected_tiny_radius
        @test expected_tiny_radius == nextfloat(0.0)

        tiny_complex_entry = ArbNumerics.ArbComplex(tiny_entry, ArbNumerics.setball(-tiny_mid, tiny_rad))
        tiny_complex_ball = BallMatrix(Float64, fill(tiny_complex_entry, 1, 1))

        @test tiny_complex_ball.c[1] == ComplexF64(0.0, 0.0)

        err_real = abs(ArbNumerics.midpoint(real(tiny_complex_entry)) - ArbNumerics.ArbReal(real(tiny_complex_ball.c[1])))
        err_imag = abs(ArbNumerics.midpoint(imag(tiny_complex_entry)) - ArbNumerics.ArbReal(imag(tiny_complex_ball.c[1])))

        total_real = ArbNumerics.radius(real(tiny_complex_entry)) + err_real
        total_imag = ArbNumerics.radius(imag(tiny_complex_entry)) + err_imag

        expected_tiny_complex_radius = BallArithmetic.setrounding(Float64, RoundUp) do
            real_hi = Float64(ArbNumerics.midpoint(total_real)) + Float64(ArbNumerics.radius(total_real))
            imag_hi = Float64(ArbNumerics.midpoint(total_imag)) + Float64(ArbNumerics.radius(total_imag))
            sqrt(real_hi^2 + imag_hi^2)
        end

        if expected_tiny_complex_radius == 0.0 && (!iszero(total_real) || !iszero(total_imag))
            expected_tiny_complex_radius = nextfloat(0.0)
        end

        @test tiny_complex_ball.r[1] == expected_tiny_complex_radius
        @test expected_tiny_complex_radius == nextfloat(0.0)
    end
end
