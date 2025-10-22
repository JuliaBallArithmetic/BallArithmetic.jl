@testset "Testing ArbNumerics external module" begin
    import ArbNumerics

    real_entry = ArbNumerics.setball(pi, 1e-30)
    complex_entry = ArbNumerics.ArbComplex(
        ArbNumerics.setball(exp(1), 2e-30),
        ArbNumerics.setball(-sqrt(2), 3e-30),
    )

    A = fill(real_entry, 1, 1)
    B = BallMatrix(A)

    @test B.c[1] == Float64(ArbNumerics.midpoint(real_entry))

    mid_err = abs(ArbNumerics.midpoint(real_entry) - ArbNumerics.ArbReal(B.c[1]))
    total_real = ArbNumerics.radius(real_entry) + mid_err
    expected_rad = BallArithmetic.setrounding(Float64, RoundUp) do
        Float64(ArbNumerics.midpoint(total_real)) + Float64(ArbNumerics.radius(total_real))
    end

    @test B.r[1] == expected_rad

    C = fill(complex_entry, 1, 1)
    D = BallMatrix(C)

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

    @test D.r[1] == expected_rad
end
