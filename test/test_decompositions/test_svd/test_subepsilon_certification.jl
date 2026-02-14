"""
Test suite for sub-machine-precision (sub-ε) circle certification.

Tests the ability to certify resolvent norms at points on circles with
radius ≤ ε_machine (≈ 2.2e-16), where Float64 cannot distinguish
individual points on the circle.

Key insight: Using BigFloat arithmetic throughout, we can:
1. Represent distinct points that collapse to the same Float64 value
2. Compute A - z*I in BigFloat to preserve sub-ε differences
3. Use center's Float64 SVD as initial guess for all points
4. Refine with Ogita's algorithm in BigFloat to get distinct σ_min values
"""

using Test
using BallArithmetic
using BallArithmetic.CertifScripts: ogita_svd_refine
using LinearAlgebra

@testset "Sub-ε Circle Certification" begin

    @testset "Float64 Collapse Detection" begin
        # Test that Float64 representations collapse at small radii
        setprecision(BigFloat, 256)

        center = BigFloat("1.5")
        num_points = 8

        # At 1e-14, some points are still distinct
        radius_14 = BigFloat("1e-14")
        offsets_14 = [radius_14 * cos(BigFloat(2π * k / num_points)) for k in 0:num_points-1]
        zs_f64_14 = Float64.(center .+ offsets_14)
        @test length(unique(zs_f64_14)) > 1

        # At 1e-16, most pure real points collapse
        radius_16 = BigFloat("1e-16")
        offsets_16 = [radius_16 * cos(BigFloat(2π * k / num_points)) for k in 0:num_points-1]
        zs_f64_16 = Float64.(center .+ offsets_16)
        @test length(unique(zs_f64_16)) <= 2  # At most ±1 ulp from center
    end

    @testset "BigFloat Points Remain Distinct" begin
        setprecision(BigFloat, 256)

        center = Complex{BigFloat}(BigFloat("1.5"), BigFloat("0.0"))
        radius = BigFloat("1e-18")
        num_points = 8

        θs = [BigFloat(2π * k / num_points) for k in 0:num_points-1]
        zs_bf = [center + radius * exp(im * θ) for θ in θs]

        # All BigFloat points should be distinct
        @test length(unique(zs_bf)) == num_points

        # But Float64 may not distinguish them all
        zs_f64 = Complex{Float64}.(zs_bf)
        # Note: complex encoding may preserve some distinction via imaginary part
    end

    @testset "Ogita Distinguishes Sub-ε Points" begin
        # This is the key test: can we get distinct σ_min for points
        # that are indistinguishable in Float64?
        setprecision(BigFloat, 256)

        # Create a simple test matrix
        n = 10
        T = randn(n, n) + 3.0 * I  # Shift to ensure invertibility
        T_bf = convert.(Complex{BigFloat}, T)

        # Center near spectrum but not on eigenvalue
        λ_max = maximum(abs.(eigvals(T)))
        center_bf = Complex{BigFloat}(BigFloat(string(λ_max + 0.1)), BigFloat("0.0"))

        # Sub-ε radius
        radius_bf = BigFloat("1e-17")
        num_points = 4

        θs = [BigFloat(2π * k / num_points) for k in 0:num_points-1]
        zs_bf = [center_bf + radius_bf * exp(im * θ) for θ in θs]

        # Get Float64 SVD at center (same for all points in Float64 view)
        A_center_f64 = Complex{Float64}.(T_bf) - Complex{Float64}(center_bf) * I
        U_init, S_init, V_init = svd(A_center_f64)

        # Certify each point
        σ_mins = BigFloat[]
        for z in zs_bf
            A_bf = T_bf - z * I

            U_bf = convert.(Complex{BigFloat}, U_init)
            S_bf = convert.(BigFloat, S_init)
            V_bf = convert.(Complex{BigFloat}, V_init)

            result = ogita_svd_refine(A_bf, U_bf, S_bf, V_bf;
                                      max_iterations=5, precision_bits=256)

            σ_min = result.Σ[end, end] - result.residual_norm
            push!(σ_mins, σ_min)
        end

        # All should be positive (matrix is invertible)
        @test all(σ_mins .> 0)

        # Key test: σ_min values should be numerically distinct in BigFloat
        # Even though radius is 1e-17, the differences should be resolvable
        # at 256-bit precision
        rel_diffs = [abs(σ_mins[i] - σ_mins[j]) / min(σ_mins[i], σ_mins[j])
                     for i in 1:num_points for j in i+1:num_points]
        @test any(rel_diffs .> BigFloat("1e-30"))  # Some pairs should differ detectably
    end

    @testset "Precision Requirements" begin
        # Test that sufficient BigFloat precision resolves sub-ε differences
        center = BigFloat("1.5")
        Δ = BigFloat("1e-18")

        for bits in [64, 128, 256]
            setprecision(BigFloat, bits)
            z1 = BigFloat("1.5")
            z2 = z1 + BigFloat("1e-18")

            # At 64 bits (~19 decimal digits), 1e-18 should be resolvable
            @test z1 != z2
            @test abs(z2 - z1) > BigFloat(0)
        end

        setprecision(BigFloat, 256)  # Reset
    end

    @testset "Convergence at Sub-ε" begin
        # Test that Ogita refinement converges even when starting from
        # center's Float64 SVD applied to a slightly different matrix
        setprecision(BigFloat, 256)

        n = 8
        T = Diagonal(collect(1.0:n)) + 0.1 * randn(n, n)  # Simple matrix
        T_bf = convert.(Complex{BigFloat}, T)

        # Two points that are identical in Float64
        z1 = Complex{BigFloat}(BigFloat("5.5"), BigFloat("0.0"))
        z2 = z1 + BigFloat("1e-20")

        @test Complex{Float64}(z1) == Complex{Float64}(z2)  # Same in Float64

        # Get shared initial SVD
        A_f64 = Complex{Float64}.(T_bf) - Complex{Float64}(z1) * I
        U_init, S_init, V_init = svd(A_f64)

        # Refine both
        results = []
        for z in [z1, z2]
            A_bf = T_bf - z * I
            U_bf = convert.(Complex{BigFloat}, U_init)
            S_bf = convert.(BigFloat, S_init)
            V_bf = convert.(Complex{BigFloat}, V_init)

            result = ogita_svd_refine(A_bf, U_bf, S_bf, V_bf;
                                      max_iterations=5, precision_bits=256)
            push!(results, result)
        end

        # Both should converge (small residuals)
        @test Float64(results[1].residual_norm) < 1e-50
        @test Float64(results[2].residual_norm) < 1e-50

        # σ_min should be slightly different
        σ1 = results[1].Σ[end, end]
        σ2 = results[2].Σ[end, end]
        @test abs(σ1 - σ2) > BigFloat(0)
    end

    @testset "Bound Certification" begin
        # Test that bounds are finite and meaningful
        setprecision(BigFloat, 256)

        n = 10
        T = randn(n, n) + 5.0 * I  # Well-conditioned
        T_bf = convert.(Complex{BigFloat}, T)

        center_bf = Complex{BigFloat}(BigFloat("6.0"), BigFloat("0.0"))
        radius_bf = BigFloat("1e-16")  # Machine precision radius
        num_points = 8

        θs = [BigFloat(2π * k / num_points) for k in 0:num_points-1]
        zs_bf = [center_bf + radius_bf * exp(im * θ) for θ in θs]

        A_center_f64 = Complex{Float64}.(T_bf) - Complex{Float64}(center_bf) * I
        U_init, S_init, V_init = svd(A_center_f64)

        bounds = Float64[]
        for z in zs_bf
            A_bf = T_bf - z * I
            U_bf = convert.(Complex{BigFloat}, U_init)
            S_bf = convert.(BigFloat, S_init)
            V_bf = convert.(Complex{BigFloat}, V_init)

            result = ogita_svd_refine(A_bf, U_bf, S_bf, V_bf;
                                      max_iterations=5, precision_bits=256)

            σ_min = result.Σ[end, end] - result.residual_norm
            bound = σ_min > 0 ? 1/Float64(σ_min) : Inf
            push!(bounds, bound)
        end

        # All bounds should be finite
        @test all(isfinite.(bounds))

        # Bounds should be reasonable (not astronomically large for well-conditioned matrix)
        @test all(bounds .< 1e10)
    end
end

println("All sub-ε certification tests passed!")
