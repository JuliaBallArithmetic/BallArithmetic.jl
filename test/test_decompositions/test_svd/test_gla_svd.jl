"""
Test suite for GenericLinearAlgebra SVD extension.

Tests the native BigFloat SVD provided by GenericLinearAlgebra.jl,
which is faster and more accurate than cascade refinement methods.

NOTE: This test requires GenericLinearAlgebra.jl to be installed:
    ] add GenericLinearAlgebra
If not installed, the test is skipped (marked as broken in test summary).
"""

using Test
using BallArithmetic
using LinearAlgebra
using Random

# Check if GLA extension is available
const HAS_GLA = hasmethod(ogita_svd_cascade_gla, Tuple{Matrix{Complex{BigFloat}}, Complex{BigFloat}})

@testset "GenericLinearAlgebra SVD" begin
    if !HAS_GLA
        # OPTIONAL DEPENDENCY: GenericLinearAlgebra.jl provides native BigFloat
        # SVD without LAPACK, achieving higher precision than Float64-based methods.
        # Install with: ] add GenericLinearAlgebra
        @warn "ogita_svd_cascade_gla not available (GenericLinearAlgebra extension not loaded)"
        @test_skip true
        return
    end

    @testset "svd_bigfloat basic" begin
        Random.seed!(42)
        setprecision(BigFloat, 256)

        A = BigFloat.(randn(20, 20))
        F = svd_bigfloat(A)

        @test size(F.U) == (20, 20)
        @test length(F.S) == 20
        @test size(F.Vt) == (20, 20)

        # Check reconstruction
        residual = A - F.U * Diagonal(F.S) * F.Vt
        @test maximum(abs.(residual)) < 1e-70
    end

    @testset "ogita_svd_cascade_gla basic" begin
        Random.seed!(42)
        setprecision(BigFloat, 256)

        n = 30
        T = randn(n, n) + 5.0 * I
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade_gla(T_bf, z_bf)

        @test result isa PrecisionCascadeSVDResult
        @test size(result.U) == (n, n)
        @test length(result.Σ) == n
        @test size(result.V) == (n, n)
        @test result.σ_min > 0
        @test Float64(result.residual_norm) < 1e-70
    end

    @testset "GLA accuracy" begin
        Random.seed!(123)
        setprecision(BigFloat, 256)

        n = 25
        T = randn(n, n) + 5.0 * I
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade_gla(T_bf, z_bf)

        # GLA should achieve very high accuracy (residual < 1e-70)
        @test Float64(result.residual_norm) < 1e-70

        # Orthogonality check
        I_n = Matrix{Complex{BigFloat}}(I, n, n)
        U_orthog = maximum(abs.(result.U' * result.U - I_n))
        V_orthog = maximum(abs.(result.V' * result.V - I_n))
        @test Float64(U_orthog) < 1e-70
        @test Float64(V_orthog) < 1e-70
    end

    @testset "GLA with refinement" begin
        Random.seed!(456)
        setprecision(BigFloat, 256)

        n = 20
        T = randn(n, n) + 5.0 * I
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        # Without refinement (default)
        r0 = ogita_svd_cascade_gla(T_bf, z_bf; refine_iterations=0)

        # With refinement
        r2 = ogita_svd_cascade_gla(T_bf, z_bf; refine_iterations=2)

        # Both should give valid results
        @test r0.σ_min > 0
        @test r2.σ_min > 0

        # σ_min should be similar
        @test abs(Float64(r0.σ_min) - Float64(r2.σ_min)) / Float64(r0.σ_min) < 1e-10
    end

    @testset "Singular value ordering" begin
        Random.seed!(789)
        setprecision(BigFloat, 256)

        n = 15
        T = randn(n, n) + 5.0 * I
        T_bf = Complex{BigFloat}.(T)
        z_bf = Complex{BigFloat}(6.0, 0.0)

        result = ogita_svd_cascade_gla(T_bf, z_bf)

        # Singular values should be in decreasing order
        for i in 1:n-1
            @test result.Σ[i] >= result.Σ[i+1]
        end
    end

end

println("All GenericLinearAlgebra SVD tests passed!")
