"""
Test suite for GenericLinearAlgebra SVD extension.

Tests the native BigFloat SVD provided by GenericLinearAlgebra.jl.

NOTE: This test requires GenericLinearAlgebra.jl to be installed:
    ] add GenericLinearAlgebra
If not installed, the test is skipped (marked as broken in test summary).
"""

using Test
using BallArithmetic
using LinearAlgebra
using Random

# Check if GLA extension is available
const HAS_GLA = hasmethod(svd_bigfloat, Tuple{Matrix{BigFloat}})

@testset "GenericLinearAlgebra SVD" begin
    if !HAS_GLA
        # OPTIONAL DEPENDENCY: GenericLinearAlgebra.jl provides native BigFloat
        # SVD without LAPACK, achieving higher precision than Float64-based methods.
        # Install with: ] add GenericLinearAlgebra
        @warn "svd_bigfloat not available (GenericLinearAlgebra extension not loaded)"
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

end

println("All GenericLinearAlgebra SVD tests passed!")
