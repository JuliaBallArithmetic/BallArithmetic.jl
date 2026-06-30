# Tests for src/norm_bounds/oishi_triangular.jl

using Test
using LinearAlgebra
using BallArithmetic

@testset "Oishi-Rump Triangular Bounds" begin

    # NOTE: `oishi_rump_bound` is DISABLED (commented out in
    # src/norm_bounds/oishi_triangular.jl) because for block size k < n it can
    # return a value BELOW the true ‖inv(T)‖₂ (NON-rigorous — it lacks the
    # ‖offdiag·inv(diag)‖<1 precondition guard that rump_oishi_triangular has).
    # The tests below are marked broken until the missing guard/fallback lands.
    @testset "oishi_rump_bound (disabled — non-rigorous for k<n)" begin
        @test_broken isdefined(BallArithmetic, :oishi_rump_bound) &&
                     hasmethod(BallArithmetic.oishi_rump_bound, Tuple{BallMatrix, Int})
    end
end
