using BallArithmetic
using Test

@testset "BallArithmetic.jl" begin
    include("test_ball/test_ball.jl")
    include("test_matrix/test_constructors.jl")
    include("test_matrix_classifier/test_matrix_classifier.jl")
    include("test_matrix/test_algebra.jl")
    include("test_eigen/test_eigen.jl")
    include("test_pseudospectra/test_pseudospectra.jl")
    include("test_fft/test_fft.jl")
    include("test_norm_bounds/test_norm_bounds.jl")
    include("test_svd/test_svd.jl")
    include("test_numerical_test/test_numerical_test.jl")
end
