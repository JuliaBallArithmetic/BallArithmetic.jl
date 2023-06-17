using BallArithmetic
using Test

@testset "BallArithmetic.jl" begin
    include("test_ball/test_ball.jl")
    include("test_matrix/test_constructors.jl")
    include("test_matrix/test_algebra.jl")
    include("test_eigen/test_eigen.jl")
    include("test_svd/test_svd.jl")
end
