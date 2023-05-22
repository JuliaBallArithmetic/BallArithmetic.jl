module BallArithmetic

include("numerical_test/multithread.jl")

using LinearAlgebra

if Sys.ARCH == :x86_64
    using OpenBLASConsistentFPCSR_jll
else
    @warn "The behaviour of multithreaded OpenBlas on this architecture is unclear,
    we will fallback to single threaded OpenBLAS"
end

function  __init__()
    if Sys.ARCH == :x86_64
        @info "Switching to OpenBLAS with ConsistentFPCSR = 1 flag enabled, guarantees
        correct floating point rounding mode over all threads."
        BLAS.lbt_forward(OpenBLASConsistentFPCSR_jll.libopenblas_path; verbose =  true)
        
        N = BLAS.get_num_threads()
        K = 1024
        if NumericalTest.rounding_test(N, K)
            @info "OpenBLAS is giving correct rounding on a ($K,$K) test matrix on $N threads"
        else
            @warn "OpenBLAS is not rounding correctly on the test matrix"
            @warn "The number of BLAS threads was set to 1 to ensure rounding mode is consistent"    
            if !NumericalTest.rounding_test(1, K)
                @warn "The rounding test failed on 1 thread"
            end
        end
    else
        BLAS.set_num_threads(1)
        @warn "The number of BLAS threads was set to 1 to ensure rounding mode is consistent"
        if !NumericalTest.rounding_test(1, 1024)
            @warn "The rounding test failed on 1 thread"
        end
    end
end

using RoundingEmulator, MacroTools, SetRounding

export Ball, BallMatrix, ±

include("rounding.jl")
include("ball.jl")
include("matrix.jl")
include("norm_bounds/rigorous_norm_bounds.jl")
include("eigenvalues/gev.jl")
include("svd/svd.jl")


end
