module BallArithmetic

include("numerical_test/multithread.jl")

using LinearAlgebra

if Sys.ARCH == :x86_64
    using OpenBLASConsistentFPCSR_jll
else
    @warn "The behaviour of multithreaded OpenBlas on this architecture is unclear,
    we will fallback to single threaded OpenBLAS

    We refer to
    https://www.tuhh.de/ti3/rump/intlab/Octave/INTLAB_for_GNU_Octave.shtml
    "
end

function __init__()
    if Sys.ARCH == :x86_64
        @info "Switching to OpenBLAS with ConsistentFPCSR = 1 flag enabled, guarantees
        correct floating point rounding mode over all threads."
        BLAS.lbt_forward(OpenBLASConsistentFPCSR_jll.libopenblas_path; verbose = true)

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

export ±, mid, rad

include("rounding/rounding.jl")
include("types/ball.jl")
export Ball, BallF64, BallComplexF64, inf, sup

include("types/array.jl")
export BallArray

include("types/matrix.jl")
export BallMatrix
include("types/vector.jl")
export BallVector
include("types/triangularize.jl")

include("types/convertpromote.jl")
include("norm_bounds/rigorous_norm.jl")
export upper_bound_norm
include("norm_bounds/rigorous_opnorm_bounds.jl")
include("norm_bounds/oishi.jl")
include("norm_bounds/oishi_triangular.jl")

export upper_bound_L1_opnorm, upper_bound_L2_opnorm, upper_bound_L_inf_opnorm
include("eigenvalues/gev.jl")
include("eigenvalues/upper_bound_spectral.jl")
include("svd/svd.jl")
include("pseudospectra/rigorous_contour.jl")
include("matrix_classifiers/is_M_matrix.jl")

include("linear_system/inflation.jl")
include("linear_system/backward_substitution.jl")
include("linear_system/Krawczyk.jl")

end
