# Verify the GPU MMul4 now fires for adjoint/transpose operands (U'·U etc.) and
# stays rigorous, rather than silently falling back to the (non-rigorous on GPU)
# generic CPU MMul4.
using BallArithmetic, CUDA, LinearAlgebra, Random, Printf
using BallArithmetic: mid, rad

# Is this BallMatrix product handled by CUDAExt's GPU MMul4?  We detect it via the
# method that `MMul4` resolves to: CUDAExt vs the generic src method.
gpu_method(A, B) = (m = which(BallArithmetic.MMul4, (typeof(A), typeof(B)));
    string(parentmodule(m)) == "CUDAExt")

function check(label, A, B)
    routed = gpu_method(A, B)
    C = A * B                                  # dispatches through MMul4
    onGPU = mid(C) isa CUDA.CuArray
    # rigor: ball must contain the BigFloat-exact midpoint product
    setprecision(BigFloat, 256) do
        trueC = (BigFloat.(Array(mid(A))) .+ 0) * (BigFloat.(Array(mid(B))))
        # account only for midpoint here (operands have small radii handled by MMul4)
        lo = BigFloat.(Array(mid(C))) .- BigFloat.(Array(rad(C)))
        hi = BigFloat.(Array(mid(C))) .+ BigFloat.(Array(rad(C)))
        contains = all(lo .<= trueC .<= hi)
        @printf("%-16s routed→CUDAExt=%-5s  resultOnGPU=%-5s  enclosesTrue=%-5s\n",
            label, routed, onGPU, contains)
    end
end

n = 256
rng = MersenneTwister(1)
M = randn(rng, n, n) ./ sqrt(n)
U = BallMatrix(CuArray(M), CUDA.fill(1e-14, n, n))
V = BallMatrix(CuArray(randn(rng, n, n) ./ sqrt(n)), CUDA.fill(1e-14, n, n))

println("Device: ", CUDA.name(CUDA.device()), "\n")
check("U * V",     U, V)            # plain × plain (baseline, already worked)
check("U' * V",    U', V)           # adjoint × plain
check("U * V'",    U, V')           # plain × adjoint
check("U' * U",    U', U)           # the SVD orthogonality-defect product
check("transpose", transpose(U), V) # transpose × plain
