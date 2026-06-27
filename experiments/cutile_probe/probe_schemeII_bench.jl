# Where does Scheme II time go, and how does it compare to Scheme I?
# Scheme II uses 14 GEMMs (vs Scheme I's 43) but a heavier reconstruction
# (Garner + Int128, here on CPU). Measure GPU-GEMM portion vs CPU-reconstruction
# vs full, against Scheme I and native FP64.

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const MODULI  = Int[256,255,253,251,247,239,233,229,227,223,217,211,199,197,193]
const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS = 7; const BASE = 2.0^B_BITS

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))
residue_i8(Ap, m) = (r = mod.(Ap, Int64(m)); Int8.(ifelse.(2 .* r .>= m, r .- m, r)))

# Scheme II GPU portion: returns the s residue matrices on CPU.
function schemeII_gemms(dA, dB, σA, σB; s=14, k=53)
    M_, N_ = size(dA,1), size(dB,2)
    Ap = Int64.(round.((2.0^k) .* (dA .* (2.0 .^ (.-σA)))))
    Bp = Int64.(round.((2.0^k) .* (dB .* (2.0 .^ (.-σB)))))
    R = Vector{Matrix{Int64}}(undef, s); Cij = CUDA.zeros(Int32, M_, N_)
    for (t,m) in enumerate(MODULI[1:s])
        At = residue_i8(Ap, m); Bt = residue_i8(Bp, m)
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), At, Bt, Int32(0), Cij)
        R[t] = Int64.(Array(mod.(Cij, Int32(m))))
    end
    return R
end

function garner_cpu(R, σA, σB; s=14, k=53)
    mods = MODULI[1:s]
    minv = [j < t ? invmod(mods[j], mods[t]) : 0 for j in 1:s, t in 1:s]
    c = Vector{Matrix{Int64}}(undef, s)
    for t in 1:s
        ct = copy(R[t])
        for j in 1:t-1; ct = mod.((ct .- c[j]) .* minv[j,t], mods[t]); end
        c[t] = ct
    end
    Mbig = prod(Int128.(mods)); X = Int128.(c[s])
    for t in s-1:-1:1; X = X .* Int128(mods[t]) .+ c[t]; end
    X = ifelse.(X .>= Mbig ÷ 2, X .- Mbig, X)
    return Float64.(X) .* (2.0^(-2k)) .* σA .* σB
end

# Scheme I (slice), GPU only, for reference.
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s); t = copy(Ahat)
    for i in 1:s; t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di; end
    return D
end
function schemeI(dA, dB; s=8, T=10)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    DA = slices(dA .* (2.0 .^ (.-σA)), s); DB = slices(dB .* (2.0 .^ (.-σB)), s)
    M_, N_ = size(dA,1), size(dB,2); S = CUDA.zeros(Float64, M_, N_); Cij = CUDA.zeros(Int32, M_, N_)
    for i in 1:s, j in 1:s
        i+j <= T || continue
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j)))
    end
    return (S .* (2.0 .^ σA)) .* (2.0 .^ σB)
end

gtime(f;nw=2,nr=6)=(for _ in 1:nw;f();CUDACore.synchronize();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();CUDACore.synchronize();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])
ctime(f;nw=1,nr=3)=(for _ in 1:nw;f();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])

function run(M, N, K)
    rng = MersenneTwister(1)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    σAc = Array(2.0 .^ σA); σBc = Array(2.0 .^ σB)

    t_fp64 = gtime(() -> dA * dB)
    t_I    = gtime(() -> schemeI(dA, dB; s=8, T=10))            # 43 gemms, full
    t_II_g = gtime(() -> schemeII_gemms(dA, dB, σA, σB; s=14))  # 14 gemms only
    R = schemeII_gemms(dA, dB, σA, σB; s=14)
    t_II_r = ctime(() -> garner_cpu(R, σAc, σBc; s=14))         # CPU reconstruction
    @printf("  %d^3\n", M)
    @printf("    native FP64 gemm                 %8.2f ms\n", t_fp64*1e3)
    @printf("    Scheme I  (43 gemms, full)       %8.2f ms\n", t_I*1e3)
    @printf("    Scheme II GPU GEMMs (14)         %8.2f ms   (%.2fx Scheme I)\n", t_II_g*1e3, t_I/t_II_g)
    @printf("    Scheme II CPU reconstruction     %8.2f ms\n", t_II_r*1e3)
    @printf("    Scheme II total (GEMM+recon)     %8.2f ms\n", (t_II_g+t_II_r)*1e3)
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
for s in [1024, 2048]; run(s, s, s); end
