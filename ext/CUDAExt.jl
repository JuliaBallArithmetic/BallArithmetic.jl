# CUDAExt — GPU rigorous BallMatrix product (MMul4) on NVIDIA INT8 tensor cores.
#
# This extension is a Julia / CUDA.jl port of RIKEN R-CCS's GEMMul8
# (https://github.com/RIKEN-RCCS/GEMMul8, MIT License, © 2025- RIKEN R-CCS),
# based on the Ozaki Scheme II (Ozaki, Uchino, Imamura, arXiv:2504.08009).
# It computes the midpoint product via exact INT8-tensor-core GEMMs + a fused
# CRT reconstruction, and a rigorous radius (also on INT8 tensor cores).
#
# Original GEMMul8 license (MIT) — reproduced per its terms:
#   MIT License. Copyright (c) 2025- RIKEN R-CCS.
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY ...
#   (full text in the upstream repository).

module CUDAExt

using BallArithmetic
using BallArithmetic: BallMatrix, Ball, mid, rad,
    SVDMethod, MiyajimaM1, RigorousSVDResult,
    upper_bound_L2_opnorm, miyajima_vbd
using CUDA
using LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!          # INT8×INT8 -> INT32 (CUBLAS_COMPUTE_32I)
# Coprime moduli in [191,256]; M = ∏ grows ~2^(7.97 s). s=16 ⇒ M~2^127.5, which
# guarantees the exact integer product (M > 2q) for any K up to ~6M at k=53.
const MODULI = Int[256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191]

# Storage types accepted by the GPU MMul4.  `adjoint(::BallMatrix)` stores the
# midpoint/radius as `Adjoint{T,<:CuArray}` (and `transpose` as `Transpose`),
# which are NOT `<:CuArray`; without matching them the dispatch would silently
# fall back to the generic CPU `MMul4` (whose `setrounding` is a no-op on the GPU,
# so that path is not even rigorous).  We match them here and densify below.
const GPUMat{T} = Union{CUDA.CuArray{T, 2},
    LinearAlgebra.Adjoint{T, <:CUDA.CuArray{T, 2}},
    LinearAlgebra.Transpose{T, <:CUDA.CuArray{T, 2}}}

# Materialize to a dense column-major CuArray.  For an Adjoint/Transpose this is a
# GPU transpose copy — an exact data reshuffle (no rounding), so rigor is preserved.
# The INT8 residue/reconstruction kernels index linearly and cannot operate on a
# lazy Adjoint/Transpose wrapper, so densification is required, not just convenient.
_dense(x::CUDA.CuArray) = x
_dense(x::Union{LinearAlgebra.Adjoint, LinearAlgebra.Transpose}) = CUDA.CuArray(x)

# cuBLAS's INT8 IMMA `gemmEx` (CUBLAS_COMPUTE_32I on the tensor cores) requires
# every dimension to be a multiple of 16.  We zero-pad the operands up to the next
# multiple of 16 and slice the result back: padding the contraction/output
# dimensions with exact zeros leaves the midpoint product unchanged and only
# inflates the (already upper) truncation/radius bounds (the bound uses the padded
# `K`), so rigor is preserved.
const INT8_TILE = 16
_next4(n::Int) = cld(n, INT8_TILE) * INT8_TILE
function _pad4(X::CUDA.CuArray{T, 2}, r::Int, c::Int) where {T}
    (size(X, 1) == r && size(X, 2) == c) && return X
    Y = CUDA.zeros(T, r, c)
    @views Y[1:size(X, 1), 1:size(X, 2)] .= X
    return Y
end

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))

#============================ midpoint (GEMMul8 port) ========================#

# Fused residue extraction (port of GEMMul8 extract_A_lo): read each element of
# X once, form A' = round(X·2^(k-σ)), write all s balanced-INT8 residues.
function _residues_kernel!(lo, X, sft, mods, nrows, sz, s, rowwise, k)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= sz
        @inbounds begin
            r = (idx - 1) % nrows + 1
            c = (idx - 1) ÷ nrows + 1
            e = rowwise ? sft[r] : sft[c]
            ap = round(X[idx] * exp2(Float64(k) - e))
            for i in 1:s
                m  = mods[i]
                rr = ap - m * floor(ap / m)
                rr = rr < 0 ? rr + m : rr
                rr = rr >= m ? rr - m : rr
                rb = (2 * rr >= m) ? rr - m : rr
                lo[idx + (i - 1) * sz] = unsafe_trunc(Int8, rb)
            end
        end
    end
    return nothing
end

function _residues!(lo, X, sft, mods_d, nrows, s, k, rowwise)
    sz = length(X); threads = 256; blocks = cld(sz, threads)
    @cuda threads=threads blocks=blocks _residues_kernel!(lo, X, sft, mods_d, nrows, sz, s, rowwise, k)
    return lo
end

# Fused direct-CRT reconstruction (port of GEMMul8 invscal_device): precomputed
# double-double weights qPi, error-free accumulation, balanced rint reduction.
function _invscal_kernel!(C, Cmid, qhi, qlo, Px, Py, invP, scaleA, scaleB, M, sizeC, s)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= sizeC
        col = (idx - 1) ÷ M + 1
        row = (idx - 1) % M + 1
        chi = 0.0; clo = 0.0
        @inbounds for i in 1:s
            a = Float64(Cmid[idx + (i-1)*sizeC])
            chi = fma(qhi[i], a, chi); clo = fma(qlo[i], a, clo)
        end
        quot = round(invP * chi)
        crt  = fma(Py, quot, fma(Px, quot, chi) + clo)
        @inbounds C[idx] = crt * scaleA[row] * scaleB[col]
    end
    return nothing
end

# Precompute double-double CRT weights with uniform-absolute (error-free) hi part.
function _crt_weights(mods)
    s = length(mods); qhi = zeros(s); qlo = zeros(s)
    P = prod(BigInt.(mods))
    rho = sum(div(m, 2) for m in mods)
    L = (ndigits(P, base=2) - 1) - 52 + ceil(Int, log2(rho))
    twoL = BigInt(1) << L
    for i in 1:s
        Pi = P ÷ mods[i]; qi = invmod(mod(Pi, mods[i]), mods[i]); qPi = qi * Pi
        n = fld(qPi + (twoL >> 1), twoL)
        qhi[i] = Float64(n) * 2.0^L; qlo[i] = Float64(qPi - n * twoL)
    end
    Px = Py = 0.0
    setprecision(BigFloat, 400) do
        Pn = -BigFloat(P); Px = Float64(Pn); Py = Float64(Pn - Px)
    end
    return qhi, qlo, Px, Py, Float64(1 / BigFloat(P))
end

# Exact-CRT midpoint product mA·mB (Float64), via s INT8 GEMMs + fused recon.
function _midpoint(dmA, dmB, σA, σB; s=16, k=53)
    M_, N_, K_ = size(dmA,1), size(dmB,2), size(dmA,2); sizeC = M_*N_
    mods = MODULI[1:s]; mods_d = CuArray(Int32.(mods))
    Alo = CuArray{Int8}(undef, M_, K_, s); Blo = CuArray{Int8}(undef, K_, N_, s)
    _residues!(Alo, dmA, vec(σA), mods_d, M_, s, k, true)
    _residues!(Blo, dmB, vec(σB), mods_d, K_, s, k, false)
    Cmid = CUDA.zeros(Int32, sizeC, s); Cij = CUDA.zeros(Int32, M_, N_)
    for (i,m) in enumerate(mods)
        At = @view Alo[:,:,i]; Bt = @view Blo[:,:,i]
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), At, Bt, Int32(0), Cij)
        @views Cmid[:, i] .= vec(mod.(Cij, Int32(m)))
    end
    qhi, qlo, Px, Py, invP = _crt_weights(mods)
    m_out = CUDA.zeros(Float64, M_, N_)
    scaleA = vec(2.0 .^ (σA .- k)); scaleB = vec(2.0 .^ (σB .- k))
    @cuda threads=256 blocks=cld(sizeC,256) _invscal_kernel!(
        m_out, Cmid, CuArray(qhi), CuArray(qlo), Px, Py, invP, scaleA, scaleB, M_, sizeC, s)
    return m_out
end

#============================ radius (INT8 upper bound) =======================#

# Rigorous elementwise UPPER bound on U·V (U,V >= 0) via ONE exact INT8 GEMM
# with ceil-rounded 7-bit slices.  We only need an upper bound, so the slice
# truncation is fine — and it runs on the INT8 tensor cores (no rounding control
# needed, unlike TF32).
function _int8_upper_product(dU, dV)
    M_ = size(dU, 1); N_ = size(dV, 2)
    mxU = maximum(dU; dims=2); mxV = maximum(dV; dims=1)
    σU = ifelse.(mxU .== 0, 0.0, ceil.(log2.(max.(mxU, floatmin(Float64)))) .+ 1.0)
    σV = ifelse.(mxV .== 0, 0.0, ceil.(log2.(max.(mxV, floatmin(Float64)))) .+ 1.0)
    U8 = Int8.(ceil.(dU .* (2.0 .^ (7 .- σU))))
    V8 = Int8.(ceil.(dV .* (2.0 .^ (7 .- σV))))
    C  = CUDA.zeros(Int32, M_, N_)
    gemmI8!('N','N', Int32(1), U8, V8, Int32(0), C)
    return (Float64.(C) .* (2.0 .^ (σU .- 7)) .* (2.0 .^ (σV .- 7))) .* (1 + 4eps())
end

#================================ MMul4 dispatch =============================#

# Full rigorous ball product on the GPU.  C = (m, rC) with
#   m       = exact-CRT INT8 midpoint of mA·mB,
#   rC      = ρ_trunc (midpoint input-scaling truncation) + rC_in (input radii),
#             both rigorous upper bounds (INT8 / row-col-sum), rounded up.
function BallArithmetic.MMul4(
        A::BallMatrix{T, T, BTA, CMA, RMA},
        B::BallMatrix{T, T, BTB, CMB, RMB}) where {
            T <: Float64, BTA, BTB,
            CMA <: GPUMat{T}, CMB <: GPUMat{T},
            RMA <: GPUMat{T}, RMB <: GPUMat{T}}
    k = 53; s = 16
    u = eps(Float64)
    # densify (no-op for plain CuArray; exact GPU transpose for Adjoint/Transpose)
    mA, rA = _dense(mid(A)), _dense(rad(A)); mB, rB = _dense(mid(B)), _dense(rad(B))
    # pad to multiples of 4 for the INT8 GEMM; remember the true output size
    M0 = size(mA, 1); N0 = size(mB, 2)
    Mp = _next4(M0); Kp = _next4(size(mA, 2)); Np = _next4(N0)
    padded = (Mp, Kp, Np) != (M0, size(mA, 2), N0)
    if padded
        mA = _pad4(mA, Mp, Kp); rA = _pad4(rA, Mp, Kp)
        mB = _pad4(mB, Kp, Np); rB = _pad4(rB, Kp, Np)
    end
    K_ = size(mA, 2)
    σA = scale_exp(mA; dims=2); σB = scale_exp(mB; dims=1)

    m = _midpoint(mA, mB, σA, σB; s=s, k=k)

    # midpoint truncation bound: |δA| <= 2^(σA-k-1) is per-row constant -> no gemm
    absA = abs.(mA); absB = abs.(mB)
    rowsumA = sum(absA; dims=2) .* (1 + K_*u)
    colsumB = sum(absB; dims=1) .* (1 + K_*u)
    δA = 2.0 .^ (σA .- k .- 1); δB = 2.0 .^ (σB .- k .- 1)
    ρ_trunc = (δA .* colsumB) .+ (rowsumA .* δB) .+ (Float64(K_) .* (δA .* δB))

    # input-radius propagation: |mA|·rB + rA·(|mB|+rB), via INT8 upper products
    rC = ρ_trunc .+ (u .* abs.(m))
    if any(!iszero, rA) || any(!iszero, rB)
        rC = rC .+ _int8_upper_product(absA, rB) .+ _int8_upper_product(rA, absB .+ rB)
    end
    rC = rC .* (1 + 8u)
    if padded
        m  = CUDA.CuArray(@view m[1:M0, 1:N0])
        rC = CUDA.CuArray(@view rC[1:M0, 1:N0])
    end
    return BallMatrix(m, rC)
end

#============================ GPU rigorous SVD ===============================#

# Wrap a host/device matrix as a zero-radius GPU ball matrix.
_gpu_ball(c::CUDA.CuArray) = BallMatrix(c, CUDA.zeros(Float64, size(c)...))
_gpu_ball(c) = BallMatrix(CUDA.CuArray(c), CUDA.zeros(Float64, size(c)...))
# Build a GPU ball matrix from explicit (midpoint, radius) host/device matrices.
_gpu_ball(m, r) = BallMatrix(CUDA.CuArray(m), CUDA.CuArray(r))
# Copy a GPU ball matrix back to the CPU.
_to_cpu(B::BallMatrix) = BallMatrix(Array(mid(B)), Array(rad(B)))

# GPU-accelerated rigorous SVD.  The `O(n³)` certification products run on the
# GPU `MMul4`; the `O(n²)` rigorous finish (norm bounds, Neumann gate,
# singular-value enclosures, optional VBD) runs on the CPU with directed
# rounding, exactly as in the CPU `rigorous_svd`.  See `rigorous_svd_gpu`.
function BallArithmetic.rigorous_svd_gpu(A::BallMatrix{Float64, Float64};
        method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = true,
        seed_on::Symbol = :cpu, alg = CUDA.CUSOLVER.QRAlgorithm())
    T = Float64
    mA = mid(A)

    # --- seed factorisation (approximate; only a starting point) -----------
    Smid, Uc, Vc, Vtc = if seed_on == :gpu
        Fg = svd(CUDA.CuArray(mA); alg = alg)          # cuSOLVER, stays on device
        (Array(Fg.S), Fg.U, Fg.V, Fg.Vt isa CUDA.CuArray ? Fg.Vt : CUDA.CuArray(Fg.Vt))
    elseif seed_on == :cpu
        F = svd(mA)                                    # CPU LAPACK, then upload
        (F.S, Matrix(F.U), Matrix(F.V), Matrix(F.Vt))
    else
        throw(ArgumentError("seed_on must be :cpu or :gpu, got $(seed_on)"))
    end

    Ug  = _gpu_ball(Uc)
    Vg  = _gpu_ball(Vc)
    Vtg = _gpu_ball(Vtc)
    Σg  = _gpu_ball(Matrix(Diagonal(Smid)))

    # --- O(n³) certification products on the GPU (rigorous MMul4) ----------
    P1 = Ug * Σg * Vtg        # ≈ A
    P2 = Vtg * Vg             # ≈ I  (right orthogonality)
    P3 = Ug' * Ug             # ≈ I  (left orthogonality)
    CUDA.synchronize()

    # --- O(n²) rigorous finish on the CPU (directed rounding) --------------
    E = _to_cpu(P1) - A
    Fdef = _to_cpu(P2) - I
    Gdef = _to_cpu(P3) - I
    normE = upper_bound_L2_opnorm(E)
    normF = upper_bound_L2_opnorm(Fdef)
    normG = upper_bound_L2_opnorm(Gdef)

    # CPU ball factors for the returned result.
    U  = BallMatrix(Array(Uc))
    V  = BallMatrix(Array(Vc))

    # Neumann gate: ‖F‖ < 1 and ‖G‖ < 1 are required for the bounds.
    if normF >= 1 || normG >= 1
        n_sv = length(Smid)
        singular_values = [Ball(Smid[i], T(Inf)) for i in 1:n_sv]
        Σ = BallArithmetic._diagonal_ball_matrix(singular_values)
        @warn "GPU SVD verification failed: ‖V'V - I‖ = $normF, ‖U'U - I‖ = $normG (both must be < 1)"
        return RigorousSVDResult(U, singular_values, Σ, V, E,
            T(Inf), normF, normG, nothing)
    end

    down, up = BallArithmetic._compute_svd_bounds(method, Smid, normE, normF, normG, T)
    mids = (down .+ up) ./ 2
    radii = setrounding(T, RoundUp) do
        [max(up[i] - mids[i], mids[i] - down[i]) for i in eachindex(mids)]
    end
    singular_values = [Ball(mids[i], radii[i]) for i in eachindex(mids)]
    Σ = BallArithmetic._diagonal_ball_matrix(singular_values)

    # Residual `U Σ Vᵀ - A` reusing the midpoint residual `E`, accounting only
    # for the widening from `Σ_mid` to the ball diagonal `Σ`.  The `O(n³)`
    # correction product `U ΔΣ Vᵀ` is formed on the GPU.
    ΔΣg = _gpu_ball(Matrix(Diagonal(mids .- Smid)), Matrix(Diagonal(radii)))
    corr = _to_cpu(Ug * ΔΣg * Vtg)
    CUDA.synchronize()
    residual = E + corr
    residual_norm = upper_bound_L2_opnorm(residual)

    vbd = apply_vbd ? miyajima_vbd(adjoint(Σ) * Σ; hermitian = true) : nothing

    return RigorousSVDResult(U, singular_values, Σ, V, residual,
        residual_norm, normF, normG, vbd)
end

end # module
