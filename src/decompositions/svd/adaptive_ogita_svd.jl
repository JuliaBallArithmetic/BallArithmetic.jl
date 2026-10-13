"""
    adaptive_ogita_svd.jl

Adaptive precision SVD with the iterative refinement of

    T. Ogita and K. Aishima, "Iterative refinement for singular value decomposition based on
    matrix multiplication", J. Comput. Appl. Math. 369 (2020) 112512,

Algorithm 1 (RefSVD), which is stated there for a real `m × n` matrix with `m ≥ n`, simple
singular values and full `m × m` and `n × n` factors. The driver `adaptive_ogita_svd`, which
alternates that refinement with a certification at increasing precision, is ours:
1. Starts with low precision (Float64)
2. Computes rigorous SVD bounds
3. If bounds are not tight enough, refines the factors by Algorithm 1 at twice the precision
4. Certifies again at that precision, and repeats
"""

"""
    _spectral_norm_bound(A::AbstractMatrix)

The floating-point value of `√(‖A‖∞ ‖A‖₁)`, with `‖A‖∞` the largest row sum of `|A|` and `‖A‖₁`
the largest column sum. In exact arithmetic that quantity is an upper bound of `‖A‖₂`; here the
sums, the product and the root are rounded to nearest and no rounding is accounted for, so the
value returned is **an estimate and not a bound**. It is used for the stopping test of the
refinement and for the diagnostic `residual_norm` of its result, and for nothing that is stated.
An upper bound of the spectral norm is [`upper_bound_L2_opnorm`](@ref) on a ball matrix.
"""
function _spectral_norm_bound(A::AbstractMatrix{T}) where T
    m, n = size(A)

    # ‖A‖∞ = max row sum of absolute values
    inf_norm = zero(real(T))
    for i in 1:m
        row_sum = sum(abs(A[i, j]) for j in 1:n)
        inf_norm = max(inf_norm, row_sum)
    end

    # ‖A‖₁ = max column sum of absolute values
    one_norm = zero(real(T))
    for j in 1:n
        col_sum = sum(abs(A[i, j]) for i in 1:m)
        one_norm = max(one_norm, col_sum)
    end

    return sqrt(inf_norm * one_norm)
end

"""
    OgitaSVDRefinementResult

Result from Ogita's iterative SVD refinement.
"""
struct OgitaSVDRefinementResult{T, UT, ΣT, VT}
    U::UT                      # Refined left singular vectors
    Σ::ΣT                      # Refined singular values
    V::VT                      # Refined right singular vectors
    iterations::Int            # Number of refinement iterations performed
    precision_used::Int        # Precision (bits) used for refinement
    residual_norm::T           # floating-point estimate of ‖A − UΣV'‖₂ (`_spectral_norm_bound`), not a bound
    converged::Bool            # Whether refinement converged
end

"""
    AdaptiveSVDResult

Result from adaptive precision SVD computation.
"""
struct AdaptiveSVDResult{T, RigT}
    rigorous_result::RigT           # Final RigorousSVDResult
    precision_levels::Vector{Int}   # Precision levels attempted
    radii_history::Vector{T}        # Maximum radii at each level
    total_iterations::Int           # Total refinement iterations
    final_precision::Int            # Final precision used
    tolerance_achieved::Bool        # Whether tolerance was met
end

"""
    ogita_iterations_for_precision(target_bits::Int; start_bits::Int=53)

Compute the number of Ogita iterations needed to achieve target precision,
based on quadratic convergence theory.

With quadratic convergence, each iteration roughly doubles the number of correct bits.
Starting from Float64 (53 bits), we need ceil(log2(target_bits / start_bits)) iterations.

# Examples
- 256 bits: ceil(log2(256/53)) = 3 iterations
- 512 bits: ceil(log2(512/53)) = 4 iterations
- 1024 bits: ceil(log2(1024/53)) = 5 iterations
"""
function ogita_iterations_for_precision(target_bits::Int; start_bits::Int=53)
    if target_bits <= start_bits
        return 0
    end
    return ceil(Int, log2(target_bits / start_bits))
end

"""
    ogita_svd_refine(A::AbstractMatrix{T}, U, Σ, V;
                     max_iterations=10, precision_bits=256,
                     check_convergence=false,
                     use_optimal_iterations=false) where {T<:AbstractFloat}

Refine an approximate SVD using Ogita's iterative method (RefSVD algorithm).

# Arguments
- `A`: Original matrix (in higher precision if needed)
- `U`: Initial left singular vectors
- `Σ`: Initial singular values
- `V`: Initial right singular vectors
- `max_iterations`: Maximum number of iterations to run (default: 10)
- `precision_bits`: Working precision in bits (default: 256)
- `check_convergence`: If `true`, check convergence via spectral norm and stop early.
                       If `false` (default), run fixed number of iterations.
- `use_optimal_iterations`: If `true`, compute optimal iteration count from quadratic
                            convergence theory and ignore max_iterations.
                            This is the fastest option for known target precision.

# Iteration count (quadratic convergence from Float64)
Based on theory, starting from Float64 (~53 bits), each iteration doubles precision:
- 3 iterations: sufficient for 256-bit (~77 decimal digits)
- 4 iterations: sufficient for 512-bit (~154 decimal digits)
- 5 iterations: sufficient for 1024-bit (~308 decimal digits)

# Returns
- `OgitaSVDRefinementResult` containing refined SVD

# References
- [OgitaAishima2020](@cite) Ogita, T. & Aishima, K. (2020), "Iterative refinement for
  singular value decomposition based on matrix multiplication",
  J. Comput. Appl. Math. 369, 112512.
"""
function ogita_svd_refine(A::AbstractMatrix{T}, U, Σ, V;
                         max_iterations::Int=10,
                         precision_bits::Int=256,
                         check_convergence::Bool=false,
                         use_optimal_iterations::Bool=false) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    RT = real(T)  # Get the real type (Float64, BigFloat, etc.)

    # Compute optimal iterations if requested
    iterations_to_run = if use_optimal_iterations
        ogita_iterations_for_precision(precision_bits)
    else
        max_iterations
    end

    # Convert to higher precision if needed
    if RT == Float64 && precision_bits > 53
        # Need to work in BigFloat
        A_high = convert.(Complex{BigFloat}, A)
        U_high = convert.(Complex{BigFloat}, U)
        Σ_high = convert.(BigFloat, Σ)
        V_high = convert.(Complex{BigFloat}, V)

        # Set precision
        # a scoped setting: a global one has no effect inside a caller's scoped block
        return setprecision(BigFloat, precision_bits) do
            result = _ogita_svd_refine_impl(A_high, U_high, Σ_high, V_high, iterations_to_run, check_convergence)
            return result
        end
    else
        return _ogita_svd_refine_impl(A, U, Σ, V, iterations_to_run, check_convergence)
    end
end

"""
    _ogita_svd_refine_impl(A, U, Σ, V, max_iterations)

Internal implementation of Ogita's iterative SVD refinement.

Based on Algorithm 1 (RefSVD) from:
Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value
decomposition based on matrix multiplication", J. Comput. Appl. Math. 369, 112512.

The algorithm computes corrections F̃ and G̃ such that:
  U' = Û(I + F̃)
  V' = V̂(I + G̃)

This converges quadratically if singular values are well-separated.
"""
function _ogita_svd_refine_impl(A::AbstractMatrix{T}, U, Σ, V,
                                max_iterations::Int,
                                check_convergence::Bool) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    m, n = size(A)
    if m < n
        # the paper's reduction: the decomposition of A* has the factors exchanged
        r = _ogita_svd_refine_impl(Matrix(A'), V, Σ, U, max_iterations, check_convergence)
        return OgitaSVDRefinementResult(r.V, r.Σ, r.U, r.iterations, r.precision_used,
            r.residual_norm, r.converged)
    end
    thin = size(U, 2) < m
    if thin
        # Algorithm 1 needs the full m × m factor (Remark 3 of the paper: it does not work for
        # the thin decomposition). The columns are completed by an approximately orthonormal
        # basis of the complement, which the refinement then corrects with the rest.
        Uc = convert.(T, U)
        U = hcat(Uc, (qr(Uc).Q * Matrix{T}(I, m, m))[:, (size(Uc, 2) + 1):m])
    end
    r = _ogita_refsvd(A, U, Σ, V, max_iterations, check_convergence)
    thin || return r
    return OgitaSVDRefinementResult(r.U[:, 1:n], r.Σ, r.V, r.iterations, r.precision_used,
        r.residual_norm, r.converged)
end

# Algorithm 1 for m ≥ n with a full m × m factor U. For complex data the transposes of the paper
# are conjugate transposes, as (26) to (28) with F* and G*.
function _ogita_refsvd(A::AbstractMatrix{T}, U, Σ, V,
                       max_iterations::Int,
                       check_convergence::Bool) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    m, n = size(A)
    min_dim = min(m, n)
    RT = real(T)  # Real type for singular values

    # Current approximations - convert to match A's precision
    U_curr = convert.(T, U)
    Σ_curr = convert.(RT, Σ)
    V_curr = convert.(T, V)

    iterations = 0
    converged = false

    # Quadratic convergence guarantees: iterations ≈ ceil(log2(precision_bits / 15))
    # - 2 iterations: ~10^-60 (enough for 256-bit)
    # - 3 iterations: ~10^-120 (saturates 256-bit, enough for 512-bit)
    # - 4 iterations: ~10^-240 (enough for 1024-bit)
    # NOTE: This iterative refinement is an ORACLE computation. The accumulated
    # errors during iteration are NOT tracked here because the final result is
    # verified A POSTERIORI by computing rigorous residual bounds. The refinement
    # improves the approximation quality, but rigor comes from the final verification.
    for iter in 1:max_iterations
        iterations = iter
        # Step 1: Compute residual matrices (Algorithm 1, line 1)
        # R = I_m - U^T U (orthogonality residual for U)
        # S = I_n - V^T V (orthogonality residual for V)
        # T = U^T A V (should be diagonal if SVD is exact)
        R = I - U_curr' * U_curr
        S = I - V_curr' * V_curr
        T_matrix = U_curr' * A * V_curr

        # Step 2: Compute refined singular values (Algorithm 1, line 2)
        # Singular values are always real
        σ_tilde = zeros(RT, min_dim)
        for i in 1:min_dim
            denom = 1 - real(R[i,i] + S[i,i]) / 2
            σ_tilde[i] = real(T_matrix[i,i] / denom)
        end

        # Steps 3-7: Compute correction matrices F̃ and G̃
        F_tilde = zeros(T, m, m)
        G_tilde = zeros(T, n, n)

        # Step 3: Diagonal parts (Algorithm 1, line 3)
        for i in 1:n
            F_tilde[i,i] = R[i,i] / 2
            G_tilde[i,i] = S[i,i] / 2
        end

        # Step 4: Off-diagonal parts of F_11 and G (Algorithm 1, line 4)
        # From Ogita & Aishima 2020:
        #   α := T_{i,j} + σ̃_j · R_{i,j}
        #   β := conj(T_{j,i}) + σ̃_j · conj(S_{j,i})
        #   F̃_{i,j} := (α·σ̃_j + β·σ̃_i) / (σ̃_j² - σ̃_i²)
        #   G̃_{i,j} := (α·σ̃_i + β·σ̃_j) / (σ̃_j² - σ̃_i²)
        for i in 1:n
            for j in 1:n
                if i != j
                    denom = σ_tilde[j]^2 - σ_tilde[i]^2
                    # Skip if singular values are too close (clustered)
                    if abs(denom) > eps(RT) * max(σ_tilde[i], σ_tilde[j])^2
                        α = T_matrix[i,j] + σ_tilde[j] * R[i,j]
                        β = conj(T_matrix[j,i]) + σ_tilde[j] * conj(S[j,i])
                        F_tilde[i,j] = (α * σ_tilde[j] + β * σ_tilde[i]) / denom
                        G_tilde[i,j] = (α * σ_tilde[i] + β * σ_tilde[j]) / denom
                    end
                end
            end
        end

        # Step 5: F_12 block (Algorithm 1, line 5)
        if m > n
            for i in 1:n
                for j in (n+1):m
                    if abs(σ_tilde[i]) > eps(RT)
                        F_tilde[i,j] = -conj(T_matrix[j,i]) / σ_tilde[i]
                    end
                end
            end
        end

        # Step 6: F_21 block (Algorithm 1, line 6)
        if m > n
            for i in (n+1):m
                for j in 1:n
                    F_tilde[i,j] = R[i,j] - conj(F_tilde[j,i])
                end
            end
        end

        # Step 7: F_22 block (Algorithm 1, line 7)
        if m > n
            for i in (n+1):m
                for j in (n+1):m
                    F_tilde[i,j] = R[i,j] / 2
                end
            end
        end

        # Step 8: Update U and V (Algorithm 1, line 8)
        U_curr = U_curr * (I + F_tilde)
        V_curr = V_curr * (I + G_tilde)
        Σ_curr = σ_tilde

        # Optional convergence check based on correction matrix norms
        if check_convergence
            correction_norm = max(_spectral_norm_bound(F_tilde), _spectral_norm_bound(G_tilde))
            if correction_norm < eps(RT) * 100
                converged = true
                break
            end
        end
    end

    # Final singular value computation using the converged U and V
    # This is necessary because Σ_curr was computed BEFORE the last U,V update
    T_final = U_curr' * A * V_curr
    R_final = I - U_curr' * U_curr
    S_final = I - V_curr' * V_curr
    for i in 1:min_dim
        denom = one(RT) - real(R_final[i,i] + S_final[i,i]) / 2
        Σ_curr[i] = abs(T_final[i,i] / denom)  # Use abs() to get the magnitude
    end

    # Phase correction for complex matrices: ensure U'AV is real positive
    # If T_final[i,i] = σ_i * e^{iθ_i}, multiply U[:,i] by e^{iθ_i} so that
    # (new U[:,i])' * A * V[:,i] = σ_i (real positive)
    if T <: Complex
        for i in 1:min_dim
            if abs(T_final[i,i]) > eps(RT)
                phase = T_final[i,i] / abs(T_final[i,i])  # e^{iθ_i}
                U_curr[:, i] .*= phase  # U[:,i] ← U[:,i] * e^{iθ_i}
            end
        end
    end

    # Mark as converged if we ran all iterations (for fixed iteration mode)
    if !check_convergence
        converged = true
    end

    precision_used = T == Float64 ? 53 : precision(BigFloat)

    # Compute final residual norm for diagnostics (only once at the end)
    residual_norm = _spectral_norm_bound(A - U_curr[:, 1:n] * Diagonal(Σ_curr) * V_curr')

    return OgitaSVDRefinementResult(
        U_curr, Diagonal(Σ_curr), V_curr,
        iterations, precision_used, residual_norm, converged
    )
end


"""
    adaptive_ogita_svd(A::BallMatrix{T};
                       tolerance=1e-10,
                       method::SVDMethod=MiyajimaM1(),
                       apply_vbd=true,
                       max_precision_bits=1024,
                       max_refinement_iterations=5) where {T}

Compute rigorous SVD bounds with adaptive precision using Ogita's refinement.

# Algorithm
1. Start in the arithmetic of `A`: Float64, or BigFloat at the precision of the midpoints of `A`.
   The result for a BigFloat matrix is in BigFloat; for a Float64 matrix it is in Float64 when
   the tolerance is met at the first level and in BigFloat otherwise. Where a level works at a
   lower precision than an entry of `A` has, the entry is rounded and the rounding error goes
   into the radii, so that what is certified holds for `A`.
2. Compute rigorous SVD bounds using `rigorous_svd`
3. Check if max(rad(σᵢ)) < tolerance
4. If not satisfied, refine using Ogita's method with doubled precision
5. Repeat until tolerance met or max precision reached

# A matrix with non-zero radii
The enclosures must contain the singular values of every matrix of the ball, so their radii cannot
be brought below what the radii of `A` impose, at any precision. The singular values are Lipschitz
with constant 1 in the 2-norm, `|σᵢ(X) − σᵢ(M)| ≤ ‖X − M‖₂`, and `‖X − M‖₂ ≤ ‖R‖₂` when
`|X − M| ≤ R` entrywise; both are proved in the docstring of [`_miyajima2014_thm10_weyl`](@ref),
which uses the same step. The precision is therefore adapted on the midpoint of `A`, and
`tolerance`, `radii_history` and `tolerance_achieved` are about the enclosures of the singular
values of the midpoint. In `rigorous_result` each of those enclosures is widened by an upper bound
of `‖rad(A)‖₂`, so that it holds for every matrix of the ball; the residual there is `U Σ Vᴴ − A`
with that `Σ`.

# Arguments
- `A`: Input ball matrix
- `tolerance`: Target tolerance for max(rad(σᵢ)), of the midpoint when `A` has radii
- `method`: SVD certification method (MiyajimaM1 for Theorem 7, MiyajimaM4 for Theorem 11)
- `apply_vbd`: Whether to apply verified block diagonalization
- `max_precision_bits`: Maximum precision to use (bits)
- `max_refinement_iterations`: Maximum number of refinement steps

# Returns
- `AdaptiveSVDResult` containing final rigorous result and adaptation history

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-8, 2, 2))
result = adaptive_ogita_svd(A; tolerance=1e-12)

# Access final result
σ = result.rigorous_result.singular_values
println("Final precision: ", result.final_precision, " bits")
println("Max radius: ", maximum(rad.(σ)))
```
"""
function adaptive_ogita_svd(A::BallMatrix{T};
                           tolerance::Real=1e-10,
                           method::SVDMethod=MiyajimaM1(),
                           apply_vbd::Bool=true,
                           max_precision_bits::Int=1024,
                           max_refinement_iterations::Int=5) where {T}

    # A matrix with radii: the precision is adapted on the midpoint, whose singular values can be
    # enclosed as tightly as asked, and each enclosure is widened by an upper bound of ‖rad(A)‖₂
    # (`_weyl_widening`, the step that `_miyajima2014_thm10_weyl` applies to Theorem 10).
    if any(!iszero, rad(A))
        inner = adaptive_ogita_svd(BallMatrix(mid(A)); tolerance, method, apply_vbd,
            max_precision_bits, max_refinement_iterations)
        r0 = inner.rigorous_result
        p = inner.final_precision
        infloat64 = eltype(rad(r0.U)) === Float64
        widen() = begin
            Ap = infloat64 && T === Float64 ? A : _convert_to_precision(A, p)
            σ = _weyl_widening(r0.singular_values, rad(Ap))
            Σ = _diagonal_ball_matrix(σ)
            residual = r0.U * Σ * r0.V' - Ap
            vbd = apply_vbd && r0.block_diagonalisation !== nothing ?
                  schur_gershgorin_enclosure(adjoint(Σ) * Σ) : nothing
            RigorousSVDResult(r0.U, σ, Σ, r0.V, residual, upper_bound_L2_opnorm(residual),
                r0.right_orthogonality_defect, r0.left_orthogonality_defect, vbd)
        end
        result = infloat64 ? widen() : setprecision(widen, BigFloat, p)
        return AdaptiveSVDResult(result, inner.precision_levels, inner.radii_history,
            inner.total_iterations, p, inner.tolerance_achieved)
    end

    # Track adaptation history
    precision_levels = Int[]
    radii_history = T[]
    total_iterations = 0

    # The first level is the arithmetic of the matrix given: Float64, or BigFloat at the
    # precision of its midpoints, so that a BigFloat matrix has a BigFloat result.
    if T === BigFloat
        current_precision = maximum(x -> precision(real(x)), mid(A); init = precision(BigFloat))
        result = setprecision(BigFloat, current_precision) do
            rigorous_svd(A; method, apply_vbd)
        end
    else
        current_precision = 53  # Float64 precision in bits
        A_working = T == Float64 ? A : _convert_to_precision(A, 53)
        result = rigorous_svd(A_working; method, apply_vbd)
    end

    push!(precision_levels, current_precision)
    max_radius = maximum(rad.(result.singular_values))
    push!(radii_history, max_radius)

    # Check if initial result is good enough
    if max_radius <= tolerance
        return AdaptiveSVDResult(
            result, precision_levels, radii_history,
            0, current_precision, true
        )
    end

    # Need refinement - iterate with increasing precision
    for refinement_step in 1:max_refinement_iterations
        # a matrix given at the largest precision allowed, or above it, is not rounded down
        current_precision >= max_precision_bits && break
        # Double precision
        current_precision = min(current_precision * 2, max_precision_bits)
        push!(precision_levels, current_precision)

        @info "Refining SVD at precision $(current_precision) bits (step $refinement_step/$max_refinement_iterations)"

        result = setprecision(BigFloat, current_precision) do
            # Convert to higher precision
            A_high = _convert_to_precision(A, current_precision)

            # Extract current SVD for refinement
            U_curr = mid(result.U)
            Σ_curr = [mid(σ) for σ in result.singular_values]
            V_curr = mid(result.V)

            # Apply Ogita refinement
            refined = ogita_svd_refine(
                mid(A_high), U_curr, Σ_curr, V_curr;
                max_iterations=10,
                precision_bits=current_precision
            )

            total_iterations += refined.iterations

            if !refined.converged
                @warn "Ogita refinement did not converge at precision $(current_precision) bits"
            end

            # Certify the refined SVD
            # Create synthetic SVD for certification
            Σ_refined = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
            svd_refined = (U=refined.U,
                          S=Σ_refined,
                          V=refined.V,
                          Vt=refined.V')

            # Certify with rigorous bounds, at the precision of this level
            certify_svd(A_high, svd_refined, method; apply_vbd)
        end

        # Check new radius
        max_radius = maximum(rad.(result.singular_values))
        push!(radii_history, max_radius)

        @info "After refinement: max radius = $(max_radius)"

        if max_radius <= tolerance
            @info "Tolerance achieved at precision $(current_precision) bits"
            return AdaptiveSVDResult(
                result, precision_levels, radii_history,
                total_iterations, current_precision, true
            )
        end

        if current_precision >= max_precision_bits
            @warn "Maximum precision reached without achieving tolerance"
            break
        end
    end

    # Tolerance not achieved
    @warn "Maximum refinement iterations reached. Final max radius: $(max_radius), target: $(tolerance)"
    return AdaptiveSVDResult(
        result, precision_levels, radii_history,
        total_iterations, current_precision, false
    )
end

# The ball matrix `A` in the arithmetic of `precision_bits` bits, which is Float64 at 53 bits or
# fewer and BigFloat above. A midpoint with more digits than that is rounded to nearest; the
# rounding error, computed exactly at the precision of the midpoint and then rounded up, is added
# to the radius, itself rounded up, so that the result contains `A`.
function _convert_to_precision(A::BallMatrix{T}, precision_bits::Int) where {T}
    RT = precision_bits <= 53 ? Float64 : BigFloat
    p = RT === Float64 ? 53 : precision_bits
    up(x::Real) = RT === Float64 ? Float64(x, RoundUp) : BigFloat(x, RoundUp; precision = p)
    nearest(x::Real) = RT === Float64 ? Float64(x) : BigFloat(x; precision = p)
    function excess(x::Real, c::Real)
        px = precision(x)
        px <= p && return up(0)
        return up(setprecision(() -> abs(BigFloat(x) - BigFloat(c)), BigFloat, px))
    end
    Ac, Ar = mid(A), rad(A)
    iscomplex = eltype(Ac) <: Complex
    c = Matrix{iscomplex ? Complex{RT} : RT}(undef, size(A))
    r = Matrix{RT}(undef, size(A))
    convert_entries() = for i in eachindex(Ac)
        x = Ac[i]
        if iscomplex
            cr, ci = nearest(real(x)), nearest(imag(x))
            c[i] = complex(cr, ci)
            r[i] = add_up(add_up(excess(real(x), cr), excess(imag(x), ci)), up(Ar[i]))
        else
            c[i] = nearest(x)
            r[i] = add_up(excess(x, c[i]), up(Ar[i]))
        end
    end
    RT === Float64 ? convert_entries() : setprecision(convert_entries, BigFloat, p)
    return BallMatrix(c, r)
end

# Export new functions
export OgitaSVDRefinementResult, AdaptiveSVDResult
export ogita_svd_refine, adaptive_ogita_svd, ogita_iterations_for_precision
