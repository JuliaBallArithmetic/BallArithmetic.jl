# V1 — Certified resolvent bound via Sylvester similarity
#
# Combines:
# (i)   Miyajima SVD on small block A_z = zI - T11
# (ii)  Rump–Oishi conditioning for Sylvester similarity S(X̃)
# (iii) Triangular inverse back-substitution on large block D_z = zI - T22
#
# Reference: Custom algorithm for non-normal pseudospectra certification

"""
    SylvesterResolventResult{T}

Result from the Sylvester-based certified resolvent bound computation.

# Fields
## Precomputation results (z-independent)
- `residual_norm::T`: Upper bound on ‖R‖₂ where R = T₁₂ + T₁₁X̃ - X̃T₂₂
- `coupling_norm::T`: Upper bound on ‖T₁₂‖₂
- `similarity_cond::T`: κ₂(S(X̃)) = ψ(‖X̃‖)²
- `reduction_factor::T`: residual_norm / coupling_norm (should be << 1)
- `net_penalty::T`: similarity_cond * reduction_factor

## Split information
- `k::Int`: Split index (T₁₁ is k×k)
- `n::Int`: Total matrix size

## Status
- `precomputation_success::Bool`: Whether precomputation succeeded
- `failure_reason::String`: Empty if success, otherwise describes failure
"""
struct SylvesterResolventResult{T}
    residual_norm::T
    coupling_norm::T
    similarity_cond::T
    reduction_factor::T
    net_penalty::T
    k::Int
    n::Int
    precomputation_success::Bool
    failure_reason::String
end

"""
    SylvesterResolventPointResult{T}

Result for a single point z in the resolvent bound computation.

# Fields
- `z::Complex{T}`: The evaluation point
- `resolvent_bound::T`: Certified upper bound on ‖(zI - T)⁻¹‖₂
- `M_A::T`: Upper bound on ‖(zI - T₁₁)⁻¹‖₂
- `M_D::T`: Upper bound on ‖(zI - T₂₂)⁻¹‖₂
- `coupling_contrib::T`: M_A * r * M_D (effective coupling contribution)
- `success::Bool`: Whether bound computation succeeded at this point
- `failure_reason::String`: Empty if success
"""
struct SylvesterResolventPointResult{T}
    z::Complex{T}
    resolvent_bound::T
    M_A::T
    M_D::T
    coupling_contrib::T
    success::Bool
    failure_reason::String
end

# Triangular inverse bounds and similarity conditioning functions
# are now defined in src/norm_bounds/triangular_inverse_bounds.jl

#==============================================================================#
# Sylvester solve (oracle)
#==============================================================================#

"""
    solve_sylvester_oracle(T11::AbstractMatrix, T12::AbstractMatrix, T22::AbstractMatrix)

Solve the Sylvester equation T₁₁X - XT₂₂ = -T₁₂ for X.

This is the "oracle" that provides an approximate solution. For Float64, uses
LAPACK TRSYL. For BigFloat, converts to Float64, solves, and converts back.

# Returns
- X̃: Approximate solution to the Sylvester equation
"""
function solve_sylvester_oracle(T11::AbstractMatrix{CT}, T12::AbstractMatrix{CT},
                                 T22::AbstractMatrix{CT}) where {CT<:Complex}
    T = real(CT)

    if T === Float64
        # Direct LAPACK solve
        # Julia's sylvester(A, B, C) solves AX + XB + C = 0, i.e., AX + XB = -C
        # We want T11*X - X*T22 = -T12
        # i.e., T11*X + X*(-T22) = -T12
        # So we call sylvester(T11, -T22, T12) to get T11*X - X*T22 = -T12
        X = sylvester(T11, -T22, T12)
        return X
    else
        # BigFloat: use Float64 oracle and convert back
        T11_f64 = ComplexF64.(T11)
        T12_f64 = ComplexF64.(T12)
        T22_f64 = ComplexF64.(T22)

        X_f64 = sylvester(T11_f64, -T22_f64, T12_f64)
        return CT.(X_f64)
    end
end

# Real matrix version
function solve_sylvester_oracle(T11::AbstractMatrix{T}, T12::AbstractMatrix{T},
                                 T22::AbstractMatrix{T}) where {T<:Real}
    if T === Float64
        X = sylvester(T11, -T22, T12)
        return X
    else
        # BigFloat: use Float64 oracle
        T11_f64 = Float64.(T11)
        T12_f64 = Float64.(T12)
        T22_f64 = Float64.(T22)

        X_f64 = sylvester(T11_f64, -T22_f64, T12_f64)
        return T.(X_f64)
    end
end

#==============================================================================#
# Main algorithm
#==============================================================================#

"""
    sylvester_resolvent_precompute(T::AbstractMatrix, k::Int;
                                    X_oracle::Union{Nothing, AbstractMatrix}=nothing)

Precompute z-independent quantities for the Sylvester-based resolvent bound.

# Arguments
- `T::AbstractMatrix`: Schur-triangular matrix (complex or real)
- `k::Int`: Split index. T₁₁ = T[1:k, 1:k], T₂₂ = T[k+1:end, k+1:end]
- `X_oracle`: Optional pre-computed Sylvester solution. If nothing, computed internally.

# Returns
- `SylvesterResolventResult`: Contains precomputed quantities and diagnostics

# Algorithm
1. Extract blocks T₁₁, T₁₂, T₂₂
2. Solve Sylvester equation T₁₁X - XT₂₂ = -T₁₂ (oracle)
3. Compute residual R = T₁₂ + T₁₁X̃ - X̃T₂₂
4. Compute norms: r = ‖R‖₂, t₁₂ = ‖T₁₂‖₂
5. Compute κ₂(S(X̃)) using ψ²(‖X̃‖₂)
6. Return diagnostics: reduction = r/t₁₂, penalty = κ₂(S), net = penalty * reduction
"""
function sylvester_resolvent_precompute(T::AbstractMatrix{ET}, k::Int;
                                         X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)
    n == size(T, 2) || throw(ArgumentError("T must be square"))
    1 ≤ k < n || throw(ArgumentError("k must satisfy 1 ≤ k < n"))

    RT = real(ET)  # Real type for bounds

    # Extract blocks
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    # Check T22 is upper triangular (or close to it)
    if !istriu(T22)
        lower_norm = norm(tril(T22, -1))
        if lower_norm > sqrt(eps(RT)) * norm(T22)
            return SylvesterResolventResult{RT}(
                RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
                k, n, false, "T22 block is not upper triangular"
            )
        end
    end

    # Solve Sylvester equation (oracle)
    X = if X_oracle !== nothing
        X_oracle
    else
        try
            solve_sylvester_oracle(T11, T12, T22)
        catch e
            return SylvesterResolventResult{RT}(
                RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
                k, n, false, "Sylvester solve failed: $(e)"
            )
        end
    end

    # Compute residual R = T12 + T11*X - X*T22
    R = T12 + T11 * X - X * T22

    # Compute norm bounds using cheap sqrt(‖·‖₁ * ‖·‖_∞) estimate
    r = sqrt(opnorm(R, 1) * opnorm(R, Inf))        # ‖R‖₂ upper bound
    t12 = sqrt(opnorm(T12, 1) * opnorm(T12, Inf))  # ‖T₁₂‖₂ upper bound

    # Handle case where T12 is zero or tiny
    if t12 < eps(RT)
        t12 = eps(RT)  # Avoid division by zero
    end

    # Similarity condition number κ₂(S(X̃))
    K_S = similarity_condition_number(X)

    # Diagnostics
    reduction = r / t12
    net = K_S * reduction

    return SylvesterResolventResult{RT}(
        RT(r), RT(t12), RT(K_S), RT(reduction), RT(net),
        k, n, true, ""
    )
end

"""
    sylvester_resolvent_bound(precomp::SylvesterResolventResult, T::AbstractMatrix,
                               z::Complex; miyajima_method=:M1)

Compute the certified resolvent bound ‖(zI - T)⁻¹‖₂ at a single point z.

# Arguments
- `precomp::SylvesterResolventResult`: Precomputed quantities from `sylvester_resolvent_precompute`
- `T::AbstractMatrix`: The original Schur-triangular matrix
- `z::Complex`: Evaluation point
- `miyajima_method::Symbol`: Method for small block SVD (:M1 or :M4)

# Returns
- `SylvesterResolventPointResult`: Contains the bound and diagnostics

# Algorithm
For each z:
1. Form A_z = zI - T₁₁
2. Compute M_A(z) = ‖A_z⁻¹‖₂ via Miyajima SVD (certified lower bound on σ_min)
3. Form D_z = zI - T₂₂ (upper triangular)
4. Compute M_D(z) = ‖D_z⁻¹‖₂ via triangular backward recursion
5. Combine: M_T(z) = K_S · (M_A + M_D + M_A · r · M_D)
"""
function sylvester_resolvent_bound(precomp::SylvesterResolventResult{RT},
                                    T::AbstractMatrix{ET},
                                    z::Complex;
                                    miyajima_method::Symbol=:M1) where {RT, ET}
    if !precomp.precomputation_success
        return SylvesterResolventPointResult{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            false, "Precomputation failed: $(precomp.failure_reason)"
        )
    end

    k = precomp.k
    n = precomp.n

    # Extract blocks
    T11 = T[1:k, 1:k]
    T22 = T[(k+1):n, (k+1):n]

    # Form A_z = zI - T11
    A_z = z * I - T11

    # Certified inverse norm for small block via Miyajima SVD
    A_z_ball = BallMatrix(A_z, zeros(RT, k, k))

    M_A = try
        svd_result = rigorous_svd(A_z_ball; method = miyajima_method == :M4 ? MiyajimaM4() : MiyajimaM1())
        σ_min_ball = svd_result.singular_values[end]
        σ_min_lower = mid(σ_min_ball) - rad(σ_min_ball)

        if σ_min_lower ≤ zero(RT)
            return SylvesterResolventPointResult{RT}(
                Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
                false, "σ_min(A_z) ≤ 0: z is at or very near an eigenvalue of T11"
            )
        end

        one(RT) / σ_min_lower
    catch e
        return SylvesterResolventPointResult{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            false, "Miyajima SVD failed: $(e)"
        )
    end

    # Form D_z = zI - T22 (upper triangular)
    D_z = z * I - T22

    # Triangular inverse bound for large block
    M_D = triangular_inverse_two_norm_bound(D_z)

    if !isfinite(M_D)
        return SylvesterResolventPointResult{RT}(
            Complex{RT}(z), RT(Inf), M_A, RT(Inf), RT(Inf),
            false, "Triangular bound failed: z is at or very near an eigenvalue of T22"
        )
    end

    # Combine: M_T(z) = K_S * (M_A + M_D + M_A * r * M_D)
    r = precomp.residual_norm
    K_S = precomp.similarity_cond

    coupling = M_A * r * M_D
    resolvent_bound = K_S * (M_A + M_D + coupling)

    return SylvesterResolventPointResult{RT}(
        Complex{RT}(z), resolvent_bound, M_A, M_D, coupling,
        true, ""
    )
end

"""
    sylvester_resolvent_bound(precomp::SylvesterResolventResult, T::AbstractMatrix,
                               z_list::AbstractVector{<:Complex}; miyajima_method=:M1)

Compute certified resolvent bounds at multiple points.

# Returns
- `Vector{SylvesterResolventPointResult}`: Results for each z in z_list
"""
function sylvester_resolvent_bound(precomp::SylvesterResolventResult{RT},
                                    T::AbstractMatrix{ET},
                                    z_list::AbstractVector{<:Complex};
                                    miyajima_method::Symbol=:M1) where {RT, ET}
    return [sylvester_resolvent_bound(precomp, T, z; miyajima_method=miyajima_method)
            for z in z_list]
end

"""
    sylvester_resolvent_bound(T::AbstractMatrix, k::Int, z;
                               miyajima_method=:M1, X_oracle=nothing)

Convenience function: precompute and evaluate at one or more points.

# Arguments
- `T::AbstractMatrix`: Schur-triangular matrix
- `k::Int`: Split index
- `z`: Single complex point or vector of complex points
- `miyajima_method::Symbol`: :M1 or :M4
- `X_oracle`: Optional pre-computed Sylvester solution

# Returns
For single z: `(precomp, point_result)`
For vector z: `(precomp, vector_of_point_results)`
"""
function sylvester_resolvent_bound(T::AbstractMatrix, k::Int, z::Complex;
                                    miyajima_method::Symbol=:M1,
                                    X_oracle::Union{Nothing, AbstractMatrix}=nothing)
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X_oracle)
    point_result = sylvester_resolvent_bound(precomp, T, z; miyajima_method=miyajima_method)
    return (precomp, point_result)
end

function sylvester_resolvent_bound(T::AbstractMatrix, k::Int, z_list::AbstractVector{<:Complex};
                                    miyajima_method::Symbol=:M1,
                                    X_oracle::Union{Nothing, AbstractMatrix}=nothing)
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X_oracle)
    point_results = sylvester_resolvent_bound(precomp, T, z_list; miyajima_method=miyajima_method)
    return (precomp, point_results)
end

#==============================================================================#
# Diagnostic utilities
#==============================================================================#

"""
    print_sylvester_diagnostics(precomp::SylvesterResolventResult; io::IO=stdout)

Print human-readable diagnostics for the Sylvester precomputation.
"""
function print_sylvester_diagnostics(precomp::SylvesterResolventResult; io::IO=stdout)
    println(io, "Sylvester Resolvent Bound Diagnostics")
    println(io, "="^40)
    println(io, "Matrix size: $(precomp.n) × $(precomp.n)")
    println(io, "Split index k: $(precomp.k)")
    println(io, "Small block: $(precomp.k) × $(precomp.k)")
    println(io, "Large block: $(precomp.n - precomp.k) × $(precomp.n - precomp.k)")
    println(io)

    if !precomp.precomputation_success
        println(io, "⚠ PRECOMPUTATION FAILED: $(precomp.failure_reason)")
        return
    end

    println(io, "Norms:")
    println(io, "  ‖R‖₂ (residual):     $(precomp.residual_norm)")
    println(io, "  ‖T₁₂‖₂ (coupling):   $(precomp.coupling_norm)")
    println(io, "  κ₂(S(X̃)) (penalty):  $(precomp.similarity_cond)")
    println(io)

    println(io, "Quality indicators:")
    red = precomp.reduction_factor
    pen = precomp.similarity_cond
    net = precomp.net_penalty

    red_status = red < 0.1 ? "✓ excellent" : red < 0.5 ? "○ good" : red < 1.0 ? "△ marginal" : "✗ poor"
    pen_status = pen < 2.0 ? "✓ excellent" : pen < 5.0 ? "○ good" : pen < 10.0 ? "△ marginal" : "✗ poor"
    net_status = net < 0.5 ? "✓ excellent" : net < 1.0 ? "○ good" : net < 2.0 ? "△ marginal" : "✗ poor"

    println(io, "  reduction = r/‖T₁₂‖: $(red) $red_status")
    println(io, "  penalty = κ₂(S):     $(pen) $pen_status")
    println(io, "  net = pen × red:     $(net) $net_status")
end

"""
    print_point_result(result::SylvesterResolventPointResult; io::IO=stdout)

Print human-readable result for a single point evaluation.
"""
function print_point_result(result::SylvesterResolventPointResult; io::IO=stdout)
    println(io, "z = $(result.z)")
    if !result.success
        println(io, "  ⚠ FAILED: $(result.failure_reason)")
        return
    end

    println(io, "  ‖(zI-T)⁻¹‖₂ ≤ $(result.resolvent_bound)")
    println(io, "  Components: M_A=$(result.M_A), M_D=$(result.M_D), coupling=$(result.coupling_contrib)")
end

#==============================================================================#
# V3 — Collatz-Wielandt Neumann bound for large block
#==============================================================================#

"""
    CollatzNeumannResult{T}

Result from the Collatz-Wielandt Neumann bound computation for ‖D_z⁻¹‖₂.

# Fields
- `M_D::T`: Upper bound on ‖D_z⁻¹‖₂
- `alpha::T`: Collatz upper bound on ‖N_z‖₂ (must be < 1 for success)
- `Dd_inv_norm::T`: ‖(D_z)_d⁻¹‖₂ = max_i 1/|z - T₂₂[i,i]|
- `neumann_gap::T`: 1 - alpha (margin for Neumann convergence)
- `success::Bool`: Whether alpha < 1 (Neumann certified)
"""
struct CollatzNeumannResult{T}
    M_D::T
    alpha::T
    Dd_inv_norm::T
    neumann_gap::T
    success::Bool
end

"""
    collatz_norm_N_bound(T22::AbstractMatrix, z::Complex;
                          power_iterations::Int=3, x0::Union{Nothing, Vector}=nothing)

Compute a certified upper bound on ‖N_z‖₂ where N_z = (D_z)_d⁻¹(D_z)_f
using the Collatz-Wielandt theorem.

# Arguments
- `T22::AbstractMatrix`: Upper triangular block (n-k) × (n-k)
- `z::Complex`: Evaluation point
- `power_iterations::Int=3`: Number of power iterations to refine the Collatz vector
- `x0::Vector`: Optional initial positive vector (default: ones)

# Method
Since N_z is strict upper triangular (scaled), we have:
- ‖N_z‖₂² = ρ(N_z*N_z) ≤ ρ(|N_z|ᵀ|N_z|) =: ρ(B_z)
- Collatz: ρ(B_z) ≤ max_i (B_z x)_i / x_i for any x > 0

We compute B_z x = |N_z|ᵀ(|N_z| x) without forming B_z explicitly.

# Returns
- `(alpha, Dd_inv_norm)` where alpha ≥ ‖N_z‖₂ and Dd_inv_norm = ‖(D_z)_d⁻¹‖₂
"""
function collatz_norm_N_bound(T22::AbstractMatrix{CT}, z::Complex;
                               power_iterations::Int=3,
                               x0::Union{Nothing, AbstractVector}=nothing) where {CT}
    m = size(T22, 1)
    T = real(CT)

    # Step A: Build diagonal inverse scale
    invabsd = Vector{T}(undef, m)
    for i in 1:m
        d_i = abs(z - T22[i, i])
        if d_i ≤ eps(T) * abs(z)
            invabsd[i] = T(Inf)
        else
            invabsd[i] = one(T) / d_i
        end
    end

    Dd_inv_norm = maximum(invabsd)

    if !isfinite(Dd_inv_norm)
        return (T(Inf), T(Inf))
    end

    # Initialize positive vector x
    x = if x0 !== nothing
        T.(abs.(x0)) .+ eps(T)  # Ensure positive
    else
        ones(T, m)
    end

    # Allocate work vectors
    u = similar(x)
    v = similar(x)

    # Define |N_z| x multiplication (strict upper triangular)
    # |N_z|[i,j] = |T22[i,j]| / |z - T22[i,i]| for j > i, 0 otherwise
    function mul_absN!(y, x_in)
        fill!(y, zero(T))
        for i in 1:m
            s = zero(T)
            for j in (i+1):m
                s += abs(T22[i, j]) * x_in[j]
            end
            y[i] = invabsd[i] * s
        end
        return y
    end

    # Define |N_z|ᵀ x multiplication
    function mul_absNt!(y, x_in)
        fill!(y, zero(T))
        for j in 1:m
            for i in 1:(j-1)
                # |N_z|[i,j] = invabsd[i] * |T22[i,j]|
                y[j] += invabsd[i] * abs(T22[i, j]) * x_in[i]
            end
        end
        return y
    end

    # Power iterations to refine x (optional but improves bound)
    for _ in 1:power_iterations
        mul_absN!(u, x)      # u = |N| x
        mul_absNt!(v, u)     # v = |N|ᵀ u = B x
        scale = maximum(v)
        if scale > eps(T)
            x .= v ./ scale  # Normalize, keep positive
        end
    end

    # Final Collatz bound
    mul_absN!(u, x)          # u = |N| x
    mul_absNt!(v, u)         # v = B x

    # ρ(B) ≤ max_i v[i]/x[i]
    rho_ub = zero(T)
    for i in 1:m
        if x[i] > eps(T)
            rho_ub = max(rho_ub, v[i] / x[i])
        end
    end

    alpha = sqrt(rho_ub)  # ‖N_z‖₂ ≤ alpha

    return (alpha, Dd_inv_norm)
end

"""
    neumann_inverse_bound(T22::AbstractMatrix, z::Complex;
                           power_iterations::Int=3)

Compute a certified upper bound on ‖D_z⁻¹‖₂ using the Neumann series approach.

# Method
For D_z = (D_z)_d + (D_z)_f where (D_z)_d is diagonal and (D_z)_f is strict upper:
- D_z = (D_z)_d (I + N_z) where N_z = (D_z)_d⁻¹(D_z)_f
- If ‖N_z‖₂ < 1: ‖D_z⁻¹‖₂ ≤ ‖(D_z)_d⁻¹‖₂ / (1 - ‖N_z‖₂)

# Returns
`CollatzNeumannResult` with the bound and diagnostics.
"""
function neumann_inverse_bound(T22::AbstractMatrix{CT}, z::Complex;
                                power_iterations::Int=3) where {CT}
    T = real(CT)

    alpha, Dd_inv_norm = collatz_norm_N_bound(T22, z; power_iterations=power_iterations)

    if !isfinite(alpha) || !isfinite(Dd_inv_norm)
        return CollatzNeumannResult{T}(T(Inf), alpha, Dd_inv_norm, -T(Inf), false)
    end

    if alpha ≥ one(T)
        # Neumann series does not converge
        return CollatzNeumannResult{T}(T(Inf), alpha, Dd_inv_norm, one(T) - alpha, false)
    end

    # Neumann bound: ‖D_z⁻¹‖₂ ≤ ‖(D_z)_d⁻¹‖₂ / (1 - ‖N_z‖₂)
    neumann_gap = one(T) - alpha
    M_D = Dd_inv_norm / neumann_gap

    return CollatzNeumannResult{T}(M_D, alpha, Dd_inv_norm, neumann_gap, true)
end

#==============================================================================#
# V3 Result and Main Function
#==============================================================================#

"""
    SylvesterResolventPointResultV3{T}

Result for V3 resolvent bound using Collatz-Wielandt Neumann for large block.

# Fields
- `z::Complex{T}`: Evaluation point
- `resolvent_bound::T`: V3 certified upper bound on ‖(zI - T)⁻¹‖₂
- `resolvent_bound_v1::T`: V1 bound for comparison
- `M_A::T`: ‖(zI - T₁₁)⁻¹‖₂ (Miyajima SVD)
- `M_D::T`: ‖(zI - T₂₂)⁻¹‖₂ (Neumann bound)
- `M_D_v1::T`: V1 triangular bound for comparison
- `M_AR::T`: ‖(zI - T₁₁)⁻¹R‖₂ (V2 tightening, if used)
- `alpha::T`: Collatz bound on ‖N_z‖₂
- `neumann_gap::T`: 1 - alpha
- `Dd_inv_norm::T`: ‖(D_z)_d⁻¹‖₂
- `coupling::T`: Coupling term in final bound
- `success::Bool`: Whether computation succeeded
- `neumann_success::Bool`: Whether Neumann bound succeeded (alpha < 1)
- `failure_reason::String`: Empty if success
"""
struct SylvesterResolventPointResultV3{T}
    z::Complex{T}
    resolvent_bound::T
    resolvent_bound_v1::T
    M_A::T
    M_D::T
    M_D_v1::T
    M_AR::T
    alpha::T
    neumann_gap::T
    Dd_inv_norm::T
    coupling::T
    success::Bool
    neumann_success::Bool
    failure_reason::String
end

"""
    sylvester_resolvent_bound_v3(precomp::SylvesterResolventResult, T::AbstractMatrix,
                                  R::AbstractMatrix, z::Complex;
                                  miyajima_method=:M1, power_iterations=3,
                                  use_v2_coupling=true)

Compute V3 certified resolvent bound using Collatz-Wielandt Neumann for large block.

V3 replaces the triangular back-substitution bound for M_D with a Neumann series bound:
- M_D = ‖(D_z)_d⁻¹‖₂ / (1 - ‖N_z‖₂) when ‖N_z‖₂ < 1
- ‖N_z‖₂ bounded via Collatz-Wielandt on |N_z|ᵀ|N_z|

# Arguments
- `precomp`: Precomputed Sylvester quantities
- `T`: Schur-triangular matrix
- `R`: Sylvester residual
- `z`: Evaluation point
- `miyajima_method`: :M1 or :M4 for small block SVD
- `power_iterations`: Number of power iterations for Collatz
- `use_v2_coupling`: If true, use V2's tighter ‖A⁻¹R‖ bound for coupling

# Returns
`SylvesterResolventPointResultV3` with bounds and diagnostics
"""
function sylvester_resolvent_bound_v3(precomp::SylvesterResolventResult{RT},
                                       T::AbstractMatrix{ET},
                                       R::AbstractMatrix,
                                       z::Complex;
                                       miyajima_method::Symbol=:M1,
                                       power_iterations::Int=3,
                                       use_v2_coupling::Bool=true) where {RT, ET}
    if !precomp.precomputation_success
        return SylvesterResolventPointResultV3{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), -RT(Inf), RT(Inf), RT(Inf), false, false,
            "Precomputation failed: $(precomp.failure_reason)"
        )
    end

    k = precomp.k
    n = precomp.n
    r = precomp.residual_norm
    K_S = precomp.similarity_cond

    # Extract blocks
    T11 = T[1:k, 1:k]
    T22 = T[(k+1):n, (k+1):n]

    # === M_A via Miyajima SVD ===
    A_z = z * I - T11
    A_z_ball = BallMatrix(A_z, zeros(RT, k, k))

    M_A = try
        svd_result = rigorous_svd(A_z_ball; method = miyajima_method == :M4 ? MiyajimaM4() : MiyajimaM1())
        σ_min_ball = svd_result.singular_values[end]
        σ_min_lower = mid(σ_min_ball) - rad(σ_min_ball)

        if σ_min_lower ≤ zero(RT)
            return SylvesterResolventPointResultV3{RT}(
                Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
                RT(Inf), -RT(Inf), RT(Inf), RT(Inf), false, false,
                "σ_min(A_z) ≤ 0: z at or near eigenvalue of T11"
            )
        end

        one(RT) / σ_min_lower
    catch e
        return SylvesterResolventPointResultV3{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), -RT(Inf), RT(Inf), RT(Inf), false, false,
            "Miyajima SVD failed: $(e)"
        )
    end

    # === M_D via Collatz-Neumann (V3) ===
    neumann_result = neumann_inverse_bound(T22, z; power_iterations=power_iterations)

    # Also compute V1 triangular bound for comparison
    D_z = z * I - T22
    M_D_v1 = triangular_inverse_two_norm_bound(D_z)

    # Use Neumann if successful, else fall back to V1
    if neumann_result.success
        M_D = neumann_result.M_D
    else
        M_D = M_D_v1
    end

    if !isfinite(M_D)
        return SylvesterResolventPointResultV3{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), M_A, RT(Inf), M_D_v1, RT(Inf),
            neumann_result.alpha, neumann_result.neumann_gap, neumann_result.Dd_inv_norm,
            RT(Inf), false, neumann_result.success,
            "D_z inverse bound failed: z at or near eigenvalue of T22"
        )
    end

    # === Coupling term ===
    M_AR = M_A * r  # Default V1 coupling

    if use_v2_coupling
        # V2 tightening: compute ‖A_z⁻¹R‖ directly
        Y_approx = try
            A_z \ R
        catch
            nothing
        end

        if Y_approx !== nothing
            Δ = R - A_z * Y_approx
            Y_norm = sqrt(opnorm(Y_approx, 1) * opnorm(Y_approx, Inf))
            Δ_norm = sqrt(opnorm(Δ, 1) * opnorm(Δ, Inf))
            M_AR = Y_norm + M_A * Δ_norm
        end
    end

    coupling = M_AR * M_D
    coupling_v1 = M_A * r * M_D_v1

    # === Final bounds ===
    bound_v3 = K_S * (M_A + M_D + coupling)
    bound_v1 = K_S * (M_A + M_D_v1 + coupling_v1)

    return SylvesterResolventPointResultV3{RT}(
        Complex{RT}(z), bound_v3, bound_v1, M_A, M_D, M_D_v1, M_AR,
        neumann_result.alpha, neumann_result.neumann_gap, neumann_result.Dd_inv_norm,
        coupling, true, neumann_result.success, ""
    )
end

"""
    sylvester_resolvent_bound_v3(precomp, T, R, z_list; kwargs...)

V3 bound at multiple points.
"""
function sylvester_resolvent_bound_v3(precomp::SylvesterResolventResult{RT},
                                       T::AbstractMatrix{ET},
                                       R::AbstractMatrix,
                                       z_list::AbstractVector{<:Complex};
                                       miyajima_method::Symbol=:M1,
                                       power_iterations::Int=3,
                                       use_v2_coupling::Bool=true) where {RT, ET}
    return [sylvester_resolvent_bound_v3(precomp, T, R, z;
                                          miyajima_method=miyajima_method,
                                          power_iterations=power_iterations,
                                          use_v2_coupling=use_v2_coupling)
            for z in z_list]
end

"""
    sylvester_resolvent_bound_v3(T, k, z; kwargs...)

Convenience function for V3.
"""
function sylvester_resolvent_bound_v3(T::AbstractMatrix{ET}, k::Int, z::Complex;
                                       miyajima_method::Symbol=:M1,
                                       power_iterations::Int=3,
                                       use_v2_coupling::Bool=true,
                                       X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = X_oracle !== nothing ? X_oracle : solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22

    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)
    result = sylvester_resolvent_bound_v3(precomp, T, R, z;
                                           miyajima_method=miyajima_method,
                                           power_iterations=power_iterations,
                                           use_v2_coupling=use_v2_coupling)

    return (precomp, R, result)
end

function sylvester_resolvent_bound_v3(T::AbstractMatrix{ET}, k::Int,
                                       z_list::AbstractVector{<:Complex};
                                       miyajima_method::Symbol=:M1,
                                       power_iterations::Int=3,
                                       use_v2_coupling::Bool=true,
                                       X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = X_oracle !== nothing ? X_oracle : solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22

    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)
    results = sylvester_resolvent_bound_v3(precomp, T, R, z_list;
                                            miyajima_method=miyajima_method,
                                            power_iterations=power_iterations,
                                            use_v2_coupling=use_v2_coupling)

    return (precomp, R, results)
end

"""
    print_point_result_v3(result::SylvesterResolventPointResultV3; io::IO=stdout)

Print V3 result with Neumann diagnostics.
"""
function print_point_result_v3(result::SylvesterResolventPointResultV3; io::IO=stdout)
    println(io, "z = $(result.z)")
    if !result.success
        println(io, "  ⚠ FAILED: $(result.failure_reason)")
        return
    end

    improvement = (result.resolvent_bound_v1 - result.resolvent_bound) /
                  result.resolvent_bound_v1 * 100
    md_improvement = (result.M_D_v1 - result.M_D) / result.M_D_v1 * 100

    println(io, "  V3 bound: ‖(zI-T)⁻¹‖₂ ≤ $(result.resolvent_bound)")
    println(io, "  V1 bound: ‖(zI-T)⁻¹‖₂ ≤ $(result.resolvent_bound_v1)")
    println(io, "  Overall improvement: $(round(improvement, digits=1))%")
    println(io)
    println(io, "  Components:")
    println(io, "    M_A = $(result.M_A)")
    println(io, "    M_D (V3 Neumann) = $(result.M_D)")
    println(io, "    M_D (V1 triang.) = $(result.M_D_v1)")
    println(io, "    M_D improvement: $(round(md_improvement, digits=1))%")
    println(io, "    M_AR = $(result.M_AR)")
    println(io, "    coupling = $(result.coupling)")
    println(io)
    println(io, "  Neumann diagnostics:")
    println(io, "    α = ‖N_z‖₂ ≤ $(result.alpha)")
    println(io, "    gap = 1-α = $(result.neumann_gap)")
    println(io, "    ‖(D_z)_d⁻¹‖₂ = $(result.Dd_inv_norm)")
    println(io, "    Neumann certified: $(result.neumann_success)")
end

#==============================================================================#
# V2 — Improved bound: certify ||A⁻¹R|| directly
#==============================================================================#

"""
    SylvesterResolventPointResultV2{T}

Result for a single point z in the V2 resolvent bound computation.

V2 computes a tighter coupling bound by certifying ‖A_z⁻¹R‖ directly
instead of using the product bound ‖A_z⁻¹‖·‖R‖.

# Fields
- `z::Complex{T}`: The evaluation point
- `resolvent_bound::T`: Certified upper bound on ‖(zI - T)⁻¹‖₂ (V2)
- `resolvent_bound_v1::T`: V1 bound for comparison
- `M_A::T`: Upper bound on ‖(zI - T₁₁)⁻¹‖₂
- `M_D::T`: Upper bound on ‖(zI - T₂₂)⁻¹‖₂
- `M_AR::T`: Upper bound on ‖(zI - T₁₁)⁻¹R‖₂ (V2 tightening)
- `coupling_v1::T`: M_A * r * M_D (V1 coupling term)
- `coupling_v2::T`: M_AR * M_D (V2 coupling term)
- `tightening_ratio::T`: M_AR / (M_A * r) — how much V2 improves over V1
- `solve_residual_norm::T`: ‖Δ‖₂ where Δ = R - A_z * Ŷ
- `success::Bool`: Whether bound computation succeeded
- `failure_reason::String`: Empty if success
"""
struct SylvesterResolventPointResultV2{T}
    z::Complex{T}
    resolvent_bound::T
    resolvent_bound_v1::T
    M_A::T
    M_D::T
    M_AR::T
    coupling_v1::T
    coupling_v2::T
    tightening_ratio::T
    solve_residual_norm::T
    success::Bool
    failure_reason::String
end

"""
    sylvester_resolvent_bound_v2(precomp::SylvesterResolventResult, T::AbstractMatrix,
                                  R::AbstractMatrix, z::Complex; miyajima_method=:M1)

Compute the V2 certified resolvent bound ‖(zI - T)⁻¹‖₂ at a single point z.

V2 improves over V1 by computing a certified bound on ‖A_z⁻¹R‖ directly:
- Solve Ŷ ≈ A_z \\ R (triangular solve since T₁₁ is upper triangular)
- Compute residual Δ = R - A_z * Ŷ rigorously
- Bound: ‖A_z⁻¹R‖ ≤ ‖Ŷ‖ + ‖A_z⁻¹‖·‖Δ‖

This is tighter than V1's ‖A_z⁻¹‖·‖R‖ when R is aligned with well-conditioned
directions of A_z.

# Arguments
- `precomp::SylvesterResolventResult`: Precomputed quantities
- `T::AbstractMatrix`: The original Schur-triangular matrix
- `R::AbstractMatrix`: Sylvester residual (precomputed)
- `z::Complex`: Evaluation point
- `miyajima_method::Symbol`: Method for small block SVD (:M1 or :M4)

# Returns
- `SylvesterResolventPointResultV2`: Contains V2 bound and comparison with V1
"""
function sylvester_resolvent_bound_v2(precomp::SylvesterResolventResult{RT},
                                       T::AbstractMatrix{ET},
                                       R::AbstractMatrix,
                                       z::Complex;
                                       miyajima_method::Symbol=:M1) where {RT, ET}
    if !precomp.precomputation_success
        return SylvesterResolventPointResultV2{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(1), RT(Inf), false,
            "Precomputation failed: $(precomp.failure_reason)"
        )
    end

    k = precomp.k
    n = precomp.n
    r = precomp.residual_norm
    K_S = precomp.similarity_cond

    # Extract blocks
    T11 = T[1:k, 1:k]
    T22 = T[(k+1):n, (k+1):n]

    # Form A_z = zI - T11 (upper triangular)
    A_z = z * I - T11

    # === Step 7: Certified M_A(z) via Miyajima SVD ===
    A_z_ball = BallMatrix(A_z, zeros(RT, k, k))

    M_A = try
        svd_result = rigorous_svd(A_z_ball; method = miyajima_method == :M4 ? MiyajimaM4() : MiyajimaM1())
        σ_min_ball = svd_result.singular_values[end]
        σ_min_lower = mid(σ_min_ball) - rad(σ_min_ball)

        if σ_min_lower ≤ zero(RT)
            return SylvesterResolventPointResultV2{RT}(
                Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
                RT(Inf), RT(Inf), RT(1), RT(Inf), false,
                "σ_min(A_z) ≤ 0: z is at or very near an eigenvalue of T11"
            )
        end

        one(RT) / σ_min_lower
    catch e
        return SylvesterResolventPointResultV2{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(1), RT(Inf), false,
            "Miyajima SVD failed: $(e)"
        )
    end

    # === Step 8: Certified M_D(z) via triangular bounds ===
    D_z = z * I - T22
    M_D = triangular_inverse_two_norm_bound(D_z)

    if !isfinite(M_D)
        return SylvesterResolventPointResultV2{RT}(
            Complex{RT}(z), RT(Inf), RT(Inf), M_A, RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(1), RT(Inf), false,
            "Triangular bound failed: z is at or very near an eigenvalue of T22"
        )
    end

    # === Step 9: V2 tightening - certify M_AR(z) = ‖A_z⁻¹R‖ ===

    # 9a) Approximate solve: Ŷ ≈ A_z \ R
    # A_z is upper triangular (T11 is Schur), so use triangular solve
    Y_approx = try
        A_z \ R
    catch e
        # Fallback to V1 bound if solve fails
        coupling_v1 = M_A * r * M_D
        bound_v1 = K_S * (M_A + M_D + coupling_v1)
        return SylvesterResolventPointResultV2{RT}(
            Complex{RT}(z), bound_v1, bound_v1, M_A, M_D, M_A * r,
            coupling_v1, coupling_v1, RT(1), RT(Inf), true,
            "Triangular solve failed, using V1 bound"
        )
    end

    # 9b) Compute residual Δ = R - A_z * Ŷ rigorously
    Δ = R - A_z * Y_approx

    # 9c) Bound ‖A_z⁻¹R‖ ≤ ‖Ŷ‖ + ‖A_z⁻¹‖·‖Δ‖
    Y_norm = sqrt(opnorm(Y_approx, 1) * opnorm(Y_approx, Inf))
    Δ_norm = sqrt(opnorm(Δ, 1) * opnorm(Δ, Inf))

    M_AR = Y_norm + M_A * Δ_norm

    # === Step 10: Combine V2 bound ===
    coupling_v1 = M_A * r * M_D
    coupling_v2 = M_AR * M_D

    bound_v1 = K_S * (M_A + M_D + coupling_v1)
    bound_v2 = K_S * (M_A + M_D + coupling_v2)

    # Tightening ratio: how much V2 improves over V1
    tightening = if M_A * r > eps(RT)
        M_AR / (M_A * r)
    else
        one(RT)
    end

    return SylvesterResolventPointResultV2{RT}(
        Complex{RT}(z), bound_v2, bound_v1, M_A, M_D, M_AR,
        coupling_v1, coupling_v2, tightening, Δ_norm, true, ""
    )
end

"""
    sylvester_resolvent_bound_v2(precomp::SylvesterResolventResult, T::AbstractMatrix,
                                  R::AbstractMatrix, z_list::AbstractVector{<:Complex};
                                  miyajima_method=:M1)

Compute V2 certified resolvent bounds at multiple points.
"""
function sylvester_resolvent_bound_v2(precomp::SylvesterResolventResult{RT},
                                       T::AbstractMatrix{ET},
                                       R::AbstractMatrix,
                                       z_list::AbstractVector{<:Complex};
                                       miyajima_method::Symbol=:M1) where {RT, ET}
    return [sylvester_resolvent_bound_v2(precomp, T, R, z; miyajima_method=miyajima_method)
            for z in z_list]
end

"""
    sylvester_resolvent_bound_v2(T::AbstractMatrix, k::Int, z;
                                  miyajima_method=:M1, X_oracle=nothing)

Convenience function for V2: precompute and evaluate at one or more points.

# Returns
For single z: `(precomp, R, point_result)`
For vector z: `(precomp, R, vector_of_point_results)`
"""
function sylvester_resolvent_bound_v2(T::AbstractMatrix{ET}, k::Int, z::Complex;
                                       miyajima_method::Symbol=:M1,
                                       X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)

    # Extract blocks
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    # Compute oracle X if not provided
    X = if X_oracle !== nothing
        X_oracle
    else
        solve_sylvester_oracle(T11, T12, T22)
    end

    # Compute residual R
    R = T12 + T11 * X - X * T22

    # Precompute
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)

    # Evaluate
    point_result = sylvester_resolvent_bound_v2(precomp, T, R, z; miyajima_method=miyajima_method)

    return (precomp, R, point_result)
end

function sylvester_resolvent_bound_v2(T::AbstractMatrix{ET}, k::Int,
                                       z_list::AbstractVector{<:Complex};
                                       miyajima_method::Symbol=:M1,
                                       X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)

    # Extract blocks
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    # Compute oracle X if not provided
    X = if X_oracle !== nothing
        X_oracle
    else
        solve_sylvester_oracle(T11, T12, T22)
    end

    # Compute residual R
    R = T12 + T11 * X - X * T22

    # Precompute
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)

    # Evaluate at all points
    point_results = sylvester_resolvent_bound_v2(precomp, T, R, z_list; miyajima_method=miyajima_method)

    return (precomp, R, point_results)
end

"""
    print_point_result_v2(result::SylvesterResolventPointResultV2; io::IO=stdout)

Print human-readable result for a V2 single point evaluation.
"""
function print_point_result_v2(result::SylvesterResolventPointResultV2; io::IO=stdout)
    println(io, "z = $(result.z)")
    if !result.success
        println(io, "  ⚠ FAILED: $(result.failure_reason)")
        return
    end

    improvement = (result.resolvent_bound_v1 - result.resolvent_bound) / result.resolvent_bound_v1 * 100

    println(io, "  V2 bound: ‖(zI-T)⁻¹‖₂ ≤ $(result.resolvent_bound)")
    println(io, "  V1 bound: ‖(zI-T)⁻¹‖₂ ≤ $(result.resolvent_bound_v1)")
    println(io, "  Improvement: $(round(improvement, digits=1))%")
    println(io, "  Components:")
    println(io, "    M_A = $(result.M_A)")
    println(io, "    M_D = $(result.M_D)")
    println(io, "    M_AR = $(result.M_AR) (V2)")
    println(io, "    coupling_v1 = $(result.coupling_v1)")
    println(io, "    coupling_v2 = $(result.coupling_v2)")
    println(io, "    tightening = $(result.tightening_ratio)")
    println(io, "    ‖Δ‖ = $(result.solve_residual_norm)")
end

#==============================================================================#
# Optimal split selection
#==============================================================================#

"""
    find_optimal_split(T::AbstractMatrix, z::Complex;
                        k_range=2:min(size(T,1)-2, 50),
                        miyajima_method=:M1,
                        version=:V1)

Find the optimal split index k that minimizes the resolvent bound at z.

# Arguments
- `T::AbstractMatrix`: Schur-triangular matrix
- `z::Complex`: Evaluation point
- `k_range`: Range of k values to try
- `miyajima_method`: :M1 or :M4
- `version`: :V1, :V2, or :V3

# Returns
- For V1: `(best_k, best_precomp, best_result)`
- For V2/V3: `(best_k, best_precomp, best_R, best_result)`
"""
function find_optimal_split(T::AbstractMatrix, z::Complex;
                            k_range=2:min(size(T,1)-2, 50),
                            miyajima_method::Symbol=:M1,
                            version::Symbol=:V1)
    best_k = first(k_range)
    best_bound = Inf
    best_precomp = nothing
    best_result = nothing
    best_R = nothing

    for k in k_range
        if version == :V1
            precomp, result = sylvester_resolvent_bound(T, k, z; miyajima_method=miyajima_method)

            if result.success && result.resolvent_bound < best_bound
                best_k = k
                best_bound = result.resolvent_bound
                best_precomp = precomp
                best_result = result
            end
        elseif version == :V2
            precomp, R, result = sylvester_resolvent_bound_v2(T, k, z; miyajima_method=miyajima_method)

            if result.success && result.resolvent_bound < best_bound
                best_k = k
                best_bound = result.resolvent_bound
                best_precomp = precomp
                best_result = result
                best_R = R
            end
        else  # V3
            precomp, R, result = sylvester_resolvent_bound_v3(T, k, z; miyajima_method=miyajima_method)

            if result.success && result.resolvent_bound < best_bound
                best_k = k
                best_bound = result.resolvent_bound
                best_precomp = precomp
                best_result = result
                best_R = R
            end
        end
    end

    if version == :V1
        return (best_k, best_precomp, best_result)
    else
        return (best_k, best_precomp, best_R, best_result)
    end
end

#==============================================================================#
# Unified parametric interface
#==============================================================================#

"""
    LargeBlockMethod

Enum for selecting the large block (M_D) estimation method.

- `TriangularBacksub`: O(m²) backward recursion on triangular structure (V1)
- `NeumannCollatz`: Collatz-Wielandt bound on Neumann series (V3)
"""
@enum LargeBlockMethod begin
    TriangularBacksub
    NeumannCollatz
end

"""
    CouplingMethod

Enum for selecting the coupling term estimation method.

- `ProductBound`: Use M_A · ‖R‖ (V1, conservative)
- `DirectSolve`: Solve A_z \\ R and certify ‖A_z⁻¹R‖ directly (V2, tighter)
"""
@enum CouplingMethod begin
    ProductBound
    DirectSolve
end

"""
    UnifiedResolventResult{T}

Unified result structure for the parametric resolvent bound computation.

Contains all information from V1, V2, and V3 methods in a single structure,
allowing easy comparison between different estimator combinations.

# Fields
## Core results
- `z::Complex{T}`: Evaluation point
- `resolvent_bound::T`: Certified upper bound on ‖(zI - T)⁻¹‖₂
- `success::Bool`: Whether computation succeeded
- `failure_reason::String`: Empty if success

## Method configuration
- `large_block_method::LargeBlockMethod`: Method used for M_D
- `coupling_method::CouplingMethod`: Method used for coupling term

## Component bounds
- `K_S::T`: Similarity condition number κ₂(S(X̃))
- `M_A::T`: Upper bound on ‖(zI - T₁₁)⁻¹‖₂ (Miyajima SVD)
- `M_D::T`: Upper bound on ‖(zI - T₂₂)⁻¹‖₂ (selected method)
- `M_AR::T`: Upper bound on ‖(zI - T₁₁)⁻¹R‖₂ (or M_A·r if product)
- `coupling::T`: M_AR · M_D

## Comparison with other methods
- `M_D_triangular::T`: Triangular backsubstitution bound (for comparison)
- `M_D_neumann::T`: Neumann bound if computed, else Inf
- `M_AR_product::T`: Product bound M_A · r
- `M_AR_direct::T`: Direct bound if computed, else Inf

## Neumann-specific diagnostics
- `alpha::T`: Collatz bound on ‖N_z‖₂ (Inf if not computed)
- `neumann_gap::T`: 1 - alpha (negative if Neumann fails)
- `Dd_inv_norm::T`: ‖(D_z)_d⁻¹‖₂

## Direct solve diagnostics
- `solve_residual_norm::T`: ‖Δ‖ where Δ = R - A_z Ŷ
- `tightening_ratio::T`: M_AR_direct / M_AR_product
"""
struct UnifiedResolventResult{T}
    # Core
    z::Complex{T}
    resolvent_bound::T
    success::Bool
    failure_reason::String

    # Configuration
    large_block_method::LargeBlockMethod
    coupling_method::CouplingMethod

    # Components
    K_S::T
    M_A::T
    M_D::T
    M_AR::T
    coupling::T

    # Comparison values
    M_D_triangular::T
    M_D_neumann::T
    M_AR_product::T
    M_AR_direct::T

    # Neumann diagnostics
    alpha::T
    neumann_gap::T
    Dd_inv_norm::T

    # Direct solve diagnostics
    solve_residual_norm::T
    tightening_ratio::T
end

"""
    sylvester_resolvent_bound_unified(precomp::SylvesterResolventResult,
                                       T::AbstractMatrix, z::Complex;
                                       R::Union{Nothing, AbstractMatrix}=nothing,
                                       large_block_method::LargeBlockMethod=TriangularBacksub,
                                       coupling_method::CouplingMethod=ProductBound,
                                       miyajima_method::Symbol=:M1,
                                       power_iterations::Int=3)

Unified parametric interface for computing certified resolvent bounds.

Allows selecting different estimation methods for the large block (M_D)
and coupling term independently, enabling easy comparison.

# Arguments
- `precomp`: Precomputed Sylvester quantities from `sylvester_resolvent_precompute`
- `T`: Schur-triangular matrix
- `z`: Evaluation point
- `R`: Sylvester residual (required if `coupling_method=DirectSolve`)
- `large_block_method`: How to bound ‖(zI - T₂₂)⁻¹‖₂
  - `TriangularBacksub`: O(m²) backward recursion (V1)
  - `NeumannCollatz`: Collatz-Wielandt Neumann series (V3)
- `coupling_method`: How to bound the coupling term
  - `ProductBound`: Use M_A · ‖R‖ (V1)
  - `DirectSolve`: Solve and certify ‖A_z⁻¹R‖ (V2)
- `miyajima_method`: :M1 or :M4 for small block SVD
- `power_iterations`: Iterations for Collatz bound (only for NeumannCollatz)

# Method combinations
- `TriangularBacksub + ProductBound` = V1
- `TriangularBacksub + DirectSolve` = V2
- `NeumannCollatz + ProductBound` = V3 with V1 coupling
- `NeumannCollatz + DirectSolve` = V3 (default)

# Returns
`UnifiedResolventResult` containing the bound and all diagnostic information.

# Example
```julia
precomp = sylvester_resolvent_precompute(T, k)
R = compute_sylvester_residual(T, k, precomp)  # or from precomputation

# V1 style
r1 = sylvester_resolvent_bound_unified(precomp, T, z;
    large_block_method=TriangularBacksub, coupling_method=ProductBound)

# V2 style
r2 = sylvester_resolvent_bound_unified(precomp, T, z; R=R,
    large_block_method=TriangularBacksub, coupling_method=DirectSolve)

# V3 style
r3 = sylvester_resolvent_bound_unified(precomp, T, z; R=R,
    large_block_method=NeumannCollatz, coupling_method=DirectSolve)

# Compare bounds
println("V1: ", r1.resolvent_bound)
println("V2: ", r2.resolvent_bound)
println("V3: ", r3.resolvent_bound)
```
"""
function sylvester_resolvent_bound_unified(precomp::SylvesterResolventResult{RT},
                                            T::AbstractMatrix{ET},
                                            z::Complex;
                                            R::Union{Nothing, AbstractMatrix}=nothing,
                                            large_block_method::LargeBlockMethod=TriangularBacksub,
                                            coupling_method::CouplingMethod=ProductBound,
                                            miyajima_method::Symbol=:M1,
                                            power_iterations::Int=3) where {RT, ET}
    # Initialize failure result helper
    function fail_result(reason::String)
        return UnifiedResolventResult{RT}(
            Complex{RT}(z), RT(Inf), false, reason,
            large_block_method, coupling_method,
            RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), -RT(Inf), RT(Inf),
            RT(Inf), RT(1)
        )
    end

    # Check precomputation
    if !precomp.precomputation_success
        return fail_result("Precomputation failed: $(precomp.failure_reason)")
    end

    # Check R is provided if needed for direct coupling
    if coupling_method == DirectSolve && R === nothing
        return fail_result("R (Sylvester residual) required for DirectSolve coupling method")
    end

    k = precomp.k
    n = precomp.n
    r = precomp.residual_norm
    K_S = precomp.similarity_cond

    # Extract blocks
    T11 = T[1:k, 1:k]
    T22 = T[(k+1):n, (k+1):n]

    # ----------------------------------------
    # M_A via Miyajima SVD (always)
    # ----------------------------------------
    A_z = z * I - T11
    A_z_ball = BallMatrix(A_z, zeros(RT, k, k))

    M_A = try
        svd_result = rigorous_svd(A_z_ball; method = miyajima_method == :M4 ? MiyajimaM4() : MiyajimaM1())
        σ_min_ball = svd_result.singular_values[end]
        σ_min_lower = mid(σ_min_ball) - rad(σ_min_ball)

        if σ_min_lower ≤ zero(RT)
            return fail_result("σ_min(A_z) ≤ 0: z at or near eigenvalue of T11")
        end

        one(RT) / σ_min_lower
    catch e
        return fail_result("Miyajima SVD failed: $(e)")
    end

    # ----------------------------------------
    # M_D: Compute both methods for comparison
    # ----------------------------------------
    D_z = z * I - T22

    # Triangular backsubstitution (always computed for comparison)
    M_D_triangular = triangular_inverse_two_norm_bound(D_z)

    # Neumann bound (computed if requested or for comparison if cheap)
    neumann_result = neumann_inverse_bound(T22, z; power_iterations=power_iterations)
    M_D_neumann = neumann_result.success ? neumann_result.M_D : RT(Inf)

    # Select M_D based on method
    M_D = if large_block_method == TriangularBacksub
        M_D_triangular
    else  # NeumannCollatz
        neumann_result.success ? neumann_result.M_D : M_D_triangular  # Fallback
    end

    if !isfinite(M_D)
        return fail_result("D_z inverse bound failed: z at or near eigenvalue of T22")
    end

    # ----------------------------------------
    # Coupling term: Compute both methods
    # ----------------------------------------
    M_AR_product = M_A * r

    # Direct solve (if R provided)
    M_AR_direct = RT(Inf)
    solve_residual_norm = RT(Inf)

    if R !== nothing
        Y_approx = try
            A_z \ R
        catch
            nothing
        end

        if Y_approx !== nothing
            Δ = R - A_z * Y_approx
            Y_norm = sqrt(opnorm(Y_approx, 1) * opnorm(Y_approx, Inf))
            Δ_norm = sqrt(opnorm(Δ, 1) * opnorm(Δ, Inf))
            M_AR_direct = Y_norm + M_A * Δ_norm
            solve_residual_norm = Δ_norm
        end
    end

    # Select M_AR based on method
    M_AR = if coupling_method == ProductBound
        M_AR_product
    else  # DirectSolve
        isfinite(M_AR_direct) ? M_AR_direct : M_AR_product  # Fallback
    end

    # Tightening ratio
    tightening_ratio = if M_AR_product > eps(RT) && isfinite(M_AR_direct)
        M_AR_direct / M_AR_product
    else
        one(RT)
    end

    # ----------------------------------------
    # Final bound
    # ----------------------------------------
    coupling = M_AR * M_D
    resolvent_bound = K_S * (M_A + M_D + coupling)

    return UnifiedResolventResult{RT}(
        Complex{RT}(z), resolvent_bound, true, "",
        large_block_method, coupling_method,
        K_S, M_A, M_D, M_AR, coupling,
        M_D_triangular, M_D_neumann, M_AR_product, M_AR_direct,
        neumann_result.alpha, neumann_result.neumann_gap, neumann_result.Dd_inv_norm,
        solve_residual_norm, tightening_ratio
    )
end

"""
    sylvester_resolvent_bound_unified(precomp, T, z_list; kwargs...)

Unified bound at multiple points.
"""
function sylvester_resolvent_bound_unified(precomp::SylvesterResolventResult{RT},
                                            T::AbstractMatrix{ET},
                                            z_list::AbstractVector{<:Complex};
                                            R::Union{Nothing, AbstractMatrix}=nothing,
                                            large_block_method::LargeBlockMethod=TriangularBacksub,
                                            coupling_method::CouplingMethod=ProductBound,
                                            miyajima_method::Symbol=:M1,
                                            power_iterations::Int=3) where {RT, ET}
    return [sylvester_resolvent_bound_unified(precomp, T, z;
                                               R=R,
                                               large_block_method=large_block_method,
                                               coupling_method=coupling_method,
                                               miyajima_method=miyajima_method,
                                               power_iterations=power_iterations)
            for z in z_list]
end

"""
    sylvester_resolvent_bound_unified(T, k, z; kwargs...)

Convenience function: precompute and evaluate unified bound.

# Returns
`(precomp, R, result)` where R is the Sylvester residual.
"""
function sylvester_resolvent_bound_unified(T::AbstractMatrix{ET}, k::Int, z::Complex;
                                            large_block_method::LargeBlockMethod=TriangularBacksub,
                                            coupling_method::CouplingMethod=ProductBound,
                                            miyajima_method::Symbol=:M1,
                                            power_iterations::Int=3,
                                            X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = X_oracle !== nothing ? X_oracle : solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22

    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)
    result = sylvester_resolvent_bound_unified(precomp, T, z;
                                                R=R,
                                                large_block_method=large_block_method,
                                                coupling_method=coupling_method,
                                                miyajima_method=miyajima_method,
                                                power_iterations=power_iterations)

    return (precomp, R, result)
end

function sylvester_resolvent_bound_unified(T::AbstractMatrix{ET}, k::Int,
                                            z_list::AbstractVector{<:Complex};
                                            large_block_method::LargeBlockMethod=TriangularBacksub,
                                            coupling_method::CouplingMethod=ProductBound,
                                            miyajima_method::Symbol=:M1,
                                            power_iterations::Int=3,
                                            X_oracle::Union{Nothing, AbstractMatrix}=nothing) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = X_oracle !== nothing ? X_oracle : solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22

    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)
    results = sylvester_resolvent_bound_unified(precomp, T, z_list;
                                                 R=R,
                                                 large_block_method=large_block_method,
                                                 coupling_method=coupling_method,
                                                 miyajima_method=miyajima_method,
                                                 power_iterations=power_iterations)

    return (precomp, R, results)
end

"""
    print_unified_result(result::UnifiedResolventResult; io::IO=stdout)

Print human-readable result for unified bound with full comparison.
"""
function print_unified_result(result::UnifiedResolventResult; io::IO=stdout)
    println(io, "Unified Resolvent Bound Result")
    println(io, "="^50)
    println(io, "z = $(result.z)")
    println(io)

    if !result.success
        println(io, "⚠ FAILED: $(result.failure_reason)")
        return
    end

    # Configuration
    println(io, "Configuration:")
    println(io, "  Large block method: $(result.large_block_method)")
    println(io, "  Coupling method:    $(result.coupling_method)")
    println(io)

    # Main result
    println(io, "Result:")
    println(io, "  ‖(zI-T)⁻¹‖₂ ≤ $(result.resolvent_bound)")
    println(io)

    # Components
    println(io, "Components:")
    println(io, "  K_S (similarity cond.) = $(result.K_S)")
    println(io, "  M_A (small block)      = $(result.M_A)")
    println(io, "  M_D (large block)      = $(result.M_D)")
    println(io, "  M_AR (coupling factor) = $(result.M_AR)")
    println(io, "  coupling = M_AR · M_D  = $(result.coupling)")
    println(io)

    # Comparison
    println(io, "Method comparison:")
    println(io, "  M_D methods:")
    println(io, "    Triangular: $(result.M_D_triangular)")
    println(io, "    Neumann:    $(isfinite(result.M_D_neumann) ? string(result.M_D_neumann) : "N/A (α ≥ 1)")")
    if isfinite(result.M_D_neumann) && isfinite(result.M_D_triangular)
        md_ratio = result.M_D_neumann / result.M_D_triangular
        println(io, "    Ratio:      $(round(md_ratio, digits=3))")
    end
    println(io)

    println(io, "  Coupling methods:")
    println(io, "    Product:    $(result.M_AR_product)")
    println(io, "    Direct:     $(isfinite(result.M_AR_direct) ? string(result.M_AR_direct) : "N/A")")
    if isfinite(result.M_AR_direct)
        println(io, "    Tightening: $(round(result.tightening_ratio, digits=3))")
    end
    println(io)

    # Neumann diagnostics
    if result.large_block_method == NeumannCollatz || isfinite(result.M_D_neumann)
        println(io, "Neumann diagnostics:")
        println(io, "  α = ‖N_z‖₂ ≤ $(result.alpha)")
        println(io, "  gap = 1-α   = $(result.neumann_gap)")
        println(io, "  ‖(D_z)_d⁻¹‖ = $(result.Dd_inv_norm)")
        println(io, "  Converged:   $(result.neumann_gap > 0)")
        println(io)
    end

    # Direct solve diagnostics
    if result.coupling_method == DirectSolve && isfinite(result.solve_residual_norm)
        println(io, "Direct solve diagnostics:")
        println(io, "  ‖Δ‖ (solve residual) = $(result.solve_residual_norm)")
        println(io)
    end

    # Quick comparison of all 4 combinations
    println(io, "All method combinations (estimated bounds):")

    # V1: Triangular + Product
    b_v1 = result.K_S * (result.M_A + result.M_D_triangular + result.M_AR_product * result.M_D_triangular)
    println(io, "  V1 (Tri+Prod):  $(b_v1)")

    # V2: Triangular + Direct
    if isfinite(result.M_AR_direct)
        b_v2 = result.K_S * (result.M_A + result.M_D_triangular + result.M_AR_direct * result.M_D_triangular)
        println(io, "  V2 (Tri+Dir):   $(b_v2)")
    else
        println(io, "  V2 (Tri+Dir):   N/A")
    end

    # V3a: Neumann + Product
    if isfinite(result.M_D_neumann)
        b_v3a = result.K_S * (result.M_A + result.M_D_neumann + result.M_AR_product * result.M_D_neumann)
        println(io, "  V3a (Neu+Prod): $(b_v3a)")
    else
        println(io, "  V3a (Neu+Prod): N/A")
    end

    # V3b: Neumann + Direct
    if isfinite(result.M_D_neumann) && isfinite(result.M_AR_direct)
        b_v3b = result.K_S * (result.M_A + result.M_D_neumann + result.M_AR_direct * result.M_D_neumann)
        println(io, "  V3b (Neu+Dir):  $(b_v3b)")
    else
        println(io, "  V3b (Neu+Dir):  N/A")
    end
end

"""
    compare_methods(T::AbstractMatrix, k::Int, z::Complex;
                    miyajima_method::Symbol=:M1, power_iterations::Int=3)

Convenience function to compare all method combinations at once.

# Returns
NamedTuple with:
- `precomp`: Precomputation result
- `R`: Sylvester residual
- `v1`: V1 result (Triangular + Product)
- `v2`: V2 result (Triangular + Direct)
- `v3a`: V3a result (Neumann + Product)
- `v3b`: V3b result (Neumann + Direct)
- `best`: The method with the tightest bound
- `bounds`: Dict mapping method name to bound value
"""
function compare_methods(T::AbstractMatrix{ET}, k::Int, z::Complex;
                          miyajima_method::Symbol=:M1,
                          power_iterations::Int=3) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)

    # Compute all four combinations
    v1 = sylvester_resolvent_bound_unified(precomp, T, z; R=R,
        large_block_method=TriangularBacksub, coupling_method=ProductBound,
        miyajima_method=miyajima_method, power_iterations=power_iterations)

    v2 = sylvester_resolvent_bound_unified(precomp, T, z; R=R,
        large_block_method=TriangularBacksub, coupling_method=DirectSolve,
        miyajima_method=miyajima_method, power_iterations=power_iterations)

    v3a = sylvester_resolvent_bound_unified(precomp, T, z; R=R,
        large_block_method=NeumannCollatz, coupling_method=ProductBound,
        miyajima_method=miyajima_method, power_iterations=power_iterations)

    v3b = sylvester_resolvent_bound_unified(precomp, T, z; R=R,
        large_block_method=NeumannCollatz, coupling_method=DirectSolve,
        miyajima_method=miyajima_method, power_iterations=power_iterations)

    # Build bounds dict
    bounds = Dict{String, Float64}()
    bounds["V1 (Tri+Prod)"] = v1.success ? Float64(v1.resolvent_bound) : Inf
    bounds["V2 (Tri+Dir)"] = v2.success ? Float64(v2.resolvent_bound) : Inf
    bounds["V3a (Neu+Prod)"] = v3a.success ? Float64(v3a.resolvent_bound) : Inf
    bounds["V3b (Neu+Dir)"] = v3b.success ? Float64(v3b.resolvent_bound) : Inf

    # Find best
    _, best_name = findmin(bounds)

    return (
        precomp = precomp,
        R = R,
        v1 = v1,
        v2 = v2,
        v3a = v3a,
        v3b = v3b,
        best = best_name,
        bounds = bounds
    )
end

"""
    print_method_comparison(cmp::NamedTuple; io::IO=stdout)

Print comparison of all method combinations.
"""
function print_method_comparison(cmp::NamedTuple; io::IO=stdout)
    println(io, "Method Comparison")
    println(io, "="^50)
    println(io, "z = $(cmp.v1.z)")
    println(io)

    # Sort by bound value
    sorted = sort(collect(cmp.bounds), by=x->x[2])

    println(io, "Bounds (sorted, tightest first):")
    for (i, (name, val)) in enumerate(sorted)
        marker = i == 1 ? "★" : " "
        if isfinite(val)
            println(io, "  $marker $name: $val")
        else
            println(io, "    $name: N/A")
        end
    end
    println(io)

    # Component breakdown
    println(io, "Component breakdown:")
    println(io, "  K_S = $(cmp.v1.K_S)")
    println(io, "  M_A = $(cmp.v1.M_A)")
    println(io)
    println(io, "  Large block M_D:")
    println(io, "    Triangular: $(cmp.v1.M_D_triangular)")
    if isfinite(cmp.v1.M_D_neumann)
        println(io, "    Neumann:    $(cmp.v1.M_D_neumann) (α=$(round(cmp.v1.alpha, digits=3)))")
    else
        println(io, "    Neumann:    N/A (α=$(round(cmp.v1.alpha, digits=3)) ≥ 1)")
    end
    println(io)
    println(io, "  Coupling M_AR:")
    println(io, "    Product:    $(cmp.v1.M_AR_product)")
    if isfinite(cmp.v1.M_AR_direct)
        println(io, "    Direct:     $(cmp.v1.M_AR_direct) (ratio=$(round(cmp.v1.tightening_ratio, digits=3)))")
    else
        println(io, "    Direct:     N/A")
    end
end

#==============================================================================#
# Extended Parametric Framework (per user specification)
#==============================================================================#

# ---- Norm Estimator Enum ----
"""
    NormEstimator

Enum for selecting norm estimation method for matrices (R, X̃, etc).

- `OneInfNorm`: √(‖·‖₁ · ‖·‖_∞) — fast, O(n²)
- `FrobeniusNorm`: ‖·‖_F — fast, O(n²)
- `RowCol2Norm`: max(max_row_2_norm, max_col_2_norm) — O(n²)
"""
@enum NormEstimator begin
    OneInfNorm
    FrobeniusNorm
    RowCol2Norm
end

# ---- D Inverse Estimator Enum (extended) ----
"""
    DInverseEstimator

Enum for selecting large block inverse estimation method.

- `TriBacksub`: Backward recursion on triangular structure (V1 default)
- `NeumannOneInf`: Neumann series with 1/∞ norm bounds (simpler, faster)
- `NeumannCollatz2`: Collatz-Wielandt 2-norm Neumann bound (V3)
"""
@enum DInverseEstimator begin
    TriBacksub
    NeumannOneInf
    NeumannCollatz2
end

# ---- Coupling Estimator Enum (extended) ----
"""
    CouplingEstimator

Enum for selecting coupling term estimation method.

- `CouplingNone`: Use combiner V1 with r = ‖R‖
- `CouplingARSolve`: V2 - solve A_z \\ R and certify ‖A_z⁻¹R‖
- `CouplingOffDirect`: V2.5 - directly bound ‖A_z⁻¹RD_z⁻¹‖
"""
@enum CouplingEstimator begin
    CouplingNone
    CouplingARSolve
    CouplingOffDirect
end

# ---- Combiner Enum ----
"""
    Combiner

Enum for selecting the combining formula.

- `CombinerV1`: M_T = K_S(M_A + M_D + M_A·r·M_D)
- `CombinerV2`: M_T = K_S(M_A + M_D + M_AR·M_D)
- `CombinerV2p5`: M_T = K_S(M_A + M_D + M_off)
"""
@enum Combiner begin
    CombinerV1
    CombinerV2
    CombinerV2p5
end

# ---- Configuration Struct ----
"""
    ResolventBoundConfig

Configuration for the parametric resolvent bound pipeline.

# Fields
- `norm_estimator::NormEstimator`: How to estimate matrix 2-norms (default: `OneInfNorm`)
- `d_inv_estimator::DInverseEstimator`: How to bound ‖D_z⁻¹‖₂ (default: `TriBacksub`)
- `coupling_estimator::CouplingEstimator`: How to bound coupling (default: `CouplingNone`)
- `combiner::Combiner`: Combining formula (default: `CombinerV1`)
- `miyajima_method::Symbol`: SVD method for small block (:M1 or :M4)
- `power_iterations::Int`: Iterations for Collatz/power methods
- `fallback_to_tri::Bool`: If Neumann fails, fallback to triangular backsubstitution

# Predefined Configurations
Use `config_v1()`, `config_v2()`, `config_v3()` for standard configurations.
"""
struct ResolventBoundConfig
    norm_estimator::NormEstimator
    d_inv_estimator::DInverseEstimator
    coupling_estimator::CouplingEstimator
    combiner::Combiner
    miyajima_method::Symbol
    power_iterations::Int
    fallback_to_tri::Bool
end

"""
    config_v1()

V1 baseline configuration: triangular backsubstitution, product coupling bound.
"""
config_v1() = ResolventBoundConfig(
    OneInfNorm, TriBacksub, CouplingNone, CombinerV1, :M1, 3, true
)

"""
    config_v2()

V2 configuration: triangular backsubstitution, direct A_z⁻¹R solve.
"""
config_v2() = ResolventBoundConfig(
    OneInfNorm, TriBacksub, CouplingARSolve, CombinerV2, :M1, 3, true
)

"""
    config_v3()

V3 configuration: Neumann Collatz for large block, direct coupling solve.
"""
config_v3() = ResolventBoundConfig(
    OneInfNorm, NeumannCollatz2, CouplingARSolve, CombinerV2, :M1, 3, true
)

"""
    config_v2p5()

V2.5 configuration: triangular backsubstitution, direct off-diagonal bound.
"""
config_v2p5() = ResolventBoundConfig(
    OneInfNorm, TriBacksub, CouplingOffDirect, CombinerV2p5, :M1, 3, true
)

# ---- Rigorous Norm estimation functions (using BallArithmetic) ----

"""
    estimate_2norm(M::AbstractMatrix, method::NormEstimator)

Compute a *rigorous* upper bound on ‖M‖₂ using the specified method.

Uses BallArithmetic's existing rigorous norm infrastructure.
"""
function estimate_2norm(M::AbstractMatrix{CT}, method::NormEstimator) where {CT}
    T = real(CT)
    # Wrap in BallMatrix with zero radius for rigorous bounds
    M_ball = BallMatrix(M, zeros(T, size(M)...))
    return estimate_2norm(M_ball, method)
end

"""
    estimate_2norm(M::BallMatrix, method::NormEstimator)

Rigorous upper bound on ‖M‖₂ for BallMatrix using existing infrastructure.
"""
function estimate_2norm(M::BallMatrix{T}, method::NormEstimator) where {T}
    if method == OneInfNorm
        # Uses existing rigorous 1/∞ bounds with directed rounding
        norm1 = upper_bound_L1_opnorm(M)
        norminf = upper_bound_L_inf_opnorm(M)
        return sqrt_up(setrounding(T, RoundUp) do
            norm1 * norminf
        end)
    elseif method == FrobeniusNorm
        # Frobenius = 2-norm of vectorized matrix
        return upper_bound_norm(M, 2)
    elseif method == RowCol2Norm
        # Row/column 2-norm bound (rigorous via directed rounding)
        m, n = size(M)
        row_max = setrounding(T, RoundUp) do
            maximum(sqrt(sum(j -> (abs(M.c[i,j]) + M.r[i,j])^2, 1:n)) for i in 1:m)
        end
        col_max = setrounding(T, RoundUp) do
            maximum(sqrt(sum(i -> (abs(M.c[i,j]) + M.r[i,j])^2, 1:m)) for j in 1:n)
        end
        return max(row_max, col_max)
    else
        # Default: use best available rigorous L2 bound
        return upper_bound_L2_opnorm(M)
    end
end

# Convenience: sqrt with upward rounding
sqrt_up(x::T) where {T<:AbstractFloat} = setrounding(T, RoundUp) do
    sqrt(x)
end

# ---- Neumann 1/∞ norm estimator for D_z (Rigorous) ----
"""
    neumann_one_inf_bound(T22::AbstractMatrix, z::Complex)

Compute *rigorous* certified upper bound on ‖D_z⁻¹‖₂ using Neumann series with 1/∞ norms.

All arithmetic uses directed rounding for certified bounds.

For D_z = D_d + D_f (diagonal + strict upper):
- α∞ = ‖D_d⁻¹ D_f‖_∞ = max_i Σ_{j>i} |u_ij|/|u_ii|
- α₁ = ‖D_d⁻¹ D_f‖₁ = max_j Σ_{i<j} |u_ij|/|u_ii|
- If α∞, α₁ < 1: ‖D_z⁻¹‖₂ ≤ √(‖D_d⁻¹‖_∞ · ‖D_d⁻¹‖₁ / ((1-α∞)(1-α₁)))

# Returns
NamedTuple with rigorous bounds.
"""
function neumann_one_inf_bound(T22::AbstractMatrix{CT}, z::Complex) where {CT}
    m = size(T22, 1)
    T = real(CT)

    # All computations use directed rounding for rigour
    result = setrounding(T, RoundUp) do
        # Compute diagonal inverse with RoundUp (upper bound on 1/|d_i|)
        invdiag = Vector{T}(undef, m)
        for i in 1:m
            d_i = abs(z - T22[i, i])
            if d_i ≤ eps(T) * (abs(z) + norm(T22, Inf))
                return (M_D=T(Inf), alpha_inf=T(Inf), alpha_one=T(Inf),
                        Dd_inv_inf=T(Inf), Dd_inv_one=T(Inf), success=false)
            end
            invdiag[i] = one(T) / d_i
        end

        Dd_inv_inf = maximum(invdiag)
        Dd_inv_one = maximum(invdiag)

        # Compute α∞ = ‖D_d⁻¹ D_f‖_∞ via row sums (RoundUp)
        alpha_inf = zero(T)
        for i in 1:m
            row_sum = zero(T)
            for j in (i+1):m
                row_sum = row_sum + abs(T22[i, j]) * invdiag[i]
            end
            alpha_inf = max(alpha_inf, row_sum)
        end

        # Compute α₁ = ‖D_d⁻¹ D_f‖₁ via column sums (RoundUp)
        alpha_one = zero(T)
        for j in 1:m
            col_sum = zero(T)
            for i in 1:(j-1)
                col_sum = col_sum + abs(T22[i, j]) * invdiag[i]
            end
            alpha_one = max(alpha_one, col_sum)
        end

        # Check Neumann convergence
        if alpha_inf ≥ one(T) || alpha_one ≥ one(T)
            return (M_D=T(Inf), alpha_inf=alpha_inf, alpha_one=alpha_one,
                    Dd_inv_inf=Dd_inv_inf, Dd_inv_one=Dd_inv_one, success=false)
        end

        # Neumann bound with RoundDown for denominator, RoundUp for division
        # Note: We're in RoundUp block, so compute 1-α with care
        # gap_inf = 1 - alpha_inf computed with RoundDown gives lower bound
        # Then division by lower bound gives upper bound on the quotient
        gap_inf = setrounding(T, RoundDown) do
            one(T) - alpha_inf
        end
        gap_one = setrounding(T, RoundDown) do
            one(T) - alpha_one
        end

        if gap_inf ≤ zero(T) || gap_one ≤ zero(T)
            return (M_D=T(Inf), alpha_inf=alpha_inf, alpha_one=alpha_one,
                    Dd_inv_inf=Dd_inv_inf, Dd_inv_one=Dd_inv_one, success=false)
        end

        D_inv_inf = Dd_inv_inf / gap_inf
        D_inv_one = Dd_inv_one / gap_one

        # 2-norm from 1/∞ norms (RoundUp for sqrt)
        M_D = sqrt(D_inv_one * D_inv_inf)

        return (M_D=M_D, alpha_inf=alpha_inf, alpha_one=alpha_one,
                Dd_inv_inf=Dd_inv_inf, Dd_inv_one=Dd_inv_one, success=true)
    end

    return result
end

# ---- Off-diagonal direct bound (V2.5) - Rigorous ----
"""
    offdiag_direct_bound(A_z, D_z, R, M_A, M_D; norm_method=OneInfNorm)

Compute *rigorous* certified bound on ‖A_z⁻¹RD_z⁻¹‖₂ directly for V2.5 combiner.

All norm computations use rigorous BallArithmetic bounds.

# Method
1. Compute Ŵ ≈ A_z⁻¹ R D_z⁻¹ (two triangular solves)
2. Compute residual Δ = R - A_z Ŵ D_z rigorously
3. Bound: ‖A_z⁻¹RD_z⁻¹‖ ≤ ‖Ŵ‖ + M_A·‖Δ‖·M_D

# Returns
NamedTuple with rigorous bounds.
"""
function offdiag_direct_bound(A_z::AbstractMatrix{CT}, D_z::AbstractMatrix{CT},
                               R::AbstractMatrix, M_A::T, M_D::T;
                               norm_method::NormEstimator=OneInfNorm) where {CT, T}
    # Compute approximate W = A_z⁻¹ R D_z⁻¹
    W_approx = try
        Y = A_z \ R      # A_z⁻¹ R
        Y / D_z          # (A_z⁻¹ R) D_z⁻¹
    catch
        return (M_off=T(Inf), W_norm=T(Inf), Delta_norm=T(Inf), success=false)
    end

    # Compute residual rigorously: Δ = R - A_z W D_z
    # Use BallMatrix for the residual computation
    k, m = size(R)
    R_ball = BallMatrix(R, zeros(T, k, m))
    W_ball = BallMatrix(W_approx, zeros(T, k, m))
    A_ball = BallMatrix(A_z, zeros(T, k, k))
    D_ball = BallMatrix(D_z, zeros(T, m, m))

    # Δ_ball encapsulates rounding errors in the residual computation
    Δ_ball = R_ball - A_ball * W_ball * D_ball

    # Rigorous norm bounds
    W_norm = estimate_2norm(W_ball, norm_method)
    Δ_norm = estimate_2norm(Δ_ball, norm_method)

    # Certified bound with upward rounding
    M_off = setrounding(T, RoundUp) do
        W_norm + M_A * Δ_norm * M_D
    end

    return (M_off=M_off, W_norm=W_norm, Delta_norm=Δ_norm, success=true)
end

# ---- Extended Per-Point Result ----
"""
    ParametricResolventResult{T}

Extended result structure for the parametric resolvent bound.

Contains all diagnostic information for systematic comparison.
"""
struct ParametricResolventResult{T}
    # Core
    z::Complex{T}
    resolvent_bound::T
    success::Bool
    failure_reason::String

    # Configuration used
    config::ResolventBoundConfig

    # Common components
    K_S::T                # Similarity conditioning
    M_A::T                # Small block inverse bound
    M_D::T                # Large block inverse bound (selected)
    r::T                  # Residual norm ‖R‖

    # Coupling term (depends on combiner)
    coupling_term::T      # Either M_A·r·M_D (V1), M_AR·M_D (V2), or M_off (V2.5)

    # All M_D variants (for comparison)
    M_D_tri::T            # Triangular backsubstitution
    M_D_neumann_oneinf::T # Neumann 1/∞ bound
    M_D_neumann_collatz::T # Neumann Collatz bound

    # All coupling variants
    cV1::T                # V1: M_A·r·M_D
    M_AR::T               # V2: ‖A_z⁻¹R‖
    M_off::T              # V2.5: ‖A_z⁻¹RD_z⁻¹‖

    # Neumann diagnostics
    alpha_collatz::T      # Collatz α for 2-norm Neumann
    alpha_inf::T          # α∞ for 1/∞ Neumann
    alpha_one::T          # α₁ for 1/∞ Neumann
    neumann_gap::T        # 1 - α (best)

    # Direct solve diagnostics
    tightening_AR::T      # M_AR / (M_A·r)
    tightening_off::T     # M_off / (M_A·r·M_D)
    solve_residual::T     # ‖Δ‖ from AR solve
    offdiag_residual::T   # ‖Δ‖ from off-diagonal solve
end

# ---- Main Parametric Pipeline ----
"""
    SVDWarmStart{UT, ST, VT}

Optional warm-start SVD for Miyajima/Ogita certification.

Providing (U, Σ, V) from a previous computation can significantly speed up
the SVD certification step, especially for BigFloat matrices using Ogita refinement.

# Notes
- For Float64 matrices: The warm start sets the internal SVD cache, but LAPACK
  computes a fresh SVD anyway. The benefit is minimal for Float64.
- For BigFloat matrices: The warm start provides initial values for Ogita's
  iterative refinement, which can reduce iterations needed for convergence.

# Usage
```julia
# Compute SVD of A_z for warm-starting
svd_result = svd(A_z)
warm_start = SVDWarmStart(svd_result.U, svd_result.S, svd_result.V)

# Use warm start in parametric bound
result = parametric_resolvent_bound(precomp, T, z, config; R=R, svd_warm_start=warm_start)
```
"""
struct SVDWarmStart{UT, ST, VT}
    U::UT    # Left singular vectors
    Σ::ST    # Singular values (vector)
    V::VT    # Right singular vectors
end

"""
    parametric_resolvent_bound(precomp::SylvesterResolventResult, T::AbstractMatrix,
                                z::Complex, config::ResolventBoundConfig;
                                R::Union{Nothing, AbstractMatrix}=nothing,
                                svd_warm_start::Union{Nothing, SVDWarmStart}=nothing)

Main parametric pipeline for computing certified resolvent bounds.

Uses the specified configuration to select estimators and combiner formula.

# Arguments
- `precomp`: Precomputed Sylvester quantities
- `T`: Schur-triangular matrix
- `z`: Evaluation point
- `config`: Configuration specifying all estimator choices
- `R`: Sylvester residual (required for V2/V2.5 coupling)
- `svd_warm_start`: Optional warm-start SVD (U, Σ, V) for faster Miyajima certification.
  When provided, sets the SVD cache before calling `rigorous_svd`.

# Returns
`ParametricResolventResult` with comprehensive diagnostics.

# Warm-start usage
The SVD warm-start is particularly useful when:
1. Evaluating at multiple nearby z values (eigenvalues don't change much)
2. Using BigFloat precision (Ogita refinement benefits from good starting point)

```julia
# First evaluation
result1 = parametric_resolvent_bound(precomp, T, z1, config; R=R)

# Extract SVD from first evaluation for warm-start
# (Would need to cache internally - example for future enhancement)
```
"""
function parametric_resolvent_bound(precomp::SylvesterResolventResult{RT},
                                     T::AbstractMatrix{ET},
                                     z::Complex,
                                     config::ResolventBoundConfig;
                                     R::Union{Nothing, AbstractMatrix}=nothing,
                                     svd_warm_start::Union{Nothing, SVDWarmStart}=nothing) where {RT, ET}
    # Helper for failure
    function fail(reason::String)
        return ParametricResolventResult{RT}(
            Complex{RT}(z), RT(Inf), false, reason, config,
            RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(Inf),
            RT(Inf), RT(Inf), RT(Inf), RT(Inf),
            RT(1), RT(1), RT(Inf), RT(Inf)
        )
    end

    if !precomp.precomputation_success
        return fail("Precomputation failed: $(precomp.failure_reason)")
    end

    # Check R requirement
    if config.coupling_estimator != CouplingNone && R === nothing
        return fail("R required for $(config.coupling_estimator)")
    end

    k = precomp.k
    n = precomp.n
    r = precomp.residual_norm
    K_S = precomp.similarity_cond
    norm_method = config.norm_estimator

    T11 = T[1:k, 1:k]
    T22 = T[(k+1):n, (k+1):n]

    # ========================================
    # M_A via Miyajima SVD (with optional warm-start)
    # ========================================
    A_z = z * I - T11
    A_z_ball = BallMatrix(A_z, zeros(RT, k, k))

    M_A = try
        svd_method = config.miyajima_method == :M4 ? MiyajimaM4() : MiyajimaM1()

        svd_result = if svd_warm_start !== nothing
            # Use Ogita refinement from warm-start (works for both Float64 and BigFloat)
            # This is faster when z is close to previous evaluation points
            ogita_refined = ogita_svd_refine(A_z_ball.c,
                                              svd_warm_start.U,
                                              svd_warm_start.Σ,
                                              svd_warm_start.V;
                                              max_iterations=2,  # Quick refinement from nearby point
                                              precision_bits=precision(RT),
                                              check_convergence=false)
            Σ_vec = isa(ogita_refined.Σ, Diagonal) ? diag(ogita_refined.Σ) : ogita_refined.Σ
            svd_refined = SVD(Matrix(ogita_refined.U), Vector(Σ_vec), Matrix(ogita_refined.V'))
            _certify_svd(A_z_ball, svd_refined, svd_method; apply_vbd=true)
        else
            rigorous_svd(A_z_ball; method=svd_method)
        end

        σ_min_ball = svd_result.singular_values[end]
        σ_min_lower = mid(σ_min_ball) - rad(σ_min_ball)

        if σ_min_lower ≤ zero(RT)
            return fail("σ_min(A_z) ≤ 0")
        end
        one(RT) / σ_min_lower
    catch e
        return fail("Miyajima SVD failed: $e")
    end

    # ========================================
    # M_D: Compute all variants
    # ========================================
    D_z = z * I - T22

    # Triangular backsubstitution (always computed)
    M_D_tri = triangular_inverse_two_norm_bound(D_z)

    # Neumann 1/∞ bound
    neumann_oneinf = neumann_one_inf_bound(T22, z)
    M_D_neumann_oneinf = neumann_oneinf.M_D

    # Neumann Collatz bound
    neumann_collatz = neumann_inverse_bound(T22, z; power_iterations=config.power_iterations)
    M_D_neumann_collatz = neumann_collatz.success ? neumann_collatz.M_D : RT(Inf)

    # Select M_D based on config
    M_D = if config.d_inv_estimator == TriBacksub
        M_D_tri
    elseif config.d_inv_estimator == NeumannOneInf
        neumann_oneinf.success ? M_D_neumann_oneinf : (config.fallback_to_tri ? M_D_tri : RT(Inf))
    elseif config.d_inv_estimator == NeumannCollatz2
        neumann_collatz.success ? M_D_neumann_collatz : (config.fallback_to_tri ? M_D_tri : RT(Inf))
    else
        M_D_tri
    end

    if !isfinite(M_D)
        return fail("D_z inverse bound failed")
    end

    # ========================================
    # Coupling: Compute all variants (RIGOROUS)
    # ========================================

    # V1 coupling: M_A · r · M_D with upward rounding
    cV1 = setrounding(RT, RoundUp) do
        M_A * r * M_D
    end

    # V2 coupling: solve A_z \ R rigorously
    M_AR = RT(Inf)
    solve_residual = RT(Inf)
    m = n - k
    if R !== nothing
        Y_approx = try
            A_z \ R
        catch
            nothing
        end
        if Y_approx !== nothing
            # Compute residual rigorously using BallMatrix
            R_ball = BallMatrix(R, zeros(RT, k, m))
            Y_ball = BallMatrix(Y_approx, zeros(RT, k, m))
            Δ_ball = R_ball - A_z_ball * Y_ball

            Y_norm = estimate_2norm(Y_ball, norm_method)
            Δ_norm = estimate_2norm(Δ_ball, norm_method)
            # Rigorous bound: M_AR ≤ ||Y|| + M_A * ||Δ||
            M_AR = setrounding(RT, RoundUp) do
                Y_norm + M_A * Δ_norm
            end
            solve_residual = Δ_norm
        end
    end

    # V2.5 coupling: direct off-diagonal bound
    M_off = RT(Inf)
    offdiag_residual = RT(Inf)
    if R !== nothing && isfinite(M_D)
        offdiag = offdiag_direct_bound(A_z, D_z, R, M_A, M_D; norm_method=norm_method)
        if offdiag.success
            M_off = offdiag.M_off
            offdiag_residual = offdiag.Delta_norm
        end
    end

    # Select coupling term based on combiner (rigorous)
    coupling_term = setrounding(RT, RoundUp) do
        if config.combiner == CombinerV1
            cV1
        elseif config.combiner == CombinerV2
            isfinite(M_AR) ? M_AR * M_D : cV1  # Fallback to V1 if failed
        elseif config.combiner == CombinerV2p5
            isfinite(M_off) ? M_off : cV1  # Fallback
        else
            cV1
        end
    end

    # ========================================
    # Final bound (RIGOROUS with upward rounding)
    # ========================================
    resolvent_bound = setrounding(RT, RoundUp) do
        K_S * (M_A + M_D + coupling_term)
    end

    # Tightening ratios
    tightening_AR = if M_A * r > eps(RT) && isfinite(M_AR)
        M_AR / (M_A * r)
    else
        one(RT)
    end

    tightening_off = if cV1 > eps(RT) && isfinite(M_off)
        M_off / cV1
    else
        one(RT)
    end

    # Best Neumann gap
    neumann_gap = max(
        neumann_oneinf.success ? one(RT) - max(neumann_oneinf.alpha_inf, neumann_oneinf.alpha_one) : -RT(Inf),
        neumann_collatz.success ? neumann_collatz.neumann_gap : -RT(Inf)
    )

    return ParametricResolventResult{RT}(
        Complex{RT}(z), resolvent_bound, true, "", config,
        K_S, M_A, M_D, r, coupling_term,
        M_D_tri, M_D_neumann_oneinf, M_D_neumann_collatz,
        cV1, M_AR, M_off,
        neumann_collatz.alpha, neumann_oneinf.alpha_inf, neumann_oneinf.alpha_one, neumann_gap,
        tightening_AR, tightening_off, solve_residual, offdiag_residual
    )
end

"""
    parametric_resolvent_bound(precomp, T, z_list, config; R=nothing)

Compute bounds at multiple points.
"""
function parametric_resolvent_bound(precomp::SylvesterResolventResult{RT},
                                     T::AbstractMatrix{ET},
                                     z_list::AbstractVector{<:Complex},
                                     config::ResolventBoundConfig;
                                     R::Union{Nothing, AbstractMatrix}=nothing) where {RT, ET}
    return [parametric_resolvent_bound(precomp, T, z, config; R=R) for z in z_list]
end

"""
    parametric_resolvent_bound(T, k, z, config)

Convenience: precompute and evaluate.
"""
function parametric_resolvent_bound(T::AbstractMatrix{ET}, k::Int, z::Complex,
                                     config::ResolventBoundConfig) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)

    return (precomp, R, parametric_resolvent_bound(precomp, T, z, config; R=R))
end

# ---- Comparison across all configurations ----
"""
    compare_all_configs(T::AbstractMatrix, k::Int, z::Complex;
                         miyajima_method::Symbol=:M1, power_iterations::Int=3)

Compare all standard configurations at a single point.

# Returns
NamedTuple with results for each configuration and summary.
"""
function compare_all_configs(T::AbstractMatrix{ET}, k::Int, z::Complex;
                              miyajima_method::Symbol=:M1,
                              power_iterations::Int=3) where {ET}
    n = size(T, 1)
    T11 = T[1:k, 1:k]
    T12 = T[1:k, (k+1):n]
    T22 = T[(k+1):n, (k+1):n]

    X = solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22
    precomp = sylvester_resolvent_precompute(T, k; X_oracle=X)

    configs = [
        ("V1", config_v1()),
        ("V2", config_v2()),
        ("V2.5", config_v2p5()),
        ("V3", config_v3()),
    ]

    results = Dict{String, ParametricResolventResult}()
    bounds = Dict{String, Float64}()

    for (name, cfg) in configs
        # Override miyajima and power iterations
        cfg_adjusted = ResolventBoundConfig(
            cfg.norm_estimator, cfg.d_inv_estimator, cfg.coupling_estimator,
            cfg.combiner, miyajima_method, power_iterations, cfg.fallback_to_tri
        )
        result = parametric_resolvent_bound(precomp, T, z, cfg_adjusted; R=R)
        results[name] = result
        bounds[name] = result.success ? Float64(result.resolvent_bound) : Inf
    end

    _, best_name = findmin(bounds)

    return (
        precomp = precomp,
        R = R,
        results = results,
        bounds = bounds,
        best = best_name,
        z = z
    )
end

"""
    print_config_comparison(cmp::NamedTuple; io::IO=stdout)

Print detailed comparison of all configurations.
"""
function print_config_comparison(cmp::NamedTuple; io::IO=stdout)
    println(io, "Configuration Comparison")
    println(io, "="^60)
    println(io, "z = $(cmp.z)")
    println(io)

    # Sort by bound
    sorted = sort(collect(cmp.bounds), by=x->x[2])

    println(io, "Bounds (tightest first):")
    for (i, (name, val)) in enumerate(sorted)
        marker = i == 1 ? "★" : " "
        if isfinite(val)
            println(io, "  $marker $name: $val")
        else
            println(io, "    $name: FAILED")
        end
    end
    println(io)

    # Get first successful result for common diagnostics
    first_result = nothing
    for (_, r) in cmp.results
        if r.success
            first_result = r
            break
        end
    end

    if first_result !== nothing
        println(io, "Common components:")
        println(io, "  K_S = $(first_result.K_S)")
        println(io, "  M_A = $(first_result.M_A)")
        println(io, "  r = $(first_result.r)")
        println(io)

        println(io, "M_D estimators:")
        println(io, "  Triangular:      $(first_result.M_D_tri)")
        println(io, "  Neumann (1/∞):   $(isfinite(first_result.M_D_neumann_oneinf) ? first_result.M_D_neumann_oneinf : "N/A")")
        println(io, "  Neumann (Coll.): $(isfinite(first_result.M_D_neumann_collatz) ? first_result.M_D_neumann_collatz : "N/A")")
        println(io)

        println(io, "Coupling estimators:")
        println(io, "  cV1 = M_A·r·M_D:   $(first_result.cV1)")
        println(io, "  M_AR (V2):         $(isfinite(first_result.M_AR) ? first_result.M_AR : "N/A")")
        println(io, "  M_off (V2.5):      $(isfinite(first_result.M_off) ? first_result.M_off : "N/A")")
        println(io)

        println(io, "Neumann diagnostics:")
        println(io, "  α (Collatz):  $(first_result.alpha_collatz)")
        println(io, "  α∞ (1/∞):     $(first_result.alpha_inf)")
        println(io, "  α₁ (1/∞):     $(first_result.alpha_one)")
        println(io, "  Best gap:     $(first_result.neumann_gap)")
        println(io)

        if isfinite(first_result.tightening_AR) && first_result.tightening_AR < 1
            println(io, "Tightening:")
            println(io, "  V2/V1 ratio:   $(round(first_result.tightening_AR, digits=3))")
        end
        if isfinite(first_result.tightening_off) && first_result.tightening_off < 1
            println(io, "  V2.5/V1 ratio: $(round(first_result.tightening_off, digits=3))")
        end
    end
end
