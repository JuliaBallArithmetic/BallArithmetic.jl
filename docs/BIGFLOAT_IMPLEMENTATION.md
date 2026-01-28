# Implementation Plan: BigFloat Support for BallArithmetic.jl

## Goal

Add support for `BigFloat` precision in BallArithmetic.jl, enabling rigorous computations with arbitrary precision. This is needed for applications where Float64 precision is insufficient for the resolvent computations.

## Current Limitations

When a `BallMatrix{BigFloat}` is passed to `run_certification`, the following error occurs:

```
MethodError: no method matching mul_up(::Float64, ::BigFloat)
```

The root cause is that several components are hardcoded for `Float64`.

---

## Implementation Tasks

### Task 1: Parametric Machine Epsilon

**File:** `src/rounding/rounding.jl`

**Current code (line 9):**
```julia
const ϵp = 2.0^-52
```

**Required change:**
```julia
# Machine epsilon for different floating-point types
machine_epsilon(::Type{Float64}) = 2.0^-52
machine_epsilon(::Type{Float32}) = Float32(2.0^-23)
machine_epsilon(::Type{BigFloat}) = BigFloat(2)^(-precision(BigFloat))

# For backwards compatibility, keep ϵp as Float64 default
const ϵp = machine_epsilon(Float64)
```

Then update all uses of `ϵp` to use `machine_epsilon(T)` where `T` is the element type of the Ball.

---

### Task 2: Parametric Rounding Operations for BigFloat

**File:** `src/rounding/rounding.jl`

BigFloat supports native rounding modes. Add:

```julia
# Rounding operations for BigFloat
function add_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x + y
    end
end

function add_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x + y
    end
end

function mul_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x * y
    end
end

function mul_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x * y
    end
end

function div_up(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundUp) do
        x / y
    end
end

function div_down(x::BigFloat, y::BigFloat)
    setrounding(BigFloat, RoundDown) do
        x / y
    end
end

function sqrt_up(x::BigFloat)
    setrounding(BigFloat, RoundUp) do
        sqrt(x)
    end
end

function sqrt_down(x::BigFloat)
    setrounding(BigFloat, RoundDown) do
        sqrt(x)
    end
end
```

---

### Task 3: Update Ball Arithmetic Operations

**File:** `src/types/ball.jl`

The `@up` macro expands to `setrounding(Float64, RoundUp)`. For BigFloat, we need type-aware rounding.

**Option A: Replace macro with function calls**

Update operations like (line 247):
```julia
function Base.:-(x::Ball{T}, y::Ball{T}) where T
    c = mid(x) - mid(y)
    eps_T = machine_epsilon(T)
    r = add_up(mul_up(eps_T, abs(c)) + rad(x), rad(y))
    Ball(c, r)
end
```

**Option B: Keep macro but make it type-aware**

```julia
macro up_T(T, expr)
    quote
        setrounding($T, RoundUp) do
            $expr
        end
    end
end
```

---

### Task 4: Update compute_schur_and_error

**File:** `src/pseudospectra/CertifScripts.jl`

**Current code (lines 508-530):**
```julia
function compute_schur_and_error(A::BallMatrix; polynomial = nothing)
    S = LinearAlgebra.schur(Complex{Float64}.(A.c))  # ISSUE: Forces Float64

    bZ = BallMatrix(S.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(S.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)  # ISSUE: Type mismatch

    # ... more Float64-specific code
```

**Required change:**
```julia
function compute_schur_and_error(A::BallMatrix{T}; polynomial = nothing) where T
    # Preserve the precision of the input matrix
    CT = Complex{T}
    S = LinearAlgebra.schur(CT.(A.c))

    bZ = BallMatrix(S.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(S.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    sigma_Z = svdbox(bZ)
    max_sigma = sigma_Z[1]
    min_sigma = sigma_Z[end]

    # Use type-appropriate rounding
    norm_Z = setrounding(T, RoundUp) do
        T(abs(max_sigma.c)) + T(max_sigma.r)
    end

    min_sigma_lower = setrounding(T, RoundDown) do
        max(T(min_sigma.c) - T(min_sigma.r), zero(T))
    end
    min_sigma_lower <= 0 && throw(ArgumentError("Schur factor has non-positive smallest singular value bound"))

    norm_Z_inv = setrounding(T, RoundUp) do
        one(T) / min_sigma_lower
    end

    # ... rest of function with T instead of Float64
end
```

---

### Task 5: Update setrounding Calls Throughout CertifScripts.jl

Search for all `setrounding(Float64, ...)` and make them type-parametric:

```julia
# Before
setrounding(Float64, RoundUp) do
    ...
end

# After
setrounding(T, RoundUp) do
    ...
end
```

Where `T` is extracted from the input `BallMatrix{T}`.

**Locations to update:**
- Lines 521-530 in `compute_schur_and_error`
- Line 594 and surrounding in `run_certification`
- Any other `setrounding(Float64, ...)` calls

---

### Task 6: Ensure Mixed-Type Operations Work

**File:** `src/types/ball.jl`

Add promotion rules for mixed Ball types:

```julia
# Promote Ball{Float64} to Ball{BigFloat} when operating together
function Base.promote_rule(::Type{Ball{Float64, S1}}, ::Type{Ball{BigFloat, S2}}) where {S1, S2}
    Ball{BigFloat, promote_type(S1, S2)}
end
```

Or alternatively, ensure operations always use the wider type:

```julia
function Base.:-(x::Ball{T1}, y::Ball{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    Ball{T}(x) - Ball{T}(y)
end
```

---

### Task 7: Update Matrix Multiplication

**Files:** `src/types/MMul/MMul*.jl`

These files use `ϵp` directly. Update to use `machine_epsilon(T)`:

```julia
# Before
rC = abs_mA * rprimeB + rA * (abs_mB + rB) .+ η / ϵp

# After
function rigorous_mmul(A::BallMatrix{T}, B::BallMatrix{T}) where T
    eps_T = machine_epsilon(T)
    # ... use eps_T instead of ϵp
end
```

---

## Testing Strategy

### Unit Tests

1. **Machine epsilon test:**
```julia
@test machine_epsilon(Float64) == 2.0^-52
@test machine_epsilon(Float32) == Float32(2.0^-23)
@test machine_epsilon(BigFloat) ≈ BigFloat(2)^(-precision(BigFloat))
```

2. **BigFloat ball arithmetic:**
```julia
setprecision(BigFloat, 256)
a = Ball(BigFloat(1.0), BigFloat(0.1))
b = Ball(BigFloat(2.0), BigFloat(0.2))
c = a + b
@test mid(c) ≈ BigFloat(3.0)
@test rad(c) > BigFloat(0.3)  # Includes roundoff
```

3. **BigFloat matrix operations:**
```julia
setprecision(BigFloat, 256)
A = BallMatrix(Complex{BigFloat}.(rand(5, 5)), BigFloat.(fill(1e-50, 5, 5)))
B = A * A
@test eltype(mid(B)) == Complex{BigFloat}
```

4. **Schur decomposition with BigFloat:**
```julia
setprecision(BigFloat, 256)
A = BallMatrix(Complex{BigFloat}.(rand(10, 10)), BigFloat.(fill(1e-100, 10, 10)))
S, errF, errT, norm_Z, norm_Z_inv = compute_schur_and_error(A)
@test eltype(S.T) == Complex{BigFloat}
```

### Integration Test

```julia
using BallArithmetic, LinearAlgebra

setprecision(BigFloat, 512)

# Create a BigFloat BallMatrix
n = 20
M = Complex{BigFloat}.(rand(n, n))
r = BigFloat.(fill(1e-100, n, n))
A = BallMatrix(M, r)

# Run certification
λ = BigFloat(0.5) + BigFloat(0.0)im
circle = CertificationCircle(ComplexF64(λ), 0.1; samples=64)
result = run_certification(A, circle)

@test result.resolvent_original isa BigFloat
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/rounding/rounding.jl` | Add `machine_epsilon(T)`, BigFloat rounding ops |
| `src/types/ball.jl` | Update arithmetic to use `machine_epsilon(T)` |
| `src/types/matrix.jl` | Update matrix arithmetic |
| `src/types/vector.jl` | Update vector arithmetic |
| `src/types/MMul/MMul*.jl` | Update matrix multiplication |
| `src/pseudospectra/CertifScripts.jl` | Make `compute_schur_and_error` and `run_certification` type-parametric |
| `src/norm_bounds/*.jl` | Update norm bound computations |

---

## Backwards Compatibility

- Keep `const ϵp = machine_epsilon(Float64)` for existing Float64 code
- Existing `Ball{Float64}` and `BallMatrix{Float64}` code should work unchanged
- New `Ball{BigFloat}` operations should "just work"

---

## Priority Order

1. **Task 2** (rounding operations) - Foundation for everything else
2. **Task 1** (machine epsilon) - Needed for error bounds
3. **Task 3** (ball arithmetic) - Core operations
4. **Task 4** (compute_schur_and_error) - Critical for certification
5. **Task 5** (setrounding calls) - Complete the certification path
6. **Task 6** (mixed types) - Nice to have for flexibility
7. **Task 7** (matrix multiplication) - Performance-critical path

---

## Example Usage After Implementation

```julia
using BallArithmetic, LinearAlgebra

# Set BigFloat precision
setprecision(BigFloat, 1024)

# Create high-precision matrix
n = 100
M_center = Complex{BigFloat}.(rand(n, n))
M_radius = BigFloat.(fill(big"1e-200", n, n))
A = BallMatrix(M_center, M_radius)

# Certification with BigFloat precision
circle = CertificationCircle(1.0 + 0.0im, 0.01; samples=256)
result = run_certification(A, circle)

println("Resolvent bound: ", result.resolvent_original)
# Should work with BigFloat precision throughout
```

---

## Task 8: Adaptive SVD Strategy with Ogita Refinement

### Motivation

Computing SVD is expensive, especially for BigFloat matrices. The Ogita-Aishima iterative refinement algorithm can refine an approximate SVD to higher precision with quadratic convergence. This enables an optimization: **compute one SVD and refine it for nearby points**.

### Benchmark Results

Testing on 30×30 upper triangular matrix, 16 sample points on a circle:

| Radius | Approach | Time | Enclosure Quality |
|--------|----------|------|-------------------|
| **1e-3** | Full Float64 SVD | 3.0s | 1.2e-12 |
| | Center + Ogita F64 | 0.12s (24x faster) | 1.2e-10 (100x worse!) |
| | Center + Ogita BigFloat | 14.5s | 2.0e-13 (6x better) |
| **1e-6** | Full Float64 SVD | 0.036s | 1.2e-12 |
| | Center + Ogita F64 | 0.022s (1.6x faster) | 1.1e-12 (similar) |
| | Center + Ogita BigFloat | 12.8s | 2.0e-13 |
| **1e-14** | Full Float64 SVD | 0.038s | 1.2e-12 |
| | Center + Ogita F64 | 0.023s (1.6x faster) | 1.1e-12 (similar) |

**Key insight**: Ogita Float64 refinement works well when the perturbation (radius) is small relative to Float64 precision. For large perturbations, enclosure quality degrades significantly.

### Adaptive Strategy for Certification

The certification algorithm uses adaptive arc bisection. This naturally provides opportunities for SVD reuse:

```
                    Arc to certify
            ┌───────────────────────────┐
            │                           │
        endpoint_a                  endpoint_b
            │                           │
            └─────────┬─────────────────┘
                      │
                  midpoint (needs SVD)
```

**Strategy 1: Endpoint-based Ogita refinement**

When bisecting an arc:
1. We already have certified SVDs at `endpoint_a` and `endpoint_b`
2. For the `midpoint`, try Ogita refinement from the nearest endpoint
3. Check enclosure quality; if acceptable, use it
4. If not acceptable, compute fresh SVD

```julia
function certify_with_adaptive_svd(T_z, svd_nearby, threshold)
    # Try Ogita refinement from nearby SVD
    refined = ogita_svd_refine(T_z, svd_nearby.U, svd_nearby.S, svd_nearby.V;
                               max_iterations=2, precision_bits=53)

    # Certify and check enclosure quality
    result = _certify_svd(BallMatrix(T_z), refined, MiyajimaM4())
    max_rad = maximum(rad.(result.singular_values))

    if max_rad < threshold
        return result, :ogita_success
    else
        # Fall back to full SVD
        svd_fresh = svd(T_z)
        result = rigorous_svd(BallMatrix(T_z); method=MiyajimaM4())
        return result, :fallback_full_svd
    end
end
```

**Strategy 2: Quality-based heuristic**

Before starting certification, test Ogita on one sample point:

```julia
function choose_svd_strategy(T, center, radius, num_samples)
    # Compute center SVD
    T_center = T - center * I
    svd_center = svd(T_center)

    # Test on one point
    θ_test = 0.0
    z_test = center + radius * exp(im * θ_test)
    T_test = T - z_test * I

    # Try Ogita refinement
    refined = ogita_svd_refine(T_test, svd_center.U, svd_center.S, svd_center.V;
                               max_iterations=2, precision_bits=53)

    # Compare to full SVD
    svd_test = svd(T_test)
    result_ogita = _certify_svd(BallMatrix(T_test), refined, MiyajimaM4())
    result_full = rigorous_svd(BallMatrix(T_test); method=MiyajimaM4())

    rad_ogita = maximum(rad.(result_ogita.singular_values))
    rad_full = maximum(rad.(result_full.singular_values))

    # Accept if enclosure is within 10x of full SVD
    if rad_ogita < 10 * rad_full
        return :use_ogita, svd_center
    else
        return :use_full_svd, nothing
    end
end
```

**Strategy 3: Propagate SVD along arc**

For sequential certification (non-parallel), propagate SVD along the arc:

```julia
function certify_arc_sequential(T, center, radius, angles)
    results = []
    svd_prev = nothing

    for θ in angles
        z = center + radius * exp(im * θ)
        T_z = T - z * I

        if svd_prev === nothing
            # First point: compute full SVD
            svd_curr = svd(T_z)
        else
            # Try Ogita from previous point
            refined = ogita_svd_refine(T_z, svd_prev.U, svd_prev.S, svd_prev.V;
                                       max_iterations=2)

            # Check quality (residual norm)
            residual = norm(T_z - refined.U * refined.Σ * refined.V')
            if residual < 1e-12 * norm(T_z)
                svd_curr = refined
            else
                svd_curr = svd(T_z)  # Fallback
            end
        end

        result = _certify_svd(BallMatrix(T_z), svd_curr, MiyajimaM4())
        push!(results, result)
        svd_prev = svd_curr
    end

    return results
end
```

### Integration with Parallel Bisection

The current parallel certification uses arc bisection. Modify to pass SVD hints:

```julia
struct ArcSegment
    θ_start::Float64
    θ_end::Float64
    svd_start::Union{Nothing, SVDResult}  # SVD at start endpoint
    svd_end::Union{Nothing, SVDResult}    # SVD at end endpoint
end

function bisect_arc(arc::ArcSegment, T, center, radius)
    θ_mid = (arc.θ_start + arc.θ_end) / 2
    z_mid = center + radius * exp(im * θ_mid)
    T_mid = T - z_mid * I

    # Try Ogita from nearest endpoint
    if arc.svd_start !== nothing
        dist_start = abs(θ_mid - arc.θ_start)
        dist_end = abs(θ_mid - arc.θ_end)

        svd_hint = dist_start < dist_end ? arc.svd_start : arc.svd_end
        refined = ogita_svd_refine(T_mid, svd_hint.U, svd_hint.S, svd_hint.V)

        # Validate quality
        if is_acceptable_quality(refined, T_mid)
            svd_mid = refined
        else
            svd_mid = svd(T_mid)
        end
    else
        svd_mid = svd(T_mid)
    end

    # Return two child arcs with SVD hints
    left_arc = ArcSegment(arc.θ_start, θ_mid, arc.svd_start, svd_mid)
    right_arc = ArcSegment(θ_mid, arc.θ_end, svd_mid, arc.svd_end)

    return left_arc, right_arc, svd_mid
end
```

### When Ogita Refinement Fails

Ogita refinement may fail (poor enclosure) when:
1. **Large perturbation**: `|z - center| / σ_min(T - center*I)` is not small
2. **Clustered singular values**: Ogita formulas have `σ_i² - σ_j²` in denominator
3. **Near-singular matrix**: Small singular values amplify errors

**Heuristic threshold**:
```julia
function should_use_ogita(radius, σ_min_center)
    # Ogita works well when relative perturbation is small
    relative_perturbation = radius / σ_min_center
    return relative_perturbation < 1e-6  # Conservative threshold
end
```

### BigFloat Considerations

For BigFloat matrices:
1. **Always use Ogita** to refine Float64 SVD to BigFloat precision
2. **Center SVD optimization** is more valuable (BigFloat SVD is very expensive)
3. **3 Ogita iterations** typically sufficient for 256-bit precision

```julia
function certify_bigfloat_circle(T_bf::Matrix{Complex{BigFloat}}, center, radius, angles)
    # Compute center SVD via Float64 + Ogita to BigFloat
    T_center_bf = T_bf - center * I
    svd_f64 = svd(ComplexF64.(T_center_bf))
    center_svd = ogita_svd_refine(T_center_bf, svd_f64.U, svd_f64.S, svd_f64.V;
                                   max_iterations=3, precision_bits=256)

    results = []
    for θ in angles
        z = center + radius * exp(im * θ)
        T_z = T_bf - z * I

        # Refine from center SVD (already BigFloat)
        refined = ogita_svd_refine(T_z, center_svd.U, diag(center_svd.Σ), center_svd.V;
                                    max_iterations=3, precision_bits=256)

        result = _certify_svd(BallMatrix(T_z), refined, MiyajimaM4())
        push!(results, result)
    end

    return results
end
```

### Summary: Recommended Strategy

| Scenario | Strategy |
|----------|----------|
| Float64, large radius (>1e-6) | Full SVD at each point |
| Float64, small radius (<1e-6) | Center SVD + Ogita (1.5-2x speedup) |
| BigFloat, any radius | Center SVD + Ogita (mandatory for performance) |
| Parallel bisection | Pass SVD hints from endpoints to midpoint |

### Integration with Existing Architecture

The current certification system has:

```
┌─────────────────────────────────────────────────────────────┐
│  Main Process                                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ adaptive_arcs_distributed!                              ││
│  │   - Manages arcs queue                                  ││
│  │   - Bisects arcs where ε = |z_b - z_a| / σ_min > η      ││
│  │   - Caches results by z coordinate                      ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│                    job_channel: (id, z)                      │
│                           ↓                                  │
├─────────────────────────────────────────────────────────────┤
│  Worker Processes                                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ dowork(jobs, results)                                   ││
│  │   - Takes (id, z) from job_channel                      ││
│  │   - Calls _evaluate_sample(T, z, id)                    ││
│  │   - Returns certified singular values                   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Workers are stateless - they don't remember previous SVDs.

### Strategy A: Worker-Local SVD Caching (Minimal Changes)

Each worker caches its last computed SVD and tries Ogita refinement first:

```julia
# Worker-local state
const _last_svd = Ref{Union{Nothing, NamedTuple}}(nothing)
const _last_z = Ref{Union{Nothing, ComplexF64}}(nothing)

function _evaluate_sample_with_ogita(T::BallMatrix{ET}, z::Number, idx::Int;
                                      ogita_threshold::Real = 1e-6) where ET
    RT = real(ET)
    CT = Complex{RT}
    z_converted = CT(z)

    # Check if we can use Ogita from last SVD
    use_ogita = false
    if _last_svd[] !== nothing && _last_z[] !== nothing
        dist = abs(z_converted - _last_z[])
        if dist < ogita_threshold
            use_ogita = true
        end
    end

    T_shifted = T.c - z_converted * I

    if use_ogita
        # Try Ogita refinement
        last = _last_svd[]
        refined = ogita_svd_refine(T_shifted, last.U, last.S, last.V;
                                    max_iterations=2, precision_bits=53)

        # Validate quality
        residual = norm(T_shifted - refined.U * refined.Σ * refined.V')
        if residual < 1e-12 * norm(T_shifted)
            # Success - use refined SVD for certification
            Σ_vec = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
            svd_result = (U=refined.U, S=Σ_vec, V=refined.V, Vt=refined.V')
            Σ = _certify_svd(BallMatrix(T_shifted), svd_result, MiyajimaM4())

            # Update cache
            _last_svd[] = (U=refined.U, S=Σ_vec, V=refined.V)
            _last_z[] = z_converted

            return _format_result(Σ, z_converted, idx)
        end
        # Fall through to full SVD if quality check fails
    end

    # Full SVD path
    svd_full = svd(T_shifted)
    A_ball = BallMatrix(T_shifted, fill(eps(RT) * norm(T_shifted), size(T_shifted)...))
    Σ = rigorous_svd(A_ball; method=MiyajimaM4(), apply_vbd=false)

    # Update cache
    _last_svd[] = (U=svd_full.U, S=svd_full.S, V=svd_full.V)
    _last_z[] = z_converted

    return _format_result(Σ, z_converted, idx)
end
```

**Pros**: Minimal changes to architecture, no serialization of SVD over network
**Cons**: Only helps when same worker processes nearby points (depends on job scheduling)

### Strategy B: Arc-Based SVD Propagation (More Changes)

Modify the job structure to include SVD hints from arc endpoints:

```julia
# New job type with SVD hint
const _RemoteJobWithHint = Tuple{Int, ComplexF64, Union{Nothing, Tuple{Matrix, Vector, Matrix}}}

function _adaptive_arcs_distributed_with_hints!(arcs, cache, pending, η, ...)
    # When bisecting arc (z_a, z_b) → (z_a, z_m), (z_m, z_b)
    # We have SVD at z_a in cache
    # Pass it as hint for computing SVD at z_m

    z_a, z_b = pop!(arcs)
    svd_hint_a = get_svd_hint_from_cache(cache, z_a)

    z_m = (z_a + z_b) / 2

    # Include SVD hint in job
    job_id = id_counter
    put!(job_channel, (job_id, z_m, svd_hint_a))
    pending[job_id] = (z_m, z_b)
    ...
end
```

**Pros**: Guarantees SVD hints are available for midpoints
**Cons**: Requires serializing SVD matrices (network overhead), more invasive changes

### Strategy C: Radius-Based Heuristic (Simplest)

Use the circle radius to decide strategy at the start:

```julia
function run_certification_adaptive(A, circle; kwargs...)
    radius = circle.radius

    if radius < 1e-6
        # Small circle: use center SVD + Ogita
        return run_certification_ogita(A, circle; kwargs...)
    else
        # Large circle: use standard approach
        return run_certification(A, circle; kwargs...)
    end
end
```

**Pros**: Simplest implementation, no architectural changes
**Cons**: Doesn't adapt during bisection (early arcs are large, later arcs are small)

### Recommended Approach: Hybrid

1. **Pre-flight check**: Before starting certification, test Ogita on a sample point
2. **If successful**: Use Strategy A (worker-local caching)
3. **If failed**: Fall back to standard approach

```julia
function run_certification_with_ogita_fallback(A, circle; η=0.5, ...)
    # Pre-flight: test Ogita viability
    z_test = circle.center + circle.radius * exp(0im)
    T_test = A.c - z_test * I

    svd_center = svd(A.c - circle.center * I)
    refined = ogita_svd_refine(T_test, svd_center.U, svd_center.S, svd_center.V;
                               max_iterations=2, precision_bits=53)

    result_ogita = _certify_svd(BallMatrix(T_test), refined, MiyajimaM4())
    result_full = rigorous_svd(BallMatrix(T_test); method=MiyajimaM4())

    rad_ratio = maximum(rad.(result_ogita.singular_values)) /
                maximum(rad.(result_full.singular_values))

    if rad_ratio < 10.0
        @info "Using Ogita optimization (enclosure ratio: $rad_ratio)"
        # Use workers with Ogita caching enabled
        return _run_certification_with_ogita_workers(A, circle; η, ...)
    else
        @info "Ogita enclosure too large ($rad_ratio), using standard approach"
        return run_certification(A, circle; η, ...)
    end
end
```

### Implementation Priority

1. **Phase 1**: Implement Strategy C (radius-based heuristic) - quick win
2. **Phase 2**: Implement Strategy A (worker-local caching) - moderate effort
3. **Phase 3**: Implement pre-flight check and adaptive fallback
4. **Phase 4 (optional)**: Strategy B for maximum optimization

---

## Implementation Status

### Completed Work

#### Phase 1 & 2: Worker-Local SVD Caching

**Files modified:**
- `src/pseudospectra/CertifScripts.jl` - Added Float64 and BigFloat Ogita caching
- `ext/DistributedExt.jl` - Added `use_bigfloat_ogita` mode for distributed certification

**Float64 Ogita Caching (implemented):**
```julia
# Worker-local SVD cache variables
const _last_svd_U = Ref{Union{Nothing, Matrix}}(nothing)
const _last_svd_S = Ref{Union{Nothing, Vector}}(nothing)
const _last_svd_V = Ref{Union{Nothing, Matrix}}(nothing)
const _last_svd_z = Ref{Union{Nothing, Number}}(nothing)

# Functions
_clear_ogita_cache!()           # Clear Float64 cache
_ogita_cache_stats()            # Get hit/miss/fallback counts
_evaluate_sample_with_ogita_cache(T, z, idx; ...)  # Cached evaluation
dowork_ogita(jobs, results; ...)  # Worker loop with Float64 caching
```

**BigFloat Ogita Caching (implemented):**
```julia
# Worker-local BigFloat SVD cache variables
const _bf_last_svd_U = Ref{Union{Nothing, Matrix}}(nothing)
const _bf_last_svd_S = Ref{Union{Nothing, Vector}}(nothing)
const _bf_last_svd_V = Ref{Union{Nothing, Matrix}}(nothing)
const _bf_last_svd_z = Ref{Union{Nothing, Number}}(nothing)

# Functions
_clear_bf_ogita_cache!()           # Clear BigFloat cache
_bf_ogita_cache_stats()            # Get hit/miss/fallback counts
_evaluate_sample_ogita_bigfloat(T, z, idx; ...)  # BigFloat cached evaluation
dowork_ogita_bigfloat(jobs, results; ...)  # Worker loop with BigFloat caching
```

**Distributed API (implemented in DistributedExt.jl):**
```julia
# New keyword arguments for run_certification
run_certification(A, circle, workers;
    use_ogita_cache = false,           # Enable Float64 Ogita caching
    ogita_distance_threshold = 1e-4,   # Max distance for cache reuse
    ogita_quality_threshold = 1e-10,   # Quality threshold for Ogita
    ogita_iterations = 2,              # Ogita iterations (Float64)
    use_bigfloat_ogita = false,        # Enable BigFloat mode
    target_precision = 256,            # BigFloat precision in bits
    max_ogita_iterations = 4           # Ogita iterations (BigFloat)
)
```

### Test Results

**Parallel BigFloat Ogita (radius = 1e-10):**
- Time: 42.74s
- Minimum singular value: ~3.1e-11 (BigFloat precision maintained)
- Enclosure radius (avg): ~1.25e-13
- Enclosure radius (max): ~1.25e-13
- Status: ✅ SUCCESS

**Float64 Ogita Cache Performance (from earlier tests):**
- Cache hit rate: ~95% during adaptive bisection
- Speedup: ~2.5x compared to full SVD at each point

### Cache Strategy Details

**How caching works:**
1. Worker processes jobs sequentially
2. After computing SVD at point `z`, cache the result
3. For next job at `z'`, check if `|z' - z| < distance_threshold`
4. If cache hit: use cached SVD as starting point for Ogita refinement
   - Float64: 2 iterations (from Float64 starting point)
   - BigFloat: 1-2 iterations (from BigFloat starting point)
5. If cache miss: compute fresh Float64 SVD, then refine with Ogita

**Cache effectiveness depends on:**
- Job scheduling (nearby points assigned to same worker)
- Adaptive bisection naturally groups nearby points
- Distance threshold tuning (default: 1e-4, or `radius * 10` for small circles)

### Known Issues

1. **Worker reuse across multiple certifications**: Workers cannot be reused
   across multiple `run_certification` calls because they exit after channels
   close. Fresh workers must be spawned for each call.

2. **Very small radii (< 1e-15)**: Testing with radii 1e-15, 1e-20, 1e-30 is
   ongoing. The adaptive bisection may require many iterations for very small
   circles.

### Ogita SVD Refinement Algorithm

The `ogita_svd_refine` function implements the Ogita-Aishima RefSVD algorithm:

```julia
ogita_svd_refine(A, U₀, Σ₀, V₀; max_iterations=4, precision_bits=256)
```

**Key properties:**
- Quadratic convergence: error decreases as O(ε²) per iteration
- From Float64 (ε ≈ 10⁻¹⁶) to 256-bit BigFloat (ε ≈ 10⁻⁷⁷): 4 iterations
- From BigFloat to BigFloat (nearby point): 1-2 iterations

**Phase correction for complex matrices:**
The algorithm includes phase correction to ensure `U'AV` has real positive diagonal:
```julia
for i in 1:n
    phase = T_final[i,i] / abs(T_final[i,i])  # e^{iθ_i}
    U[:, i] .*= phase  # Rotate U column to make diagonal real
end
```

### Future Improvements

1. **Pre-flight quality check**: Test Ogita on one sample before committing
   to the strategy

2. **Adaptive threshold**: Adjust `distance_threshold` based on observed
   cache hit rate

3. **Arc-based SVD hints**: Pass SVD from arc endpoints to midpoint
   (Strategy B - requires serialization overhead)

4. **Hybrid Float64/BigFloat**: Use Float64 Ogita for initial exploration,
   switch to BigFloat for final certification
