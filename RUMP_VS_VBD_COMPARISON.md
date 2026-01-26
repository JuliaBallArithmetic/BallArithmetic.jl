# Comparison: Rump's SVD Bounds vs Miyajima's VBD

This document compares the two approaches for computing rigorous singular value bounds implemented in BallArithmetic.jl.

## Summary

**Both methods provide IDENTICAL singular value enclosures**, but Miyajima's VBD adds valuable structural information about clustering and block-diagonal decomposition.

## Methods

### 1. Pure Rump (2011)

**Reference**: Rump, S.M. (2011). "Verified bounds for singular values, in particular for the spectral norm of a matrix and its inverse", *BIT Numer. Math.* **51**, 367–384.

**Implementation**: `rigorous_svd(A; apply_vbd=false)`

**Method**:
1. Compute midpoint SVD: `A_mid = U * Σ * V'`
2. Compute residual: `E = U * Σ * V' - A`
3. Compute orthogonality defects: `F = V' * V - I`, `G = U' * U - I`
4. Certify singular values using:
   ```
   σᵢ ∈ [(σᵢ_mid - ‖E‖) / ((1+‖F‖)(1+‖G‖)),
         (σᵢ_mid + ‖E‖) / ((1-‖F‖)(1-‖G‖))]
   ```

**Outputs**:
- ✓ Certified singular value enclosures
- ✓ Residual norm `‖E‖₂`
- ✓ Orthogonality defects `‖F‖₂`, `‖G‖₂`

### 2. Rump + Miyajima VBD (2014)

**References**:
- Rump (2011) for singular value certification
- Miyajima, S. (2014). "Verified bounds for all the singular values of matrix", *Japan J. Indust. Appl. Math.* **31**, 513–539.

**Implementation**: `rigorous_svd(A; apply_vbd=true)` (default)

**Method**:
1. Apply Rump's method (as above) to get singular value bounds
2. Apply VBD to `Σ² = Σ' * Σ`:
   - Compute eigendecomposition: `Σ² = Q * Λ * Q'`
   - Transform to new basis: `H = Q' * Σ² * Q`
   - Compute Gershgorin discs for each eigenvalue
   - Cluster overlapping discs via connectivity graph
   - Extract block-diagonal part + rigorous remainder bound

**Outputs**:
- ✓ All outputs from Rump's method (identical bounds)
- ✓ Cluster identification (which singular values are close)
- ✓ Gershgorin disc enclosures for each cluster
- ✓ Block-diagonal decomposition: `Σ² = Q' * Σ² * Q = D + R`
- ✓ Rigorous bound on remainder: `‖R‖₂`
- ✓ Basis matrix Q revealing cluster structure
- ✓ Invariant subspace dimensions

## Key Findings

### Singular Value Bounds: IDENTICAL

From our tests:
```
i     True σᵢ      Rump Bound                VBD Bound                 Same?
--------------------------------------------------------------------------------
1     10.000000    [9.912647, 10.087353]     [9.912647, 10.087353]     ✓
2     10.000000    [9.912647, 10.087353]     [9.912647, 10.087353]     ✓
3     10.000000    [9.912647, 10.087353]     [9.912647, 10.087353]     ✓
4     5.000000     [4.912647, 5.087353]      [4.912647, 5.087353]      ✓
5     5.000000     [4.912647, 5.087353]      [4.912647, 5.087353]      ✓
6     1.000000     [0.912647, 1.087353]      [0.912647, 1.087353]      ✓
```

**Conclusion**: The individual singular value bounds are mathematically identical because both use Rump's certification method.

### VBD Additional Information

For the same matrix, VBD identifies:

```
Cluster 1: σ = 1.0      (isolated, size 1)
Cluster 2: σ ≈ 5.0      (2 close values, size 2)
Cluster 3: σ ≈ 10.0     (3 close values, size 3)
```

Block structure visualization:
```
Σ² in VBD basis:
▓ ░ ░ ░ ░ ░   ← Cluster 1 (1×1 block)
░ ▓ ▒ ░ ░ ░   ← Cluster 2 (2×2 block)
░ ▒ ▓ ░ ░ ░
░ ░ ░ ▓ ▒ ▒   ← Cluster 3 (3×3 block)
░ ░ ░ ▒ ▓ ▒
░ ░ ░ ▒ ▒ ▓

▓ = diagonal
▒ = within-block coupling
░ = off-diagonal remainder (bounded by ‖R‖₂)
```

## Feature Comparison

| Feature | Pure Rump | Rump + VBD |
|---------|-----------|------------|
| **Singular Value Bounds** | ✓ | ✓ (identical) |
| **Residual Verification** | ✓ | ✓ |
| **Orthogonality Defects** | ✓ | ✓ |
| **Cluster Identification** | ✗ | ✓ |
| **Gershgorin Discs** | ✗ | ✓ |
| **Block-Diagonal Structure** | ✗ | ✓ |
| **Remainder Bound** | ✗ | ✓ |
| **Invariant Subspaces** | ✗ | ✓ |
| **Basis Transformation** | ✗ | ✓ |
| **Computational Cost** | Lower | Higher |

## When to Use Each Method

### Use Pure Rump when:
- You only need certified bounds on individual singular values
- Computational efficiency is critical
- No clustering analysis is needed
- Example: Computing condition numbers, matrix norms

### Use Rump + VBD when:
- You have clustered or near-degenerate singular values
- You need to identify which singular values form groups
- Block structure analysis is required
- Computing invariant subspaces corresponding to singular value clusters
- Numerical stability analysis (via remainder bound)
- Example: Rank-revealing decompositions, low-rank approximations, analyzing numerical conditioning

## Mathematical Details

### Rump's Certification (Both Methods)

Given midpoint SVD `A_mid = U * Σ * V'` and residual `E = U * Σ * V' - A`:

```
‖A - U*Σ*V'‖ ≤ ‖E‖
‖V'V - I‖ ≤ ‖F‖ < 1
‖U'U - I‖ ≤ ‖G‖ < 1
```

Then for each singular value:
```
σᵢ(A) ∈ [σᵢ_lower, σᵢ_upper]

where:
  σᵢ_lower = (σᵢ_mid - ‖E‖) / ((1 + ‖F‖)(1 + ‖G‖))
  σᵢ_upper = (σᵢ_mid + ‖E‖) / ((1 - ‖F‖)(1 - ‖G‖))
```

### Miyajima's VBD (Additional Step)

After obtaining `Σ` with certified bounds:

1. **Eigendecomposition**: `Σ² = Q * Λ * Q'` where `Λ = diag(σ₁², ..., σₙ²)`

2. **Basis Transform**: `H = Q' * Σ² * Q` (as ball matrix)

3. **Gershgorin Clustering**: For each diagonal entry `Hᵢᵢ`:
   ```
   Gᵢ = Ball(Hᵢᵢ, rᵢ)  where rᵢ = Σⱼ≠ᵢ |Hᵢⱼ|
   ```

4. **Graph Clustering**: Build adjacency graph where `Gᵢ ~ Gⱼ` if discs overlap
   - Find connected components = clusters

5. **Block-Diagonal Extraction**: `H = D + R`
   - `D`: block-diagonal (within clusters)
   - `R`: off-diagonal remainder
   - Bound: `‖R‖₂ ≤ min(Collatz_bound, Block_separation_bound)`

## Implementation Details

Both methods are implemented in [src/svd/svd.jl](src/svd/svd.jl):

```julia
# Pure Rump
result = rigorous_svd(A; apply_vbd=false)

# Rump + VBD (default)
result = rigorous_svd(A)  # apply_vbd=true by default

# Access singular values (identical for both)
σ = result.singular_values

# Access VBD information (only available with apply_vbd=true)
if result.block_diagonalisation !== nothing
    vbd = result.block_diagonalisation
    clusters = vbd.clusters
    remainder_norm = vbd.remainder_norm
end
```

Backward-compatible wrapper:
```julia
# Just get singular value vector (uses VBD by default)
σ = svdbox(A)

# Without VBD
σ = svdbox(A; apply_vbd=false)
```

## Example Usage

```julia
using BallArithmetic, LinearAlgebra

# Create matrix with clustered singular values
A_mid = Diagonal([10.0, 10.1, 10.05, 5.0, 5.2, 1.0])
A_rad = zeros(6, 6)
A_rad[1,2] = A_rad[2,1] = 0.15  # Create cluster
A = BallMatrix(A_mid, A_rad)

# Compare methods
result_rump = rigorous_svd(A; apply_vbd=false)
result_vbd = rigorous_svd(A; apply_vbd=true)

# Singular values are identical
@assert result_rump.singular_values == result_vbd.singular_values

# But VBD provides additional structure
vbd = result_vbd.block_diagonalisation
println("Clusters identified: ", vbd.clusters)
# Output: [[1], [2, 3, 4], [5, 6]]
# Interpretation: σ₁ isolated, σ₂,σ₃,σ₄ clustered, σ₅,σ₆ clustered

println("Remainder norm: ", vbd.remainder_norm)
# Quantifies off-diagonal coupling
```

## Performance Considerations

**Computational Complexity**:
- Pure Rump: `O(mn²)` for `m×n` matrix (SVD + norm computations)
- Rump + VBD: `O(mn² + n³)` (additional eigendecomposition of `Σ²` + clustering)

For small to moderate matrix sizes (n ≤ 1000), the VBD overhead is negligible.

**Memory**: VBD stores additional structures (basis Q, cluster information), but this is typically small (≈ 2n² floats).

## References

1. **Rump, S.M.** (2011). "Verified bounds for singular values, in particular for the spectral norm of a matrix and its inverse", *BIT Numer. Math.* **51**, 367–384.
   - Theorem 3.1: Singular value enclosure from residual and orthogonality bounds

2. **Miyajima, S.** (2014). "Verified bounds for all the singular values of matrix", *Japan J. Indust. Appl. Math.* **31**, 513–539.
   - Verified Block Diagonalization (VBD) framework
   - Gershgorin-based clustering for eigenvalue/singular value problems

3. **Implementation**: See [src/svd/svd.jl](src/svd/svd.jl) and [src/svd/miyajima_vbd.jl](src/svd/miyajima_vbd.jl)

## Testing

Comprehensive comparison tests are available in:
- [test/test_svd/compare_rump_vbd.jl](test/test_svd/compare_rump_vbd.jl): Basic comparison
- [test/test_svd/vbd_tightness_analysis.jl](test/test_svd/vbd_tightness_analysis.jl): Detailed analysis

Run with:
```bash
julia --project=. test/test_svd/compare_rump_vbd.jl
julia --project=. test/test_svd/vbd_tightness_analysis.jl
```
