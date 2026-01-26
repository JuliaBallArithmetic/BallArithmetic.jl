using BallArithmetic
using LinearAlgebra
using Test

"""
Compare Rump's singular value bounds with and without Miyajima's VBD clustering.

This test demonstrates the difference between:
1. Pure Rump (2011): Basic singular value enclosures from residual/orthogonality
2. Rump + Miyajima VBD (2014): Same enclosures + block-diagonal clustering
"""

println("\n" * "="^80)
println("COMPARISON: Rump's SVD Bounds vs Rump + Miyajima VBD")
println("="^80)

# Test 1: Matrix with well-separated singular values
println("\n--- Test 1: Well-separated singular values ---")
A1 = BallMatrix([3.0 0.1; 0.0 1.0], [0.01 0.0; 0.0 0.01])

result_no_vbd = rigorous_svd(A1; apply_vbd=false)
result_with_vbd = rigorous_svd(A1; apply_vbd=true)

println("\nSingular value enclosures (both methods give same bounds):")
println("  Pure Rump:      ", result_no_vbd.singular_values)
println("  Rump + VBD:     ", result_with_vbd.singular_values)
println("  Identical?      ", result_no_vbd.singular_values == result_with_vbd.singular_values)

println("\nVerification bounds:")
println("  Residual norm:           ", result_no_vbd.residual_norm)
println("  Right orthog. defect:    ", result_no_vbd.right_orthogonality_defect)
println("  Left orthog. defect:     ", result_no_vbd.left_orthogonality_defect)

println("\nVBD-specific information:")
if result_with_vbd.block_diagonalisation !== nothing
    vbd = result_with_vbd.block_diagonalisation
    println("  Number of clusters:      ", length(vbd.clusters))
    println("  Cluster indices:         ", vbd.clusters)
    println("  Cluster intervals:       ", vbd.cluster_intervals)
    println("  Remainder norm (‖R‖₂):   ", vbd.remainder_norm)
    println("\n  Interpretation: Each singular value is isolated (2 clusters of size 1)")
else
    println("  VBD not computed (apply_vbd=false)")
end

# Test 2: Matrix with clustered singular values
println("\n\n--- Test 2: Clustered singular values ---")
# Create matrix with two close singular values
A2_mid = Matrix(Diagonal([5.0, 5.1, 1.0]))
A2_rad = zeros(size(A2_mid))
A2_rad[1, 2] = A2_rad[2, 1] = 0.15  # Create coupling between first two
A2 = BallMatrix(A2_mid, A2_rad)

result2_no_vbd = rigorous_svd(A2; apply_vbd=false)
result2_with_vbd = rigorous_svd(A2; apply_vbd=true)

println("\nSingular value enclosures:")
for i in 1:3
    sv_rump = result2_no_vbd.singular_values[i]
    sv_vbd = result2_with_vbd.singular_values[i]
    println("  σ[$i]: mid=$(mid(sv_rump)), rad=$(rad(sv_rump))")
end

println("\nVBD clustering analysis:")
if result2_with_vbd.block_diagonalisation !== nothing
    vbd = result2_with_vbd.block_diagonalisation
    println("  Number of clusters:      ", length(vbd.clusters))
    println("  Cluster structure:       ", vbd.clusters)

    # Cluster 1 (should contain the two close singular values)
    for (k, cluster) in enumerate(vbd.clusters)
        println("\n  Cluster $k (indices $cluster):")
        println("    Size:                  ", length(cluster))
        println("    Interval enclosure:    ", vbd.cluster_intervals[k])
        println("    Eigenvalues in cluster:", vbd.eigenvalues[cluster])
    end

    println("\n  Off-diagonal remainder:")
    println("    ‖R‖₂ bound:            ", vbd.remainder_norm)
    println("\n  Interpretation: VBD identifies that σ₁≈5.0 and σ₂≈5.1 form a cluster")
    println("                  while σ₃≈1.0 is isolated")
end

# Test 3: Larger matrix with multiple clusters
println("\n\n--- Test 3: Multiple clusters ---")
A3_mid = Matrix(Diagonal([10.0, 10.1, 10.05, 5.0, 5.2, 1.0]))
A3_rad = zeros(size(A3_mid))
# Create overlaps within clusters
A3_rad[1, 2] = A3_rad[2, 1] = 0.15  # Cluster {10.0, 10.1, 10.05}
A3_rad[2, 3] = A3_rad[3, 2] = 0.10
A3_rad[4, 5] = A3_rad[5, 4] = 0.25  # Cluster {5.0, 5.2}
A3 = BallMatrix(A3_mid, A3_rad)

result3_with_vbd = rigorous_svd(A3; apply_vbd=true)

println("\nMatrix size: 6×6")
println("Expected clusters: {σ₁,σ₂,σ₃}≈10, {σ₄,σ₅}≈5, {σ₆}≈1")

if result3_with_vbd.block_diagonalisation !== nothing
    vbd = result3_with_vbd.block_diagonalisation
    println("\nVBD identified ", length(vbd.clusters), " clusters:")

    for (k, cluster) in enumerate(vbd.clusters)
        cluster_size = length(cluster)
        interval = vbd.cluster_intervals[k]
        eigenvals = vbd.eigenvalues[cluster]

        println("\n  Cluster $k:")
        println("    Indices:               ", cluster)
        println("    Size:                  ", cluster_size)
        println("    Gershgorin interval:   ", interval)
        println("    Eigenvalues (Σ²):      ", round.(eigenvals, digits=4))
        println("    Singular values:       ", round.(sqrt.(eigenvals), digits=4))
    end

    println("\n  Block-diagonal structure:")
    println("    Remainder ‖R‖₂:        ", vbd.remainder_norm)

    # Show the block structure
    println("\n  Transformed matrix Σ² = Q'(Σ²)Q has block structure:")
    for (k, cluster) in enumerate(vbd.clusters)
        println("    Block $k: $(cluster.start):$(cluster.stop) (size $(length(cluster))×$(length(cluster)))")
    end
end

# Test 4: Quantitative comparison of bounds
println("\n\n--- Test 4: Quantitative bound comparison ---")
println("\nKey differences between methods:")
println("┌─────────────────────────────────────┬──────────────┬──────────────────┐")
println("│ Property                            │ Pure Rump    │ Rump + VBD       │")
println("├─────────────────────────────────────┼──────────────┼──────────────────┤")
println("│ Singular value bounds               │ ✓ Provided   │ ✓ Same bounds    │")
println("│ Residual/orthogonality verification │ ✓ Provided   │ ✓ Same bounds    │")
println("│ Clustering information              │ ✗ Not avail. │ ✓ VBD clusters   │")
println("│ Block-diagonal structure            │ ✗ Not avail. │ ✓ Q'ΣQ = D + R   │")
println("│ Off-diagonal remainder bound        │ ✗ Not avail. │ ✓ ‖R‖₂ bounded   │")
println("│ Gershgorin disc enclosures          │ ✗ Not avail. │ ✓ Per cluster    │")
println("│ Computational cost                  │ Lower        │ Higher (+ VBD)   │")
println("└─────────────────────────────────────┴──────────────┴──────────────────┘")

println("\n\nConclusion:")
println("• Both methods provide identical certified singular value enclosures")
println("• Miyajima's VBD adds clustering structure: Σ² = Q'(Σ²)Q = Block_Diag + Remainder")
println("• VBD is valuable when:")
println("    - Multiple singular values are close (clusters)")
println("    - Need block structure for further analysis")
println("    - Want rigorous bounds on off-diagonal coupling")
println("• Pure Rump is sufficient when only singular value bounds are needed")

println("\n" * "="^80)
