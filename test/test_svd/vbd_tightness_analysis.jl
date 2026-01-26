using BallArithmetic
using LinearAlgebra
using Printf

"""
Analyze the tightness and additional information from Miyajima's VBD
compared to pure Rump bounds for singular values.
"""

println("\n" * "="^80)
println("VBD TIGHTNESS ANALYSIS: How VBD Improves Understanding of Singular Values")
println("="^80)

# Example: Matrix with near-degenerate singular values
println("\n--- Case Study: Matrix with near-degenerate singular values ---\n")

# Create a matrix where we KNOW the exact singular values
# Use a diagonal matrix plus small perturbation
true_singular_values = [10.0, 10.0, 10.0, 5.0, 5.0, 1.0]
n = length(true_singular_values)

# Midpoint is exactly diagonal
A_mid = Matrix(Diagonal(true_singular_values))

# Add small radii to make it interesting for VBD
A_rad = zeros(n, n)
# Cluster 1: Three singular values at 10.0 (add coupling)
A_rad[1, 2] = A_rad[2, 1] = 0.05
A_rad[2, 3] = A_rad[3, 2] = 0.05
A_rad[1, 3] = A_rad[3, 1] = 0.03

# Cluster 2: Two singular values at 5.0 (add coupling)
A_rad[4, 5] = A_rad[5, 4] = 0.08

# Isolated: One singular value at 1.0 (no coupling)

A = BallMatrix(A_mid, A_rad)

println("True singular values: ", true_singular_values)
println("Matrix has 3 natural clusters: {10,10,10}, {5,5}, {1}")
println()

# Compute with both methods
result_rump = rigorous_svd(A; apply_vbd=false)
result_vbd = rigorous_svd(A; apply_vbd=true)

# Compare singular value enclosures
println("="^80)
println("SINGULAR VALUE ENCLOSURES (Individual)")
println("="^80)
println()
@printf("%-5s %-12s %-25s %-25s %-10s\n", "i", "True σᵢ", "Rump Bound", "VBD Bound", "Same?")
println("-"^80)

for i in 1:n
    true_val = true_singular_values[i]
    rump_bound = result_rump.singular_values[i]
    vbd_bound = result_vbd.singular_values[i]

    rump_str = @sprintf("[%.6f, %.6f]", inf(rump_bound), sup(rump_bound))
    vbd_str = @sprintf("[%.6f, %.6f]", inf(vbd_bound), sup(vbd_bound))
    same = rump_bound == vbd_bound ? "✓" : "✗"

    @printf("%-5d %-12.6f %-25s %-25s %-10s\n", i, true_val, rump_str, vbd_str, same)

    # Verify true value is contained
    if !(true_val ∈ rump_bound)
        println("  ⚠ Warning: True value not in Rump bound!")
    end
    if !(true_val ∈ vbd_bound)
        println("  ⚠ Warning: True value not in VBD bound!")
    end
end

println()
println("Conclusion: Individual singular value bounds are IDENTICAL for both methods.")
println()

# Now show VBD's additional information
if result_vbd.block_diagonalisation !== nothing
    vbd = result_vbd.block_diagonalisation

    println("="^80)
    println("VBD ADDITIONAL INFORMATION (Not available in pure Rump)")
    println("="^80)
    println()

    println("Number of clusters identified: ", length(vbd.clusters))
    println()

    for (k, cluster) in enumerate(vbd.clusters)
        println("Cluster $k: Indices $cluster (size $(length(cluster)))")
        println("-"^80)

        # Show eigenvalues of Σ² in this cluster
        eigenvals_in_cluster = vbd.eigenvalues[cluster]
        singular_vals_in_cluster = sqrt.(eigenvals_in_cluster)

        println("  Singular values in cluster:")
        for (j, idx) in enumerate(cluster)
            @printf("    σ[%d] = %.6f (from Σ²: %.6f)\n",
                    idx, singular_vals_in_cluster[j], eigenvals_in_cluster[j])
        end

        # Gershgorin interval for this cluster
        interval = vbd.cluster_intervals[k]
        println("\n  Gershgorin disc enclosure (for Σ²):")
        @printf("    Interval: [%.6f, %.6f]\n", inf(interval), sup(interval))
        @printf("    Center:   %.6f\n", mid(interval))
        @printf("    Radius:   %.6f\n", rad(interval))

        # Show what singular values this corresponds to
        sqrt_inf = sqrt(max(0, inf(interval)))
        sqrt_sup = sqrt(sup(interval))
        @printf("\n  Corresponding singular value range: [%.6f, %.6f]\n",
                sqrt_inf, sqrt_sup)

        println()
    end

    # Block structure information
    println("="^80)
    println("BLOCK-DIAGONAL STRUCTURE")
    println("="^80)
    println()
    println("The VBD provides a decomposition: Σ² = Q'(Σ²)Q = D + R")
    println("where D is block-diagonal and R is the off-diagonal remainder.")
    println()
    @printf("Remainder norm bound: ‖R‖₂ ≤ %.6e\n", vbd.remainder_norm)
    println()
    println("Block structure:")
    for (k, cluster) in enumerate(vbd.clusters)
        @printf("  Block %d: rows/cols %d:%d (size %d×%d)\n",
                k, cluster.start, cluster.stop, length(cluster), length(cluster))
    end

    # Practical interpretation
    println()
    println("="^80)
    println("PRACTICAL INTERPRETATION")
    println("="^80)
    println()
    println("What VBD tells us that pure Rump doesn't:")
    println()
    println("1. CLUSTERING: We can rigorously identify which singular values are")
    println("   close together (forming clusters) vs well-separated.")
    println()
    println("2. BLOCK STRUCTURE: The matrix Σ² can be transformed to block-diagonal")
    println("   form with rigorous bounds on the off-diagonal coupling.")
    println()
    println("3. INVARIANT SUBSPACES: Each cluster corresponds to an invariant")
    println("   subspace of Σ² with known dimension and enclosure.")
    println()
    println("4. NUMERICAL STABILITY: The remainder bound ‖R‖₂ quantifies how")
    println("   'block-diagonal' the matrix is, indicating potential numerical")
    println("   difficulties if R is large.")
    println()

    # Show the actual block structure visually
    println("="^80)
    println("VISUAL REPRESENTATION OF BLOCK STRUCTURE")
    println("="^80)
    println()
    println("Σ² in VBD basis (• = diagonal block entry, ○ = small remainder entry):")
    println()

    # Create a visual representation
    for i in 1:n
        for j in 1:n
            # Check which cluster i and j belong to
            i_cluster = findfirst(c -> i ∈ c, vbd.clusters)
            j_cluster = findfirst(c -> j ∈ c, vbd.clusters)

            if i == j
                print("▓")  # Diagonal
            elseif i_cluster == j_cluster
                print("▒")  # Within same block
            else
                print("░")  # Off-diagonal (remainder)
            end
            print(" ")
        end
        # Annotate which cluster this row belongs to
        i_cluster = findfirst(c -> i ∈ c, vbd.clusters)
        println("  ← Cluster $i_cluster")
    end

    println()
    println("Legend: ▓ = diagonal, ▒ = within-block, ░ = off-diagonal remainder")
    println()

    # Summary table
    println("="^80)
    println("SUMMARY: What Each Method Provides")
    println("="^80)
    println()
    println("┌────────────────────────────────────────────┬─────────┬─────────┐")
    println("│ Information                                │  Rump   │   VBD   │")
    println("├────────────────────────────────────────────┼─────────┼─────────┤")
    @printf("│ Certified singular value bounds            │    ✓    │    ✓    │\n")
    @printf("│ Residual norm ‖U·Σ·V' - A‖                │    ✓    │    ✓    │\n")
    @printf("│ Orthogonality defects ‖U'U-I‖, ‖V'V-I‖    │    ✓    │    ✓    │\n")
    println("├────────────────────────────────────────────┼─────────┼─────────┤")
    @printf("│ Clustering identification                  │    ✗    │    ✓    │\n")
    @printf("│ Gershgorin disc enclosures                 │    ✗    │    ✓    │\n")
    @printf("│ Block-diagonal decomposition               │    ✗    │    ✓    │\n")
    @printf("│ Remainder bound ‖R‖₂                       │    ✗    │    ✓    │\n")
    @printf("│ Invariant subspace dimensions              │    ✗    │    ✓    │\n")
    @printf("│ Basis for block structure (Q matrix)      │    ✗    │    ✓    │\n")
    println("└────────────────────────────────────────────┴─────────┴─────────┘")
    println()
    println("Computational cost: VBD requires additional eigendecomposition + clustering")
    println("                    Worth it when clustering structure is needed!")
end

println("\n" * "="^80)
println("CONCLUSION")
println("="^80)
println()
println("• Singular value BOUNDS are identical (both use Rump's certification)")
println("• VBD adds STRUCTURAL information:")
println("    - Which singular values cluster together")
println("    - Rigorous block-diagonal decomposition")
println("    - Basis transformation that reveals cluster structure")
println("    - Quantification of off-diagonal coupling")
println()
println("• Use pure Rump when: You only need singular value bounds")
println("• Use Rump+VBD when: You need clustering or block structure analysis")
println("="^80)
println()
