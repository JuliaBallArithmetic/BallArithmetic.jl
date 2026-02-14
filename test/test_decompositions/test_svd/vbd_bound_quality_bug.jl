using BallArithmetic
using LinearAlgebra
using Printf

"""
Demonstrate that VBD does NOT provide tighter bounds than pure Rump.
This appears to be a bug or incomplete implementation of Miyajima 2014.
"""

println("\n" * "="^80)
println("BUG REPORT: VBD Does Not Refine Singular Value Bounds")
println("="^80)
println()

# Test case: Matrix with interval uncertainty
A_mid = Diagonal([10.0, 5.0, 1.0])
A_rad = [0.0 0.5 0.1;
         0.5 0.0 0.2;
         0.1 0.2 0.0]
A = BallMatrix(A_mid, A_rad)

println("Test matrix: Diagonal([10, 5, 1]) with off-diagonal radii")
println()

# Compute with both methods
result_rump = rigorous_svd(A; apply_vbd=false)
result_vbd = rigorous_svd(A; apply_vbd=true)

# Issue 1: Bounds are identical (VBD doesn't refine)
println("ISSUE 1: Singular value bounds are IDENTICAL")
println("-"^80)
println()
@printf("%-5s %-30s %-30s %-10s\n", "i", "Rump Bound", "VBD Bound", "Same?")
println("-"^80)

for i in 1:3
    rump_b = result_rump.singular_values[i]
    vbd_b = result_vbd.singular_values[i]

    rump_str = @sprintf("[%.6f, %.6f]", inf(rump_b), sup(rump_b))
    vbd_str = @sprintf("[%.6f, %.6f]", inf(vbd_b), sup(vbd_b))
    same = (rump_b == vbd_b) ? "YES" : "NO"

    @printf("%-5d %-30s %-30s %-10s\n", i, rump_str, vbd_str, same)
end

println()
println("Conclusion: VBD does not refine the bounds from Rump's method.")
println()

# Issue 2: Gerschgorin bounds are actually WORSE
if result_vbd.block_diagonalisation !== nothing
    vbd = result_vbd.block_diagonalisation

    println()
    println("ISSUE 2: Gershgorin bounds from VBD are WIDER than Rump")
    println("-"^80)
    println()

    @printf("%-10s %-30s %-15s %-30s %-15s\n",
            "Cluster", "Rump Bound", "Width", "Gershgorin Bound", "Width")
    println("-"^80)

    for (k, cluster) in enumerate(vbd.clusters)
        if length(cluster) == 1
            # Get the Rump bound (WARNING: indices may not match due to permutation!)
            idx = cluster[1]
            # FIXME: This assumes no permutation - may be wrong!

            # Gershgorin bound for this cluster
            interval = vbd.cluster_intervals[k]
            gershgorin_lower = sqrt(max(0, inf(interval)))
            gershgorin_upper = sqrt(sup(interval))
            gershgorin_width = gershgorin_upper - gershgorin_lower

            # Try to find corresponding Rump bound
            # This is problematic because of VBD permutation!
            eigenval = vbd.eigenvalues[idx]
            approx_sigma = sqrt(eigenval)

            # Find closest Rump bound
            closest_idx = argmin([abs(sqrt(eigenval) - mid(result_rump.singular_values[i]))
                                  for i in 1:3])
            rump_b = result_rump.singular_values[closest_idx]
            rump_lower = inf(rump_b)
            rump_upper = sup(rump_b)
            rump_width = rump_upper - rump_lower

            rump_str = @sprintf("[%.6f, %.6f]", rump_lower, rump_upper)
            gersh_str = @sprintf("[%.6f, %.6f]", gershgorin_lower, gershgorin_upper)

            comparison = gershgorin_width > rump_width ? "WORSE" : "BETTER"

            @printf("%-10d %-30s %-15.6f %-30s %-15.6f %s\n",
                    k, rump_str, rump_width, gersh_str, gershgorin_width, comparison)
        end
    end

    println()
    println("Conclusion: Gershgorin bounds are WIDER, not tighter!")
    println()
end

# Issue 3: Permutation mismatch
println()
println("ISSUE 3: Index mismatch between VBD and singular values")
println("-"^80)
println()
println("VBD permutes eigenvalues to make clusters contiguous, but")
println("singular_values array remains in original SVD order.")
println()
println("This creates confusion when interpreting cluster indices:")
println()

if result_vbd.block_diagonalisation !== nothing
    vbd = result_vbd.block_diagonalisation

    println("VBD clusters (permuted order):")
    for (k, cluster) in enumerate(vbd.clusters)
        eigenvals = vbd.eigenvalues[cluster]
        println("  Cluster $k: indices $cluster, eigenvalues(Σ²) = ", eigenvals)
    end

    println()
    println("Singular values (original SVD order):")
    for i in 1:3
        sv = result_vbd.singular_values[i]
        println("  σ[$i] = [", inf(sv), ", ", sup(sv), "]")
    end

    println()
    println("⚠️  These indices don't correspond! Cluster 1 doesn't mean σ[1].")
end

println()
println("="^80)
println("SUMMARY")
println("="^80)
println()
println("1. VBD does NOT refine/tighten singular value bounds")
println("   - Bounds are computed entirely by Rump's method")
println("   - VBD is only used for clustering analysis")
println()
println("2. Gershgorin bounds from VBD are WIDER than Rump bounds")
println("   - This suggests VBD is not being used optimally")
println("   - OR Miyajima 2014 doesn't claim to improve bounds")
println()
println("3. Index mismatch between VBD clusters and singular_values array")
println("   - VBD permutes to make clusters contiguous")
println("   - singular_values remains in SVD order")
println("   - Makes interpretation confusing/error-prone")
println()
println("QUESTIONS:")
println("- Does Miyajima 2014 provide an alternative bound computation method?")
println("- Should VBD Gershgorin bounds be used instead of Rump for tightness?")
println("- Is the current implementation complete/correct per the paper?")
println("="^80)
println()
