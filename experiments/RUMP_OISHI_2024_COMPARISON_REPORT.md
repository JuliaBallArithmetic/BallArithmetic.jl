# Rump-Oishi 2024 vs SVD for Pseudospectra Certification

## Executive Summary

**The Rump-Oishi 2024 Schur complement method is NOT suitable for pseudospectra certification.**

The method was designed for a specific class of matrices (linearized Galerkin equations) with:
- Increasing diagonal elements (1, 2, 3, ..., n)
- Diagonal dominance in the D block
- Specific decay structure in off-diagonal blocks

The matrices arising in pseudospectra certification (T - zI where T is Schur form) do NOT have this structure.

## Experimental Results

### Success Rate: 0/21 tests (excluding timing tests)

The contraction parameter γ = ‖D_d⁻¹(D_f - CA⁻¹B)‖ must be < 1 for the method to work.
In our tests:
- Well-separated eigenvalues: γ ≈ 4.7 - 7.5
- Clustered eigenvalues: γ ≈ 230 - 17,000
- Large matrices: γ ≈ 54 - 916

### Why It Fails

For a complex upper triangular matrix T - zI, the real 2n×2n representation is:
```
[Re(T-zI)  -Im(T-zI)]
[Im(T-zI)   Re(T-zI)]
```

This structure:
1. Is NOT upper triangular
2. Does NOT have diagonal dominance
3. Has cross-coupling between real and imaginary parts that creates large off-diagonal elements

### One Partial Success

For n=20 with well-separated eigenvalues, the method worked (γ = 0.309) but:
- Gave a looser bound (ratio 1.96x worse than SVD)
- Was slower (18.56ms vs 0.68ms for SVD)

## Recommendations

### Keep Current SVD-based Approach
The rigorous SVD method works reliably and provides tight bounds:
- Success rate: 100%
- Bound quality: Exact to machine precision
- Timing: O(n³) but well-optimized

### Potential Alternatives to Explore

1. **Triangular-specific bounds**: The `backward_singular_value_bound` function uses the triangular structure directly for upper triangular matrices. This could be adapted for complex triangular matrices.

2. **Gerschgorin-type bounds**: For upper triangular T, singular values can be bounded using diagonal elements and row/column sums.

3. **Shifted inverse iteration**: For computing σ_min specifically, inverse iteration on (T-zI)*(T-zI)^H could be efficient.

### When Rump-Oishi 2024 IS Useful

The Schur complement method remains valuable for:
1. Linearized Galerkin equations (its original purpose)
2. Matrices with known diagonal dominance structure
3. Infinite-dimensional operators (as shown in Section 2 of the paper)

## Files Created

- `experiments/rump_oishi_2024_comparison.jl` - Experiment script
- `src/norm_bounds/oishi_2023_schur.jl` - Implementation (updated with RO2024)
- `test/test_norm_bounds/test_oishi_2023_schur.jl` - Tests (updated)
- `docs/src/api/core.md` - Documentation (updated)

## Timing Comparison (for reference)

| n | SVD (ms) | Schur (ms) | Schur verified |
|---|----------|------------|----------------|
| 20 | 0.68 | 18.56 | Yes (γ=0.31) |
| 50 | 3.56 | 3.40 | No (γ=4.64) |
| 100 | 14.20 | 5.87 | No (γ=20.3) |
| 150 | 29.85 | 13.66 | No (γ=9.61) |
| 200 | 59.96 | 19.51 | No (γ=91.5) |

Even when the Schur method is faster (for larger n), it fails to produce valid bounds.

## Conclusion

For pseudospectra certification, continue using the SVD-based approach. The Rump-Oishi 2024 implementation is still valuable for other applications where the matrix has the appropriate diagonal-dominant structure.
