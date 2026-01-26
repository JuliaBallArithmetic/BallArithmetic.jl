# Implementable Methods from Horáček's PhD Thesis
## "Interval linear and nonlinear systems" (2012)

This document summarizes algorithms, methods, and techniques from Jaroslav Horáček's PhD thesis that could be implemented in BallArithmetic.jl.

---

## 1. Interval Linear Systems - Square Systems

### 1.1 Basic Solution Methods

#### **Krawczyk Method**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Enclose solution set of Ax = b where A ∈ IR^(n×n), b ∈ IR^n
- **Key Formula**:
  ```
  K(x, A, b) = x̃ + (I - CA)(x - x̃) + C(b - Ax̃)
  ```
  where C is a preconditioner (typically C ≈ A_c^(-1))
- **Complexity**: Polynomial time per iteration
- **Implementation notes**:
  - Requires good preconditioner selection
  - ε-inflation can prevent empty intersection
  - Iterative refinement possible

#### **Interval Jacobi Method**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Iterative enclosure refinement
- **Key Formula**:
  ```
  x_i^(k+1) = (b_i - Σ_{j≠i} a_{ij}x_j^(k)) / a_{ii}
  ```
- **Complexity**: Polynomial time per iteration
- **Implementation notes**:
  - Component-wise iteration
  - May not converge for all matrices
  - Works well for diagonally dominant matrices

#### **Interval Gauss-Seidel Method**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Improved iterative enclosure using most recent values
- **Key Formula**:
  ```
  x_i^(k+1) = (b_i - Σ_{j<i} a_{ij}x_j^(k+1) - Σ_{j>i} a_{ij}x_j^(k)) / a_{ii}
  ```
- **Complexity**: Polynomial time per iteration
- **Implementation notes**:
  - Uses updated components immediately
  - Generally faster convergence than Jacobi
  - Order of variables can affect convergence

#### **Hansen-Bliek-Rohn (HBR) Method**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Tight enclosure using extremal systems
- **Key idea**: Solve 2n real systems at vertices
- **Complexity**: O(n^4) - polynomial but expensive
- **Implementation notes**:
  - Provides tighter enclosures than Krawczyk
  - Computationally intensive
  - Best for high-accuracy requirements

#### **Gaussian Elimination Method**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Direct solution via elimination
- **Complexity**: O(n^3) - polynomial
- **Implementation notes**:
  - Can detect singularity during elimination
  - Produces enclosure of solution set
  - Susceptible to overestimation without preconditioning

### 1.2 Refinement Techniques

#### **ε-Inflation**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Ensure non-empty intersection in iterative methods
- **Key Formula**: Inflate intervals by small ε before intersection
- **Implementation notes**:
  - Prevents premature termination
  - ε selection affects accuracy
  - Essential for robust convergence

#### **Shaving Method with Sherman-Morrison Formula**
- **Reference**: Part 4, Chapter 5
- **Purpose**: Remove provably infeasible parts of solution enclosure
- **Key Formula** (Sherman-Morrison):
  ```
  (A + uv^T)^(-1) = A^(-1) - (A^(-1)uv^T A^(-1))/(1 + v^T A^(-1)u)
  ```
- **Complexity**: O(n^2) per update (vs O(n^3) for full inverse)
- **Implementation notes**:
  - Significantly more efficient than recomputing inverse
  - Useful for boundary constraint propagation
  - Can be applied iteratively to each variable boundary
  - Particularly effective when combined with Krawczyk iterations

---

## 2. Interval Linear Systems - Overdetermined Systems

### 2.1 Subsquares Approach

#### **Subsquares Enclosure Method**
- **Reference**: Part 5, Chapter 6
- **Purpose**: Solve overdetermined system Ax = b where A ∈ IR^(m×n), m > n
- **Key idea**: Consider all (n choose m) square subsystems
- **Complexity**: Combinatorial - O(C(m,n) × n^3)
- **Implementation notes**:
  - Solution exists iff at least one subsystem is solvable
  - Union of all subsystem solutions gives enclosure
  - Computationally expensive for large m
  - Can detect unsolvability

### 2.2 Multi-Jacobi Method

#### **Multi-Jacobi for Overdetermined Systems**
- **Reference**: Part 5, Chapter 6
- **Purpose**: Iterative method for overdetermined systems
- **Key Formula**:
  ```
  x_j^(k+1) = ⋂_{i: a_{ij} ≠ 0} [(b_i - Σ_{l≠j} a_{il}x_l^(k)) / a_{ij}]
  ```
- **Complexity**: Polynomial per iteration
- **Implementation notes**:
  - Intersection over all equations containing variable x_j
  - May not converge for all overdetermined systems
  - Empty intersection indicates unsolvability

### 2.3 Least Squares Approach

#### **Interval Least Squares**
- **Reference**: Part 5, Chapter 6
- **Purpose**: Minimize ||Ax - b||² over interval matrix/vector
- **Key Formula**: Related to normal equations A^T Ax = A^T b
- **Complexity**: Depends on specific formulation
- **Implementation notes**:
  - Useful when exact solution doesn't exist
  - Can provide error bounds
  - Multiple formulations possible (tolerance/control)

---

## 3. Matrix Property Verification

### 3.1 Regularity Testing

#### **Sufficient Condition for Regularity (Theorem 11.12)**
- **Reference**: Part 10, Chapter 11
- **Purpose**: Verify that all matrices in [A] are nonsingular
- **Condition**: `λ_max(A_∆^T A_∆) < λ_min(A_c^T A_c)`
- **Complexity**: O(n^3) - polynomial (eigenvalue computation)
- **Implementation notes**:
  - Sufficient but not necessary
  - Very efficient when applicable
  - Requires symmetric eigenvalue computations

#### **Sufficient Condition for Singularity (Theorem 11.13)**
- **Reference**: Part 10, Chapter 11
- **Purpose**: Verify that at least one matrix in [A] is singular
- **Complexity**: O(n^3) - polynomial
- **Implementation notes**:
  - Dual to regularity condition
  - Useful for detecting degenerate cases

### 3.2 Solvability/Unsolvability Detection

#### **Gaussian Elimination Unsolvability Test**
- **Reference**: Part 6, Chapter 7
- **Purpose**: Detect when Ax = b has no solution
- **Key idea**: Zero appears on diagonal during elimination
- **Complexity**: O(n^3) - polynomial
- **Implementation notes**:
  - Can fail early on detection
  - Provides certificate of unsolvability
  - Not complete (may miss some unsolvable cases)

#### **Subsquares Unsolvability Test**
- **Reference**: Part 6, Chapter 7
- **Purpose**: For overdetermined systems, check all subsystems
- **Key idea**: If all subsystems unsolvable, system is unsolvable
- **Complexity**: O(C(m,n) × n^3)
- **Implementation notes**:
  - More expensive but more comprehensive
  - Particularly useful for overdetermined systems

### 3.3 Full Column Rank Verification

#### **Full Column Rank Test**
- **Reference**: Part 10, Chapter 11
- **Purpose**: Verify A has full column rank for all A ∈ [A]
- **Complexity**: coNP-complete (no known polynomial algorithm)
- **Implementation notes**:
  - Can use sufficient conditions for special cases
  - Important for least squares problems
  - May require exponential time in general case

---

## 4. Interval Determinant Computation

### 4.1 Direct Methods

#### **Gaussian Elimination Determinant**
- **Reference**: Part 6-7, Chapters 7-8
- **Purpose**: Compute det([A]) via elimination
- **Key Formula**: Product of diagonal elements after elimination
- **Complexity**: O(n^3) - polynomial
- **Implementation notes**:
  - Can detect singularity during process
  - Overestimation due to wrapping effect
  - Standard method but not always tightest

#### **Hadamard's Inequality**
- **Reference**: Part 6, Chapter 7
- **Purpose**: Upper bound on |det(A)|
- **Key Formula**: `|det(A)| ≤ ∏_{i=1}^n ||a_i||`
- **Complexity**: O(n^2) - very fast
- **Implementation notes**:
  - Only provides upper bound
  - Very efficient for quick checks
  - Can be used to prove nonsingularity

#### **Cramer's Rule Based**
- **Reference**: Part 6, Chapter 7
- **Purpose**: Compute determinant via cofactor expansion
- **Complexity**: O(n!) - exponential
- **Implementation notes**:
  - Only practical for very small n (n ≤ 4)
  - Exact interval arithmetic result
  - Too slow for general use

### 4.2 Eigenvalue-Based Methods

#### **Eigenvalue Product Method**
- **Reference**: Part 7, Chapter 8
- **Purpose**: det(A) = ∏ λ_i
- **Complexity**: O(n^3) for eigenvalue computation
- **Implementation notes**:
  - Requires interval eigenvalue computation
  - Can provide tight bounds for symmetric matrices
  - Complex eigenvalues require careful handling

### 4.3 Gerschgorin-Based Bounds

#### **Gerschgorin Discs for Determinant**
- **Reference**: Part 6, Chapter 7
- **Purpose**: Bound eigenvalues, hence determinant
- **Key Formula**: `λ ∈ ⋃_i {z : |z - a_{ii}| ≤ Σ_{j≠i} |a_{ij}|}`
- **Complexity**: O(n^2) - very fast
- **Implementation notes**:
  - Provides rough bounds quickly
  - Tighter bounds for diagonally dominant matrices
  - Can detect nonsingularity

---

## 5. Preconditioning Strategies

### 5.1 Midpoint Inverse Preconditioning

#### **Standard Midpoint Preconditioner**
- **Reference**: Part 3, Chapter 4
- **Purpose**: C = A_c^(-1) for reducing interval width
- **Complexity**: O(n^3) for computing inverse
- **Implementation notes**:
  - Most common choice
  - Works well when A_c is well-conditioned
  - Single computation before iteration

### 5.2 LU-Based Preconditioning

#### **LU Decomposition Preconditioner**
- **Reference**: Part 3, Chapter 4
- **Purpose**: C from LU factorization of A_c
- **Complexity**: O(n^3) for LU, O(n^2) per solve
- **Implementation notes**:
  - More stable than direct inverse
  - Can reuse factorization
  - Better for iterative methods

### 5.3 LDLT Preconditioning

#### **LDLT Preconditioner for Symmetric Matrices**
- **Reference**: Part 3, Chapter 4
- **Purpose**: Exploit symmetry for efficiency
- **Complexity**: O(n^3/6) - half of LU cost
- **Implementation notes**:
  - Only for symmetric [A]
  - More efficient than LU
  - Numerically stable

---

## 6. Special Matrix Classes - Efficient Algorithms

### 6.1 Bidiagonal Systems

#### **Strongly Polynomial Algorithm for Bidiagonal Systems**
- **Reference**: Part 10, Proposition 11.18
- **Purpose**: Solve bidiagonal Ax = b
- **Complexity**: Strongly polynomial O(n)
- **Implementation notes**:
  - Forward/backward substitution
  - Very efficient
  - Exact interval result without iteration

### 6.2 Inverse Nonnegative Matrices

#### **Polynomial Algorithm for A_c = I Case**
- **Reference**: Part 10, Theorem 11.21
- **Purpose**: When center is identity matrix
- **Complexity**: Strongly polynomial
- **Implementation notes**:
  - Special case but important
  - Can compute matrix inverse efficiently
  - Useful for perturbation problems

### 6.3 Diagonal Dominance

#### **Efficient Methods for SDD Matrices**
- **Reference**: Various chapters
- **Purpose**: Exploit strict diagonal dominance
- **Key property**: Jacobi/Gauss-Seidel guaranteed to converge
- **Implementation notes**:
  - Check condition: `|a_{ii}| > Σ_{j≠i} |a_{ij}|` for all i
  - Many methods have better performance
  - Gerschgorin bounds are tight

---

## 7. Constraint Satisfaction and Linearization

### 7.1 Beaumont's Theorem for Absolute Value

#### **Linearization of |y|**
- **Reference**: Part 9, Chapter 10
- **Purpose**: Linear relaxation of absolute value constraints
- **Key Formula**: For any y ∈ IR,
  ```
  |y| ≤ αy + β
  where α = (|ȳ| - |y|)/(ȳ - y)
        β = (ȳ|y| - y|ȳ|)/(ȳ - y)
  ```
- **Complexity**: O(1) per constraint
- **Implementation notes**:
  - Enables LP solver use for interval CSP
  - Tightness depends on inner point selection
  - Can combine multiple linearizations

### 7.2 Advanced Linearization (Proposition 10.3)

#### **Linearization Coefficients**
- **Reference**: Part 9, Proposition 10.3
- **Key Formula**:
  ```
  α_i = (x_i^c - x_i^0) / x_i^∆
  v_i = (x_i^c x_i^0 - x_i x_i) / x_i^∆
  ```
  where x_i^0 is inner point, x_i^c is center, x_i^∆ is radius
- **Implementation notes**:
  - Choice of x_i^0 affects tightness
  - Can use vertex selection strategies
  - Useful for nonlinear interval problems

### 7.3 Combination of Centers (Proposition 10.4)

#### **Multiple Linearization Combination**
- **Reference**: Part 9, Proposition 10.4
- **Purpose**: Tighter enclosure via multiple inner points
- **Key idea**: Take intersection of multiple linearizations
- **Complexity**: Linear in number of linearizations
- **Implementation notes**:
  - More linearizations = tighter bounds
  - Diminishing returns after few linearizations
  - Balance accuracy vs computation cost

### 7.4 Convex Case Optimization

#### **Convex Linearization (Proposition 10.5)**
- **Reference**: Part 9, Proposition 10.5
- **Purpose**: Optimal linearization for convex functions
- **Key property**: Convexity ensures global bounds
- **Implementation notes**:
  - More efficient than general case
  - Applicable to many practical problems
  - Can use gradient information

---

## 8. Tolerance and Control Solutions

### 8.1 Tolerance Solution

#### **Polynomial-Time Tolerance Solution**
- **Reference**: Part 10, Chapter 11
- **Purpose**: Find x such that `|A_c x - b_c| ≤ -A_∆|x| + δ`
- **Complexity**: Polynomial time (via LP)
- **Implementation notes**:
  - Tractable problem
  - δ is tolerance vector
  - Can use standard LP solvers
  - Provides robustness guarantees

### 8.2 Control Solution

#### **Control Problem Formulation**
- **Reference**: Part 10, Chapter 11
- **Purpose**: Find x such that `|A_c x - b_c| ≤ A_∆|x| - δ`
- **Complexity**: NP-complete
- **Implementation notes**:
  - Harder than tolerance problem
  - May require heuristics for large problems
  - Important in control theory applications
  - Sign flip from tolerance makes it intractable

---

## 9. Complexity Results - Implementation Guidance

### 9.1 Tractable Problems (Polynomial Time)

These can be implemented efficiently:

1. **Regular matrix verification** (sufficient condition): O(n^3)
2. **Singular matrix verification** (sufficient condition): O(n^3)
3. **Bidiagonal system solution**: O(n)
4. **Tolerance solution**: Polynomial (LP-based)
5. **Gerschgorin bounds**: O(n^2)
6. **Hadamard inequality**: O(n^2)
7. **Determinant via Gaussian elimination**: O(n^3)
8. **A_c = I inverse computation**: Strongly polynomial

### 9.2 Intractable Problems (NP-hard/coNP-hard)

These require heuristics or approximations:

1. **General regularity verification**: coNP-complete
2. **Optimal solution enclosure**: NP-hard
3. **Full column rank verification**: coNP-complete
4. **Control solution**: NP-complete
5. **Exact solution set boundary**: Generally intractable

### 9.3 Implementation Strategy

- **For tractable problems**: Implement exact algorithms
- **For intractable problems**:
  - Implement sufficient conditions (fast, incomplete)
  - Provide heuristic methods for small dimensions
  - Document computational complexity clearly
  - Consider approximation algorithms

---

## 10. LIME² Toolbox - Reference Implementation

### 10.1 Package Structure Lessons

#### **Modular Organization** (Part 11, Chapter 12)
- **ils**: Interval linear systems (square)
- **oils**: Overdetermined interval linear systems
- **idet**: Interval determinant
- **iest**: Interval estimation/regression
- **ieig**: Interval eigenvalues
- **iviz**: Visualization
- **useful**: Utility functions
- **ocdoc**: Documentation

**Implementation notes for BallArithmetic.jl**:
- Consider similar modular structure
- Separate square vs overdetermined systems
- Group related functionality
- Provide visualization utilities

### 10.2 Key Functions from LIME²

#### **Square Systems (ils)**
- `ilsjacobienc`: Jacobi method
- `ilsgsenc`: Gauss-Seidel method
- `ilsgeenc`: Gaussian elimination
- `ilskrawczykenc`: Krawczyk method
- `ilshbrenc`: Hansen-Bliek-Rohn method
- `ilshullver`: Hull verification
- `isuns`: Unsolvability test
- `issolvable`: Solvability test

#### **Overdetermined Systems (oils)**
- `oilssubsqenc`: Subsquares method
- `oilsmultijacenc`: Multi-Jacobi method
- `oilslsqenc`: Least squares

#### **Determinant (idet)**
- `idethad`: Hadamard inequality
- `idetcram`: Cramer's rule
- `idetgauss`: Gaussian elimination
- `idetgersch`: Gerschgorin bounds
- `idetencsym`: Enclosure for symmetric matrices

**Implementation notes**:
- Provides naming convention examples
- Shows separation of concerns
- Indicates which methods are worth implementing

---

## 11. Practical Applications Identified

### 11.1 Medical Signal Processing
- **Reference**: Part 7, Chapter 8
- **Application**: Breath detection from monitoring signals
- **Techniques used**: Interval regression, interval least squares
- **Implementation notes**:
  - Real-world validation of methods
  - Shows importance of overdetermined system solvers
  - Demonstrates noise handling with intervals

### 11.2 Interval Regression
- **Reference**: Part 8, Chapter 9
- **Purpose**: Fit linear model with interval data
- **Key idea**: Find coefficients that satisfy all interval constraints
- **Implementation notes**:
  - Can use linear programming
  - Related to tolerance problem
  - Useful for data with known uncertainties

---

## 12. Implementation Priority Recommendations

### High Priority (Core Functionality)

1. **Krawczyk method** - Essential, widely used, polynomial time
2. **Gaussian elimination** - Fundamental, detects singularity
3. **Gauss-Seidel method** - Iterative refinement, often converges fast
4. **ε-inflation** - Necessary for robust iteration
5. **Midpoint preconditioning** - Standard, effective
6. **Regularity sufficient conditions** - Fast, practical checks
7. **Gerschgorin bounds** - Cheap, useful for many properties

### Medium Priority (Extended Functionality)

1. **Jacobi method** - Parallelizable alternative to Gauss-Seidel
2. **HBR method** - Tighter enclosures when precision needed
3. **Shaving with Sherman-Morrison** - Efficient boundary refinement
4. **Subsquares method** - Overdetermined systems support
5. **Hadamard inequality** - Fast determinant bounds
6. **Bidiagonal solver** - Efficient special case
7. **Tolerance solution** - Practical robustness analysis

### Low Priority (Specialized/Expensive)

1. **Multi-Jacobi** - Overdetermined systems, may not converge
2. **Least squares** - Specialized use case
3. **Cramer's determinant** - Only n ≤ 4
4. **Control solution** - NP-complete, limited practical use
5. **Full eigenvalue methods** - Complex, expensive

### Research/Experimental

1. **Beaumont linearization** - For interval CSP, nonlinear problems
2. **Advanced linearization combinations** - Cutting-edge techniques
3. **Convex optimization approaches** - Specialized applications

---

## 13. Key Formulas Summary

### Oettli-Prager Theorem
```
Ax = b has a solution ⟺ |A_c x - b_c| ≤ A_∆|x| + b_∆
```

### Krawczyk Operator
```
K(x, A, b) = x̃ + (I - CA)(x - x̃) + C(b - Ax̃)
```

### Sherman-Morrison Formula
```
(A + uv^T)^{-1} = A^{-1} - (A^{-1}uv^T A^{-1})/(1 + v^T A^{-1}u)
```

### Beaumont Linearization
```
|y| ≤ αy + β
α = (|ȳ| - |y|)/(ȳ - y)
β = (ȳ|y| - y|ȳ|)/(ȳ - y)
```

### Regularity Sufficient Condition
```
λ_max(A_∆^T A_∆) < λ_min(A_c^T A_c) ⟹ A is regular
```

### Gerschgorin Theorem
```
Every eigenvalue λ satisfies: λ ∈ ⋃_i {z : |z - a_{ii}| ≤ Σ_{j≠i} |a_{ij}|}
```

---

## 14. References for Implementation

- **Primary source**: Horáček, J. (2012). Interval linear and nonlinear systems. PhD thesis, Charles University in Prague.
- **LIME² toolbox**: https://kam.mff.cuni.cz/~horacek/lime (Octave implementation)
- **Total bibliography entries**: 222 references (see part 12 for complete list)

---

## Notes

- Complexity classifications help prioritize which methods to implement exactly vs approximately
- Many "expensive" methods have practical value for small dimensions (n ≤ 10)
- Sufficient conditions are valuable even when not necessary - fast negative/positive results
- LIME² provides a reference implementation structure to learn from
- Medical application shows real-world value of interval methods for uncertainty handling
