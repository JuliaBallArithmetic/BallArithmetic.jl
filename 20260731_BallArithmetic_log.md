# 2026-07-31 — BallArithmetic.jl log

## Type-preservation audit of Ball array constructors

Probed `BallArray`/`BallMatrix`/`BallVector` construction, indexing and
propagation against structured (`Diagonal`, `*Triangular`, `Symmetric`,
`Hermitian`, `Tridiagonal`, `Bidiagonal`, `Adjoint`, `Transpose`), sparse,
view/range, and `UniformScaling` inputs. Scripts: `scratchpad/audit{,2,3}.jl`.

### Central result

The storage machinery is *fully* structure-polymorphic — but only through the
**two-argument** constructor. `BallMatrix(Diagonal, Diagonal)` keeps
`Diagonal` in both `c` and `r`, and `+`, `*` and `upper_bound_L2_opnorm` all
preserve it end to end (likewise `SparseMatrixCSC`, `UpperTriangular`).

The **one-argument** constructor destroys it: `rad(A) = zeros(T, size(A))`
(`array.jl:50`, `matrix.jl:85`, `vector.jl:52`) always returns a dense
`Matrix`. So `c` keeps its type and `r` goes dense.

Measured densification cost, `sprand(n, n, 0.002)`:

| n | nnz | `c` bytes | `r` bytes | ratio |
|---|-----|-----------|-----------|-------|
| 500 | 508 | 13 128 | 2 000 048 | 152× |
| 2000 | 7 864 | 146 656 | 32 000 048 | 218× |
| 5000 | 49 923 | 848 496 | 200 000 048 | 236× |

Note `SparseArrays` is not a dependency and no file in `src/` references it —
sparse support is incidental via `AbstractMatrix`, not designed or tested.

### Rigor findings (no false enclosures observed)

Products of structured-backed ball matrices were checked against BigFloat
ground truth for `Diagonal`, `UpperTriangular`, `Symmetric`, sparse and
`Adjoint`: **0/64 bad entries in every case**. No rigor violation found.

But construction performs **no validation**, contradicting the comment at
`matrix.jl:62-65` which claims sizes, element types and non-negativity of `r`
are checked:

- `BallMatrix(rand(3,3), rand(2,2))` builds an object reporting `size (3,3)`
  with a 2×2 radius; `X[3,3]` then throws `BoundsError`, `upper_bound_L2_opnorm`
  throws `DimensionMismatch`.
- `Ball(1.0, -1.0)` and `BallMatrix(c, -ones(3,3))` are accepted — a negative
  radius is not an enclosure. `Y*Y` on it returned radius 15.0 (garbage, not an
  error).
- `BallMatrix(c, fill(NaN,3,3))` accepted.

### Hard errors

- `permutedims(B)` throws `MethodError: BallArray(::Float64, ::Float64)`.
  Root cause: the catch-all `getindex(M::BallArray, inds...)` (`array.jl:99`)
  assumes the sliced result is an array and wraps it in `BallArray`. Any index
  pattern mixing scalars with `CartesianIndex{0}` — which `permutedims!` uses
  internally — yields scalars. `B[1, CartesianIndex()]` fails the same way.
- `BallMatrix(rand(1:5,3,3))` fails: no `rad(::Matrix{Int})`.
- `BallMatrix(c::Float64, r::Float32)` and `(BigFloat, Float64)` fail — the
  inner constructor demands identical `T`.
- `strides(::BallMatrix)` unimplemented.

### Type-preservation gaps (return `Matrix{Ball}`, not `BallMatrix`)

`one`, `zero`, `similar`, `collect`, `hcat`, `vcat`, `kron`, `repeat`,
`reverse`, `circshift`, `triu`, `tril`, broadcasting (`B .+ 1`), and
`transpose` (returns a lazy `Transpose` wrapper; `adjoint` correctly returns a
`BallMatrix`). These fall through to the generic `AbstractArray` path — still
rigorous, since element-wise `Ball` arithmetic is used, but no BLAS and the
wrong type.

`imag` of a *real* ball array loses the float type: `zeros(size(A))` at
`array.jl:144` is unparameterised, so `Float32` and `BigFloat` inputs both
return `Float64` storage. `real` is unaffected.

### Slices — correct

`B[1,:]`, `B[:,1]`, `B[1:2,1:2]`, `B[:]`, `diag(B)`, 3-D `[:,:,1]` and
logical indexing all return proper `BallVector`/`BallMatrix`.

`BallMatrix * v` is typed `v::Vector` only, so a view, range or sparse vector
falls back to generic `Vector{Ball}` scalar multiplication instead of the
rigorous BLAS kernel — correct but slow, and the wrong return type.

### UniformScaling

`BallMatrix(I)` / `BallMatrix(2.0I)` have no method (no size — arguably
correct). `B + I`, `B - I`, `B * I`, `I - B` all work and return `BallMatrix`.
`2.0*I(3)` works because it is already a `Diagonal`.

## Fix applied: structure-preserving `rad`

`rad` now returns `_zero_radius(A, T) = fill!(similar(A, T), zero(T))`.

Candidates compared across 15 real and 8 complex layouts
(`scratchpad/radcand.jl`):

| candidate | verdict |
|---|---|
| `zero(A)` | ✗ keeps the **complex** eltype; radius must be real |
| `zero(similar(A, T))` | ✗ `UndefRefError` on `Diagonal{BigFloat}` — reads the undefined refs `similar` leaves |
| `real(zero(A))` | ✓ everywhere, but allocates a complex zero first (costly for BigFloat) |
| `fill!(similar(A, T), zero(T))` | ✓ everywhere — **chosen** |

`UnitUpperTriangular`/`UnitLowerTriangular` fail under *all four* (the unit
diagonal cannot be set to zero), so they get an explicit method storing the
radius in the corresponding non-unit type — the unit diagonal is exactly one,
hence radius exactly zero, which `UpperTriangular` can hold.

Blast radius is small: for a dense `A`, `fill!(similar(A,T),zero(T))` is
identical to the old `zeros(T, size(A))`. Only structured/sparse inputs change.
`Adjoint`, `Transpose`, `SubArray` and ranges still give a dense radius, since
`similar` already returns a plain array for them.

### Results

Sparse radius, `sprand(n, n, 0.002)` — was fully dense:

| n | nnz | `c` | `r` before | `r` after |
|---|-----|-----|-----------|-----------|
| 500 | 498 | 13 048 | 2 000 048 | 12 136 |
| 2000 | 7 996 | 147 712 | 32 000 048 | 144 104 |
| 5000 | 50 186 | 850 600 | 200 000 048 | 843 144 |

237× less radius memory at n=5000.

Rigor re-checked against BigFloat ground truth for `Diagonal`, both
triangular kinds, `UnitUpperTriangular`, `Symmetric`, `Tridiagonal`,
`SymTridiagonal`, `Bidiagonal`, sparse, `Adjoint`, `Hermitian{Complex}` and
`sparse{Complex}`: squares, sums, differences and ball-scalar products all gave
**0 enclosure violations**, and every `upper_bound_L2_opnorm` dominated the true
2-norm. Specifically checked that a `Symmetric`/`Hermitian` radius wrapper —
which reads only one triangle — does not mirror away an inflation when the
other operand is unsymmetric (`Symmetric*dense`, `dense*Symmetric`,
`sparse*dense`: 0 bad entries).

Only one in-place radius write exists in the package
(`norm_bounds/oishi.jl:56`); its matrix comes from the two-argument
constructor with dense broadcasts, so it is unaffected.

`SparseArrays` added to `test/Project.toml` (it is still not referenced
anywhere in `src/`, so no main-project dependency was added). New file
`test/test_types/test_structured_radius.jl`, 81 tests.

Full suite: **4171 pass, 15 broken (pre-existing), 0 fail**, 6m45s
(was 4090 pass / 15 broken).

### Still open (audit findings not yet fixed)

`permutedims` throwing, the missing construction validation, `imag` losing the
float type, and `*` on views/ranges falling back to the generic path.
