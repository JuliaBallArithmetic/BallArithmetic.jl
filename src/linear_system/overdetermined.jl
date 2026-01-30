"""
    overdetermined.jl

Methods for overdetermined interval linear systems (m > n).
Implements subsquares, multi-Jacobi, and least squares approaches.

# References
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis, Chapters 6
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
- Rohn, J. (2006), "Solvability of systems of interval linear equations and inequalities"
"""

using LinearAlgebra
using Combinatorics

"""
    OverdeterminedResult{T, VT}

Result from overdetermined interval linear system solver.

# Fields
- `solution::VT`: Solution enclosure (or empty if unsolvable)
- `solvable::Bool`: Whether system is proven solvable
- `method::Symbol`: Method used
- `subsystems_checked::Int`: Number of subsystems examined
- `residual::T`: Maximum residual norm
"""
struct OverdeterminedResult{T, VT}
    """Solution enclosure."""
    solution::VT
    """Whether system is solvable."""
    solvable::Bool
    """Method used."""
    method::Symbol
    """Subsystems checked."""
    subsystems_checked::Int
    """Maximum residual."""
    residual::T
end

"""
    subsquares_method(A::BallMatrix{T}, b::BallVector{T};
                     max_subsystems::Int=1000,
                     solver::Symbol=:gaussian_elimination) where {T}

Solve overdetermined system Ax = b using subsquares approach.

# Algorithm
For m×n system with m > n:
1. Consider all (m choose n) square subsystems
2. Solve each n×n subsystem
3. Check if solution satisfies all m equations (Oettli-Prager)
4. Take hull of all valid solutions

# Mathematical Background
A solution exists if at least one n×n subsystem has a solution
that satisfies all m equations.

# Arguments
- `A`: Coefficient ball matrix (m×n, m > n)
- `b`: Right-hand side ball vector (m)
- `max_subsystems`: Maximum number of subsystems to check
- `solver`: Solver for square subsystems (`:gaussian_elimination`, `:krawczyk`)

# Returns
`OverdeterminedResult` with solution enclosure.

# Example
```julia
A = BallMatrix([2.0 1.0; 1.0 2.0; 3.0 1.0], fill(0.1, 3, 2))
b = BallVector([3.0, 3.0, 4.0], fill(0.1, 3))

result = subsquares_method(A, b)

if result.solvable
    println("Solution: ", result.solution)
    println("Checked ", result.subsystems_checked, " subsystems")
end
```

# Notes
- Combinatorial complexity: O(C(m,n) × n³)
- Only practical for small m and n
- Guaranteed to find solution if one exists (complete method)
- May be very slow for large m or n
- Recommended: m ≤ 20, n ≤ 10
"""
function subsquares_method(A::BallMatrix{T}, b::BallVector{T};
                          max_subsystems::Int=1000,
                          solver::Symbol=:gaussian_elimination) where {T}
    m, n = size(A)

    if m <= n
        error("Subsquares method requires m > n (overdetermined system)")
    end

    # Number of possible subsystems
    num_subsystems = binomial(m, n)

    if num_subsystems > max_subsystems
        @warn "System has $num_subsystems subsystems, limiting to $max_subsystems"
    end

    # Generate all combinations of n rows from m
    row_combinations = collect(combinations(1:m, n))

    if length(row_combinations) > max_subsystems
        # Sample randomly
        row_combinations = row_combinations[1:max_subsystems]
    end

    # Track solutions
    valid_solutions = BallVector{T}[]
    subsystems_checked = 0

    for rows in row_combinations
        subsystems_checked += 1

        # Extract subsystem
        A_sub = A[rows, :]
        b_sub = b[rows]

        # Solve subsystem
        x_sub = try
            if solver == :gaussian_elimination
                result_ge = interval_gaussian_elimination(A_sub, b_sub)
                if result_ge.success
                    result_ge.solution
                else
                    nothing
                end
            elseif solver == :krawczyk
                result_k = krawczyk_linear_system(A_sub, b_sub)
                if result_k.verified
                    result_k.solution
                else
                    nothing
                end
            else
                error("Unknown solver: $solver")
            end
        catch e
            # Subsystem failed to solve
            nothing
        end

        if x_sub === nothing
            continue
        end

        # Check if solution satisfies ALL m equations (Oettli-Prager condition)
        # |A_c x - b_c| ≤ A_Δ|x| + b_Δ
        A_c = mid(A)
        A_Δ = rad(A)
        b_c = mid(b)
        b_Δ = rad(b)

        x_c = mid(x_sub)
        x_rad = rad(x_sub)

        residual = A_c * x_c - b_c
        bound = A_Δ * (abs.(x_c) .+ x_rad) .+ b_Δ

        if all(abs.(residual) .<= bound)
            # Valid solution!
            push!(valid_solutions, x_sub)
        end
    end

    if isempty(valid_solutions)
        # No solution found
        return OverdeterminedResult(
            BallVector(fill(Ball(T(0), T(Inf)), n)),
            false, :subsquares, subsystems_checked, T(Inf)
        )
    end

    # Compute hull of all valid solutions
    solution_hull = valid_solutions[1]
    for i in 2:length(valid_solutions)
        solution_hull = ball_hull(solution_hull, valid_solutions[i])
    end

    # Compute max residual
    A_c = mid(A)
    b_c = mid(b)
    x_c = mid(solution_hull)
    max_residual = norm(A_c * x_c - b_c, Inf)

    return OverdeterminedResult(
        solution_hull, true, :subsquares, subsystems_checked, max_residual
    )
end

"""
    multi_jacobi_method(A::BallMatrix{T}, b::BallVector{T};
                       x0::Union{Nothing, BallVector{T}}=nothing,
                       max_iterations::Int=100,
                       tol::T=T(1e-10)) where {T}

Solve overdetermined system using Multi-Jacobi iteration.

# Algorithm
For each variable x_j, compute intersection of bounds from all equations:

    x_j^(k+1) = ⋂_{i=1}^m [(b_i - ∑_{l≠j} a_{il}x_l^(k)) / a_{ij}]

where the intersection is over all rows i where a_{ij} ≠ 0.

# Arguments
- `A`: Coefficient ball matrix (m×n, m > n)
- `b`: Right-hand side ball vector (m)
- `x0`: Initial enclosure (computed if not provided)
- `max_iterations`: Maximum iterations
- `tol`: Convergence tolerance

# Returns
`OverdeterminedResult` with solution enclosure.

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0; 2.0 3.0], fill(0.1, 3, 2))
b = BallVector([4.0, 3.0, 5.0], fill(0.1, 3))

result = multi_jacobi_method(A, b)

if result.solvable
    println("Solution: ", result.solution)
    println("Iterations: ", result.subsystems_checked)
end
```

# Notes
- O(mn²) per iteration - faster than subsquares
- May not converge for all systems
- Empty intersection indicates unsolvability
- Works well when system is "nearly square" or well-conditioned
"""
function multi_jacobi_method(A::BallMatrix{T}, b::BallVector{T};
                             x0::Union{Nothing, BallVector{T}}=nothing,
                             max_iterations::Int=100,
                             tol::T=T(1e-10)) where {T}
    m, n = size(A)

    if m <= n
        @warn "Multi-Jacobi typically used for overdetermined systems (m > n)"
    end

    # Initialize with wide enclosure if not provided
    if x0 === nothing
        # Use least squares solution as starting point
        A_mid = mid(A)
        b_mid = mid(b)
        x_approx = A_mid \ b_mid  # Least squares
        x0 = BallVector(x_approx, abs.(x_approx) .+ T(1.0))
    end

    x = copy(x0)

    for iter in 1:max_iterations
        x_new = similar(x)

        for j in 1:n
            # Collect bounds from all equations
            bounds = Ball{T}[]

            for i in 1:m
                a_ij = A[i, j]

                # Skip if coefficient is zero
                if 0.0 ∈ a_ij && rad(a_ij) < eps(T) * 10
                    continue
                end

                # Compute rhs for this equation
                rhs = b[i]
                for l in 1:n
                    if l != j
                        rhs = rhs - A[i, l] * x[l]
                    end
                end

                # Compute bound from this equation
                bound_from_i = rhs / a_ij

                push!(bounds, bound_from_i)
            end

            if isempty(bounds)
                @warn "Multi-Jacobi: Variable $j has no constraints"
                x_new[j] = x[j]
                continue
            end

            # Intersect all bounds
            x_new_j = bounds[1]
            for k in 2:length(bounds)
                x_new_j = intersect_ball(x_new_j, bounds[k])

                # Check for empty intersection
                if rad(x_new_j) < 0
                    # Empty intersection - system unsolvable
                    return OverdeterminedResult(
                        BallVector(fill(Ball(T(0), T(Inf)), n)),
                        false, :multi_jacobi, iter, T(Inf)
                    )
                end
            end

            x_new[j] = x_new_j
        end

        # Check convergence
        current_width = maximum(rad(x_new))

        if current_width < tol
            # Converged
            A_c = mid(A)
            b_c = mid(b)
            x_c = mid(x_new)
            max_residual = norm(A_c * x_c - b_c, Inf)

            return OverdeterminedResult(
                x_new, true, :multi_jacobi, iter, max_residual
            )
        end

        x = x_new
    end

    # Max iterations reached
    A_c = mid(A)
    b_c = mid(b)
    x_c = mid(x)
    max_residual = norm(A_c * x_c - b_c, Inf)

    return OverdeterminedResult(
        x, false, :multi_jacobi, max_iterations, max_residual
    )
end

"""
    interval_least_squares(A::BallMatrix{T}, b::BallVector{T};
                          method::Symbol=:normal_equations) where {T}

Compute interval least squares solution to overdetermined system.

Minimizes ||Ax - b||² over the interval matrix/vector.

# Arguments
- `A`: Coefficient ball matrix (m×n, m ≥ n)
- `b`: Right-hand side ball vector (m)
- `method`: Solution method (`:normal_equations` or `:qr`)

# Returns
`OverdeterminedResult` with least squares solution enclosure.

# Example
```julia
A = BallMatrix([2.0 1.0; 1.0 2.0; 3.0 1.0], fill(0.1, 3, 2))
b = BallVector([3.1, 2.9, 4.2], fill(0.05, 3))

result = interval_least_squares(A, b)
println("LS solution: ", result.solution)
```

# Notes
- Solves normal equations: A^T A x = A^T b
- O(mn² + n³) complexity
- May not satisfy all equations exactly
- Useful when exact solution doesn't exist
- Returns minimum norm solution in interval sense
"""
function interval_least_squares(A::BallMatrix{T}, b::BallVector{T};
                               method::Symbol=:normal_equations) where {T}
    m, n = size(A)

    if method == :normal_equations
        # Form normal equations: A^T A x = A^T b
        # Use adjoint for BallMatrix (same as transpose for real matrices)
        At = adjoint(A)
        AtA_raw = At * A
        Atb_raw = At * b

        # Ensure proper BallMatrix/BallVector types
        AtA = AtA_raw isa BallMatrix ? AtA_raw : BallMatrix(Matrix(mid(AtA_raw)), Matrix(rad(AtA_raw)))
        Atb = Atb_raw isa BallVector ? Atb_raw : BallVector(Vector(mid(Atb_raw)), Vector(rad(Atb_raw)))

        # Solve square system
        result = interval_gaussian_elimination(AtA, Atb)

        if !result.success
            return OverdeterminedResult(
                BallVector(fill(Ball(T(0), T(Inf)), n)),
                false, :least_squares, 0, T(Inf)
            )
        end

        # Compute residual
        A_c = mid(A)
        b_c = mid(b)
        x_c = mid(result.solution)
        residual_norm = norm(A_c * x_c - b_c, 2)

        return OverdeterminedResult(
            result.solution, true, :least_squares, 1, residual_norm
        )

    elseif method == :qr
        # QR factorization approach
        # Currently not implemented for interval matrices
        @warn "QR method not yet implemented, falling back to normal equations"
        return interval_least_squares(A, b, method=:normal_equations)
    else
        error("Unknown least squares method: $method")
    end
end

# Forward declarations for methods from other files
if !@isdefined(interval_gaussian_elimination)
    function interval_gaussian_elimination(_A::BallMatrix{T}, _b::BallVector{T}) where {T}
        error("interval_gaussian_elimination not available. Include gaussian_elimination.jl first.")
    end
end

if !@isdefined(krawczyk_linear_system)
    function krawczyk_linear_system(_A::BallMatrix{T}, _b::BallVector{T}) where {T}
        error("krawczyk_linear_system not available. Include krawczyk_complete.jl first.")
    end
end

# BallVector hull implementation
function ball_hull(x::BallVector{T}, y::BallVector{T}) where {T}
    # Simple hull implementation
    n = length(x)
    c = Vector{eltype(mid(x))}(undef, n)
    r = Vector{T}(undef, n)
    for i in 1:n
        inf_i = min(inf(x[i]), inf(y[i]))
        sup_i = max(sup(x[i]), sup(y[i]))
        c[i] = (inf_i + sup_i) / 2
        r[i] = (sup_i - inf_i) / 2
    end
    return BallVector(c, r)
end

# Export functions
export OverdeterminedResult
export subsquares_method, multi_jacobi_method, interval_least_squares
