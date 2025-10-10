function abs_preserving_structure(A::LinearAlgebra.AbstractTriangular)
    data = abs.(Matrix(A))
    if LinearAlgebra.istriu(A)
        LinearAlgebra.triu!(data)
    else
        LinearAlgebra.tril!(data)
    end
    return typeof(A)(data)
end

function abs_preserving_structure(A::LinearAlgebra.Symmetric)
    data = abs_preserving_structure(LinearAlgebra.parent(A))
    uplo = A.uplo isa Symbol ? A.uplo : Symbol(String(A.uplo))
    return LinearAlgebra.Symmetric(data, uplo)
end

function abs_preserving_structure(A::LinearAlgebra.Hermitian)
    data = abs_preserving_structure(LinearAlgebra.parent(A))
    uplo = A.uplo isa Symbol ? A.uplo : Symbol(String(A.uplo))
    return LinearAlgebra.Hermitian(data, uplo)
end

function abs_preserving_structure(A::LinearAlgebra.Adjoint{<:Any, <:AbstractMatrix})
    return adjoint(abs_preserving_structure(LinearAlgebra.parent(A)))
end

function abs_preserving_structure(A::LinearAlgebra.Transpose{<:Any, <:AbstractMatrix})
    return transpose(abs_preserving_structure(LinearAlgebra.parent(A)))
end

function abs_preserving_structure(A::LinearAlgebra.Diagonal)
    return LinearAlgebra.Diagonal(abs.(LinearAlgebra.diag(A)))
end

abs_preserving_structure(A::AbstractMatrix) = abs.(A)
