# TODO: should it be mutable?

mutable struct TransformedList{T,N} <: AbstractArray{T,N}
    items::AbstractArray{T,N}
    transforms
end

function Base.getindex(X::TransformedList, i)
    item = X.items[i]
    transformed = item .|> X.transforms
    if X.transforms isa AbstractArray
        # Multiple transforms, return a tuple with
        # the result of each transform
        transformed = tuple(transformed...)
    end
    return transformed
end

function Base.setindex!(X::TransformedList, v, i)
    X.items[i] = v
end
Base.firstindex(X::TransformedList) = 1
Base.lastindex(X::TransformedList) = length(X.items)
Base.size(X::TransformedList) = Base.size(X.items)


function Base.show(io::IO, ::MIME"text/plain", v::TransformedList)
    println(io, "Instance of TransformedList:")
    print(io, "\titems= ")
    show(io, v.items)
    println(io, "\n\ttransforms=$(v.transforms)")
end