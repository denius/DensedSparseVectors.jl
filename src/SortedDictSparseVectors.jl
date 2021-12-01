
#module SortedDictSparseVectors
#export SortedDictSparseVector

using DocStringExtensions
using DataStructures
using SparseArrays
using Random

import Base: ForwardOrdering, Forward
const FOrd = ForwardOrdering


abstract type AbstractSortedDictSparseVector{Tv,Ti} <: AbstractSparseVector{Tv,Ti} end

"""
The `SortedSetSparseIndex` is for fast indices creating and saving.

It is the same as the `SortedDictSparseVector` but without data storing.

$(TYPEDEF)
Struct fields:
$(TYPEDFIELDS)
"""
struct SortedSetSparseIndex{Ti<:Integer} <: AbstractSortedDictSparseVector{Bool,Ti}
    "Length of sparse vector"
    n::Int
    "Stored indices"
    data::SortedSet{Ti,FOrd}
    SortedSetSparseIndex{Ti}(n::Integer) where Ti = new{Ti}(Int(n), SortedSet{Ti,FOrd}(Forward))
end

"""
The `SortedDictSparseVector` is the simple and trivial realization of `SparseVector` based on `SortedDict`.

The main purpose is fast creating big sparse vectors in random order.

$(TYPEDEF)
Struct fields:
$(TYPEDFIELDS)
"""
struct SortedDictSparseVector{Tv,Ti<:Integer} <: AbstractSortedDictSparseVector{Tv,Ti}
    "Length of sparse vector"
    n::Int
    "Stored indices and respective values"
    data::SortedDict{Ti,Tv,FOrd}
    SortedDictSparseVector{Tv,Ti}(n::Integer) where {Tv,Ti} = new{Tv,Ti}(Int(n), SortedDict{Ti,Tv,FOrd}(Forward))
end

SortedSetSparseIndex(n::Integer) = SortedSetSparseIndex{Int}(n)
SortedDictSparseVector(n::Integer) = SortedDictSparseVector{Float64,Int}(n)

Base.length(v::AbstractSortedDictSparseVector) = v.n
SparseArrays.nnz(v::AbstractSortedDictSparseVector) = length(v.data)
Base.isempty(v::AbstractSortedDictSparseVector) = nnz(v) == 0
Base.size(v::AbstractSortedDictSparseVector) = (v.n,)
Base.axes(v::AbstractSortedDictSparseVector) = (Base.OneTo(v.n),)
Base.ndims(::AbstractSortedDictSparseVector) = 1
Base.ndims(::Type{AbstractSortedDictSparseVector}) = 1
Base.strides(v::AbstractSortedDictSparseVector) = (1,)
Base.eltype(v::AbstractSortedDictSparseVector{Tv,Ti}) where {Tv,Ti} = Tv
Base.IndexStyle(::AbstractSortedDictSparseVector) = IndexLinear()
Base.deepcopy(v::T) where {T<:AbstractSortedDictSparseVector{Tv,Ti}} where {Tv,Ti} = T{Tv,Ti}(v.n, deepcopy(v.data))


@inline Base.isstored(v::AbstractSortedDictSparseVector, i::Integer) = haskey(v.data, i)
@inline Base.haskey(v::AbstractSortedDictSparseVector, i::Integer) = haskey(v.data, i)
@inline Base.getindex(v::SortedSetSparseIndex, i::Integer) = haskey(v.data, i)
@inline Base.getindex(v::AbstractSortedDictSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti} = get(v.data, i, zero(Tv))
@inline Base.push!(v::SortedSetSparseIndex, i::Integer) = (push!(v.data, i); v)
@inline Base.setindex!(v::SortedSetSparseIndex, value, i::Integer) = (push!(v.data, i); v)
@inline Base.setindex!(v::AbstractSortedDictSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti} = (setindex!(v.data, Tv(value), i); v)
@inline Base.delete!(v::AbstractSortedDictSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti} = delete!(v.data, i)


Base.similar(v::T) where {T<:AbstractSortedDictSparseVector{Tv,Ti}} where {Tv,Ti} = deepcopy(v)
Base.similar(v::SortedSetSparseIndex{Ti}, ::Type{ElType}) where {Ti,ElType} = deepcopy(v)
function Base.similar(v::SortedDictSparseVector{Tv,Ti}, ::Type{ElType}) where {Tv,Ti,ElType}
    data = SortedDict{Ti,ElType,FOrd}(Forward)
    for (k,d) in v.data
        data[k] = ElType(d)
    end
    return SortedDictSparseVector{ElType,Ti}(v.n, data)
end


SparseArrays.nonzeroinds(v::SortedSetSparseIndex) = [i for i in v.data]
SparseArrays.nonzeros(v::SortedSetSparseIndex) = [true for i in v.data]
SparseArrays.nonzeroinds(v::AbstractSortedDictSparseVector{Tv,Ti}) where {Tv,Ti} = collect(keys(v.data))
SparseArrays.nonzeros(v::AbstractSortedDictSparseVector{Tv,Ti}) where {Tv,Ti} = collect(values(v.data))

SparseArrays.findnz(v::AbstractSortedDictSparseVector) = (SparseArrays.nonzeroinds(v), SparseArrays.nonzeros(v))



#
#  Testing functions
#

function testfun_create(T::Type, n = 500_000, density = 0.9)
    dsv = T(n)
    Random.seed!(1234)
    for i in shuffle(randsubseq(1:n, density))
        dsv[i] = rand()
    end
    dsv
end

function testfun_create_seq(T::Type, n = 500_000, density = 0.9)
    dsv = T(n)
    Random.seed!(1234)
    for i in randsubseq(1:n, density)
        dsv[i] = rand()
    end
    dsv
end

function testfun_create_dense(T::Type, n = 500_000, nchunks = 800, density = 0.95)
    dsv = T(n)
    chunklen = max(1, floor(Int, n / nchunks))
    Random.seed!(1234)
    for i = 0:nchunks-1
        len = floor(Int, chunklen*density + randn() * chunklen * min(0.1, (1.0-density), density))
        len = max(1, min(chunklen-2, len))
        for j = 1:len
            dsv[i*chunklen + j] = rand()
        end
    end
    dsv
end


function testfun_delete!(dsv)
    Random.seed!(1234)
    indices = shuffle(SparseArrays.nonzeroinds(dsv))
    for i in indices
        delete!(dsv, i)
    end
    dsv
end


function testfun_getindex(sv)
    S = 0.0
    for i = 1:length(sv)
        S += sv[i]
    end
    (0, S)
end


#end  # of module SortedDictSparseVectors
