
# TODO:
# * Introduce two macros: `@inzeros` and `@zeroscheck` like `@inbounds` and `@boundscheck`
#   to ommit sparse similarity checking and fixing. Then BZP is not need.
#   Or may be do `@inbounds` do this?
#
# * Introduce offsets fields to all types to have indexable iterator
#   nonzeros(::AbstractAllDensedSparseVector): getindex(it::NZValues, i) and
#   fast `_are_same_sparse_indices()`
#
# * Introduce ChainIndex: ((chain_index, element_index), LinearIndex) like CartesianIndex
#   for fast AbstractAllDensedSparseVector access without searchsortedlast and so on.
#
# * Test https://github.com/JuliaSIMD/StrideArrays.jl instead of StaticArrays.
#
# * May be all iterators should returns `view(chunk, :)`?
#
# * Introducing ArrayInterface.jl allows automatic broadcast by FastBroadcast.jl. Isn't it?
#   Try to implement `MatrixIndex` from ArrayInterface.jl -- is it unusefull?
#
#
#
# Notes:
#
# * `map[!]` for SparseVector work different from DenseVector:
#   1-length Vector and scalars are allowed, but 1-length SparseVector is not allowed.
#
# 
# * Broadcast is more fragile than `map`. `map` should be more agile in terms of vectors dimensions and
#   other differences in argumens. Broadcast allows scalars and zero-dimensions/one-length arrays as arguments.
#   In broadcast axes should be same, sparse indices are not should be same! There should be two branches
#   for same indices and not.
#   The `map` try to calculate anyway: in complex cases map will `convert()` to Vector and apply function.
#   ```julia
#   julia> map(+, (2,3,4), [1,2])
#   2-element Vector{Int64}:
#    3
#    5
#   ```
#   In both cases the `firstindex(v)` should be same. In broadcast length should be same.
#
#   From official help:
#     `map`: Transform collection c by applying f to each element. For multiple
#     collection arguments, apply f elementwise, and stop when when any of them is
#     exhausted.
#     When acting on multi-dimensional arrays of the same ndims, they must all
#     have the same axes, and the answer will too.
#     See also broadcast, which allows mismatched sizes.
#
#     Broadcast the function f over the arrays, tuples, collections, Refs and/or
#     scalars As.
#     Broadcasting applies the function f over the elements of the container
#     arguments and the scalars themselves in As. Singleton and missing dimensions
#     are expanded to match the extents of the other arguments by virtually
#     repeating the value.


#
# * Shape is an like (2,3) in tuple in `reshape(V, (2,3))` and same operations.
#


module DensedSparseVectors

export AbstractAllDensedSparseVector
export DensedSparseVector, FixedDensedSparseVector, DynamicDensedSparseVector
export DensedSVSparseVector, DensedVLSparseVector
export nzpairs, nzpairsview, nzvalues, nzvaluesview, nzindices, nzchunks, nzchunkspairs
export startindex
export findfirstnz, findlastnz, findfirstnzindex, findlastnzindex
export iteratenzpairs, iteratenzpairsview, iteratenzvalues, iteratenzvaluesview, iteratenzindices
export is_broadcast_zero_preserve
export get_iterable, iterateempty


import Base: ForwardOrdering, Forward
const FOrd = ForwardOrdering

import Base.Broadcast: BroadcastStyle
using Base.Broadcast: AbstractArrayStyle, Broadcasted, DefaultArrayStyle
using DocStringExtensions
using DataStructures
#using FillArrays
using IterTools
using OffsetArrays
using Setfield
using SparseArrays
using StaticArrays
import SparseArrays: indtype, nonzeroinds, nonzeros
using Random


## TODO: use Espresso.jl package
#
## via base/Base.jl:25
#macro inzeros()   Expr(:meta, :inzeros)   end
#
### via essentials.jl:676
##macro inzeros(blk)
##    return Expr(:block,
##        Expr(:inzeros, true),
##        Expr(:local, Expr(:(=), :val, esc(blk))),
##        Expr(:inzeros, :pop),
##        :val)
##end
#
## via essentials.jl:644
#macro zeroscheck(blk)
#    return Expr(:if, Expr(:inzeros), esc(blk))
#    #return Expr(:if, Expr(:zeroscheck), esc(blk))
#end

# https://github.com/JuliaLang/julia/issues/39952
basetype(::Type{T}) where T = Base.typename(T).wrapper

abstract type AbstractAllDensedSparseVector{Tv,Ti,BZP} <: AbstractSparseVector{Tv,Ti} end

"Vector alike DensedSparseVector kind"
abstract type AbstractDensedSparseVector{Tv,Ti,BZP} <: AbstractAllDensedSparseVector{Tv,Ti,BZP} end
"Matrix alike Vector of Vectors kind"
abstract type AbstractDensedBlockSparseVector{Tv,Ti,BZP} <: AbstractAllDensedSparseVector{Tv,Ti,BZP} end

"Simple VectorDensedSparseVector kind"
abstract type AbstractSimpleDensedSparseVector{Tv,Ti,BZP} <: AbstractDensedSparseVector{Tv,Ti,BZP} end
"Based on SortedDict VectorDensedSparseVector kind"
abstract type AbstractSDictDensedSparseVector{Tv,Ti,BZP} <: AbstractDensedSparseVector{Tv,Ti,BZP} end


"All Vector alike types `<: AbstractAllDensedSparseVector`"
const AbstractVecbasedDensedSparseVector{Tv,Ti,BZP} = Union{AbstractSimpleDensedSparseVector{Tv,Ti,BZP}, AbstractDensedBlockSparseVector{Tv,Ti,BZP}}


"""
The `DensedSparseVector` is alike the `Vector` but have the omits in stored indices/data.
It is the subtype of `AbstractSparseVector`. The speed of `Broadcasting` on `DensedSparseVector`
is almost the same as on the `Vector`, but the speed by direct index access is almost few times
slower then the for `Vector`'s one.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSparseVector{Tv,Ti,BZP} <: AbstractSimpleDensedSparseVector{Tv,Ti,BZP}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s"
    nzchunks::Vector{Vector{Tv}}
    "Vector length"
    n::Ti
    #"Vector range, `firstindex(V) = first(V.axes1)` and so on"
    # see https://github.com/JuliaArrays/CustomUnitRanges.jl
    #axes1::UnitRange{Ti}
    "Number of stored non-zero elements"
    nnz::Int

    DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = DensedSparseVector{Tv,Ti,Val{false}}(n)
    DensedSparseVector{Tv,Ti,BZP}(n::Integer = 0) where {Tv,Ti,BZP} = new{Tv,Ti,BZP}(0, Vector{Ti}(), Vector{Vector{Tv}}(), n, 0)

    DensedSparseVector{Tv,Ti}(n::Integer, nzind, nzchunks) where {Tv,Ti} =
        DensedSparseVector{Tv,Ti,Val{false}}(n, nzind, nzchunks)
    DensedSparseVector{Tv,Ti,BZP}(n::Integer, nzind, nzchunks) where {Tv,Ti,BZP} =
        new{Tv,Ti,BZP}(0, nzind, nzchunks, n, foldl((s,c)->(s+length(c)), nzchunks; init=0))

end

#DensedSparseVector(n::Integer = 0) = DensedSparseVector{Float64,Int,Val{false}}(n)
#DensedSparseVector{Tv,Ti}(V) where {Tv,Ti} = DensedSparseVector{Tv,Ti,Val{false}}(V)

DensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}) where {Tv,Ti,BZP} = DensedSparseVector{Tv,Ti,BZP}(V)

function DensedSparseVector{Tv,Ti,BZP}(V::AbstractAllDensedSparseVector) where {Tv,Ti,BZP}
    nzind = Vector{Ti}(undef, nnzchunks(V))
    nzchunks = Vector{Vector{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = k
        nzchunks[i] = Vector{Tv}(d)
    end
    return DensedSparseVector{Tv,Ti,BZP}(length(V), nzind, nzchunks)
end

#"View for DensedSparseVector"
#struct DensedSparseVectorView{Tv,Ti,T,Tc} <: AbstractVecbasedDensedSparseVector{Tv,Ti,BZP}
#    "Index of first chunk in `view` V"
#    firstnzchunk_index::Int
#    "Index of last chunk in `view` V"
#    lastnzchunk_index::Int
#    "View on DensedSparseVector"
#    V::Tc
#end



"""
The `FixedDensedSparseVector` is the `DensedSparseVector` without the ability to change shape after creating.
It should be slightly faster then `DensedSparseVector` because data locality and omitted one reference lookup.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct FixedDensedSparseVector{Tv,Ti,BZP} <: AbstractSimpleDensedSparseVector{Tv,Ti,BZP}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s"
    nzchunks::Vector{Tv}
    "Offsets of starts of vestors in `nzchunks` like in CSC matrix structure"
    offsets::Vector{Int}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    FixedDensedSparseVector{Tv,Ti,BZP}(n::Integer, nzind, nzchunks, offsets) where {Tv,Ti,BZP} =
        new{Tv,Ti,BZP}(0, nzind, nzchunks, offsets, n, length(nzchunks))
end


FixedDensedSparseVector{Tv,Ti}(V) where {Tv,Ti} = FixedDensedSparseVector{Tv,Ti,Val{false}}(V)
FixedDensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}) where {Tv,Ti,BZP} = FixedDensedSparseVector{Tv,Ti,BZP}(V)

function FixedDensedSparseVector{Tv,Ti,BZP}(V::AbstractAllDensedSparseVector) where {Tv,Ti,BZP}
    nzind = Vector{Ti}(undef, nnzchunks(V))
    nzchunks = Vector{Tv}(undef, nnz(V))
    offsets = Vector{Int}(undef, nnzchunks(V)+1)
    offsets[1] = 1
    for (i, (k,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = k
        offsets[i+1] = offsets[i] + length(d)
        @view(nzchunks[offsets[i]:offsets[i+1]-1]) .= Tv.(d)
    end
    return FixedDensedSparseVector{Tv,Ti,BZP}(length(V), nzind, nzchunks, offsets)
end



"""
The `DensedSVSparseVector` is the version of `DensedSparseVector` with `SVector` as elements
and alike `Matrix` with sparse first dimension and with dense `SVector` in second dimension.
See `DensedSparseVector` for details.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSVSparseVector{Tv,Ti,m,BZP} <: AbstractDensedBlockSparseVector{Tv,Ti,BZP}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    nzchunks::Vector{Vector{SVector{m,Tv}}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    DensedSVSparseVector{Tv,Ti,m,BZP}(n::Integer, nzind, nzchunks) where {Tv,Ti,m,BZP} =
        new{Tv,Ti,m,BZP}(0, nzind, nzchunks, n, foldl((s,c)->(s+length(c)), nzchunks; init=0))
    DensedSVSparseVector{Tv,Ti,m,BZP}(n::Integer = 0) where {Tv,Ti,m,BZP} =
        new{Tv,Ti,m,BZP}(0, Vector{Ti}(), Vector{Vector{Tv}}(), n, 0)
end

DensedSVSparseVector{Tv,Ti,m}(V) where {Tv,Ti,m} = DensedSVSparseVector{Tv,Ti,m,Val{false}}(V)
DensedSVSparseVector(m::Integer, n::Integer = 0) = DensedSVSparseVector{Float64,Int,m,Val{false}}(n)



"""
The `DensedVLSparseVector` is the version of `DensedSparseVector` with variable length elements
and alike `Matrix` with sparse first dimension and with variable length dense vectors in second dimension.
See `DensedSparseVector` for details.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedVLSparseVector{Tv,Ti,BZP} <: AbstractDensedBlockSparseVector{Tv,Ti,BZP}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    nzchunks::Vector{Vector{Tv}}
    "Offsets of starts of variable length vestors in `nzchunks`"
    offsets::Vector{Vector{Int}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int
    "Dummy for empty `getindex` returns"
    dummy::Vector{Tv}

    DensedVLSparseVector{Tv,Ti,BZP}(n::Integer = 0) where {Tv,Ti,BZP} =
        new{Tv,Ti,BZP}(0, Vector{Ti}(), Vector{Vector{Tv}}(), Vector{Vector{Int}}(), n, 0, Tv[])
end

DensedVLSparseVector(n::Integer = 0) = DensedVLSparseVector{Float64,Int,Val{false}}(n)
DensedVLSparseVector{Tv,Ti}(V) where {Tv,Ti} = DensedVLSparseVector{Tv,Ti,Val{false}}(V)



"""
The `DynamicDensedSparseVector` is alike the `SparseVector` but should have the almost all indices are consecuitive stored.
The speed of `Broadcasting` on `DynamicDensedSparseVector` is almost the same as
on the `Vector` excluding the cases where the indices are wide broaded and
there is no consecuitive ranges of indices. The speed by direct index access is ten or
more times slower then the for `Vector`'s one. The main purpose of this type is
the construction of the `DynamicDensedSparseVector` vectors with further conversion to `DensedSparseVector`.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DynamicDensedSparseVector{Tv,Ti,BZP} <: AbstractSDictDensedSparseVector{Tv,Ti,BZP}
    "Index of last used chunk"
    lastusedchunkindex::DataStructures.Tokens.IntSemiToken
    "Storage for indices of the first element of non-zero chunks and corresponding chunks as `SortedDict(Int=>Vector)`"
    nzchunks::SortedDict{Ti,Vector{Tv},FOrd}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    DynamicDensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = DynamicDensedSparseVector{Tv,Ti,Val{false}}(n)
    function DynamicDensedSparseVector{Tv,Ti,BZP}(n::Integer = 0) where {Tv,Ti,BZP}
        nzchunks = SortedDict{Ti,Vector{Tv},FOrd}(Forward)
        new{Tv,Ti,BZP}(beforestartsemitoken(nzchunks), nzchunks, n, 0)
    end

    DynamicDensedSparseVector{Tv,Ti}(n::Integer, nzchunks::SortedDict{K,V}) where {Tv,Ti,K,V<:AbstractVector} =
        DynamicDensedSparseVector{Tv,Ti,Val{false}}(n, nzchunks)
    DynamicDensedSparseVector{Tv,Ti,BZP}(n::Integer, nzchunks::SortedDict{K,V}) where {Tv,Ti,BZP,K,V<:AbstractVector} =
        new{Tv,Ti,BZP}(beforestartsemitoken(nzchunks), nzchunks, n, foldl((s,c)->(s+length(c)), values(nzchunks); init=0))

end

#DynamicDensedSparseVector(n::Integer = 0) = DynamicDensedSparseVector{Float64,Int}(n)
#DynamicDensedSparseVector{Tv,Ti}(V) where {Tv,Ti} = DynamicDensedSparseVector{Tv,Ti,Val{false}}(V)

DynamicDensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}) where {Tv,Ti,BZP} = DynamicDensedSparseVector{Tv,Ti,BZP}(V)
function DynamicDensedSparseVector{Tv,Ti,BZP}(V::AbstractAllDensedSparseVector) where {Tv,Ti,BZP}
    nzchunks = SortedDict{Ti, Vector{Tv}, FOrd}(Forward)
    for (k,d) in nzchunkspairs(V)
        nzchunks[k] = Vector{Tv}(d)
    end
    return DynamicDensedSparseVector{Tv,Ti,BZP}(length(V), nzchunks)
end

"""
Convert any particular `AbstractSparseVector`s to corresponding `AbstractAllDensedSparseVector`:

    DensedSparseVector(sv)

"""
function (::Type{T})(V::AbstractSparseVector{Tv,Ti}) where {T<:AbstractAllDensedSparseVector,Tv,Ti}
    sv = T{Tv,Ti,Val{false}}(length(V))
    for (i,d) in zip(nonzeroinds(V), nonzeros(V))
        sv[i] = d
    end
    return sv
end

"""
Convert any `AbstractSparseVector`s to particular `AbstractAllDensedSparseVector`:

    DensedSparseVector{Float64,Int}(sv)

"""
(::Type{T})(V::AbstractSparseVector) where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti} = basetype(T){Tv,Ti,Val{false}}(V)
#(::Type{T})(V::AbstractSparseVector) where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti} = T{Tv,Ti,Val{false}}(V)
function (::Type{T})(V::AbstractSparseVector) where {T<:AbstractAllDensedSparseVector{Tv,Ti,BZP}} where {Tv,Ti,BZP}
    sv = T(length(V))
    for (i,d) in zip(nonzeroinds(V), nonzeros(V))
        sv[i] = d
    end
    return sv
end


(::Type{T})(V::DenseVector{Tv}) where {T<:AbstractAllDensedSparseVector,Tv} = T{Tv,Int,Val{false}}(V)
(::Type{T})(V::DenseVector) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}} = T{Tv,Ti,Val{false}}(V)
function (::Type{T})(V::DenseVector) where {Tv,Ti,BZP,T<:AbstractAllDensedSparseVector{Tv,Ti,BZP}}
    dsv = T(length(V))
    for (i,d) in enumerate(V)
        dsv[i] = d
    end
    return dsv
    #nzind = ones(Ti, 1)
    #nzchunks = Vector{Vector{Tv}}(undef, length(nzind))
    #nzchunks[1] = Vector{Tv}(V)
    #return DensedSparseVector{Tv,Ti,BZP}(length(V), nzind, nzchunks)
end

function raw_index(V, i)
    idxchunk = searchsortedlast_nzchunk(V, i)
    if idxchunk != pastendnzchunk_index(V)
        indices = get_nzchunk_indices(V, idxchunk)
        if checkindex(Bool, indices, i) #key <= i < key + length(chunk)
            return Pair(idxchunk, Int(i - first(indices) + 1))
        end
    end
    throw(BoundsError(V, i))
end

idxcompare(V::AbstractAllDensedSparseVector, i, j) = cmp(i, j)
idxcompare(V::DynamicDensedSparseVector, i, j) = compare(V.nzchunks, i, j)

function rawindexcompare(V, i, j)
    c = idxcompare(V, first(i), first(j))
    if c < 0
        return -1
    elseif c == 0
        return cmp(last(i), last(j))
    else
        return 1
    end
end

advancerawindex(V::AbstractAllDensedSparseVector) =
    nnz(V) > 0 ? Pair(firstnzchunk_index(V), 1) : Pair(pastendnzchunk_index(V), 0)

function advancerawindex(V::AbstractAllDensedSparseVector, i::Pair)
    if last(i) != 0
        indices = get_nzchunk_indices(V, first(i))
        if last(i) < length(indices)
            return Pair(first(i), last(i)+1)
        elseif (st = advance(V, first(i))) != pastendnzchunk_index(V)
            return Pair(st, 1)
        else
            return Pair(pastendnzchunk_index(V), 0)
        end
    else
        return Pair(pastendnzchunk_index(V), 0)
    end
end


Base.to_index(V::AbstractAllDensedSparseVector{Tv,Ti}, idx::Pair) where {Tv,Ti} = Ti(get_nzchunk_key(V, first(idx))) + Ti(last(idx)) - Ti(1)

#is_broadcast_zero_preserve(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}) where {Tv,Ti,BZP} = BZP != Val{false}
is_broadcast_zero_preserve(V::AbstractAllDensedSparseVector) = false
is_broadcast_zero_preserve(V::AbstractAllDensedSparseVector{<:Any,<:Any,<:Val{true}}) = true

Base.length(V::AbstractAllDensedSparseVector) = getfield(V, :n)
Base.@propagate_inbounds SparseArrays.nnz(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector} =
        foldl((s,c)->(s+Int(length(c))), nzchunks(V); init=Int(0))
Base.@propagate_inbounds SparseArrays.nnz(V::OffsetArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector} = nnz(parent(V))
SparseArrays.nnz(V::AbstractAllDensedSparseVector) = getfield(V, :nnz)
Base.isempty(V::AbstractAllDensedSparseVector) = nnz(V) == 0
Base.size(V::AbstractAllDensedSparseVector) = (length(V),)
Base.axes(V::AbstractAllDensedSparseVector) = (Base.OneTo(length(V)),)
Base.ndims(::AbstractAllDensedSparseVector) = 1
Base.ndims(::Type{AbstractAllDensedSparseVector}) = 1
Base.strides(V::AbstractAllDensedSparseVector) = (1,)
Base.eltype(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = Tv
SparseArrays.indtype(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = Ti
Base.IndexStyle(::AbstractAllDensedSparseVector) = IndexLinear()

Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}) where {Tv,Ti,BZP} = similar(V, Tv, Ti, BZP)
Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}, ::Type{TvNew}) where {Tv,Ti,BZP,TvNew} = similar(V, TvNew, Ti, BZP)
Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti,BZP}, ::Type{TvNew}, ::Type{TiNew}) where {Tv,Ti,BZP,TvNew,TiNew} = similar(V, TvNew, TiNew, BZP)

function Base.similar(V::DensedSparseVector, ::Type{TvNew}, ::Type{TiNew}, ::Type{BZP}) where {TvNew,TiNew,BZP}
    nzind = similar(V.nzind, TiNew)
    nzchunks = similar(V.nzchunks)
    for (i, (ids,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = first(ids)
        nzchunks[i] = similar(d, TvNew)
    end
    return DensedSparseVector{TvNew,TiNew,BZP}(length(V), nzind, nzchunks)
end
function Base.similar(V::FixedDensedSparseVector, ::Type{TvNew}, ::Type{TiNew}, ::Type{BZP}) where {TvNew,TiNew,BZP}
    nzind = Vector{TiNew}(V.nzind)
    nzchunks = similar(V.nzchunks, TvNew)
    offsets = Vector{Int}(V.offsets)
    return FixedDensedSparseVector{TvNew,TiNew,BZP}(length(V), nzind, nzchunks, offsets)
end
function Base.similar(V::DynamicDensedSparseVector, ::Type{TvNew}, ::Type{TiNew}, ::Type{BZP}) where {TvNew,TiNew,BZP}
    nzchunks = SortedDict{TiNew, Vector{TvNew}, FOrd}(Forward)
    for (ids,d) in nzchunkspairs(V)
        nzchunks[first(ids)] = similar(d, TvNew)
    end
    return DynamicDensedSparseVector{TvNew,TiNew,BZP}(length(V), nzchunks)
end

function Base.copy(V::T) where {T<:DensedSparseVector}
    nzind = copy(V.nzind)
    nzchunks = copy(V.nzchunks)
    for (i, (ids,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = first(ids)
        nzchunks[i] = copy(d)
    end
    return T(length(V), nzind, nzchunks)
end
Base.copy(V::T) where {T<:FixedDensedSparseVector} = T(length(V), copy(V.nzind), copy(V.nzchunks), copy(V.offsets))
function Base.copy(V::DynamicDensedSparseVector{Tv,Ti,BZP}) where {Tv,Ti,BZP}
    nzchunks = SortedDict{Ti, Vector{Tv}, FOrd}(Forward)
    for (ids,d) in nzchunkspairs(V)
        nzchunks[first(ids)] = copy(d)
    end
    return DynamicDensedSparseVector{Tv,Ti,BZP}(length(V), nzchunks)
end

function Base.collect(::Type{ElType}, V::AbstractAllDensedSparseVector) where ElType
    res = zeros(ElType, length(V))
    for (i,V) in nzpairs(V)
        res[i] = ElType(V)
    end
    return res
end
Base.collect(V::AbstractAllDensedSparseVector) = collect(eltype(V), V)

@inline nnzchunks(V::Vector) = 1
@inline function nnzchunks(V::SparseVector)
    nnz(V) == 0 && return 0
    nzinds = SparseArrays.nonzeroinds(V)
    len = 1
    prev = nzinds[1]
    for i = 2:nnz(V)
        if prev + 1 != (prev = nzinds[i])
            len += 1
        end
    end
    return len
end
@inline nnzchunks(V::AbstractAllDensedSparseVector) = length(getfield(V, :nzchunks))
@inline nnzchunks(V::FixedDensedSparseVector) = length(getfield(V, :nzind))
function nnzchunks(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector}
    length(V) == 0 && return 0
    idx1 = searchsortedlast_nzchunk(parent(V), first(parentindices(V)[1]))
    idx2 = searchsortedlast_nzchunk(parent(V), last(parentindices(V)[1]))
    if idx1 == idx2
        return get_nzchunk_length(V, idx1) == 0 ? 0 : 1
    else
        return idx2 - idx1 - 1 + (get_nzchunk_length(V, idx1) > 0) + (get_nzchunk_length(V, idx2) > 0)
    end
end
@inline nnzchunks(V::OffsetArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector} = nnzchunks(parent(V))
@inline length_of_that_nzchunk(V::AbstractVecbasedDensedSparseVector, chunk) = length(chunk) # TODO: Is it need?
@inline length_of_that_nzchunk(V::DynamicDensedSparseVector, chunk) = length(chunk)
@inline get_nzchunk_length(V::AbstractVecbasedDensedSparseVector, i) = size(V.nzchunks[i])[1]
@inline get_nzchunk_length(V::FixedDensedSparseVector, i) = V.offsets[i+1] - V.offsets[i]
@inline get_nzchunk_length(V::DensedVLSparseVector, i) = size(V.offsets[i])[1] - 1
@inline get_nzchunk_length(V::DynamicDensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = size(deref_value((V.nzchunks, i)))[1]
@inline get_nzchunk_length(V::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractAllDensedSparseVector} = length(get_nzchunk(V, i))
@inline get_nzchunk(V::Number, i) = Ref(V)
@inline get_nzchunk(V::Vector, i) = view(V, :)
@inline get_nzchunk(V::SparseVector, i) = view(nonzeros(V), i[1]:i[1]+i[2]-1)
@inline get_nzchunk(V::AbstractVecbasedDensedSparseVector, i) = view(V.nzchunks[i], :)
@inline get_nzchunk(V::FixedDensedSparseVector, i) = @view( V.nzchunks[ V.offsets[i]:V.offsets[i+1] - 1 ] )
@inline get_nzchunk(V::DynamicDensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = view(deref_value((V.nzchunks, i)), :)
###@inline function get_nzchunk(V::SubArray{<:Any,<:Any,<:T}, i) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
###    idx1 = first(parentindices(V)[1])
###    idx2 = last(parentindices(V)[1])
###    key, chunk = get_key_and_nzchunk(parent(V), i)
###    len = Ti(length(chunk))
###    if key <= idx1 < key+len && key <= idx2 < key+len
###        return view(chunk, idx1-key+Ti(1):idx2-key+Ti(1))
###    elseif key <= idx1 < key+len
###        return @view(chunk[idx1-key+Ti(1):end])
###    elseif key <= idx2 < key+len
###        return view(chunk, Ti(1):(idx2-key+Ti(1)))
###    elseif (key > idx1 && key > idx2) || (idx1 >= key+len && idx2 >= key+len)
###        return @view(chunk[end:Ti(0)])
###    else
###        return @view(chunk[Ti(1):end])
###    end
###end
@inline function get_nzchunk(V::SubArray{<:Any,<:Any,<:T}, i) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    idx1 = first(parentindices(V)[1])
    idx2 = last(parentindices(V)[1])
    indices, chunk = get_indices_and_nzchunk(parent(V), i)
    index1 = first(indices)
    index1 = last(indices)
    if checkindex(Bool, indices, idx1) && checkindex(Bool, indices, idx2)
        return view(chunk, idx1-index1+Ti(1):idx2-index1+Ti(1))
    elseif checkindex(Bool, indices, idx1)
        return @view(chunk[idx1-index1+Ti(1):end])
    elseif checkindex(Bool, indices, idx2)
        return view(chunk, Ti(1):(idx2-index1+Ti(1)))
    elseif (idx1 < index1 && idx2 < index1) || (idx1 > index2 && idx2 > index2)
        return @view(chunk[end:Ti(0)])
    else
        return @view(chunk[Ti(1):end])
    end
end
@inline get_nzchunk_key(V::Vector, i) = i
@inline get_nzchunk_key(V::SparseVector, i) = V.nzind[i]
@inline get_nzchunk_key(V::AbstractVecbasedDensedSparseVector, i) = V.nzind[i]
@inline get_nzchunk_key(V::DynamicDensedSparseVector, i) = deref_key((V.nzchunks, i))
@inline function get_nzchunk_key(V::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractAllDensedSparseVector}
    indices = get_nzchunk_indices(parent(V), i)
    if checkindex(Bool, indices, first(parentindices(V)[1]))
        return first(parentindices(V)[1])
    else
        return key
    end
end

@inline get_nzchunk_indices(V::Vector, i) = UnitRange{Int}(1, length(V))
@inline get_nzchunk_indices(V::SparseVector, i) = V.nzind[i], V.nzind[i] # FIXME:
@inline get_nzchunk_indices(V::AbstractVecbasedDensedSparseVector{Tv,Ti}, i) where {Tv,Ti} =
    UnitRange{Ti}(V.nzind[i], V.nzind[i]+length(V.nzchunks[i])-1)
@inline get_nzchunk_indices(V::FixedDensedSparseVector{Tv,Ti}, i) where {Tv,Ti} =
    UnitRange{Ti}(V.nzind[i], V.nzind[i]+(V.offsets[i+1]-V.offsets[i])-1)
@inline get_nzchunk_indices(V::DynamicDensedSparseVector{Tv,Ti}, i) where {Tv,Ti} =
    ((key, chunk) = deref((V.nzchunks, i));
     return UnitRange{Ti}(key, key+length(chunk)-1))
@inline function get_nzchunk_indices(V::SubArray{<:Any,<:Any,<:T}, i) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    idx1 = first(parentindices(V)[1])
    idx2 = last(parentindices(V)[1])
    indices = get_nzchunk_indices(parent(V), i)
    index1 = first(indices)
    index1 = last(indices)
    if checkindex(Bool, indices, idx1) && checkindex(Bool, indices, idx2)
        return UnitRange{Ti}(idx1, idx2)
    elseif checkindex(Bool, indices, idx1)
        return UnitRange{Ti}(index1, idx2)
    elseif checkindex(Bool, indices, idx2)
        return UnitRange{Ti}(idx1, index2)
    elseif idx1 < index1 && idx2 < index1
        return UnitRange{Ti}(index1, index1-Ti(1))
    elseif idx1 > index2 && idx2 > index2
        return UnitRange{Ti}(index2, index2-Ti(1))
    else
        return UnitRange{Ti}(index1, index2)
    end
end
@inline get_key_and_nzchunk(V::Vector, i) = (i, view(V, :))
@inline get_key_and_nzchunk(V::SparseVector, i) = (V.nzind[i], view(V.nzchunks, i:i)) # FIXME:
@inline get_key_and_nzchunk(V::AbstractVecbasedDensedSparseVector, i) = (V.nzind[i], view(V.nzchunks[i], :))
@inline get_key_and_nzchunk(V::FixedDensedSparseVector, i) = (V.nzind[i], @view(V.nzchunks[V.offsets[i]:V.offsets[i+1]-1]))
@inline get_key_and_nzchunk(V::DynamicDensedSparseVector, i) =
    ((key, chunk) = deref((V.nzchunks, i));
     return (key, view(chunk, :)))

@inline get_key_and_nzchunk(V::Vector) = (1, view(eltype(V)[], 1:0))
@inline get_key_and_nzchunk(V::SparseVector{Tv,Ti}) where {Tv,Ti} = (Ti(1), view(Tv[], 1:0))
@inline get_key_and_nzchunk(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = (Ti(1), view(Tv[], 1:0))

@inline get_indices_and_nzchunk(V::Vector, i) = (i, view(V, :))
@inline get_indices_and_nzchunk(V::SparseVector, i) = (V.nzind[i], view(V.nzchunks, i:i)) # FIXME:
@inline get_indices_and_nzchunk(V::AbstractVecbasedDensedSparseVector{Tv,Ti}, i) where {Tv,Ti} =
    (UnitRange{Ti}(V.nzind[i], V.nzind[i]+length(V.nzchunks[i])-1), view(V.nzchunks[i], :))
@inline get_indices_and_nzchunk(V::FixedDensedSparseVector{Tv,Ti}, i) where {Tv,Ti} =
    (UnitRange{Ti}(V.nzind[i], V.nzind[i]+(V.offsets[i+1]-V.offsets[i])-1), @view(V.nzchunks[V.offsets[i]:V.offsets[i+1]-1]))
@inline get_indices_and_nzchunk(V::DynamicDensedSparseVector{Tv,Ti}, i) where {Tv,Ti} =
    ((key, chunk) = deref((V.nzchunks, i));
     return (UnitRange{Ti}(key, key+length(chunk)-1), view(chunk, :)))

@inline get_indices_and_nzchunk(V::Vector) = (UnitRange(1,0), view(eltype(V)[], 1:0))
@inline get_indices_and_nzchunk(V::SparseVector{Tv,Ti}) where {Tv,Ti} = (UnitRange{Ti}(1,0), view(Tv[], 1:0))
@inline get_indices_and_nzchunk(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    (UnitRange{Ti}(1,0), view(Tv[], 1:0))

@inline get_key_and_nzchunk_and_length(V::Vector, i) = (i, view(V, :), length(V))
@inline get_key_and_nzchunk_and_length(V::SparseVector, i) = (V.nzind[i], view(V.nzchunks, i:i), 1)
@inline get_key_and_nzchunk_and_length(V::AbstractVecbasedDensedSparseVector, i) = (V.nzind[i], V.nzchunks[i], length(V.nzchunks[i]))
@inline get_key_and_nzchunk_and_length(V::FixedDensedSparseVector, i) =
        (V.nzind[i], @view(V.nzchunks[V.offsets[i]:V.offsets[i+1]-1]), V.offsets[i+1]-V.offsets[i])
        @inline get_key_and_nzchunk_and_length(V::DynamicDensedSparseVector, i) = ((key, chunk) = deref((V.nzchunks, i)); return (key, view(chunk, :), length(chunk)))

@inline is_in_nzchunk(V::Vector, i, key) = key in first(axes(V))
@inline is_in_nzchunk(V::SparseVector, i, key) = V.nzind[i] == key
@inline is_in_nzchunk(V::AbstractVecbasedDensedSparseVector, i, key) = V.nzind[i] <= key < V.nzind[i] + length(V.nzchunks[i])
@inline is_in_nzchunk(V::FixedDensedSparseVector, i) = V.nzind[i] <= key < V.nzind[i] + V.offsets[i+1]-V.offsets[i]
@inline is_in_nzchunk(V::DynamicDensedSparseVector, i, key) = ((ichunk, chunk) = deref((V.nzchunks, i)); return (ichunk <= key < ichunk + length(chunk)))

@inline firstnzchunk_index(V::AbstractVecbasedDensedSparseVector) = firstindex(V.nzind)
@inline firstnzchunk_index(V::AbstractSDictDensedSparseVector) = startof(V.nzchunks)
@inline lastnzchunk_index(V::AbstractVecbasedDensedSparseVector) = lastindex(V.nzind) # getfield(V, :n)
@inline lastnzchunk_index(V::AbstractSDictDensedSparseVector) = lastindex(V.nzchunks)

@inline beforestartnzchunk_index(V::AbstractVecbasedDensedSparseVector) = firstnzchunk_index(V) - 1
@inline beforestartnzchunk_index(V::AbstractSDictDensedSparseVector) = beforestartsemitoken(V.nzchunks)
@inline pastendnzchunk_index(V::AbstractVecbasedDensedSparseVector) = lastnzchunk_index(V) + 1
@inline pastendnzchunk_index(V::AbstractSDictDensedSparseVector) = pastendsemitoken(V.nzchunks)

@inline returnzero(V::DensedSVSparseVector) = zero(eltype(eltype(V.nzchunks)))
@inline returnzero(V::AbstractAllDensedSparseVector) = zero(eltype(V))

@inline DataStructures.advance(V::AbstractVecbasedDensedSparseVector, state) = state + 1
@inline DataStructures.advance(V::AbstractSDictDensedSparseVector, state) = advance((V.nzchunks, state))
@inline DataStructures.regress(V::AbstractVecbasedDensedSparseVector, state) = state - 1
@inline DataStructures.regress(V::AbstractSDictDensedSparseVector, state) = regress((V.nzchunks, state))

"`searchsortedlast(V.nzind)`"
@inline searchsortedlast_nzind(V::AbstractVecbasedDensedSparseVector, i) = searchsortedlast(V.nzind, i)
@inline searchsortedlast_nzind(V::AbstractSDictDensedSparseVector, i) = searchsortedlast(V.nzchunks, i)

"""
Returns nzchunk_index which on vector index `i`, or after `i`.
Slightly differs from `searchsortedfirst(V.nzind)`.
"""
@inline function searchsortedlast_nzchunk(V::AbstractAllDensedSparseVector, i::Integer)
    if i == 1 # most of use cases
        return nnz(V) == 0 ? pastendnzchunk_index(V) : firstnzchunk_index(V)
    elseif nnz(V) != 0
        st = searchsortedlast_nzind(V, i)
        if st != beforestartnzchunk_index(V)
            if is_in_nzchunk(V, st, i)
                return st
            else
                return advance(V, st)
            end
        else
            return firstnzchunk_index(V)
        end
    else
        return beforestartnzchunk_index(V)
    end
end

"""
Returns nzchunk_index which on vector index `i`, or before `i`.
Slightly differs from `searchsortedlast(V.nzind)`.
"""
@inline function searchsortedfirst_nzchunk(V::AbstractAllDensedSparseVector, i::Integer)
    return searchsortedlast_nzind(V, i)
    #=
    if nnz(V) != 0
        return searchsortedlast_nzind(V, i)
    else
        return beforestartnzchunk_index(V)
    end
    =#
end

@inline SparseArrays.sparse(V::AbstractAllDensedSparseVector) =
    SparseVector(length(V), nonzeroinds(V), nonzeros(V))

function SparseArrays.nonzeroinds(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    ret = Vector{Ti}()
    for (ids,_) in nzchunkspairs(V)
        append!(ret, ids)
    end
    return ret
end
function SparseArrays.nonzeros(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    ret = Vector{Tv}()
    for d in nzchunks(V)
        append!(ret, collect(d))
    end
    return ret
end
function SparseArrays.nonzeroinds(V::DensedVLSparseVector{Tv,Ti}) where {Tv,Ti}
    ret = Vector{Ti}()
    for (k,d) in zip(V.nzind, V.offsets)
        append!(ret, (k:k+length(d)-1-1))
    end
    return ret
end
#SparseArrays.findnz(V::AbstractAllDensedSparseVector) = (nzindices(V), nzvalues(V))
SparseArrays.findnz(V::AbstractAllDensedSparseVector) = (nonzeroinds(V), nonzeros(V))



"Returns the index of first non-zero element in sparse vector."
@inline findfirstnzindex(V::SparseVector) = nnz(V) > 0 ? V.nzind[1] : nothing
@inline findfirstnzindex(V::AbstractVecbasedDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    nnz(V) > 0 ? Ti(V.nzind[1]) : nothing
@inline findfirstnzindex(V::AbstractSDictDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    nnz(V) > 0 ? Ti(deref_key((V.nzchunks, startof(V.nzchunks)))) : nothing
function findfirstnzindex(V::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(parent(V)) == 0 && return nothing
    ifirst, ilast = first(parentindices(V)[1]), last(parentindices(V)[1])
    st = searchsortedlast_nzchunk(parent(V), ifirst)
    st == pastendnzchunk_index(parent(V)) && return nothing
    key = get_nzchunk_key(parent(V), st)
    len = get_nzchunk_length(parent(V), st)
    if key <= ifirst < key + len  # ifirst index within nzchunk range
        return Ti(1)
    elseif ifirst <= key <= ilast  # nzchunk[1] somewhere in ifirst:ilast range
        return Ti(key-ifirst+1)
    else
        return nothing
    end
end

"Returns the index of last non-zero element in sparse vector."
@inline findlastnzindex(V::SparseVector) = nnz(V) > 0 ? V.nzind[end] : nothing
@inline findlastnzindex(V::AbstractVecbasedDensedSparseVector) =
    nnz(V) > 0 ? V.nzind[end] + length(V.nzchunks[end]) - 1 : nothing
@inline function findlastnzindex(V::AbstractSDictDensedSparseVector)
    if nnz(V) > 0
        lasttoken = lastindex(V.nzchunks)
        return deref_key((V.nzchunks, lasttoken)) + length(deref_value((V.nzchunks, lasttoken))) - 1
    else
        return nothing
    end
end
function findlastnzindex(V::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(parent(V)) == 0 && return nothing
    ifirst, ilast = first(parentindices(V)[1]), last(parentindices(V)[1])
    st = searchsortedfirst_nzchunk(parent(V), ilast)
    st == beforestartnzchunk_index(parent(V)) && return nothing
    key = get_nzchunk_key(parent(V), st)
    len = get_nzchunk_length(parent(V), st)
    if key <= ilast < key + len  # ilast index within nzchunk range
        return Ti(ilast - ifirst + 1)
    elseif ifirst <= key+len-1 <= ilast  # nzchunk[end] somewhere in ifirst:ilast range
        return Ti(key+len-1 - ifirst+1)
    else
        return nothing
    end
end

"Returns value of first non-zero element in the sparse vector."
@inline findfirstnz(V::AbstractSparseVector) = nnz(V) > 0 ? V[findfirstnzindex(V)] : nothing
function findfirstnz(V::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(parent(V)) == 0 && return nothing
    ifirst, ilast = first(parentindices(V)[1]), last(parentindices(V)[1])
    st = searchsortedlast_nzchunk(parent(V), ifirst)
    st == pastendnzchunk_index(parent(V)) && return nothing
    key, chunk, len = get_key_and_nzchunk_and_length(parent(V), st)
    if key <= ifirst < key + len  # ifirst index within nzchunk range
        return chunk[ifirst-key+1]
    elseif ifirst <= key <= ilast  # nzchunk[1] somewhere in ifirst:ilast range
        return chunk[1]
    else
        return nothing
    end
end

"Returns value of last non-zero element in the sparse vector."
@inline findlastnz(V::AbstractSparseVector) = nnz(V) > 0 ? V[findlastnzindex(V)] : nothing
function findlastnz(V::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(parent(V)) == 0 && return nothing
    ifirst, ilast = first(parentindices(V)[1]), last(parentindices(V)[1])
    st = searchsortedfirst_nzchunk(parent(V), ilast)
    st == beforestartnzchunk_index(parent(V)) && return nothing
    key, chunk, len = get_key_and_nzchunk_and_length(parent(V), st)
    if key <= ilast < key + len  # ilast index within nzchunk range
        return chunk[ilast-key+1]
    elseif ifirst <= key+len-1 <= ilast  # nzchunk[end] somewhere in ifirst:ilast range
        return chunk[end]
    else
        return nothing
    end
end


@inline function Base.findfirst(testf::Function, V::AbstractAllDensedSparseVector)
    for p in nzpairs(V)
        testf(last(p)) && return first(p)
    end
    return nothing
end

@inline Base.findall(testf::Function, V::AbstractAllDensedSparseVector) = collect(first(p) for p in nzpairs(V) if testf(last(p)))
# from SparseArrays/src/sparsevector.jl:830
@inline Base.findall(p::Base.Fix2{typeof(in)}, x::AbstractAllDensedSparseVector) =
    invoke(findall, Tuple{Base.Fix2{typeof(in)}, AbstractArray}, p, x)




# FIXME: Type piracy!!!
Base.@propagate_inbounds SparseArrays.nnz(V::DenseArray) = length(V)

Base.@propagate_inbounds function iteratenzchunks(V::AbstractVecbasedDensedSparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (state, state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(V::AbstractSDictDensedSparseVector, state = beforestartsemitoken(V.nzchunks))
    state = advance((V.nzchunks, state))
    if state != pastendsemitoken(V.nzchunks)
        return (state, state)
    else
        return nothing
    end
end

"`iteratenzchunkspairs(V::AbstractVector)` iterates over non-zero chunks and returns indices of elements in chunk and chunk"
Base.@propagate_inbounds function iteratenzchunkspairs(V::AbstractVecbasedDensedSparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (Pair(get_indices_and_nzchunk(V, state)...), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunkspairs(V::AbstractSDictDensedSparseVector, state = beforestartsemitoken(V.nzchunks))
    state = advance((V.nzchunks, state))
    if state != pastendsemitoken(V.nzchunks)
        return (Pair(get_indices_and_nzchunk(V, state)...), state)
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzchunkspairs(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector}
    state = length(V) > 0 ? regress(parent(V), searchsortedlast_nzchunk(parent(V), first(parentindices(V)[1]))) :
                            beforestartnzchunk_index(parent(V))
    return iteratenzchunkspairs(V, state)
end
Base.@propagate_inbounds function iteratenzchunkspairs(V::SubArray{<:Any,<:Any,<:T}, state) where {T<:AbstractAllDensedSparseVector}
    state = advance(parent(V), state)
    if state != pastendnzchunk_index(parent(V))
        indices = get_nzchunk_indices(parent(V), state)
        if first(indices) <= last(parentindices(V)[1])
            return (Pair(get_nzchunk_indices(V, state), get_nzchunk(V, state)), state)
        else
            return nothing
        end
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzchunkspairs(V::SparseVector)
    nn = nnz(V)
    return nn == 0 ? nothing : iteratenzchunkspairs(V, (1, 0))
end
Base.@propagate_inbounds function iteratenzchunkspairs(V::SparseVector, state)
    nzinds = SparseArrays.nonzeroinds(V)
    N = length(nzinds)
    i1, len = state
    i1 += len
    if i1 <= N
        len = N + 1 - i1
        val = nzinds[i1]
        @inbounds for i = i1+1:N
            if nzinds[i] - 1 > val
                len = i - i1
                break
            end
            val = nzinds[i]
        end
        return (Pair(UnitRange(nzinds[i1], nzinds[i1]+len-1), @view(SparseArrays.nonzeros(V)[i1:i1+len-1])), (i1, len))
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzchunkspairs(V::Vector, state = 0)
    state += 1
    if length(V) == 1
        return (Pair(1:1, V), state)
    elseif state <= 1
        return ((1:length(V), V), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzchunkspairs(V::Number, state = V) = (Pair(V, V), state)

"`iteratenzpairs(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and value"
function iteratenzpairs end
"`iteratenzpairsview(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and `view` to value"
function iteratenzpairsview end
"`iteratenzvalues(V::AbstractAllDensedSparseVector)` iterates over non-zero elements of vector and returns value"
function iteratenzvalues end
"`iteratenzvaluesview(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns `view` of value"
function iteratenzvaluesview end
"`iteratenzindices(V::AbstractAllDensedSparseVector)` iterates over non-zero elements of vector and returns its indices"
function iteratenzindices end

#
# iteratenzSOMEs() iterators for `Number`, `Vector` and `SparseVector`
#

Base.@propagate_inbounds function iteratenzpairs(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (Pair(@inbounds V.nzind[state], @inbounds V.nzval[state]), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzpairsview(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (Pair(@inbounds V.nzind[state], @inbounds view(V.nzval, state:state)), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzpairs(V::Vector, state = 0)
    if state < length(V)
        state += 1
        #return ((state, @inbounds V[state]), state)
        return (Pair(state, @inbounds V[state]), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzpairsview(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (Pair(state, @inbounds view(V, state:state)), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzpairs(V::Number, state = 0) = (Pair(state+1, V), state+1)

Base.@propagate_inbounds function iteratenzvalues(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzval)
        return (@inbounds V.nzval[state], state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzvaluesview(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzval)
        return (@inbounds view(V.nzval, state:state), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzvalues(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (@inbounds V[state], state)
    elseif length(V) == 1
        return (@inbounds V[1], state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzvaluesview(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (@inbounds view(V, state:state), state)
    elseif length(V) == 1
        return (@inbounds view(V, 1:1), state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzvalues(V::Number, state = 0) = (V, state+1)

Base.@propagate_inbounds function iteratenzindices(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (@inbounds V.nzind[state], state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzindices(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (state, state)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzindices(V::Number, state = 0) = (state+1, state+1)


#
# `AbstractAllDensedSparseVector` iteration functions
#

struct ADSVIteratorState{T,Ti,Td,Tit}
    position::Int          # position of current element in the current chunk
    indices::UnitRange{Ti} # the indices of first and last elements in current chunk
    chunk::Td              # current chunk is the view into nzchunk
    idxchunk::Tit          # nzchunk iterator state (Int or Semitoken) in nzchunks
end

SparseArrays.indtype(it::ADSVIteratorState{T,Ti,Td,Tit}) where {T,Ti,Td,Tit} = Ti
Base.eltype(it::ADSVIteratorState{T,Ti,Td,Tit}) where {T,Ti,Td,Tit} = eltype(Td) # FIXME: That's wrong for BlockSparseVectors

@inline function nziteratorstate(V::Union{T,SubArray{<:Any,<:Any,<:T}}, position, indices, chunk::Tvv, it::Tit) where
                                          {T<:AbstractVecbasedDensedSparseVector{Tv,Ti},Tvv,Tit} where {Tv,Ti}
    ADSVIteratorState{T,Ti,Tvv,Tit}(position, indices, chunk, it)
end

# `ADSVIteratorState` is an NamedTuple
#@inline nziteratorstate(V, position, indices, chunk, idxchunk) =
#    (position=position, indices=indices, chunk=chunk, idxchunk=idxchunk)

# Start iterations from `i` index, i.e. `i` is `firstindex(V)`. That's option for `SubArray` and restarts.
startindex(V) = startindex(parent(V), first(parentindices(V)[1]))
function startindex(V, i)
    idxchunk = searchsortedlast_nzchunk(V, i)
    if idxchunk != pastendnzchunk_index(V)
        indices, chunk = get_indices_and_nzchunk(V, idxchunk)
        if checkindex(Bool, indices, i) #key <= i < key + length(chunk)
            return nziteratorstate(V, Int(i - first(indices)), indices, chunk, idxchunk)
        else
            return nziteratorstate(V, 0, indices, chunk, idxchunk)
        end
    else
        indices, chunk = get_indices_and_nzchunk(V)
        return nziteratorstate(V, 0, indices, chunk, idxchunk)
    end
end

# TODO: FIXME: Add simple :iterate
for (fn, ret1) in
        ((:iteratenzpairs    ,  :(indices[position] => chunk[position])                ),
         (:iteratenzpairsview,  :(indices[position] => view(chunk, position:position)) ),
         (:iteratenzvalues   ,  :(chunk[position])                                      ),
         (:iteratenzvaluesview, :(view(chunk, position:position))                       ),
         (:iteratenzindices  ,  :(indices[position])                                   ))

    @eval Base.@propagate_inbounds function $fn(V::Union{T,SubArray{<:Any,<:Any,<:T}}, state = startindex(V)) where
                                                {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Ti,Tv}
        position, indices, chunk, idxchunk = fieldvalues(state)
        position += 1
        if position <= length(indices)
            return ($ret1, nziteratorstate(V, position, indices, chunk, idxchunk))
        elseif (st = iteratenzchunkspairs(V, idxchunk)) !== nothing
            ((indices, chunk), idxchunk) = st
            position = 1
            return ($ret1, nziteratorstate(V, position, indices, chunk, idxchunk))
        else
            return nothing
        end
    end
end


###for (fn, ret1) in
###        ((:iteratenzpairs    ,  :(Ti(indices[position]-first(parentindices(V)[1])+1) => chunk[position])                ),
###         (:iteratenzpairsview,  :(Ti(indices[position]-first(parentindices(V)[1])+1) => view(chunk, position:position)) ),
###         (:iteratenzvalues   ,  :(chunk[position])                                                            ),
###         (:iteratenzvaluesview, :(view(chunk, position:position))                                             ),
###         (:iteratenzindices  ,  :(Ti(indices[position]-first(parentindices(V)[1])+1))                                   ))
###
###    @eval Base.@propagate_inbounds function $fn(V::SubArray{<:Any,<:Any,<:T},
###                                                state = startindex(parent(V), first(parentindices(V)[1]))) where
###                                                {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
###        position, indices, chunk, idxchunk = fieldvalues(state)
###        position += 1
###        if first(indices) + position >= last(parentindices(V)[1])
###            return nothing
###        elseif position <= length(indices)
###            return ($ret1, nziteratorstate(V, position, indices, chunk, idxchunk))
###            # TODO: FIXME: fix iteratenzchunkspairs() for last in SubArray nzchunk
###        #elseif (st = iteratenzchunkspairs(V, idxchunk)) !== nothing
###        elseif (st = iteratenzchunkspairs(parent(V), idxchunk)) !== nothing
###            ((indices, chunk), idxchunk) = st
###            position = 1
###            return ($ret1, nziteratorstate(V, position, indices, chunk, idxchunk))
###        else
###            return nothing
###        end
###    end
###end

#
#  Iterators
#
# TODO: Try IterTools.@ifsomething

"""
    get_iterable(it) = getfield(it, 1)

Get Iterator's iterable object.
This simple function just get first field of Iterator.
"""
@inline get_iterable(it) = getfield(it, 1)

struct NZChunks{It}
    itr::It
end
"`nzchunks(V::AbstractAllDensedSparseVector)` is the `Iterator` over chunks of nonzeros and
 returns tuple of start index and chunk vector"
@inline nzchunks(itr) = NZChunks(itr)
@inline function Base.iterate(it::NZChunks, state...)
    y = iteratenzchunkspairs(it.itr, state...)
    if y !== nothing
        return y[1][2], y[2]
    else
        return nothing
    end
end
SparseArrays.indtype(it::NZChunks) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZChunks{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZChunks{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZChunks}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZChunks}) = 1
Base.length(it::NZChunks) = nnzchunks(it.itr)
Base.size(it::NZChunks) = (nnzchunks(it.itr),)
Base.first(it::NZChunks) = first(iterate(nzchunks(it.itr)))
#Iterators.reverse(it::NZChunks) = NZChunks(Iterators.reverse(it.itr))


struct NZChunksPairs{It}
    itr::It
end
"`nzchunkspairs(V::AbstractAllDensedSparseVector)` is the `Iterator` over non-zero chunks,
 returns `Pair` of `UnitRange` first-last indices and view to vector of non-zero values."
@inline nzchunkspairs(itr) = NZChunksPairs(itr)
@inline Base.iterate(it::NZChunksPairs, state...) = iteratenzchunkspairs(it.itr, state...)
SparseArrays.indtype(it::NZChunksPairs) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZChunksPairs{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZChunksPairs{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZChunksPairs}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZChunksPairs}) = 1
Base.length(it::NZChunksPairs) = nnzchunks(it.itr)
Base.size(it::NZChunksPairs) = (nnzchunks(it.itr),)
#Iterators.reverse(it::NZChunksPairs) = NZChunksPairs(Iterators.reverse(it.itr))


struct NZIndices{It}
    itr::It
end
"`nzindices(V::AbstractVector)` is the `Iterator` over non-zero indices of vector `V`."
nzindices(itr) = NZIndices(itr)
@inline Base.iterate(it::NZIndices, state...) = iteratenzindices(it.itr, state...)
SparseArrays.indtype(it::NZIndices) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZIndices{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZIndices{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZIndices}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZIndices}) = 1
Base.length(it::NZIndices) = nnz(it.itr)
Base.size(it::NZIndices) = (nnz(it.itr),)
#Iterators.reverse(it::NZIndices) = NZIndices(Iterators.reverse(it.itr))
@inline Base.keys(V::AbstractAllDensedSparseVector) = nzindices(V)


struct NZValues{It}
    itr::It
end
"`nzvalues(V::AbstractVector)` is the `Iterator` over non-zero values of `V`."
nzvalues(itr) = NZValues(itr)
@inline Base.iterate(it::NZValues, state...) = iteratenzvalues(it.itr, state...)
SparseArrays.indtype(it::NZValues) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZValues{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZValues{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZValues}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZValues}) = 1
Base.length(it::NZValues) = nnz(it.itr)
Base.size(it::NZValues) = (nnz(it.itr),)
#Base.getindex(it::NZValues, i) = TODO
#Iterators.reverse(it::NZValues) = NZValues(Iterators.reverse(it.itr))


struct NZValuesView{It}
    itr::It
end
"""
`NZValuesView(V::AbstractVector)` is the `Iterator` over non-zero values of `V`,
returns the `view(V, idx:idx)` of iterated values.
"""
nzvaluesview(itr) = NZValuesView(itr)
@inline Base.iterate(it::NZValuesView, state...) = iteratenzvaluesview(it.itr, state...)
SparseArrays.indtype(it::NZValuesView) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZValuesView{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZValuesView{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZValuesView}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZValuesView}) = 1
Base.length(it::NZValuesView) = nnz(it.itr)
Base.size(it::NZValuesView) = (nnz(it.itr),)
#Iterators.reverse(it::NZValuesView) = NZValuesView(Iterators.reverse(it.itr))


# TODO: Create multiargs version of NZPairs like zip, zip_longest?
struct NZPairs{It}
    itr::It
end
"`nzpairs(V::AbstractVector)` is the `Iterator` over nonzeros of `V`, returns pair of index and value."
@inline nzpairs(itr) = NZPairs(itr)
@inline Base.iterate(it::NZPairs, state...) = iteratenzpairs(it.itr, state...)
SparseArrays.indtype(it::NZPairs) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZPairs{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZPairs{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZPairs}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZPairs}) = 1
Base.length(it::NZPairs) = nnz(it.itr)
Base.size(it::NZPairs) = (nnz(it.itr),)
#Iterators.reverse(it::NZPairs) = NZPairs(Iterators.reverse(it.itr))


struct NZPairsView{It}
    itr::It
end
"`nzpairsview(V::AbstractVector)` is the `Iterator` over nonzeros of `V`,
 returns pair of index and view `view(V, idx:idx)` on value to be mutable."
@inline nzpairsview(itr) = NZPairsView(itr)
@inline Base.iterate(it::NZPairsView, state...) = iteratenzpairsview(it.itr, state...)
SparseArrays.indtype(it::NZPairsView) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZPairsView{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZPairsView{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZPairsView}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZPairsView}) = 1
Base.length(it::NZPairsView) = nnz(it.itr)
Base.size(it::NZPairsView) = (nnz(it.itr),)
#Iterators.reverse(it::NZPairsView) = NZPairsView(Iterators.reverse(it.itr))


#
# Assignments
#


@inline function Base.isstored(V::AbstractAllDensedSparseVector, i::Integer)
    st = searchsortedlast_nzind(V, i)
    if st == beforestartnzchunk_index(V)  # the index `i` is before first index
        return false
    elseif i >= get_nzchunk_key(V, st) + get_nzchunk_length(V, st)
        # the index `i` is outside of data chunk indices
        return false
    end
    return true
end

@inline Base.in(i, V::AbstractAllDensedSparseVector) = Base.isstored(V, i)

function checkbounds(V, i::Pair)
    (idxcompare(V, first(i), beforestartnzchunk_index(V)) > 0 &&
     idxcompare(V, first(i), pastendnzchunk_index(V)) < 0) || throw(BoundsError(V, i))
    indices = get_nzchunk_indices(V, first(i))
    last(i) > length(indices) && throw(BoundsError(V, i))
    return nothing
end

@inline function Base.getindex(V::AbstractAllDensedSparseVector, i::Pair)
    @boundscheck checkbounds(V, i)
    return get_nzchunk(V, first(i))[last(i)]
end

@inline function Base.getindex(V::AbstractAllDensedSparseVector, i::Integer)
    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            return chunk[i - ifirst + 1]
        end
    end
    # cached chunk index miss or index not stored
    st = searchsortedlast_nzind(V, i)
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if i < ifirst + len  # is the index `i` inside of data chunk indices range
            V.lastusedchunkindex = st
            return chunk[i - ifirst + 1]
        end
    end
    V.lastusedchunkindex = beforestartnzchunk_index(V)
    return returnzero(V)
end


@inline Base.getindex(V::DensedSVSparseVector, i::Integer, j::Integer) = getindex(V, i)[j]


@inline function Base.getindex(V::DensedVLSparseVector, i::Integer)
    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, offsets = V.nzind[st], V.nzchunks[st], V.offsets[st]
        if ifirst <= i < ifirst + length(offsets)-1
            offs = offsets[i-ifirst+1]
            len = offsets[i-ifirst+1+1] - offsets[i-ifirst+1]
            return @view(chunk[offs:offs+len-1])
        end
    end
    # cached chunk index miss or index not stored
    st = searchsortedlast(V.nzind, i)
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, offsets = V.nzind[st], V.nzchunks[st], V.offsets[st]
        if i < ifirst + length(offsets)-1  # is the index `i` inside of data chunk indices range
            V.lastusedchunkindex = st
            offs = offsets[i-ifirst+1]
            len = offsets[i-ifirst+1+1] - offsets[i-ifirst+1]
            return @view(chunk[offs:offs+len-1])
        end
    end
    V.lastusedchunkindex = beforestartnzchunk_index(V)
    return view(V.dummy, 1:0)
end

@inline function Base.getindex(V::DensedVLSparseVector, i::Integer, j::Integer)
    vv = getindex(V, i)
    if j <= length(vv)
        return vv[j]
    else
        return returnzero(V)
    end
end


@inline function Base.setindex!(V::AbstractAllDensedSparseVector, value, i::Pair)
    @boundscheck checkbounds(V, i)
    get_nzchunk(V, first(i))[last(i)] = value
    return V
end

@inline function Base.setindex!(V::FixedDensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = Tv(value)

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            chunk[i - ifirst + 1] = val
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            chunk[i - ifirst + 1] = val
            V.lastusedchunkindex = st
            return V
        end
    end

    V.lastusedchunkindex = beforestartnzchunk_index(V)

    throw(BoundsError(V, i))
end



@inline function Base.setindex!(V::DensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = Tv(value)

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            chunk[i - ifirst + 1] = val
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk = V.nzind[st], V.nzchunks[st]
        if i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            V.lastusedchunkindex = st
            return V
        end
    end

    if V.nnz == 0
        push!(V.nzind, Ti(i))
        push!(V.nzchunks, [val])
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = 1
        return V
    end

    if st == beforestartnzchunk_index(V)  # the index `i` is before the first index
        inextfirst = V.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(V.nzind, i)
            pushfirst!(V.nzchunks, [val])
        else
            V.nzind[1] -= 1
            pushfirst!(V.nzchunks[1], val)
        end
        V.nnz += 1
        V.lastusedchunkindex = 1
        return V
    end

    ifirst, chunk = V.nzind[st], V.nzchunks[st]

    if i >= V.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(V.nzind, i)
            push!(V.nzchunks, [val])
        else  # just append to last chunk
            push!(V.nzchunks[st], val)
        end
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = length(V.nzind)
        return V
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = st + 1
    inextfirst = V.nzind[stnext]

    if inextfirst - ilast == 2  # join nzchunks
        append!(V.nzchunks[st], [val], V.nzchunks[stnext])
        deleteat!(V.nzind, stnext)
        deleteat!(V.nzchunks, stnext)
        V.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        push!(V.nzchunks[st], val)
        V.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        V.nzind[stnext] -= 1
        pushfirst!(V.nzchunks[stnext], val)
        V.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(V.nzind, stnext, Ti(i))
        insert!(V.nzchunks, stnext, [val])
        V.lastusedchunkindex = stnext
    end

    V.nnz += 1
    return V

end



@inline function Base.setindex!(V::DensedSVSparseVector{Tv,Ti,m}, vectorvalue::AbstractVector, i::Integer) where {Tv,Ti,m}
    sv = eltype(eltype(V.nzchunks))(vectorvalue)

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            chunk[i - ifirst + 1] = sv
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        #ifirst, chunk = V.nzind[st], V.nzchunks[st]
        if i < ifirst + len
            chunk[i - ifirst + 1] = sv
            V.lastusedchunkindex = st
            return V
        end
    end

    if V.nnz == 0
        push!(V.nzind, Ti(i))
        push!(V.nzchunks, [sv])
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = 1
        return V
    end

    if st == beforestartnzchunk_index(V)  # the index `i` is before the first index
        inextfirst = V.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(V.nzind, i)
            pushfirst!(V.nzchunks, [sv])
        else
            V.nzind[1] -= 1
            pushfirst!(V.nzchunks[1], sv)
        end
        V.nnz += 1
        V.lastusedchunkindex = 1
        return V
    end

    ifirst, chunk = V.nzind[st], V.nzchunks[st]

    if i >= V.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(V.nzind, i)
            push!(V.nzchunks, [sv])
        else  # just append to last chunk
            push!(V.nzchunks[st], sv)
        end
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = length(V.nzind)
        return V
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = st + 1
    inextfirst = V.nzind[stnext]

    if inextfirst - ilast == 2  # join nzchunks
        append!(V.nzchunks[st], [sv], V.nzchunks[stnext])
        deleteat!(V.nzind, stnext)
        deleteat!(V.nzchunks, stnext)
        V.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        push!(V.nzchunks[st], sv)
        V.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        V.nzind[stnext] -= 1
        pushfirst!(V.nzchunks[stnext], sv)
        V.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(V.nzind, stnext, Ti(i))
        insert!(V.nzchunks, stnext, [sv])
        V.lastusedchunkindex = stnext
    end

    V.nnz += 1
    return V

end

@inline function Base.setindex!(V::DensedSVSparseVector{Tv,Ti,m}, value, i::Integer, j::Integer) where {Tv,Ti,m}
    val = Tv(value)

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            sv = chunk[i - ifirst + 1]
            chunk[i - ifirst + 1] = @set sv[j] = val
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if i < ifirst + len
            sv = chunk[i - ifirst + 1]
            chunk[i - ifirst + 1] = @set sv[j] = val
            V.lastusedchunkindex = st
            return V
        end
    end

    sv = zeros(eltype(eltype(V.nzchunks)))
    sv = @set sv[j] = val

    if V.nnz == 0
        push!(V.nzind, Ti(i))
        push!(V.nzchunks, [sv])
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = 1
        return V
    end

    if st == beforestartnzchunk_index(V)  # the index `i` is before the first index
        inextfirst = V.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(V.nzind, i)
            pushfirst!(V.nzchunks, [sv])
        else
            V.nzind[1] -= 1
            pushfirst!(V.nzchunks[1], sv)
        end
        V.nnz += 1
        V.lastusedchunkindex = 1
        return V
    end

    ifirst, chunk = V.nzind[st], V.nzchunks[st]

    if i >= V.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(V.nzind, i)
            push!(V.nzchunks, [sv])
        else  # just append to last chunk
            push!(V.nzchunks[st], sv)
        end
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = length(V.nzind)
        return V
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = st + 1
    inextfirst = V.nzind[stnext]

    if inextfirst - ilast == 2  # join nzchunks
        append!(V.nzchunks[st], [sv], V.nzchunks[stnext])
        deleteat!(V.nzind, stnext)
        deleteat!(V.nzchunks, stnext)
        V.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        push!(V.nzchunks[st], sv)
        V.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        V.nzind[stnext] -= 1
        pushfirst!(V.nzchunks[stnext], sv)
        V.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(V.nzind, stnext, Ti(i))
        insert!(V.nzchunks, stnext, [sv])
        V.lastusedchunkindex = stnext
    end

    V.nnz += 1
    return V

end

@inline function Base.setindex!(V::DensedVLSparseVector{Tv,Ti}, vectorvalue::AbstractVector, i::Integer) where {Tv,Ti}

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk, offsets = V.nzind[st], V.nzchunks[st], V.offsets[st]
        if ifirst <= i < ifirst + length(offsets)-1
            lenvalue = length(vectorvalue)
            pos1 = i-ifirst+1
            offs1 = offsets[pos1]
            offs2 = offsets[pos1+1] - 1
            len = offs2 - offs1 + 1
            if lenvalue == len
                chunk[offs1:offs2] .= vectorvalue
            elseif lenvalue < len
                deleteat!(chunk, offs1+lenvalue-1:offs2-1)
                @view(offsets[pos1+1:end]) .-= len - lenvalue
                offs2 = offsets[pos1+1] - 1
                chunk[offs1:offs2] .= vectorvalue
            else
                resize!(chunk, length(chunk) + lenvalue - len)
                @view(offsets[pos1+1:end]) .+= lenvalue - len
                offs2 = offsets[pos1+1] - 1
                for i = length(chunk):-1:offs2 + 1
                    chunk[i] = chunk[i - (lenvalue - len)]
                end
                chunk[offs1:offs2] .= vectorvalue
            end
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, offsets = V.nzind[st], V.nzchunks[st], V.offsets[st]
        if ifirst <= i < ifirst + length(offsets)-1
            lenvalue = length(vectorvalue)
            pos1 = i-ifirst+1
            offs1 = offsets[pos1]
            offs2 = offsets[pos1+1] - 1
            len = offs2 - offs1 + 1
            if lenvalue == len
                chunk[offs1:offs2] .= vectorvalue
            elseif lenvalue < len
                deleteat!(chunk, offs1+lenvalue-1:offs2-1)
                @view(offsets[pos1+1:end]) .-= len - lenvalue
                offs2 = offsets[pos1+1] - 1
                chunk[offs1:offs2] .= vectorvalue
            else
                resize!(chunk, length(chunk) + lenvalue - len)
                @view(offsets[pos1+1:end]) .+= lenvalue - len
                offs2 = offsets[pos1+1] - 1
                for i = length(chunk):-1:offs2 + 1
                    chunk[i] = chunk[i - (lenvalue - len)]
                end
                chunk[offs1:offs2] .= vectorvalue
            end
            return V
        end
    end


    if V.nnz == 0
        push!(V.nzind, Ti(i))
        push!(V.nzchunks, Vector(vectorvalue))
        push!(V.offsets, [1])
        append!(V.offsets[1], length(vectorvalue)+1)
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = 1
        return V
    end

    if st == beforestartnzchunk_index(V)  # the index `i` is before the first index
        inextfirst = V.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(V.nzind, i)
            pushfirst!(V.nzchunks, Vector(vectorvalue))
            pushfirst!(V.offsets, [1])
            append!(V.offsets[1], length(vectorvalue)+1)
        else
            V.nzind[1] -= 1
            prepend!(V.nzchunks[1], vectorvalue)
            @view(V.offsets[1][2:end]) .+= length(vectorvalue)
            insert!(V.offsets[1], 2, length(vectorvalue)+1)
        end
        V.nnz += 1
        V.lastusedchunkindex = 1
        return V
    end

    ifirst, chunk, offsets = V.nzind[st], V.nzchunks[st], V.offsets[st]

    if i >= V.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(offsets)-1  # there is will be the gap in indices after inserting
            push!(V.nzind, i)
            push!(V.nzchunks, Vector(vectorvalue))
            push!(V.offsets, [1])
            push!(V.offsets[end], length(vectorvalue)+1)
        else  # just append to last chunk
            append!(V.nzchunks[st], vectorvalue)
            push!(V.offsets[st], V.offsets[st][end]+length(vectorvalue))
        end
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = length(V.nzind)
        return V
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(offsets)-1 - 1
    stnext = st + 1
    inextfirst = V.nzind[stnext]

    if inextfirst - ilast == 2  # join nzchunks
        append!(V.nzchunks[st], vectorvalue, V.nzchunks[stnext])
        V.offsets[stnext] .+= V.offsets[st][end]-1 + length(vectorvalue)
        append!(V.offsets[st], V.offsets[stnext])
        deleteat!(V.nzind, stnext)
        deleteat!(V.nzchunks, stnext)
        deleteat!(V.offsets, stnext)
        V.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        append!(V.nzchunks[st], vectorvalue)
        push!(V.offsets[st], V.offsets[st][end]+length(vectorvalue))
        V.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        V.nzind[stnext] -= 1
        prepend!(V.nzchunks[stnext], vectorvalue)
        @view(V.offsets[stnext][2:end]) .+= length(vectorvalue)
        insert!(V.offsets[stnext], 2, length(vectorvalue)+1)
        V.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(V.nzind, stnext, Ti(i))
        insert!(V.nzchunks, stnext, Vector(vectorvalue))
        insert!(V.offsets, stnext, [1])
        push!(V.offsets[stnext], length(vectorvalue)+1)
        V.lastusedchunkindex = stnext
    end

    V.nnz += 1
    return V

end



@inline function Base.setindex!(V::DynamicDensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = eltype(V)(value)

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartsemitoken(V.nzchunks)
        ifirst, chunk, len = get_key_and_nzchunk_and_length(V, st)
        if ifirst <= i < ifirst + len
            chunk[i - ifirst + 1] = val
            return V
        end
    end

    st = searchsortedlast(V.nzchunks, i)

    sstatus = status((V.nzchunks, st))
    @boundscheck if sstatus == 0 # invalid semitoken
        throw(KeyError(i))
    end

    # check the index exist and update its data
    if V.nnz > 0 && sstatus != 2  # the index `i` is not before the first index
        ifirst, chunk = deref((V.nzchunks, st))
        if ifirst + length(chunk) > i
            chunk[i - ifirst + 1] = val
            V.lastusedchunkindex = st
            return V
        end
    end

    if V.nnz == 0
        V.nzchunks[i] = [val]
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = startof(V.nzchunks)  # firstindex(V.nzchunks)
        return V
    end

    if sstatus == 2  # the index `i` is before the first index
        stnext = startof(V.nzchunks)
        inextfirst = deref_key((V.nzchunks, stnext))
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            V.nzchunks[i] = [val]
        else
            V.nzchunks[i] = pushfirst!(deref_value((V.nzchunks, stnext)), val)
            delete!((V.nzchunks, stnext))
        end
        V.nnz += 1
        V.lastusedchunkindex = startof(V.nzchunks)
        return V
    end

    ifirst, chunk = deref((V.nzchunks, st))

    if i >= deref_key((V.nzchunks, lastindex(V.nzchunks))) #lastkey(V) # the index `i` is after the last key index
        if ifirst + length(chunk) < i  # there is will be the gap in indices after inserting
            V.nzchunks[i] = [val]
        else  # just append to last chunk
            V.nzchunks[st] = push!(chunk, val)
        end
        V.nnz += 1
        V.n = max(V.n, i)
        V.lastusedchunkindex = lastindex(V.nzchunks)
        return V
    end

    V.lastusedchunkindex = beforestartsemitoken(V.nzchunks)

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = advance((V.nzchunks, st))
    inextfirst = deref_key((V.nzchunks, stnext))

    if inextfirst - ilast == 2  # join nzchunks
        V.nzchunks[st] = append!(chunk, [val], deref_value((V.nzchunks, stnext)))
        delete!((V.nzchunks, stnext))
    elseif i - ilast == 1  # append to left chunk
        V.nzchunks[st] = push!(chunk, val)
        V.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        V.nzchunks[i] = pushfirst!(deref_value((V.nzchunks, stnext)), val)
        delete!((V.nzchunks, stnext))
    else  # insert single element chunk
        V.nzchunks[i] = [val]
    end

    V.nnz += 1
    return V

end

#function Base.setindex!(V::AbstractAllDensedSparseVector{Tv,Ti}, data::AbstractAllDensedSparseVector, index::Integer) where {Tv,Ti}
#    i0 = Ti(index-1)
#    if V === data
#        cdata = deepcopy(data)
#        for (i,d) in nzpairs(cdata)
#            V[i0+i] = Tv(d)
#        end
#    else
#        for (i,d) in nzpairs(data)
#            V[i0+i] = Tv(d)
#        end
#    end
#    return V
#end

@inline function SparseArrays.dropstored!(V::AbstractVecbasedDensedSparseVector, i::Integer)

    V.nnz == 0 && return V

    st = searchsortedlast(V.nzind, i)

    if st == beforestartnzchunk_index(V)  # the index `i` is before first index
        return V
    end

    ifirst = V.nzind[st]
    lenchunk = length(V.nzchunks[st])

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return V
    end

    if lenchunk == 1
        deleteat!(V.nzchunks[st], 1)
        deleteat!(V.nzind, st)
        deleteat!(V.nzchunks, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        pop!(V.nzchunks[st])
    elseif i == ifirst  # first element in chunk
        V.nzind[st] += 1
        popfirst!(V.nzchunks[st])
    else
        insert!(V.nzind, st+1, i+1)
        insert!(V.nzchunks, st+1, V.nzchunks[st][i-ifirst+1+1:end])
        resize!(V.nzchunks[st], i-ifirst+1 - 1)
    end

    V.nnz -= 1
    V.lastusedchunkindex = 0

    return V
end

@inline function SparseArrays.dropstored!(V::DensedVLSparseVector, i::Integer)

    V.nnz == 0 && return V

    st = searchsortedlast(V.nzind, i)

    if st == beforestartnzchunk_index(V)  # the index `i` is before first index
        return V
    end

    ifirst = V.nzind[st]
    lenchunk = length(V.offsets[st]) - 1

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return V
    end

    if lenchunk == 1
        deleteat!(V.nzchunks[st], 1)
        deleteat!(V.nzind, st)
        deleteat!(V.nzchunks, st)
        deleteat!(V.offsets, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        len = V.offsets[st][end] - V.offsets[st][end-1]
        resize!(V.nzchunks[st], length(V.nzchunks[st]) - len)
        pop!(V.offsets[st])
        @assert(length(V.nzchunks[st]) == V.offsets[st][end]-1)
    elseif i == ifirst  # first element in chunk
        V.nzind[st] += 1
        len = V.offsets[st][2] - V.offsets[st][1]
        deleteat!(V.nzchunks[st], 1:len)
        popfirst!(V.offsets[st])
        V.offsets[st] .-= V.offsets[st][1] - 1
        @assert(length(V.nzchunks[st]) == V.offsets[st][end]-1)
    else
        insert!(V.nzind, st+1, i+1)
        insert!(V.nzchunks, st+1, V.nzchunks[st][V.offsets[st][i-ifirst+1+1]:end])
        resize!(V.nzchunks[st], V.offsets[st][i-ifirst+1] - 1)
        insert!(V.offsets, st+1, V.offsets[st][i-ifirst+1 + 1:end])
        resize!(V.offsets[st], i-ifirst+1)
        V.offsets[st+1] .-= V.offsets[st+1][1] - 1
        @assert(length(V.nzchunks[st]) == V.offsets[st][end]-1)
        @assert(length(V.nzchunks[st+1]) == V.offsets[st+1][end]-1)
    end

    V.nnz -= 1
    V.lastusedchunkindex = 0

    return V
end

@inline function SparseArrays.dropstored!(V::DynamicDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

    V.nnz == 0 && return V

    st = searchsortedlast(V.nzchunks, i)

    if st == beforestartnzchunk_index(V)  # the index `i` is before first index
        return V
    end

    ifirst, chunk = deref((V.nzchunks, st))

    if i >= ifirst + length(chunk)  # the index `i` is outside of data chunk indices
        return V
    end

    if length(chunk) == 1
        deleteat!(chunk, 1)
        delete!(V.nzchunks, i)
    elseif i == ifirst + length(chunk) - 1  # last index in chunk
        pop!(chunk)
        V.nzchunks[st] = chunk
    elseif i == ifirst
        popfirst!(chunk)
        V.nzchunks[i+1] = chunk
        delete!(V.nzchunks, i)
    else
        V.nzchunks[i+1] = chunk[i-ifirst+1+1:end]
        V.nzchunks[st] = resize!(chunk, i-ifirst+1 - 1)
    end

    V.nnz -= 1
    V.lastusedchunkindex = beforestartsemitoken(V.nzchunks)

    return V
end




function _expand_full!(V::DensedSparseVector{Tv,Ti}) where {Tv,Ti}
    isempty(V) || empty!(V)
    resize!(V.nzind, 1)
    V.nzind[1] = firstindex(V)
    resize!(V.nzchunks, 1)
    V.nzchunks[1] = Vector{Tv}(undef, length(V))
    V.nnz = length(V)
    return V
end
function _expand_full!(V::DynamicDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    isempty(V) || empty!(V)
    V.nzchunks[firstindex(V)] = Vector{Tv}(undef, length(V))
    V.nnz = length(V)
    return V
end
_expand_full!(V::FixedDensedSparseVector) = throw(MethodError("attempt to reshape $(typeof(V)) vector"))


function Base.fill!(V::AbstractAllDensedSparseVector{Tv,Ti}, value) where {Tv,Ti}
    nnz(V) == length(V) != 0 || _expand_full!(V)
    fill!(first(nzchunks(V)), Tv(value))
    V
end
function Base.fill!(V::SubArray{<:Any,<:Any,<:T}, value) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    # TODO: FIXME: redo with broadcast
    for i in eachindex(V)
        V[i] = Tv(value)
    end
    V
end


function _similar_resize!(C::DensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    C.n = A.n
    C.nnz = A.nnz
    nnzch = nnzchunks(A)
    resize!(C.nzind, nnzch)
    resize!(C.nzchunks, nnzch)
    for (i, (indices,chunk)) in enumerate(nzchunkspairs(A))
        if isassigned(C.nzchunks, i)
            resize!(C.nzchunks[i], length(chunk))
        else
            C.nzchunks[i] = Vector{Tv}(undef, length(chunk))
        end
        C.nzind[i] = Ti(first(indices))
    end
    return C
end

function _similar_sparse_indices!(C::DensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    isempty(C) && return _similar_resize!(C, A)
    C.n = A.n
    C.nnz = A.nnz == 0 && return empty!(C)
    resizes = 0
    nresizes = 3

    iC = firstnzchunk_index(C)
    iA = firstnzchunk_index(A)

    while iC != pastendnzchunk_index(C) && iA != pastendnzchunk_index(A)
        kC, chunkC = get_key_and_nzchunk(C, iC)
        kA, chunkA = get_key_and_nzchunk(A, iA)
        if kC == kA
            length(chunkC) == length(chunkA) || resize!(chunkC, length(chunkA))
            iC = advance(C, iC)
            iA = advance(A, iA)
        elseif kC < kA
            deleteat!(C.nzind, iC)
            deleteat!(C.nzchunks, iC)
            (resizes += 1) > nresizes && return _similar_resize!(C, A)
        else #if kC > kA
            insert!(C.nzind, iC, kA)
            insert!(C.nzchunks, iC, Vector{Tv}(undef, length(chunkA)))
            (resizes += 1) > nresizes && return _similar_resize!(C, A)
            iC = advance(C, iC)
            iA = advance(A, iA)
        end
    end

    nnzch = nnzchunks(A)
    resize!(C.nzind, nnzch)
    resize!(C.nzchunks, nnzch)

    while iA != pastendnzchunk_index(A)
        kA, chunkA = get_key_and_nzchunk(A, iA)
        C.nzchunks[kA] = Vector{Tv}(undef, length(chunkA))
        iA = advance(A, iA)
    end

    return C
end

_similar_sparse_indices!(C::FixedDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti} = (_check_same_sparse_indices(C, A); return C)

_similar_sparse_indices!(C::AbstractDensedSparseVector{Tv,Ti,BZP}, A::AbstractDensedSparseVector) where {Tv,Ti,BZP<:Val{true}} = C

function _similar_sparse_indices!(C::DynamicDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    C.n = A.n
    C.nnz = A.nnz == 0 && return empty!(C)

    iC = firstnzchunk_index(C)
    iA = firstnzchunk_index(A)

    while iC != pastendnzchunk_index(C) && iA != pastendnzchunk_index(A)
        kC, chunkC = get_key_and_nzchunk(C, iC)
        kA, chunkA = get_key_and_nzchunk(A, iA)
        if kC == kA
            resize!(chunkC, length(chunkA))
            iC = advance(C, iC)
            iA = advance(A, iA)
        elseif kC < kA
            delete!((C, iC))
            iC = searchsortedlast_nzchunk(C, kC)
        else #if kC > kA
            C.nzchunks[kA] = Vector{Tv}(undef, length(chunkA))
            iC = searchsortedlast_nzchunk(C, kA)
            iC = advance(C, iC)
            iA = advance(A, iA)
        end
    end

    for st in semitokens(exclusive(C.nzchunks,iC,pastendnzchunk_index(C)))
        delete!((C, st))
    end

    while iA != pastendnzchunk_index(A)
        kA, chunkA = get_key_and_nzchunk(A, iA)
        C.nzchunks[kA] = Vector{Tv}(undef, length(chunkA))
        iA = advance(A, iA)
    end

    return C
end


# derived from SparseArrays/src/higherorderfns.jl
@inline _aresameshape(A) = true
@inline _aresameshape(A, B) = size(A) == size(B)
@inline _aresameshape(A, B, Cs...) = _aresameshape(A, B) ? _aresameshape(B, Cs...) : false

#@inline function _are_same_sparse_indices(A, B, Cs...)
#    if !reduce((a,b) -> length(a) == length(b)      , (A,B,Cs...)) ||
#       !reduce((a,b) -> nnz(a) == nnz(b)            , (A,B,Cs...)) ||
#       !reduce((a,b) -> nnzchunks(a) == nnzchunks(b), (A,B,Cs...))
#        return false
#    end
#
#    # WTF?????
#    # there is the same number of chunks thus `zip` is enough
#    for (ks, chunks) in zip(map(nzchunkspairs, (A,B,Cs...)))
#        if !reduce(==, ks) || !reduce((a,b) -> length(a) == length(b), chunks)
#            return false
#        end
#    end
#    return true
#end

@inline _are_same_sparse_indices(A) = true
@inline function _are_same_sparse_indices(A, B)
    _aresameshape(A, B) || return false
    if nnz(A) != nnz(B) || nnzchunks(A) != nnzchunks(B)
        return false
    end
    for ((idsA,_), (idsB,_)) in zip(nzchunkspairs(A), nzchunkspairs(B)) # there is the same number of chunks thus `zip` is good
        if first(idsA) != first(idsB) || last(idsA) != last(idsB)
            return false
        end
    end
    return true
end
@inline _are_same_sparse_indices(A, B, Cs...) = _are_same_sparse_indices(A, B) ? _are_same_sparse_indices(B, Cs...) : false

_check_same_sparse_indices(As...) = _are_same_sparse_indices(As...) || throw(DimensionMismatch("argument shapes must match"))

@inline _copy_chunk_to!(C::DensedSparseVector{Tv,Ti}, i, k, chunk) where {Tv,Ti} = (C.nzind[i] = Ti(k); C.nzchunks[i] .= Tv.(chunk))
@inline _copy_chunk_to!(C::FixedDensedSparseVector{Tv,Ti}, i, k, chunk) where {Tv,Ti} = @view(C.nzchunks[C.offsets[i]:C.offsets[i+1]-1]) .= Tv.(chunk)
@inline _copy_chunk_to!(C::DynamicDensedSparseVector{Tv,Ti}, i, k, chunk) where {Tv,Ti} = C.nzchunks[Ti(k)] .= Tv.(chunk)

function Base.copyto!(C::AbstractDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    _similar_sparse_indices!(C, A)
    for (i, (ids,chunk)) in enumerate(nzchunkspairs(A))
        _copy_chunk_to!(C, i, first(ids), chunk)
    end
    return C
end



function Base.empty!(V::Union{DensedSparseVector,DensedSVSparseVector})
    empty!(V.nzind); empty!(V.nzchunks);
    V.lastusedchunkindex = beforestartnzchunk_index(V)
    V.nnz = 0
    V
end
function Base.empty!(V::DensedVLSparseVector)
    empty!(V.nzind); empty!(V.nzchunks); empty!(V.offsets)
    V.lastusedchunkindex = beforestartnzchunk_index(V)
    V.nnz = 0
    V
end
function Base.empty!(V::DynamicDensedSparseVector)
    empty!(V.nzchunks)
    V.lastusedchunkindex = beforestartnzchunk_index(V)
    V.nnz = 0
    V
end
Base.empty!(V::FixedDensedSparseVector) = throw(MethodError("attempt to empty $(typeof(V)) vector"))


#
#  Broadcasting
#
# TODO: NZChunksStyle and NZValuesStyle
#

include("higherorderfns.jl")

include("show.jl")



end  # of module DensedSparseVectors
