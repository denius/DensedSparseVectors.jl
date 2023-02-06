
# TODO: Create ConsDensedSparseVector{Tv,Ti} with single consecuitive nzchunks and offsets[1:n+1]. Maybe FixedDensedSparseVector?
#
# TODO: Introduce offsets fields to all types to have indexable iterator nonzeros(::AbstractAllDensedSparseVector): getindex(it::NZValues, i)
#
# TODO: Introduce ChainIndex: ((chain_ind, element_ind), LinearIndex) like CartesianIndex
#       for fast AbstractAllDensedSparseVector access without searchsortedlast and so on.

module DensedSparseVectors
export AbstractAllDensedSparseVector
export DensedSparseVector, FixedDensedSparseVector, DynamicDensedSparseVector
export DensedSVSparseVector, DensedVLSparseVector
export nzpairs, nzvalues, nzvaluesview, nzindices, nzchunks, nzchunkspairs
export findfirstnz, findlastnz, findfirstnzindex, findlastnzindex
export iteratenzpairs, iteratenzpairsview, iteratenzvalues, iteratenzvaluesview, iteratenzindices


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


abstract type AbstractAllDensedSparseVector{Tv,Ti} <: AbstractSparseVector{Tv,Ti} end

"Vector alike DensedSparseVector kind"
abstract type AbstractDensedSparseVector{Tv,Ti} <: AbstractAllDensedSparseVector{Tv,Ti} end
"Matrix alike Vector of Vectors kind"
abstract type AbstractDensedBlockSparseVector{Tv,Ti} <: AbstractAllDensedSparseVector{Tv,Ti} end

"Simple VectorDensedSparseVector kind"
abstract type AbstractSimpleDensedSparseVector{Tv,Ti} <: AbstractDensedSparseVector{Tv,Ti} end
"Based on SortedDict VectorDensedSparseVector kind"
abstract type AbstractSDictDensedSparseVector{Tv,Ti} <: AbstractDensedSparseVector{Tv,Ti} end


"All Vector alike types `<: AbstractAllDensedSparseVector`"
const AbstractVecbasedDensedSparseVector{Tv,Ti} = Union{AbstractSimpleDensedSparseVector{Tv,Ti}, AbstractDensedBlockSparseVector{Tv,Ti}}


"""
The `DensedSparseVector` is alike the `Vector` but have the omits in stored indices/data.
It is the subtype of `AbstractSparseVector`. The speed of `Broadcasting` on `DensedSparseVector`
is almost the same as on the `Vector`, but the speed by direct index access is almost few times
slower then the for `Vector`'s one.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSparseVector{Tv,Ti} <: AbstractSimpleDensedSparseVector{Tv,Ti}
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

    DensedSparseVector{Tv,Ti}(n::Integer, nzind, nzchunks) where {Tv,Ti} =
        new{Tv,Ti}(0, nzind, nzchunks, n, foldl((s,c)->(s+length(c)), nzchunks; init=0))
    DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = new{Tv,Ti}(0, Vector{Ti}(), Vector{Vector{Tv}}(), n, 0)
end

DensedSparseVector(n::Integer = 0) = DensedSparseVector{Float64,Int}(n)

function DensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(V))
    nzchunks = Vector{Vector{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = k
        nzchunks[i] = Vector{Tv}(d)
    end
    return DensedSparseVector{Tv,Ti}(length(V), nzind, nzchunks)
end

#"View for DensedSparseVector"
#struct DensedSparseVectorView{Tv,Ti,T,Tc} <: AbstractVecbasedDensedSparseVector{Tv,Ti}
#    "Index of first chunk in `view` V"
#    firstnzchunk_index::Int
#    "Index of last chunk in `view` V"
#    lastnzchunk_index::Int
#    "View on DensedSparseVector"
#    V::Tc
#end



"""
The `FixedDensedSparseVector` is the `DensedSparseVector` without the ability to change shape after creating.
It should be slightly faster then `DensedSparseVector` because data locality and omit one reference lookup.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct FixedDensedSparseVector{Tv,Ti} <: AbstractSimpleDensedSparseVector{Tv,Ti}
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

    FixedDensedSparseVector{Tv,Ti}(n::Integer, nzind, nzchunks, offsets) where {Tv,Ti} =
        new{Tv,Ti}(0, nzind, nzchunks, offsets, n, length(nzchunks))
end


FixedDensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = FixedDensedSparseVector{Tv,Ti}(V)

function FixedDensedSparseVector{Tv,Ti}(V::AbstractAllDensedSparseVector) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(V))
    nzchunks = Vector{Tv}(undef, nnz(V))
    offsets = Vector{Int}(undef, nnzchunks(V)+1)
    offsets[1] = 1
    for (i, (k,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = k
        offsets[i+1] = offsets[i] + length(d)
        @view(nzchunks[offsets[i]:offsets[i+1]-1]) .= Tv.(d)
    end
    return FixedDensedSparseVector{Tv,Ti}(length(V), nzind, nzchunks, offsets)
end



"""
The `DensedSVSparseVector` is the version of `DensedSparseVector` with `SVector` as elements
and alike `Matrix` with sparse first dimension and with dense `SVector` in second dimension.
See `DensedSparseVector` for details.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSVSparseVector{Tv,Ti,m} <: AbstractDensedBlockSparseVector{Tv,Ti}
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

    DensedSVSparseVector{Tv,Ti,m}(n::Integer, nzind, nzchunks) where {Tv,Ti,m} =
        new{Tv,Ti,m}(0, nzind, nzchunks, n, foldl((s,c)->(s+length(c)), nzchunks; init=0))
    DensedSVSparseVector{Tv,Ti,m}(n::Integer = 0) where {Tv,Ti,m} =
        new{Tv,Ti,m}(0, Vector{Ti}(), Vector{Vector{Tv}}(), n, 0)
end

DensedSVSparseVector(m::Integer, n::Integer = 0) = DensedSVSparseVector{Float64,Int,m}(n)



"""
The `DensedVLSparseVector` is the version of `DensedSparseVector` with variable length elements
and alike `Matrix` with sparse first dimension and with variable length dense vectors in second dimension.
See `DensedSparseVector` for details.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedVLSparseVector{Tv,Ti} <: AbstractDensedBlockSparseVector{Tv,Ti}
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

    DensedVLSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
        new{Tv,Ti}(0, Vector{Ti}(), Vector{Vector{Tv}}(), Vector{Vector{Int}}(), n, 0, Tv[])
end

DensedVLSparseVector(n::Integer = 0) = DensedVLSparseVector{Float64,Int}(n)



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
mutable struct DynamicDensedSparseVector{Tv,Ti} <: AbstractSDictDensedSparseVector{Tv,Ti}
    "Index of last used chunk"
    lastusedchunkindex::DataStructures.Tokens.IntSemiToken
    "Storage for indices of the first element of non-zero chunks and corresponding chunks as `SortedDict(Int=>Vector)`"
    nzchunks::SortedDict{Ti,Vector{Tv},FOrd}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    function DynamicDensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
        nzchunks = SortedDict{Ti,Vector{Tv},FOrd}(Forward)
        new{Tv,Ti}(beforestartsemitoken(nzchunks), nzchunks, n, 0)
    end
    DynamicDensedSparseVector{Tv,Ti}(n::Integer, nzchunks::SortedDict{K,V}) where {Tv,Ti,K,V<:AbstractVector} =
        new{Tv,Ti}(beforestartsemitoken(nzchunks), nzchunks, n, foldl((s,c)->(s+length(c)), values(nzchunks); init=0))
end

DynamicDensedSparseVector(n::Integer = 0) = DynamicDensedSparseVector{Float64,Int}(n)

function DynamicDensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzchunks = SortedDict{Ti, Vector{Tv}, FOrd}(Forward)
    for (k,d) in nzchunkspairs(V)
        nzchunks[k] = Vector{Tv}(d)
    end
    return DynamicDensedSparseVector{Tv,Ti}(length(V), nzchunks)
end

"""
Convert any particular `AbstractSparseVector`s to corresponding `AbstractAllDensedSparseVector`:

    DensedSparseVector(sv)

"""
function (::Type{T})(V::AbstractSparseVector{Tv,Ti}) where {T<:AbstractAllDensedSparseVector,Tv,Ti}
    sv = T{Tv,Ti}(length(V))
    for (i,d) in zip(nonzeroinds(V), nonzeros(V))
        sv[i] = d
    end
    return sv
end

"""
Convert any `AbstractSparseVector`s to particular `AbstractAllDensedSparseVector`:

    DensedSparseVector{Float64,Int}(sv)

"""
function (::Type{T})(V::AbstractSparseVector) where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    sv = T(length(V))
    for (i,d) in zip(nonzeroinds(V), nonzeros(V))
        sv[i] = d
    end
    return sv
end





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

Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = similar(V, Tv)
Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti}, ElType::Type) where {Tv,Ti} = similar(V, ElType, Ti)
function Base.similar(V::DensedSparseVector, Tvn::Type, Tin::Type)
    nzind = similar(V.nzind, Tin)
    nzchunks = similar(V.nzchunks)
    for (i, (k,d)) in enumerate(nzchunkspairs(V))
        nzind[i] = k
        nzchunks[i] = similar(d, Tvn)
    end
    return DensedSparseVector{Tvn,Tin}(length(V), nzind, nzchunks)
end
function Base.similar(V::DynamicDensedSparseVector, Tvn::Type, Tin::Type)
    nzchunks = SortedDict{Tin, Vector{Tvn}, FOrd}(Forward)
    for (k,d) in nzchunkspairs(V)
        nzchunks[k] = similar(d, Tvn)
    end
    return DynamicDensedSparseVector{Tvn,Tin}(length(V), nzchunks)
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
#@inline nnzchunks(V::SparseVector) = length(iteratenzchunks(V))
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
@inline nnzchunks(V::AbstractAllDensedSparseVector) = length(getfield(V, :nzind))
#@inline nnzchunks(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector} = foldl((s,c)->(s+1), nzchunks(V); init=0)
function nnzchunks(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector}
    length(V) == 0 && return 0
    idx1 = searchsortedlast_nzchunk(V.parent, first(V.indices[1]))
    idx2 = searchsortedlast_nzchunk(V.parent, last(V.indices[1]))
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
@inline get_nzchunk(V::Number, i) = V
@inline get_nzchunk(V::Vector, i) = V
@inline get_nzchunk(V::SparseVector, i) = view(nonzeros(V), i[1]:i[1]+i[2]-1)
@inline get_nzchunk(V::AbstractVecbasedDensedSparseVector, i) = V.nzchunks[i]
@inline get_nzchunk(V::FixedDensedSparseVector, i) = @view( V.nzchunks[ V.offsets[i]:V.offsets[i+1] - 1 ] )
@inline get_nzchunk(V::DynamicDensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = deref_value((V.nzchunks, i))
@inline function get_nzchunk(V::SubArray{<:Any,<:Any,<:T}, i) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    idx1 = first(V.indices[1])
    idx2 = last(V.indices[1])
    key, chunk = get_key_and_nzchunk(V.parent, i)
    len = Ti(length(chunk))
    if key <= idx1 < key+len && key <= idx2 < key+len
        return view(chunk, idx1-key+Ti(1):idx2-key+Ti(1))
    elseif key <= idx1 < key+len
        return @view(chunk[idx1-key+Ti(1):end])
    elseif key <= idx2 < key+len
        return view(chunk, Ti(1):(idx2-key+Ti(1)))
    elseif (key > idx1 && key > idx2) || (idx1 >= key+len && idx2 >= key+len)
        return @view(chunk[end:Ti(0)])
    else
        return @view(chunk[Ti(1):end])
    end
end
@inline get_nzchunk_key(V::Vector, i) = i
@inline get_nzchunk_key(V::SparseVector, i) = V.nzind[i]
@inline get_nzchunk_key(V::AbstractVecbasedDensedSparseVector, i) = V.nzind[i]
@inline get_nzchunk_key(V::DynamicDensedSparseVector, i) = deref_key((V.nzchunks, i))
@inline function get_nzchunk_key(V::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractVecbasedDensedSparseVector}
    if get_nzchunk_key(V.parent, i) <= first(V.indices[1]) < get_nzchunk_key(V.parent, i) + get_nzchunk_length(V.parent, i)
        return first(V.indices[1])
    else
        return get_nzchunk_key(V.parent, i) # CHECKME:
    end
end
@inline function get_nzchunk_key(V::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractSDictDensedSparseVector}
    key, chunk = deref((V.parent.nzchunks, i))
    len = length(chunk)
    if key <= first(V.indices[1]) < key + len
        return first(V.indices[1])
    else
        return key
    end
end
@inline get_key_and_nzchunk(V::Vector, i) = (i, V)
@inline get_key_and_nzchunk(V::SparseVector, i) = (V.nzind[i], view(V.nzchunks, i:i))
@inline get_key_and_nzchunk(V::AbstractVecbasedDensedSparseVector, i) = (V.nzind[i], V.nzchunks[i])
@inline get_key_and_nzchunk(V::FixedDensedSparseVector, i) = (V.nzind[i], @view(V.nzchunks[V.offsets[i]:V.offsets[i+1]-1]))
@inline get_key_and_nzchunk(V::DynamicDensedSparseVector, i) = deref((V.nzchunks, i))

@inline get_key_and_nzchunk(V::Vector) = (1, eltype(V)[])
@inline get_key_and_nzchunk(V::SparseVector) = (eltype(V.nzind)(1), view(V.nzchunks, 1:0))
@inline get_key_and_nzchunk(V::AbstractVecbasedDensedSparseVector) = (valtype(V.nzind)(1), valtype(V.nzchunks)())
@inline get_key_and_nzchunk(V::DynamicDensedSparseVector) = (keytype(V.nzchunks)(1), valtype(V.nzchunks)())

@inline getindex_nzchunk(V::AbstractVecbasedDensedSparseVector, chunk, i) = chunk[i]
@inline getindex_nzchunk(V::DynamicDensedSparseVector, chunk, i) = chunk[i]

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
            key = get_nzchunk_key(V, st)
            len = get_nzchunk_length(V, st)
            if i < key + len
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
    for (k,d) in nzchunkspairs(V)
        append!(ret, (k:k+length_of_that_nzchunk(V,d)-1))
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
    nnz(V.parent) == 0 && return nothing
    ifirst, ilast = first(V.indices[1]), last(V.indices[1])
    st = searchsortedlast_nzchunk(V.parent, ifirst)
    st == pastendnzchunk_index(V.parent) && return nothing
    key = get_nzchunk_key(V.parent, st)
    len = get_nzchunk_length(V.parent, st)
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
    nnz(V) > 0 ? V.nzind[end] + length_of_that_nzchunk(V, V.nzchunks[end]) - 1 : nothing
@inline function findlastnzindex(V::AbstractSDictDensedSparseVector)
    if nnz(V) > 0
        lasttoken = lastindex(V.nzchunks)
        return deref_key((V.nzchunks, lasttoken)) + length_of_that_nzchunk(V, deref_value((V.nzchunks, lasttoken))) - 1
    else
        return nothing
    end
end
function findlastnzindex(V::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(V.parent) == 0 && return nothing
    ifirst, ilast = first(V.indices[1]), last(V.indices[1])
    st = searchsortedfirst_nzchunk(V.parent, ilast)
    st == beforestartnzchunk_index(V.parent) && return nothing
    key = get_nzchunk_key(V.parent, st)
    len = get_nzchunk_length(V.parent, st)
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
    nnz(V.parent) == 0 && return nothing
    ifirst, ilast = first(V.indices[1]), last(V.indices[1])
    st = searchsortedlast_nzchunk(V.parent, ifirst)
    st == pastendnzchunk_index(V.parent) && return nothing
    key, chunk = get_key_and_nzchunk(V.parent, st)
    len = length_of_that_nzchunk(V.parent, chunk)
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
    nnz(V.parent) == 0 && return nothing
    ifirst, ilast = first(V.indices[1]), last(V.indices[1])
    st = searchsortedfirst_nzchunk(V.parent, ilast)
    st == beforestartnzchunk_index(V.parent) && return nothing
    key, chunk = get_key_and_nzchunk(V.parent, st)
    len = length_of_that_nzchunk(V.parent, chunk)
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

"`iteratenzchunks(V::AbstractVector)` iterates over non-zero chunks and returns start index of elements in chunk and chunk"
Base.@propagate_inbounds function iteratenzchunks(V::AbstractVecbasedDensedSparseVector, state = 1)
    if state <= length(V.nzind)
        return (state, state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(V::AbstractSDictDensedSparseVector, state = startof(V.nzchunks))
    if state != pastendsemitoken(V.nzchunks)
        stnext = advance((V.nzchunks, state))
        return (state, stnext)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector}
    state = length(V) > 0 ? searchsortedlast_nzchunk(V.parent, first(V.indices[1])) : pastendnzchunk_index(V.parent)
    return iteratenzchunks(V, state)
end
Base.@propagate_inbounds function iteratenzchunks(V::SubArray{<:Any,<:Any,<:T}, state) where {T<:AbstractAllDensedSparseVector}
    if state != pastendnzchunk_index(V.parent)
        key = get_nzchunk_key(V.parent, state)
        len = get_nzchunk_length(V.parent, state)
        if last(V.indices[1]) >= key + len
            return (state, advance(V.parent, state))
        elseif key <= last(V.indices[1]) < key + len
            return (state, advance(V.parent, state))
        else
            return nothing
        end
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzchunks(V::SparseVector)
    nn = nnz(V)
    return nn == 0 ? nothing : iteratenzchunks(V, 1)
end
Base.@propagate_inbounds function iteratenzchunks(V::SparseVector, state)
    nzinds = SparseArrays.nonzeroinds(V)
    N = length(nzinds)
    i1 = state
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
        return ((i1, len), i1+len)
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzchunks(V::Vector, state = (1, length(V)))
    i, len = state
    if len == 1
        return (1, state)
    elseif i == 1
        return (i, (i + 1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzchunks(V::Number, state = 1) = (1, state)

"`iteratenzpairs(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and value"
function iteratenzpairs end
"`iteratenzpairsview(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and `view` to value"
function iteratenzpairsview end
"`iteratenzvalues(V::AbstractAllDensedSparseVector)` iterates over non-zero elements of vector and returns value"
function iteratenzvalues end
"`iteratenzvaluesview(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and `view` of value"
function iteratenzvaluesview end
"`iteratenzindices(V::AbstractAllDensedSparseVector)` iterates over non-zero elements of vector and returns indices"
function iteratenzindices end

#
# iteratenzSOMEs() iterators for `Number`, `Vector` and `SparseVector`
#

Base.@propagate_inbounds function iteratenzpairs(V::SparseVector, state = (1, length(V.nzind)))
    i, len = state
    if i <= len
        return ((@inbounds V.nzind[i], @inbounds V.nzval[i]), (i+1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzpairs(V::Vector, state = 1)
    if state-1 < length(V)
        return ((state, @inbounds V[state]), state + 1)
        #return (Pair(state, @inbounds V[state]), state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzpairs(V::Number, state = 1) = ((state, V), state+1)

Base.@propagate_inbounds function iteratenzvalues(V::SparseVector, state = (1, length(V.nzind)))
    i, len = state
    if i <= len
        return (@inbounds V.nzval[i], (i+1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzvalues(V::Vector, state = 1)
    if state-1 < length(V)
        return (@inbounds V[state], state + 1)
    elseif length(V) == 1
        return (@inbounds V[1], state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzvalues(V::Number, state = 1) = (V, state+1)

Base.@propagate_inbounds function iteratenzindices(V::SparseVector, state = (1, length(V.nzind)))
    i, len = state
    if i-1 < len
        return (@inbounds V.nzind[i], (i+1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzindices(V::Vector, state = 1)
    if state-1 < length(V)
        return (state, state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzindices(V::Number, state = 1) = (state, state+1)


#
# `AbstractAllDensedSparseVector` iteration functions
#


struct ADSVIteratorState{Tn,Ti,Td}
    next::Tn         # index (Int or Semitoken) of next chunk
    nextpos::Int     # position in the current chunk of element will be get
    currentkey::Ti   # the index of first element in current chunk
    chunk::Td        # current chunk
    chunklen::Int    # current chunk length
end

@inline function ADSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where
                                          {T<:AbstractVecbasedDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    ADSVIteratorState{Int, Ti, Vector{Tv}}(next, nextpos, currentkey, chunk, chunklen)
end
@inline function ADSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where
                                          {T<:AbstractSDictDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    ADSVIteratorState{DataStructures.Tokens.IntSemiToken, Ti, Vector{Tv}}(next, nextpos, currentkey, chunk, chunklen)
end

# Start iterations from `i` index, i.e. `i` is `firstindex(V)`. Thats option for `SubArray`
function get_iterator_init_state(V::T, i::Integer = 1) where {T<:AbstractAllDensedSparseVector}
    st = searchsortedlast_nzchunk(V, i)
    if (ret = iteratenzchunks(V, st)) !== nothing
        idxchunk, next = ret
        key, chunk = get_key_and_nzchunk(V, idxchunk)
        if i > key # SubArray starts in middle of chunk
            return ADSVIteratorState{T}(next, i - key + 1, key, chunk, length_of_that_nzchunk(V, chunk))
        else
            return ADSVIteratorState{T}(next, 1, key, chunk, length_of_that_nzchunk(V, chunk))
        end
    else
        key, chunk = get_key_and_nzchunk(V)
        return ADSVIteratorState{T}(1, 1, key, chunk, 0)
    end
end

for (fn, ret1, ret2) in
        ((:iteratenzpairs    ,  :((Ti(key+nextpos-1), chunk[nextpos]))              , :((key, chunk[1]))         ),
         (:iteratenzpairsview,  :((Ti(key+nextpos-1), view(chunk, nextpos:nextpos))), :((key, view(chunk, 1:1))) ),
         #(:(Base.iterate)    ,  :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvalues   ,  :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvaluesview, :(view(chunk, nextpos:nextpos))                     , :(view(chunk, 1:1))        ),
         (:iteratenzindices  ,  :(Ti(key+nextpos-1))                                , :(key)                     ))

    @eval Base.@propagate_inbounds function $fn(V::T, state = get_iterator_init_state(V)) where
                                                {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Ti,Tv}
        next, nextpos, key, chunk, chunklen = fieldvalues(state)
        if nextpos <= chunklen
            return ($ret1, ADSVIteratorState{T}(next, nextpos + 1, key, chunk, chunklen))
        elseif (ret = iteratenzchunks(V, next)) !== nothing
            i, next = ret
            key, chunk = get_key_and_nzchunk(V, i)
            return ($ret2, ADSVIteratorState{T}(next, 2, key, chunk, length_of_that_nzchunk(V, chunk)))
        else
            return nothing
        end
    end
end


for (fn, ret1, ret2) in
        ((:iteratenzpairs    ,  :((Ti(key+nextpos-1-first(V.indices[1])+1), chunk[nextpos]))              ,
                                :((Ti(key-first(V.indices[1])+1), chunk[1]))                                 ),
         (:iteratenzpairsview,  :((Ti(key+nextpos-1-first(V.indices[1])+1), view(chunk, nextpos:nextpos))),
                                :((Ti(key-first(V.indices[1])+1), view(chunk, 1:1)))                         ),
         #(:(Base.iterate)    ,  :(chunk[nextpos])                                                         ,
         #                       :(chunk[1])                                                                  ),
         (:iteratenzvalues   ,  :(chunk[nextpos])                                                         ,
                                :(chunk[1])                                                                  ),
         (:iteratenzvaluesview, :(view(chunk, nextpos:nextpos))                                           ,
                                :(view(chunk, 1:1))                                                          ),
         (:iteratenzindices  ,  :(Ti(key+nextpos-1)-first(V.indices[1])+1)                                ,
                                :(Ti(key-first(V.indices[1])+1))                                             ))

    @eval Base.@propagate_inbounds function $fn(V::SubArray{<:Any,<:Any,<:T},
                                                state = get_iterator_init_state(V.parent, first(V.indices[1]))) where
                                                {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
        next, nextpos, key, chunk, chunklen = fieldvalues(state)
        if key+nextpos-1 > last(V.indices[1])
            return nothing
        elseif nextpos <= chunklen
            return ($ret1, ADSVIteratorState{T}(next, nextpos + 1, key, chunk, chunklen))
        elseif (ret = iteratenzchunks(V.parent, next)) !== nothing
            i, next = ret
            key, chunk = get_key_and_nzchunk(V.parent, i)
            return ($ret2, ADSVIteratorState{T}(next, 2, key, chunk, length_of_that_nzchunk(V.parent, chunk)))
        else
            return nothing
        end
    end
end

#
#  Iterators
#

struct NZChunks{It}
    itr::It
end
"`nzchunks(V::AbstractAllDensedSparseVector)` is the `Iterator` over chunks of nonzeros and
 returns tuple of start index and chunk vector"
@inline nzchunks(itr) = NZChunks(itr)
@inline function Base.iterate(it::NZChunks, state...)
    y = iteratenzchunks(it.itr, state...)
    if y !== nothing
        return (get_nzchunk(it.itr, y[1]), y[2])
    else
        return nothing
    end
end
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
 returns tuple of start index and vector of non-zero values."
@inline nzchunkspairs(itr) = NZChunksPairs(itr)
@inline function Base.iterate(it::NZChunksPairs, state...)
    y = iteratenzchunks(it.itr, state...)
    if y !== nothing
        return (Pair(get_key_and_nzchunk(it.itr, y[1])...), y[2])
    else
        return nothing
    end
end
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
@inline function Base.iterate(it::NZIndices, state...)
    y = iteratenzindices(it.itr, state...)
    if y !== nothing
        return (y[1], y[2])
    else
        return nothing
    end
end
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
@inline function Base.iterate(it::NZValues, state...)
    y = iteratenzvalues(it.itr, state...)
    if y !== nothing
        return (y[1], y[2])
    else
        return nothing
    end
end
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
@inline function Base.iterate(it::NZValuesView, state...)
    y = iteratenzvaluesview(it.itr, state...)
    if y !== nothing
        return (y[1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZValuesView{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZValuesView{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZValuesView}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZValuesView}) = 1
Base.length(it::NZValuesView) = nnz(it.itr)
Base.size(it::NZValuesView) = (nnz(it.itr),)
#Iterators.reverse(it::NZValuesView) = NZValuesView(Iterators.reverse(it.itr))


struct NZPairs{It}
    itr::It
end
"`nzpairs(V::AbstractVector)` is the `Iterator` over nonzeros of `V`, returns pair of index and value."
@inline nzpairs(itr) = NZPairs(itr)
@inline function Base.iterate(it::NZPairs, state...)
    y = iteratenzpairs(it.itr, state...)
    if y !== nothing
        return (Pair(y[1]...), y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZPairs{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZPairs{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZPairs}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZPairs}) = 1
Base.length(it::NZPairs) = nnz(it.itr)
Base.size(it::NZPairs) = (nnz(it.itr),)
#Iterators.reverse(it::NZPairs) = NZPairs(Iterators.reverse(it.itr))


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


@inline function Base.getindex(V::AbstractAllDensedSparseVector, i::Integer)
    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if ifirst <= i < ifirst + length_of_that_nzchunk(V, chunk)
            return getindex_nzchunk(V, chunk, i - ifirst + 1)
        end
    end
    # cached chunk index miss or index not stored
    st = searchsortedlast_nzind(V, i)
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if i < ifirst + length_of_that_nzchunk(V, chunk)  # is the index `i` inside of data chunk indices range
            V.lastusedchunkindex = st
            return getindex_nzchunk(V, chunk, i - ifirst + 1)
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



@inline function Base.setindex!(V::DensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = Tv(value)

    # fast check for cached chunk index
    if (st = V.lastusedchunkindex) != beforestartnzchunk_index(V)
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if ifirst <= i < ifirst + length(chunk)
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
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if ifirst <= i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = sv
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk = get_key_and_nzchunk(V, st)
        #ifirst, chunk = V.nzind[st], V.nzchunks[st]
        if i < ifirst + length(chunk)
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
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if ifirst <= i < ifirst + length(chunk)
            sv = chunk[i - ifirst + 1]
            chunk[i - ifirst + 1] = @set sv[j] = val
            return V
        end
    end

    st = searchsortedlast(V.nzind, i)

    # check the index exist and update its data
    if st != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if i < ifirst + length(chunk)
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
        ifirst, chunk = get_key_and_nzchunk(V, st)
        if ifirst <= i < ifirst + length(chunk)
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




function _fill_empty!(V::DensedSparseVector{Tv,Ti}, value) where {Tv,Ti}
    @assert isempty(V)
    resize!(V.nzind, 1)
    V.nzind[1] = firstindex(V)
    resize!(V.nzchunks, 1)
    V.nzchunks[1] = fill(Tv(value), length(V))
    V.nnz = length(V)
    return V
end
function _fill_empty!(V::DynamicDensedSparseVector{Tv,Ti}, value) where {Tv,Ti}
    @assert isempty(V)
    V.nzchunks[firstindex(V)] = fill(Tv(value), length(V))
    V.nnz = length(V)
    return V
end


function Base.fill!(V::AbstractAllDensedSparseVector{Tv,Ti}, value) where {Tv,Ti}
    if nnz(V) == length(V) != 0
        fill!(first(nzchunks(V)), Tv(value))
    else
        empty!(V)
        _fill_empty!(V, Tv(value))
    end
    V
end
function Base.fill!(V::SubArray{<:Any,<:Any,<:T}, value) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
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
    for (i, (k,chunk)) in enumerate(nzchunkspairs(A))
        if isassigned(C.nzchunks, i)
            resize!(C.nzchunks[i], length(chunk))
        else
            C.nzchunks[i] = Vector{Tv}(undef, length(chunk))
        end
        C.nzind[i] = Ti(k)
    end
    return C
end

function _similar_shape!(C::DensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
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

function _similar_shape!(C::DynamicDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
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

@inline _copy_chunk_to!(C::DensedSparseVector{Tv,Ti}, i, k, chunk) where {Tv,Ti} = (C.nzind[i] = Ti(k); C.nzchunks[i] .= Tv.(chunk))
@inline _copy_chunk_to!(C::DynamicDensedSparseVector{Tv,Ti}, i, k, chunk) where {Tv,Ti} = C.nzchunks[Ti(k)] .= Tv.(chunk)

function Base.copyto!(C::AbstractDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    _similar_shape!(C, A)
    for (i, (k,chunk)) in enumerate(nzchunkspairs(A))
        _copy_chunk_to!(C, i, k, chunk)
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


#
#  Broadcasting
#
# TODO: NZChunksStyle and NZValuesStyle
#

include("higherorderfns.jl")

include("show.jl")



end  # of module DensedSparseVectors
