
# TODO:
# * Introduce two macros: `@inzeros` and `@zeroscheck` like `@inbounds` and `@boundscheck`
#   to ommit sparse similarity checking and fixing. Then .zpbc field is not need.
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
# * Add SDSVIteratorState instead of .lastusedchunkindex to improve cached data locality. DONE!
#   May be stack of few SDSVIteratorState of previous assesses ranged by access frequency
#   or size of chunk?
#
# * Add pastendnzchunk_index in all AbstractAllDensedSparseVector
#   to have the fast iteration stop checking.
#
# * Add iterators like for MethodSpecializations in base/reflection.jl with
#   `iterate(specs::MethodSpecializations, ::Nothing) = nothing`
#   Then there are may be type stable even for Tuple/Vector of iterators.
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
export nzpairs, nzpairsview, nzblocks, nzvalues, nzvaluesview, nzindices, nzchunks, nzchunkspairs
export startindex
export rawindex, from_rawindex, rawindex_advance, rawindex_possible_advance, rawindex_compare, rawindex_view
export firstrawindex, lastrawindex, pastendrawindex
export nziterator, nziterator_advance, nziterator_possible_advance, nziterator_view
export firstnziterator, pastendnziterator
export findfirstnz, findlastnz, findfirstnzindex, findlastnzindex
export iterate_nzpairs, iterate_nzpairsview, iterate_nzvalues, iterate_nzvaluesview, iterate_nzindices
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


## TODO: use Espresso.jl and MacroTools.jl packages. And maybe FastBroadcast.jl as the base for @zeropresbc macro.
## IT'S IMPOSSIBE!!!
# Only one scenario is the save .zpbc props from DSVs from broadcast expression, store .zpbc=true and run expression.
# Afterwards returns the saved .zpbc property values.
#

# https://github.com/JuliaLang/julia/issues/39952
basetype(::Type{T}) where T = Base.typename(T).wrapper

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
const AbstractCompressedDensedSparseVector{Tv,Ti} = Union{AbstractSimpleDensedSparseVector{Tv,Ti}, AbstractDensedBlockSparseVector{Tv,Ti}}



# struct ChunkLastUsed{Ti,Td}
#     indices::UnitRange{Ti} # the indices of first and last elements in current chunk
#     chunk::Td              # current chunk is the view into nzchunk
# end
struct ChunkLastUsed{Ti,Tit}
    indices::UnitRange{Ti} # the indices of first and last elements in current chunk
    itchunk::Tit          # nzchunk position state (Int or Semitoken) in nzchunks
end
# struct BlockChunkLastUsed{Ti,Tit}
#     indices::UnitRange{Ti}       # the indices of first and last elements in current chunk
#     offsetrange::UnitRange{Int}  # offsets range: offsets[first(indices)]:offsets[last(indices)]-1
#     itchunk::Tit                # nzchunk position state (Int or Semitoken) in nzchunks
# end

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
    lastused::ChunkLastUsed{Ti,Int}
    "Storage for indices of the non-zero chunks"
    nzranges::Vector{UnitRange{Ti}}  # Vector of chunk's indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s"
    nzchunks::Vector{Vector{Tv}}
    "Vector length"
    n::Ti
    #"Vector range, `firstindex(V) = first(V.axes1)` and so on"
    # see https://github.com/JuliaArrays/CustomUnitRanges.jl
    #axes1::UnitRange{Ti}
    "Number of stored non-zero elements"
    nnz::Int
    "Zero Preserve Broadcast, by default false. If true, then no new elements inserts are introduced during broadcast."
    zpbc::Bool

    DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
        new{Tv,Ti}(lostused(Ti,Int), Vector{UnitRange{Ti}}(), Vector{Vector{Tv}}(), n, 0, false)

    DensedSparseVector{Tv,Ti}(n::Integer, nzranges, nzchunks) where {Tv,Ti} =
        new{Tv,Ti}(lostused(Ti,Int), nzranges, nzchunks, n, foldl((s,c)->(s+length(c)), nzchunks; init=0), false)

end


DensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = DensedSparseVector{Tv,Ti}(V)

function DensedSparseVector{Tv,Ti}(V::AbstractAllDensedSparseVector) where {Tv,Ti}
    nzranges = Vector{UnitRange{Ti}}(undef, nnzchunks(V))
    nzchunks = Vector{Vector{Tv}}(undef, length(nzranges))
    for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
        nzranges[itc] = UnitRange{Ti}(ids)
        nzchunks[itc] = Vector{Tv}(d)
    end
    return DensedSparseVector{Tv,Ti}(length(V), nzranges, nzchunks)
end

#"View for DensedSparseVector"
#struct DensedSparseVectorView{Tv,Ti,T,Tc} <: AbstractCompressedDensedSparseVector{Tv,Ti}
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
mutable struct FixedDensedSparseVector{Tv,Ti} <: AbstractSimpleDensedSparseVector{Tv,Ti}
    "Index of last used chunk" # TODO: it may be immutable struct
    lastused::ChunkLastUsed{Ti,Int}
    "Storage for indices of the first element of non-zero chunks"
    nzranges::Vector{UnitRange{Ti}}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s"
    nzchunks::Vector{Tv}
    "Offsets of starts of vestors in `nzchunks` like in CSC matrix structure"
    offsets::Vector{Int}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int
    "Zero Preserve Broadcast, by default false. If true, then no new elements inserts are introduced during broadcast."
    zpbc::Bool

    FixedDensedSparseVector{Tv,Ti}(n::Integer, nzranges, nzchunks, offsets) where {Tv,Ti} =
        new{Tv,Ti}(lostused(Ti,Int), nzranges, nzchunks, offsets, n, length(nzchunks), true)
end


FixedDensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = FixedDensedSparseVector{Tv,Ti}(V)

function FixedDensedSparseVector{Tv,Ti}(V::AbstractAllDensedSparseVector) where {Tv,Ti}
    nzranges = Vector{UnitRange{Ti}}(undef, nnzchunks(V))
    nzchunks = Vector{Tv}(undef, nnz(V))
    offsets = Vector{Int}(undef, nnzchunks(V)+1)
    offsets[1] = 1
    for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
        nzranges[itc] = UnitRange{Ti}(ids)
        offsets[itc+1] = offsets[itc] + length(d)
        @view(nzchunks[offsets[itc]:offsets[itc+1]-1]) .= Tv.(d)
    end
    return FixedDensedSparseVector{Tv,Ti}(length(V), nzranges, nzchunks, offsets)
end



"""
The `DensedSVSparseVector` is the version of `DensedSparseVector` with `SVector` as elements
and alike `Matrix` with sparse first dimension and with dense `SVector` in second dimension.
See `DensedSparseVector` for details.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSVSparseVector{Tv,Ti,m} <: AbstractDensedBlockSparseVector{Tv,Ti} # TODO: is it should be AbstractSimple...
    "Index of last used chunk"
    lastused::ChunkLastUsed{Ti,Int}
    "Storage for indices of the first element of non-zero chunks"
    nzranges::Vector{UnitRange{Ti}}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    nzchunks::Vector{Vector{SVector{m,Tv}}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int
    "Zero Preserve Broadcast, by default false. If true, then no new elements inserts are introduced during broadcast."
    zpbc::Bool

    DensedSVSparseVector{Tv,Ti,m}(n::Integer, nzranges, nzchunks) where {Tv,Ti,m} =
        new{Tv,Ti,m}(lostused(Ti,Int), nzranges, nzchunks, n, foldl((s,c)->(s+length(c)), nzchunks; init=0), false)

    DensedSVSparseVector{Tv,Ti,m}(n::Integer = 0) where {Tv,Ti,m} =
        new{Tv,Ti,m}(lostused(Ti,Int), Vector{UnitRange{Ti}}(), Vector{Vector{Tv}}(), n, 0, false)
end

DensedSVSparseVector{Tv,Ti}(m::Integer, n::Integer = 0) where {Tv,Ti} = DensedSVSparseVector{Tv,Ti,m}(n)
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
    lastused::ChunkLastUsed{Ti,Int}
    "Storage for indices of the first element of non-zero chunks"
    nzranges::Vector{UnitRange{Ti}}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    nzchunks::Vector{Vector{Tv}}
    "Offsets of starts of variable length vestors in `nzchunks`"
    offsets::Vector{Vector{Int}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int
    "Zero Preserve Broadcast, by default false. If true, then no new elements inserts are introduced during broadcast."
    zpbc::Bool
    "Dummy for empty `getindex` returns"
    dummy::Vector{Tv}

    DensedVLSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
        new{Tv,Ti}(lostused(Ti,Int), Vector{UnitRange{Ti}}(), Vector{Vector{Tv}}(), Vector{Vector{Int}}(), n, 0, false, Tv[])

    DensedVLSparseVector{Tv,Ti}(n::Integer, nzranges::AbstractVector{TR}, nzchunks::AbstractVector, offsets::AbstractVector) where {TR<:UnitRange,Tv,Ti} =
        new{Tv,Ti}(lostused(Ti,Int), nzranges, nzchunks, offsets, n, foldl((s,c)->(s+length(c)-1), offsets; init=0), false, Tv[])

    DensedVLSparseVector{Tv,Ti}(n::Integer, ifirsts::AbstractVector, nzchunks::AbstractVector, offsets::AbstractVector) where {Tv,Ti} =
        new{Tv,Ti}(lostused(Ti,Int), [UnitRange{Ti}(ifirsts[i],ifirsts[i]+(length(offsets[i])-1)-1) for i=1:length(ifirsts)],
                   nzchunks, offsets, n, foldl((s,c)->(s+length(c)-1), offsets; init=0), false, Tv[])
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
    lastused::ChunkLastUsed{Ti,DataStructures.Tokens.IntSemiToken}
    "Storage for indices of the first element of non-zero chunks and corresponding chunks as `SortedDict(Int=>Vector)`"
    nzchunks::SortedDict{Ti,Vector{Tv},FOrd}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int
    "Zero Preserve Broadcast, by default false. If true, then no new elements inserts are introduced during broadcast."
    zpbc::Bool

    function DynamicDensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
        nzchunks = SortedDict{Ti,Vector{Tv},FOrd}(Forward)
        new{Tv,Ti}(lostused(Ti,beforestartsemitoken(nzchunks)), nzchunks, n, 0, false)
    end

    DynamicDensedSparseVector{Tv,Ti}(n::Integer, nzchunks::SortedDict{K,V}) where {Tv,Ti,K,V<:AbstractVector} =
        new{Tv,Ti}(lostused(Ti,beforestartsemitoken(nzchunks)), nzchunks, n, foldl((s,c)->(s+length(c)), values(nzchunks); init=0), false)

end

#DynamicDensedSparseVector(n::Integer = 0) = DynamicDensedSparseVector{Float64,Int}(n)

DynamicDensedSparseVector(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = DynamicDensedSparseVector{Tv,Ti}(V)
function DynamicDensedSparseVector{Tv,Ti}(V::AbstractAllDensedSparseVector) where {Tv,Ti}
    nzchunks = SortedDict{Ti, Vector{Tv}, FOrd}(Forward)
    for (ids,d) in nzchunkspairs(V)
        nzchunks[first(ids)] = Vector{Tv}(d)
    end
    return DynamicDensedSparseVector{Tv,Ti}(length(V), nzchunks)
end

#=
@inline lastused(V::AbstractAllDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
    ChunkLastUsed{Ti,Vector{Tv}}(get_indices_and_nzchunk(V, itc)...)
@inline lastused(V::AbstractAllDensedSparseVector{Tv,Ti}, indices, chunk) where {Tv,Ti} =
    ChunkLastUsed{Ti,Vector{Tv}}(indices, chunk)
@inline lostused(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    ChunkLastUsed{Ti,Vector{Tv}}(UnitRange{Ti}(Ti(1),Ti(0)), Tv[])
@inline lostused(::Type{Ti}, ::Type{Tv}) where {Ti,Tv} =
    ChunkLastUsed{Ti,Vector{Tv}}(UnitRange{Ti}(Ti(1),Ti(0)), Tv[])
=#

@inline lastused(V::AbstractAllDensedSparseVector{Tv,Ti}, itc::Tit) where {Tv,Ti,Tit} =
    ChunkLastUsed{Ti,Tit}(get_nzchunk_indices(V, itc), itc)
@inline lastused(::AbstractAllDensedSparseVector{Tv,Ti}, indices::UnitRange, itc::Tit) where {Tv,Ti,Tit} =
    ChunkLastUsed{Ti,Tit}(indices, itc)
@inline lostused(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    ChunkLastUsed{Ti,Int}(UnitRange{Ti}(Ti(1),Ti(0)), beforestartnzchunk_index(V))
@inline lostused(V::DynamicDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    ChunkLastUsed{Ti,DataStructures.Tokens.IntSemiToken}(UnitRange{Ti}(Ti(1),Ti(0)), beforestartnzchunk_index(V))
@inline lostused(::Type{Ti}, ::Type{Tit}) where {Ti,Tit<:Integer} =
    ChunkLastUsed{Ti,Tit}(UnitRange{Ti}(Ti(1),Ti(0)), 0)
@inline lostused(::Type{Ti}, itc::Tit) where {Ti,Tit<:DataStructures.Tokens.IntSemiToken} =
    ChunkLastUsed{Ti,Tit}(UnitRange{Ti}(Ti(1),Ti(0)), itc)

@inline set_lastused!(V::AbstractAllDensedSparseVector{Tv,Ti}, itc::Tit) where {Tv,Ti,Tit} =
    (V.lastused = ChunkLastUsed{Ti,Tit}(get_nzchunk_indices(V, itc), itc); return nothing)
@inline set_lastused!(V::AbstractAllDensedSparseVector{Tv,Ti}, indices::UnitRange, itc::Tit) where {Tv,Ti,Tit} =
    (V.lastused = ChunkLastUsed{Ti,Tit}(indices, itc); return nothing)
@inline set_lostused!(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    (V.lastused = ChunkLastUsed{Ti,Int}(UnitRange{Ti}(Ti(1),Ti(0)), beforestartnzchunk_index(V)); return nothing)
@inline set_lostused!(V::DynamicDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    (V.lastused = ChunkLastUsed{Ti,DataStructures.Tokens.IntSemiToken}(UnitRange{Ti}(Ti(1),Ti(0)), beforestartnzchunk_index(V)); return nothing)

# @inline lastused(V::DensedVLSparseVector{Tv,Ti}, itc::Tit, i = 1) where {Tv,Ti,Tit} =
#     BlockChunkLastUsed{Ti,Tit}(get_nzchunk_indices(V, itc), get_nzchunk_offsets(V, itc, i), itc)
# @inline lostused(V::DensedVLSparseVector{Tv,Ti}) where {Tv,Ti} =
#     BlockChunkLastUsed{Ti,Int}(UnitRange{Ti}(Ti(1),Ti(0)), UnitRange{Int}(1,0), beforestartnzchunk_index(V))
# @inline blocklostused(::Type{DensedVLSparseVector{Tv,Ti}}) where {Tv,Ti} =
#     BlockChunkLastUsed{Ti,Int}(UnitRange{Ti}(Ti(1),Ti(0)), UnitRange{Int}(1,0), 0)
# @inline blocklostused(::Type{Ti}, ::Type{Tit}) where {Ti,Tit<:Integer} =
#     BlockChunkLastUsed{Ti,Tit}(UnitRange{Ti}(Ti(1),Ti(0)), UnitRange{Tit}(1,0), 0)



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


function (::Type{T})(V::DenseVector) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    dsv = T(length(V))
    for (i,d) in enumerate(V)
        dsv[i] = d
    end
    return dsv
    #nzranges = ones(Ti, 1)
    #nzchunks = Vector{Vector{Tv}}(undef, length(nzranges))
    #nzchunks[1] = Vector{Tv}(V)
    #return DensedSparseVector{Tv,Ti}(length(V), nzranges, nzchunks)
end


is_broadcast_zero_preserve(V::AbstractAllDensedSparseVector) = V.zpbc

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
# Base.IndexStyle(::AbstractAllDensedSparseVector) = IndexLinear()
Base.IndexStyle(::AbstractAllDensedSparseVector) = IndexCartesian() #?

Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = similar(V, Tv, Ti)
Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti}, ::Type{TvNew}) where {Tv,Ti,TvNew} = similar(V, TvNew, Ti)
Base.similar(V::AbstractAllDensedSparseVector{Tv,Ti}, ::Type{TvNew}, ::Type{TiNew}) where {Tv,Ti,TvNew,TiNew} = similar(V, TvNew, TiNew)

function Base.similar(V::DensedSparseVector, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    nzranges = similar(V.nzranges, UnitRange{TiNew})
    nzchunks = similar(V.nzchunks)
    for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
        nzranges[itc] = UnitRange{TiNew}(ids)
        nzchunks[itc] = similar(d, TvNew)
    end
    return DensedSparseVector{TvNew,TiNew}(length(V), nzranges, nzchunks)
end
function Base.similar(V::FixedDensedSparseVector, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    nzranges = Vector{UnitRange{TiNew}}(V.nzranges)
    nzchunks = similar(V.nzchunks, TvNew)
    offsets = Vector{Int}(V.offsets)
    return FixedDensedSparseVector{TvNew,TiNew}(length(V), nzranges, nzchunks, offsets)
end
function Base.similar(V::DynamicDensedSparseVector, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    nzchunks = SortedDict{TiNew, Vector{TvNew}, FOrd}(Forward)
    for (ids,d) in nzchunkspairs(V)
        nzchunks[first(ids)] = similar(d, TvNew)
    end
    return DynamicDensedSparseVector{TvNew,TiNew}(length(V), nzchunks)
end
function Base.similar(V::DensedSVSparseVector{Tv,Ti,m}, ::Type{TvNew}, ::Type{TiNew}) where {Tv,Ti,m,TvNew,TiNew}
    nzranges = similar(V.nzranges, UnitRange{TiNew})
    nzchunks = similar(V.nzchunks)
    for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
        nzranges[itc] = ids
        nzchunks[itc] = [SVector(ntuple(_->TvNew(0), m)) for _ in d]
    end
    return DensedSVSparseVector{TvNew,TiNew,m}(length(V), nzranges, nzchunks)
end
function Base.similar(V::DensedVLSparseVector, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    nzranges = similar(V.nzranges, UnitRange{TiNew})
    nzchunks = Vector{Vector{TvNew}}(undef, length(V.nzchunks))
    offsets = deepcopy(V.offsets)
    for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
        nzranges[itc] = ids
        nzchunks[itc] = Vector{TvNew}(undef, length(d))
    end
    return DensedVLSparseVector{TvNew,TiNew}(length(V), nzranges, nzchunks, offsets)
end


function Base.copy(V::T) where {T<:Union{DensedSparseVector,DensedSVSparseVector}}
    nzranges = copy(V.nzranges)
    nzchunks = copy(V.nzchunks)
    for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
        nzranges[itc] = ids
        nzchunks[itc] = copy(d)
    end
    return T(length(V), nzranges, nzchunks)
end
function Base.copy(V::T) where {T<:DensedVLSparseVector}
    nzranges = copy(V.nzranges)
    nzchunks = copy(V.nzchunks)
    offsets = copy(V.offsets)
    # for (itc, (ids,d)) in enumerate(nzchunkspairs(V))
    for (itc, (ids,d,o)) in enumerate(zip(V.nzranges, V.nzchunks, V.offsets))
        nzranges[itc] = ids
        nzchunks[itc] = copy(d)
        offsets[itc] = copy(o)
    end
    return T(length(V), nzranges, nzchunks, offsets)
end
Base.copy(V::T) where {T<:FixedDensedSparseVector} = T(length(V), copy(V.nzranges), copy(V.nzchunks), copy(V.offsets))
function Base.copy(V::DynamicDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzchunks = SortedDict{Ti, Vector{Tv}, FOrd}(Forward)
    for (ids,d) in nzchunkspairs(V)
        nzchunks[first(ids)] = copy(d)
    end
    return DynamicDensedSparseVector{Tv,Ti}(length(V), nzchunks)
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
@inline nnzchunks(V::FixedDensedSparseVector) = length(getfield(V, :nzranges))
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
@inline length_of_that_nzchunk(V::AbstractCompressedDensedSparseVector, chunk) = length(chunk) # TODO: Is it need?
@inline length_of_that_nzchunk(V::DynamicDensedSparseVector, chunk) = length(chunk)
@inline get_nzchunk_length(V::AbstractCompressedDensedSparseVector, itc) = size(V.nzchunks[itc])[1]
@inline get_nzchunk_length(V::FixedDensedSparseVector, itc) = V.offsets[itc+1] - V.offsets[itc]
@inline get_nzchunk_length(V::DensedVLSparseVector, itc) = length(V.offsets[itc]) - 1
@inline get_nzchunk_length(V::DynamicDensedSparseVector, itc::DataStructures.Tokens.IntSemiToken) = size(deref_value((V.nzchunks, itc)))[1]
@inline get_nzchunk_length(V::SubArray{<:Any,<:Any,<:T}, itc) where {T<:AbstractAllDensedSparseVector} = length(get_nzchunk(V, itc))
@inline get_nzchunk(V::Number, i) = Ref(V)
@inline get_nzchunk(V::Vector, i) = V
@inline get_nzchunk(V::SparseVector, i) = @inbounds view(nonzeros(V), i[1]:i[1]+i[2]-1)
@inline get_nzchunk(V::AbstractCompressedDensedSparseVector, itc) = @inbounds V.nzchunks[itc]
@inline get_nzchunk(V::FixedDensedSparseVector, itc) = @inbounds @view( V.nzchunks[ V.offsets[itc]:V.offsets[itc+1] - 1 ] )
@inline get_nzchunk(V::DynamicDensedSparseVector, itc::DataStructures.Tokens.IntSemiToken) = @inbounds deref_value((V.nzchunks, itc))
###@inline function get_nzchunk(V::SubArray{<:Any,<:Any,<:T}, itc) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
###    idx1 = first(parentindices(V)[1])
###    idx2 = last(parentindices(V)[1])
###    key, chunk = get_key_and_nzchunk(parent(V), itc)
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
@inline function get_nzchunk(V::SubArray{<:Any,<:Any,<:T}, itc) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    idx1 = first(parentindices(V)[1])
    idx2 = last(parentindices(V)[1])
    indices, chunk = get_indices_and_nzchunk(parent(V), itc)
    index1 = first(indices)
    index2 = last(indices)
    if checkindex(Bool, indices, idx1) && checkindex(Bool, indices, idx2)
        return @inbounds view(chunk, idx1-index1+Ti(1):idx2-index1+Ti(1))
    elseif checkindex(Bool, indices, idx1)
        return @inbounds @view(chunk[idx1-index1+Ti(1):end])
    elseif checkindex(Bool, indices, idx2)
        return @inbounds view(chunk, Ti(1):(idx2-index1+Ti(1)))
    elseif (idx1 < index1 && idx2 < index1) || (idx1 > index2 && idx2 > index2)
        return @inbounds @view(chunk[end:Ti(0)])
    else
        return @inbounds @view(chunk[Ti(1):end])
    end
end
@inline get_nzchunk_key(::Vector, i) = i
@inline get_nzchunk_key(V::SparseVector, i) = V.nzind[i]
@inline get_nzchunk_key(V::AbstractCompressedDensedSparseVector, itc) = first(V.nzranges[itc])
@inline get_nzchunk_key(V::DynamicDensedSparseVector, itc) = deref_key((V.nzchunks, itc))
@inline function get_nzchunk_key(V::SubArray{<:Any,<:Any,<:T}, itc) where {T<:AbstractAllDensedSparseVector}
    indices = get_nzchunk_indices(parent(V), itc)
    if checkindex(Bool, indices, first(parentindices(V)[1]))
        return first(parentindices(V)[1])
    else
        return key # FIXME:
    end
end

@inline get_nzchunk_indices(V::Vector, i) = UnitRange{Int}(1, length(V))
@inline get_nzchunk_indices(V::SparseVector{Tv,Ti}, i) where {Tv,Ti} = @inbounds UnitRange{Ti}(V.nzind[i], V.nzind[i]) # FIXME:
@inline get_nzchunk_indices(V::AbstractCompressedDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} = @inbounds V.nzranges[itc]
@inline function get_nzchunk_offsets(V::AbstractCompressedDensedSparseVector{Tv,Ti}, itc, i) where {Tv,Ti}
    ifirst = @inbounds first(V.nzranges[itc])
    @inbounds V.offsets[itc][i-ifirst+1]:V.offsets[itc][i-ifirst+1+1]-1
end
@inline get_nzchunk_indices(V::DynamicDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
    ((key, chunk) = deref((V.nzchunks, itc));
     return UnitRange{Ti}(key, key+length(chunk)-1))
@inline function get_nzchunk_indices(V::SubArray{<:Any,<:Any,<:T}, itc) where {Tv,Ti,T<:AbstractAllDensedSparseVector{Tv,Ti}}
    idx1 = first(parentindices(V)[1])
    idx2 = last(parentindices(V)[1])
    indices = get_nzchunk_indices(parent(V), itc)
    index1 = first(indices)
    index2 = last(indices)
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
@inline get_key_and_nzchunk(V::Vector, i) = (i, V)
@inline get_key_and_nzchunk(V::SparseVector, i) = (V.nzind[i], view(V.nzchunks, i:i)) # FIXME:
@inline get_key_and_nzchunk(V::AbstractCompressedDensedSparseVector, itc) = @inbounds (first(V.nzranges[itc]), V.nzchunks[itc])
@inline get_key_and_nzchunk(V::FixedDensedSparseVector, itc) = @inbounds (first(V.nzranges[itc]), @view(V.nzchunks[V.offsets[itc]:V.offsets[itc+1]-1]))
@inline get_key_and_nzchunk(V::DynamicDensedSparseVector, itc) =
    ((key, chunk) = deref((V.nzchunks, itc));
     return (key, chunk))

@inline get_key_and_nzchunk(V::Vector) = (1, eltype(V)[])
@inline get_key_and_nzchunk(::SparseVector{Tv,Ti}) where {Tv,Ti} = (Ti(1), Tv[])
@inline get_key_and_nzchunk(::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} = (Ti(1), Tv[])

@inline get_indices_and_nzchunk(V::Vector, i) = (i:i, V)
@inline get_indices_and_nzchunk(V::SparseVector, i) = @inbounds (V.nzind[i]:V.nzind[i], view(V.nzchunks, i:i)) # FIXME:
@inline get_indices_and_nzchunk(V::AbstractCompressedDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
    @inbounds (V.nzranges[itc], V.nzchunks[itc])
@inline get_indices_and_nzchunk(V::FixedDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
     @inbounds (V.nzranges[itc], @view(V.nzchunks[V.offsets[itc]:V.offsets[itc+1]-1]))
@inline get_indices_and_nzchunk(V::DynamicDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
    ((key, chunk) = deref((V.nzchunks, itc));
     return (UnitRange{Ti}(key, key+length(chunk)-1), chunk))

@inline get_indices_and_nzchunk(V::Vector) = (UnitRange(length(V)+1,length(V)), eltype(V)[])
@inline get_indices_and_nzchunk(V::SparseVector{Tv,Ti}) where {Tv,Ti} = (UnitRange{Ti}(length(V)+1,length(V)), Tv[])
@inline get_indices_and_nzchunk(V::AbstractAllDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    (UnitRange{Ti}(length(V)+1,length(V)), Tv[])

@inline get_indices(V::AbstractCompressedDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
    @inbounds V.nzranges[itc]
@inline get_indices(V::DynamicDensedSparseVector{Tv,Ti}, itc) where {Tv,Ti} =
    ((key, chunk) = deref((V.nzchunks, itc));
    return UnitRange{Ti}(key, key+length(chunk)-1))

@inline get_key_and_nzchunk_and_length(V::Vector, i) = (i, V, length(V))
@inline get_key_and_nzchunk_and_length(V::SparseVector, i) = (V.nzind[i], view(V.nzchunks, i:i), 1)
@inline get_key_and_nzchunk_and_length(V::AbstractCompressedDensedSparseVector, itc) = @inbounds (first(V.nzranges[itc]), V.nzchunks[itc], length(V.nzranges[itc]))
@inline get_key_and_nzchunk_and_length(V::FixedDensedSparseVector, itc) =
    (first(V.nzranges[itc]), @view(V.nzchunks[V.offsets[itc]:V.offsets[itc+1]-1]), length(V.nzranges[itc]))
@inline get_key_and_nzchunk_and_length(V::DynamicDensedSparseVector, itc) = ((key, chunk) = deref((V.nzchunks, itc)); return (key, chunk, length(chunk)))

@inline get_cached(V::AbstractAllDensedSparseVector) = @inbounds (V.lastused.indices, V.lastused.itchunk)
@inline get_cached_indices(V::AbstractAllDensedSparseVector) = @inbounds V.lastused.indices
@inline get_cached_itchunk(V::AbstractAllDensedSparseVector) = @inbounds V.lastused.itchunk

@inline is_in_nzchunk(V::Vector, i, key) = key in first(axes(V))
@inline is_in_nzchunk(V::SparseVector, i, key) = V.nzind[i] == key
@inline is_in_nzchunk(V::AbstractCompressedDensedSparseVector, itc, key) = key in V.nzranges[itc]
@inline is_in_nzchunk(V::FixedDensedSparseVector, itc, key) = key in V.nzranges[itc]
@inline is_in_nzchunk(V::DynamicDensedSparseVector, itc, key) = ((ichunk, chunk) = deref((V.nzchunks, itc)); return (ichunk <= key < ichunk + length(chunk)))

@inline firstnzchunk_index(V::SparseVector) = firstindex(V.nzind)
@inline firstnzchunk_index(V::AbstractCompressedDensedSparseVector) = firstindex(V.nzranges)
@inline firstnzchunk_index(V::AbstractSDictDensedSparseVector) = startof(V.nzchunks)
@inline lastnzchunk_index(V::SparseVector) = lastindex(V.nzind)
@inline lastnzchunk_index(V::AbstractCompressedDensedSparseVector) = lastindex(V.nzranges) # getfield(V, :n)
@inline lastnzchunk_index(V::AbstractSDictDensedSparseVector) = lastindex(V.nzchunks)

@inline beforestartnzchunk_index(V::SparseVector) = firstnzchunk_index(V) - 1
@inline beforestartnzchunk_index(V::AbstractCompressedDensedSparseVector) = firstnzchunk_index(V) - 1
@inline beforestartnzchunk_index(V::AbstractSDictDensedSparseVector) = beforestartsemitoken(V.nzchunks)
@inline pastendnzchunk_index(V::SparseVector) = lastnzchunk_index(V) + 1
@inline pastendnzchunk_index(V::AbstractCompressedDensedSparseVector) = lastnzchunk_index(V) + 1
@inline pastendnzchunk_index(V::AbstractSDictDensedSparseVector) = pastendsemitoken(V.nzchunks)

@inline returnzero(V::DensedSVSparseVector) = zero(eltype(eltype(V.nzchunks)))
@inline returnzero(V::AbstractAllDensedSparseVector) = zero(eltype(V))

@inline DataStructures.advance(::AbstractCompressedDensedSparseVector, state) = state + 1
@inline DataStructures.advance(V::AbstractSDictDensedSparseVector, state) = advance((V.nzchunks, state))
@inline DataStructures.regress(::AbstractCompressedDensedSparseVector, state) = state - 1
@inline DataStructures.regress(V::AbstractSDictDensedSparseVector, state) = regress((V.nzchunks, state))

"`searchsortedlast(V.nzranges, i)`"
@inline searchsortedlast_ranges(V::AbstractCompressedDensedSparseVector, i) = searchsortedlast(V.nzranges, i, by=first)
@inline searchsortedlast_ranges(V::AbstractSDictDensedSparseVector, i) = searchsortedlast(V.nzchunks, i)

"""
Returns nzchunk_index which on vector index `i`, or after `i`.
Slightly differs from `searchsortedfirst(V.nzranges)`.
"""
@inline function searchsortedlast_nzchunk(V::AbstractAllDensedSparseVector, i::Integer)
    if i == 1 # most of use cases
        return nnz(V) == 0 ? pastendnzchunk_index(V) : firstnzchunk_index(V)
    elseif nnz(V) != 0
        itc = searchsortedlast_ranges(V, i)
        if itc != beforestartnzchunk_index(V)
            if is_in_nzchunk(V, itc, i)
                return itc
            else
                return advance(V, itc)
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
Slightly differs from `searchsortedlast(V.nzranges)`.
"""
@inline function searchsortedfirst_nzchunk(V::AbstractAllDensedSparseVector, i::Integer)
    return searchsortedlast_ranges(V, i)
    #=
    if nnz(V) != 0
        return searchsortedlast_ranges(V, i)
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


#SparseArrays.findnz(V::AbstractAllDensedSparseVector) = (nzindices(V), nzvalues(V))
SparseArrays.findnz(V::AbstractAllDensedSparseVector) = (nonzeroinds(V), nonzeros(V))



"Returns the index of first non-zero element in sparse vector."
@inline findfirstnzindex(V::SparseVector) = nnz(V) > 0 ? V.nzind[1] : nothing
@inline findfirstnzindex(V::AbstractCompressedDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    nnz(V) > 0 ? first(V.nzranges[1]) : nothing
@inline findfirstnzindex(V::AbstractSDictDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    nnz(V) > 0 ? Ti(deref_key((V.nzchunks, startof(V.nzchunks)))) : nothing
function findfirstnzindex(V::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(parent(V)) == 0 && return nothing
    ifirst, ilast = first(parentindices(V)[1]), last(parentindices(V)[1])
    itc = searchsortedlast_nzchunk(parent(V), ifirst)
    itc == pastendnzchunk_index(parent(V)) && return nothing
    key = get_nzchunk_key(parent(V), itc)
    len = get_nzchunk_length(parent(V), itc)
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
@inline findlastnzindex(V::AbstractCompressedDensedSparseVector) =
    nnz(V) > 0 ? last(V.nzranges[end]) : nothing
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
    itc = searchsortedfirst_nzchunk(parent(V), ilast)
    itc == beforestartnzchunk_index(parent(V)) && return nothing
    key = get_nzchunk_key(parent(V), itc)
    len = get_nzchunk_length(parent(V), itc)
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
    itc = searchsortedlast_nzchunk(parent(V), ifirst)
    itc == pastendnzchunk_index(parent(V)) && return nothing
    key, chunk, len = get_key_and_nzchunk_and_length(parent(V), itc)
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
    itc = searchsortedfirst_nzchunk(parent(V), ilast)
    itc == beforestartnzchunk_index(parent(V)) && return nothing
    key, chunk, len = get_key_and_nzchunk_and_length(parent(V), itc)
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

Base.@propagate_inbounds function iterate_nzchunks(V::AbstractCompressedDensedSparseVector, state = 0)
    state += 1
    if state <= length(V.nzranges)
        return (state, state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzchunks(V::AbstractSDictDensedSparseVector, state = beforestartsemitoken(V.nzchunks))
    state = advance((V.nzchunks, state))
    if state != pastendsemitoken(V.nzchunks)
        return (state, state)
    else
        return nothing
    end
end

"`iterate_nzchunkspairs(V::AbstractVector)` iterates over non-zero chunks and returns indices of elements in chunk and chunk"
Base.@propagate_inbounds function iterate_nzchunkspairs(V::AbstractCompressedDensedSparseVector, state = 0)
    state += 1
    if state <= length(V.nzranges)
        return (Pair(get_indices_and_nzchunk(V, state)...), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzchunkspairs(V::AbstractSDictDensedSparseVector, state = beforestartsemitoken(V.nzchunks))
    state = advance((V.nzchunks, state))
    if state != pastendsemitoken(V.nzchunks)
        return (Pair(get_indices_and_nzchunk(V, state)...), state)
    else
        return nothing
    end
end

Base.@propagate_inbounds function iterate_nzchunkspairs(V::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAllDensedSparseVector}
    state = length(V) > 0 ? regress(parent(V), searchsortedlast_nzchunk(parent(V), first(parentindices(V)[1]))) :
                            beforestartnzchunk_index(parent(V))
    return iterate_nzchunkspairs(V, state)
end
Base.@propagate_inbounds function iterate_nzchunkspairs(V::SubArray{<:Any,<:Any,<:T}, state) where {T<:AbstractAllDensedSparseVector}
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

Base.@propagate_inbounds function iterate_nzchunkspairs(V::SparseVector)
    nn = nnz(V)
    return nn == 0 ? nothing : iterate_nzchunkspairs(V, (1, 0))
end
Base.@propagate_inbounds function iterate_nzchunkspairs(V::SparseVector, state)
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

Base.@propagate_inbounds function iterate_nzchunkspairs(V::Vector, state = 0)
    state += 1
    if length(V) == 1
        return (Pair(1:1, V), state)
    elseif state <= 1
        return ((1:length(V), V), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds iterate_nzchunkspairs(V::Number, state = V) = (Pair(V, V), state)

"`iterate_nzpairs(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and value"
function iterate_nzpairs end
"`iterate_nzpairsview(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and `view` to value"
function iterate_nzpairsview end
"`iterate_nzvalues(V::AbstractAllDensedSparseVector)` iterates over non-zero elements of vector and returns value"
function iterate_nzvalues end
"`iterate_nzvaluesview(V::AbstractAllDensedSparseVector)` iterates over non-zero elements
 of vector and returns `view` of value"
function iterate_nzvaluesview end
"`iterate_nzindices(V::AbstractAllDensedSparseVector)` iterates over non-zero elements of vector and returns its indices"
function iterate_nzindices end

#
# iterate_nzSOMEs() iterators for `Number`, `Vector` and `SparseVector`
#

Base.@propagate_inbounds function iterate_nzpairs(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (Pair(@inbounds V.nzind[state], @inbounds V.nzval[state]), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzpairsview(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (Pair(@inbounds V.nzind[state], @inbounds view(V.nzval, state:state)), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzpairs(V::Vector, state = 0)
    if state < length(V)
        state += 1
        #return ((state, @inbounds V[state]), state)
        return (Pair(state, @inbounds V[state]), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzpairsview(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (Pair(state, @inbounds view(V, state:state)), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds iterate_nzpairs(V::Number, state = 0) = (Pair(state+1, V), state+1)

Base.@propagate_inbounds function iterate_nzvalues(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzval)
        return (@inbounds V.nzval[state], state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzvaluesview(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzval)
        return (@inbounds view(V.nzval, state:state), state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzvalues(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (@inbounds V[state], state)
    elseif length(V) == 1
        return (@inbounds V[1], state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzvaluesview(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (@inbounds view(V, state:state), state)
    elseif length(V) == 1
        return (@inbounds view(V, 1:1), state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iterate_nzvalues(V::Number, state = 0) = (V, state+1)

Base.@propagate_inbounds function iterate_nzindices(V::SparseVector, state = 0)
    state += 1
    if state <= length(V.nzind)
        return (@inbounds V.nzind[state], state)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iterate_nzindices(V::Vector, state = 0)
    if state < length(V)
        state += 1
        return (state, state)
    else
        return nothing
    end
end
Base.@propagate_inbounds iterate_nzindices(V::Number, state = 0) = (state+1, state+1)

#
# `AbstractAllDensedSparseVector` iteration functions
#

abstract type AbstractDSVIteratorState{Ti,Td,Tit} end

"Scalar DSV Iterator"
struct SDSVIteratorState{Ti,Td,Tit} <: AbstractDSVIteratorState{Ti,Td,Tit}
    "position of current scalar/block element in the current chunk"
    itblock::Int
    "the indices of first and last elements in current chunk"
    indices::UnitRange{Ti}
    "current chunk is the view into nzchunk"
    chunk::Td
    "nzchunk iterator state (Int or Semitoken) in nzchunks"
    itchunk::Tit
end

"Block DSV Iterator"
struct BDSVIteratorState{Ti,Td,Tit} <: AbstractDSVIteratorState{Ti,Td,Tit}
    "position of current scalar/block element in the current chunk"
    itblock::Int
    "the indices of first and last elements in current chunk"
    indices::UnitRange{Ti}
    "position of current element in current block (for scalar DSV it is always 1 of 1)"
    i::Int
    "the indices of first and last elements of the block in current chunk, i.e. `.offsets[itblock]:.offsets[itblock+1]-1`"
    blockindices::UnitRange{Int}
    "current chunk is the view into nzchunk"
    chunk::Td
    "nzchunk iterator state (Int or Semitoken) in nzchunks"
    itchunk::Tit
end


SparseArrays.indtype(it::AbstractDSVIteratorState{Ti,Td,Tit}) where {Ti,Td,Tit} = Ti
Base.eltype(it::AbstractDSVIteratorState{Ti,Td,Tit}) where {Ti,Td,Tit} = eltype(Td)


@inline function nziteratorstate(::Union{Type{T},Type{SubArray{<:Any,<:Any,<:T}}}, itblock, indices, chunk::Tvv, it::Tit) where
                                          {T<:AbstractCompressedDensedSparseVector{Tv,Ti},Tvv,Tit} where {Tv,Ti}
    SDSVIteratorState{Ti,Tvv,Tit}(itblock, indices, chunk, it)
end


# Start iterations from `i` index, i.e. `i` is `firstindex(V)`. That's option for `SubArray` and restarts.
startindex(V) = startindex(parent(V), first(parentindices(V)[1]))
function startindex(V, i)
    itchunk = searchsortedlast_nzchunk(V, i)
    if itchunk != pastendnzchunk_index(V)
        indices, chunk = get_indices_and_nzchunk(V, itchunk)
        if checkindex(Bool, indices, i) #key <= i < key + length(chunk)
            return nziteratorstate(typeof(V), Int(i - first(indices)), indices, chunk, itchunk)
        else
            return nziteratorstate(typeof(V), 0, indices, chunk, itchunk)
        end
    else
        indices, chunk = get_indices_and_nzchunk(V)
        return nziteratorstate(typeof(V), 0, indices, chunk, itchunk)
    end
end

# TODO: FIXME: Add simple :iterate
for (fn, ret1, ret2) in
    ((:iterate_nzpairs     ,  :((indices[itblock] => chunk[itblock], nzit))                 , :(nothing)              ),
     (:iterate_nzpairsview ,  :((indices[itblock] => view(chunk, itblock:itblock), nzit))   , :(nothing)              ),
     (:iterate_nzpairsref  ,  :((indices[itblock] => Ref(chunk, itblock, nzit))             , :(nothing)              ),
     (:iterate_nzblocks    ,  :((view(chunk, get_nzchunk_offsets(V, itchunk, indices[itblock])), nzit)), :(nothing)     ),
     (:iterate_nzvalues    ,  :((chunk[itblock], nzit))                                     , :(nothing)              ),
     (:iterate_nzvaluesview,  :((view(chunk, itblock:itblock), nzit))                       , :(nothing)              ),
     (:iterate_nzvaluesref ,  :((Ref(chunk, itblock), nzit))                                , :(nothing)              ),
     (:iterate_nzindices   ,  :((indices[itblock], nzit))                                   , :(nothing)              ),
     (:iterate_nziterator  ,  :((nzit, nzit))                                               , :(nothing)              ),
     (:nziterator_advance  ,  :(nzit)                                                       , :(pastendnziterator(V)) ) )

    @eval Base.@propagate_inbounds function $fn(V::Union{T,SubArray{<:Any,<:Any,<:T}}, state = startindex(V)) where
                                                {T<:AbstractAllDensedSparseVector{Tv,Ti}} where {Ti,Tv}
        itblock, indices, chunk, itchunk = fieldvalues(state)
        itblock += 1
        if itblock <= length(indices)
            nzit = nziteratorstate(typeof(V), itblock, indices, chunk, itchunk)
            return $ret1
        elseif (st = iterate_nzchunkspairs(V, itchunk)) !== nothing
            ((indices, chunk), itchunk) = st
            itblock = 1
            nzit = nziteratorstate(typeof(V), itblock, indices, chunk, itchunk)
            return $ret1
        else
            return $ret2
        end
    end
end


for (fn, ret1, ret2) in
    (#(:iterate_nzpairs     ,  :((indices[itblock] => chunk[itblock], nzit))                  , :(nothing)              ),
     #(:iterate_nzpairsview ,  :((indices[itblock] => view(chunk, itblock:itblock), nzit))    , :(nothing)              ),
     #(:iterate_nzblocks    ,  :((view(chunk, get_nzchunk_offsets(V, itchunk, indices[itblock])), nzit)), :(nothing)     ),
     (:iterate_nzvalues    ,  :((chunk[itblock], nzit))                                       , :(nothing)              ),
     (:iterate_nzvaluesview,  :((view(chunk, itblock:itblock), nzit))                         , :(nothing)              ),
     (:iterate_nzvaluesref ,  :((Ref(chunk, itblock), nzit))                                  , :(nothing)              ),
     #(:iterate_nzindices   ,  :((indices[itblock], nzit))                                      , :(nothing)              ),
     #(:iterate_nziterator  ,  :((nzit, nzit))                                                  , :(nothing)              ),
     #(:nziterator_advance  ,  :(nzit)                                                          , :(pastendnziterator(V)) )
    )

    @eval Base.@propagate_inbounds function $fn(V::Union{T,SubArray{<:Any,<:Any,<:T}}, state = startindex(V)) where
                                                {T<:DensedVLSparseVector{Tv,Ti}} where {Ti,Tv}
        itblock, indices, chunk, itchunk = fieldvalues(state)
        itblock += 1
        # if itblock <= length(indices)
        if itblock <= V.offsets[itchunk][length(indices)+1]-1
            nzit = nziteratorstate(typeof(V), itblock, indices, chunk, itchunk)
            return $ret1
        elseif (st = iterate_nzchunkspairs(V, itchunk)) !== nothing
            ((indices, chunk), itchunk) = st
            itblock = 1
            nzit = nziteratorstate(typeof(V), itblock, indices, chunk, itchunk)
            return $ret1
        else
            return $ret2
        end
    end
end


#=
nziterator_advance(V::AbstractAllDensedSparseVector) = firstnziterator(V)

function nziterator_advance(V::AbstractAllDensedSparseVector, nzit::SDSVIteratorState)
    if nzit.itblock < length(nzit.indices)
        return nziteratorstate(typeof(V), nzit.itblock + 1, nzit.indices, nzit.chunk, nzit.itchunk)
    elseif (itchunk = advance(V, nzit.itchunk)) != pastendnzchunk_index(V)
        return nziteratorstate(typeof(V), 1, get_indices_and_nzchunk(V, itchunk)..., itchunk)
    else
        return pastendnziterator(V)
    end
end
=#

function nziterator_advance(V::AbstractAllDensedSparseVector, nzit::SDSVIteratorState, step)
    #@boundscheck step >= 0 || throw(ArgumentError("step $step must be non-negative"))
    step == 0 && return nzit
    if step + nzit.itblock <= length(nzit.indices)
            return nziteratorstate(typeof(V), nzit.itblock+step, nzit.indices, nzit.chunk, nzit.itchunk)
    elseif (itchunk = advance(V, nzit.itchunk)) != pastendnzchunk_index(V)
        if step + nzit.itblock == 1 + Int(length(nzit.indices))
            return nziteratorstate(typeof(V), 1, get_indices_and_nzchunk(V, itchunk)..., itchunk)
        else
            return nziterator_advance(V, nziteratorstate(typeof(V), 1, get_indices_and_nzchunk(V, itchunk)..., itchunk),
                                      max(0, Int(step - (length(nzit.indices)-nzit.itblock) - 1)) )
        end
    else
        return pastendnziterator(V)
    end
end

function nziterator_possible_advance(V, nzit::SDSVIteratorState)
    if nzit.itblock != 0
        if nzit.itblock <= length(nzit.indices)
            return Int(length(nzit.indices)) - nzit.itblock + 1
        elseif (itchunk = advance(V, nzit.itchunk)) != pastendnzchunk_index(V)
            return Int(length(get_nzchunk_indices(V, itchunk)))
        else
            return 0
        end
    else
        return 0
    end
end


#
#  rawindex
#

idxcompare(V::AbstractSparseVector, i, j) = cmp(i, j)
idxcompare(V::DynamicDensedSparseVector, i, j) = compare(V.nzchunks, i, j)

function rawindex_compare(V, i, j)
    c = idxcompare(V, first(i), first(j))
    if c < 0
        return -1
    elseif c == 0
        return cmp(last(i), last(j))
    else
        return 1
    end
end

"""
RawIndex is an `Pair` of idx to chunk and value position which points directly to value in AbstractAllDensedSparseVector.
For `DensedSparseVector` it will be `Pair{Int,Int}`.
For `DynamicDensedSparseVector` it stay `Pair{DataStructures.Tokens.IntSemiToken,Int}`

TODO: Create struct RawIndex idx::It, i::Int end
TODO: RawIndex must contain index of accessed cell of array to checking for vector changes.
"""
function rawindex(V, i)
    itchunk = searchsortedlast_nzchunk(V, i)
    if itchunk != pastendnzchunk_index(V)
        indices = get_nzchunk_indices(V, itchunk)
        if checkindex(Bool, indices, i) #key <= i < key + length(chunk)
            return Pair(itchunk, Int(i - first(indices) + 1))
        end
    end
    throw(BoundsError(V, i))
end

@inline firstrawindex(V::AbstractVector) = nnz(V) > 0 ? Pair(firstnzchunk_index(V), 1) : pastendrawindex(V)
@inline pastendrawindex(V::AbstractVector) = Pair(pastendnzchunk_index(V), 0)

@inline function lastrawindex(V::AbstractVector)
    if nnz(V) > 0
        li = lastnzchunk_index(V)
        return Pair(li, length(get_nzchunk_indices(V, li)))
    else
        return pastendrawindex(V)
    end
end

# RawIndex for SparseVector is just Pair(index,1)
rawindex_advance(V::SparseVector) = firstrawindex(V)
rawindex_advance(V::SparseVector, i) = first(i) < nnz(V) ? Pair(first(i) + oftype(first(i), 1), 1) : pastendrawindex(V)
function rawindex_advance(V::SparseVector, i, step)
    @boundscheck step >= 0 || throw(ArgumentError("step $step must be non-negative"))
    if first(i) + step - 1 < nnz(V)
        Pair(first(i) + oftype(first(i), min(nnz(V)-first(i)+1, step)), 1)
    else
        pastendrawindex(V)
    end
end

rawindex_advance(V::AbstractAllDensedSparseVector) = firstrawindex(V)

function rawindex_advance(V::AbstractAllDensedSparseVector, i::Pair)
    if last(i) != 0
        indices = get_nzchunk_indices(V, first(i))
        if last(i) < length(indices)
            return Pair(first(i), last(i)+1)
        elseif (itc = advance(V, first(i))) != pastendnzchunk_index(V)
            return Pair(itc, 1)
        else
            return pastendrawindex(V)
        end
    else
        return pastendrawindex(V)
    end
end

function rawindex_advance(V::AbstractAllDensedSparseVector, i::Pair, step)
    @boundscheck step >= 0 || throw(ArgumentError("step $step must be non-negative"))
    step == 0 && return i
    if last(i) != 0
        indices = get_nzchunk_indices(V, first(i))
        if last(i) < length(indices)
            if step + last(i) <= length(indices)
                return Pair(first(i), last(i)+step)
            else
                return rawindex_advance(V, Pair(first(i), Int(length(indices))),
                                        max(0, Int(step - (length(indices)-last(i)) )) )
            end
        elseif (itc = advance(V, first(i))) != pastendnzchunk_index(V)
            return rawindex_advance(V, Pair(itc, 1), max(0, step - 1) )
        else
            return pastendrawindex(V)
        end
    else
        return pastendrawindex(V)
    end
end

function rawindex_possible_advance(V, i::Pair)
    if last(i) != 0
        indices = get_nzchunk_indices(V, first(i))
        if last(i) <= length(indices)
            return Int(length(indices) - last(i) + 1)
        elseif (itc = advance(V, first(i))) != pastendnzchunk_index(V)
            return Int(length(get_nzchunk_indices(V, itc)))
        else
            return 0
        end
    else
        return 0
    end
end

"Return `view` on pointed data with `step` length"
@inline rawindex_view(V::AbstractAllDensedSparseVector, i::Pair, step) =
    @view(get_nzchunk(V, first(i))[last(i):last(i)+step-1])

#from_rawindex(V::SparseVector{Tv,Ti}, idx::Pair) where {Tv,Ti} = Ti(first(idx))
#function from_rawindex(V::AbstractAllDensedSparseVector{Tv,Ti}, idx::Pair) where {Tv,Ti}
function from_rawindex(V::AbstractSparseVector{Tv,Ti}, idx::Pair) where {Tv,Ti}
    if first(idx) != pastendnzchunk_index(V)
        return Ti(get_nzchunk_key(V, first(idx))) + Ti(last(idx)) - Ti(1)
    else
        li = lastindex(V)
        return li + oftype(li, 1)
    end
end

#
#  nziterator
#

function nziterator(V, i)
    itchunk = searchsortedlast_nzchunk(V, i)
    if itchunk != pastendnzchunk_index(V)
        indices = get_nzchunk_indices(V, itchunk)
        if checkindex(Bool, indices, i) #key <= i < key + length(chunk)
            return nziteratorstate(typeof(V), Int(i - first(indices) + 1), indices, get_nzchunk(V, itchunk), itchunk)
        end
    end
    throw(BoundsError(V, i))
end

function firstnziterator(V::AbstractVector)
    if nnz(V) > 0
        itchunk = firstnzchunk_index(V)
        nziteratorstate(typeof(V), 1, get_indices_and_nzchunk(V, itchunk)..., itchunk)
    else
        pastendnziterator(V)
    end
end
pastendnziterator(V::AbstractVector) = nziteratorstate(typeof(V), 0, get_indices_and_nzchunk(V)..., pastendnzchunk_index(V))

Base.@propagate_inbounds function Base.to_index(nzit::SDSVIteratorState)
    if nzit.itblock != 0
        @inbounds nzit.indices[nzit.itblock]
    else
        first(nzit.indices)
    end
end


"Return `view` on pointed data with `step` length"
@inline nziterator_view(V::AbstractAllDensedSparseVector, nzit::SDSVIteratorState, step) =
    @inbounds @view(nzit.chunk[nzit.itblock:nzit.itblock+step-1])

"Return `view` on data in `nziterator` diapason"
@inline nziterator_view(V::AbstractAllDensedSparseVector, nzit::SDSVIteratorState) =
    @inbounds @view(nzit.chunk[nzit.indices])



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
    y = iterate_nzchunkspairs(it.itr, state...)
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
@inline Base.iterate(it::NZChunksPairs, state...) = iterate_nzchunkspairs(it.itr, state...)
SparseArrays.indtype(it::NZChunksPairs) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZChunksPairs{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZChunksPairs{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZChunksPairs}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZChunksPairs}) = 1
Base.length(it::NZChunksPairs) = nnzchunks(it.itr)
Base.size(it::NZChunksPairs) = (nnzchunks(it.itr),)
#Iterators.reverse(it::NZChunksPairs) = NZChunksPairs(Iterators.reverse(it.itr))


struct NZBlocks{It}
    itr::It
end
"`nzblocks(V::AbstractVector)` is the `Iterator` over non-zero values of `V`."
nzblocks(itr) = NZBlocks(itr)
@inline Base.iterate(it::NZBlocks, state...) = iterate_nzblocks(it.itr, state...)
SparseArrays.indtype(it::NZBlocks) = SparseArrays.indtype(it.itr)
Base.eltype(::Type{NZBlocks{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZBlocks{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZBlocks}) = Base.HasShape{1}()
Base.ndims(::Type{<:NZBlocks}) = 1
Base.length(it::NZBlocks) = nnz(it.itr)
Base.size(it::NZBlocks) = (nnz(it.itr),)
#Base.getindex(it::NZBlocks, i) = TODO
#Iterators.reverse(it::NZBlocks) = NZBlocks(Iterators.reverse(it.itr))


struct NZIndices{It}
    itr::It
end
"`nzindices(V::AbstractVector)` is the `Iterator` over non-zero indices of vector `V`."
nzindices(itr) = NZIndices(itr)
@inline Base.iterate(it::NZIndices, state...) = iterate_nzindices(it.itr, state...)
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
@inline Base.iterate(it::NZValues, state...) = iterate_nzvalues(it.itr, state...)
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
@inline Base.iterate(it::NZValuesView, state...) = iterate_nzvaluesview(it.itr, state...)
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
@inline Base.iterate(it::NZPairs, state...) = iterate_nzpairs(it.itr, state...)
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
@inline Base.iterate(it::NZPairsView, state...) = iterate_nzpairsview(it.itr, state...)
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
    itc = searchsortedlast_ranges(V, i)
    if itc == beforestartnzchunk_index(V)  # the index `i` is before first index
        return false
    elseif i >= get_nzchunk_key(V, itc) + get_nzchunk_length(V, itc)
        # the index `i` is outside of data chunk indices
        return false
    end
    return true
end


function checkbounds(V, i::Pair)
    (idxcompare(V, first(i), beforestartnzchunk_index(V)) > 0 &&
     idxcompare(V, first(i), pastendnzchunk_index(V)) < 0) || throw(BoundsError(V, i))
    indices = get_nzchunk_indices(V, first(i))
    last(i) > length(indices) && throw(BoundsError(V, i))
    return nothing
end

# Potential type piracy!
@inline Base.getindex(V::SparseVector, i::Pair) = V[first(i)]

@inline function Base.getindex(V::AbstractAllDensedSparseVector, i::Pair)
    @boundscheck checkbounds(V, i)
    return @inbounds get_nzchunk(V, first(i))[last(i)]
end

@inline Base.getindex(V::AbstractAllDensedSparseVector, nzit::SDSVIteratorState) = @inbounds nzit.chunk[nzit.itblock]


@inline function Base.getindex(V::AbstractAllDensedSparseVector{Tv,Ti}, idx::Integer) where {Tv,Ti}
    i = Ti(idx)
    # fast check for cached chunk index
    indices, itc = get_cached(V)
    if i in indices
        return get_nzchunk(V, itc)[i - first(indices) + oneunit(Ti)]
    end
    # cached chunk index miss or index is not stored
    itc = searchsortedlast_ranges(V, i)
    if itc != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        # ifirst, chunk, len = get_key_and_nzchunk_and_length(V, itc)
        # if i < ifirst + len  # is the index `i` inside of data chunk indices range
        #     set_lastused!(V, itc)
        #     return chunk[i - ifirst + oneunit(Ti)]
        # end
        indices = get_indices(V, itc)
        if i <= last(indices)  # is the index `i` inside of data chunk indices range
            set_lastused!(V, itc)
            return get_nzchunk(V, itc)[i - first(indices) + oneunit(Ti)]
        end
    end
    set_lostused!(V)
    return returnzero(V)
end


@inline Base.getindex(V::DensedSVSparseVector, i::Integer, j::Integer) = getindex(V, i)[j]


@inline function Base.getindex(V::DensedVLSparseVector, i::Integer)
    # fast check for cached chunk index
    indices = get_cached_indices(V)
    if i in indices
        itc = get_cached_itchunk(V)
        offs = get_nzchunk_offsets(V, itc, i)
        chunk = V.nzchunks[itc]
        return @view(chunk[offs])
    end
    # cached chunk index miss or index not stored
    itc = searchsortedlast_ranges(V, i)
    if itc != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ids = V.nzranges[itc]
        if i <= last(ids) # i < ifirst + length(offsets)-1  # is the index `i` inside of data chunk indices range
            set_lastused!(V, itc)
            offs = get_nzchunk_offsets(V, itc, i)
            chunk = V.nzchunks[itc]
            return @view(chunk[offs])
        end
    end
    set_lostused!(V)
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


# Potential type piracy!
@inline Base.setindex!(V::SparseVector, value, i::Pair) = (V[first(i)] = value; V)

@inline function Base.setindex!(V::AbstractAllDensedSparseVector, value, i::Pair)
    @boundscheck checkbounds(V, i)
    @inbounds get_nzchunk(V, first(i))[last(i)] = value
    return V
end

@inline function Base.setindex!(V::AbstractAllDensedSparseVector, value, nzit::SDSVIteratorState)
    @boundscheck checkbounds(V, nzit)
    @inbounds get_nzchunk(V, nzit)[nzit.itblock] = value
    return V
end



"Extending by append one of ranges in `nzranges` with `itc` index."
appendnzrangesat!(nzranges::Vector{UnitRange{Ti}}, itc, len=1) where {Ti} =
    nzranges[itc] = UnitRange{Ti}(first(nzranges[itc]), last(nzranges[itc])+oneunit(Ti)*len)

"Extending by prepend one of ranges in `nzranges` with `itc` index."
prependnzrangesat!(nzranges::Vector{UnitRange{Ti}}, itc, len=1) where {Ti} =
    nzranges[itc] = UnitRange{Ti}(first(nzranges[itc])-oneunit(Ti)*len, last(nzranges[itc]))

"Shrinking from end one of ranges in `nzranges` with `itc` index."
popnzrangesat!(nzranges::Vector{UnitRange{Ti}}, itc, len=1) where {Ti} =
    nzranges[itc] = UnitRange{Ti}(first(nzranges[itc]), last(nzranges[itc])-oneunit(Ti)*len)

"Shrinking from begin one of ranges in `nzranges` with `itc` index."
popfirstnzrangesat!(nzranges::Vector{UnitRange{Ti}}, itc, len=1) where {Ti} =
    nzranges[itc] = UnitRange{Ti}(first(nzranges[itc])+oneunit(Ti)*len, last(nzranges[itc]))



function _setindex!(V::AbstractAllDensedSparseVector{Tv,Ti}, val, idx::Integer) where {Tv,Ti}
    # val = Tv(value)
    i = Ti(idx)

    # fast check for cached chunk index
    indices, itc = get_cached(V)
    if i in indices
        chunk = get_nzchunk(V, itc)
        chunk[i - first(indices) + oneunit(Ti)] = val
        return V
    end

    itc = searchsortedlast_ranges(V, i)

    # check the index exist and update its data
    if itc != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        indices = V.nzranges[itc]
        if i <= last(indices)
            chunk = get_nzchunk(V, itc)
            chunk[i - first(indices) + oneunit(Ti)] = val
            set_lastused!(V, itc)
            return V
        end
    end

    @boundscheck begin
        isa(V, FixedDensedSparseVector) && throw(BoundsError(V))
    end

    if V.nnz == 0
        push!(V.nzranges, UnitRange{Ti}(i,i))
        push!(V.nzchunks, [val])
        V.nnz += 1
        V.n = max(V.n, i)
        set_lastused!(V, 1)
        return V
    end

    if itc == beforestartnzchunk_index(V)  # the index `i` is before the first index
        inextfirst = first(V.nzranges[1])
        if inextfirst - i > oneunit(Ti)  # there is will be gap in indices after inserting
            pushfirst!(V.nzranges, UnitRange{Ti}(i,i))
            pushfirst!(V.nzchunks, [val])
        else
            prependnzrangesat!(V.nzranges, 1)
            pushfirst!(V.nzchunks[1], val)
        end
        V.nnz += 1
        set_lastused!(V, 1)
        return V
    end

    indices = V.nzranges[itc]
    ifirst = first(indices)
    ilast = last(indices)

    if i >= first(V.nzranges[end])  # the index `i` is after the last key index
        if i > ilast + oneunit(Ti)  # there is will be the gap in indices after inserting
            push!(V.nzranges, UnitRange{Ti}(i,i))
            push!(V.nzchunks, [val])
        else  # just append to last chunk
            appendnzrangesat!(V.nzranges, itc)
            push!(V.nzchunks[itc], val)
        end
        V.nnz += 1
        V.n = max(V.n, i)
        set_lastused!(V, length(V.nzranges))
        return V
    end

    # the index `i` is somewhere between indices
    stnext = itc + 1
    inextfirst = first(V.nzranges[stnext])

    if inextfirst - ilast == Ti(2)  # join nzchunks
        V.nzranges[itc] = UnitRange{Ti}(first(V.nzranges[itc]), last(V.nzranges[stnext]))
        append!(V.nzchunks[itc], [val], V.nzchunks[stnext])
        deleteat!(V.nzranges, stnext)
        deleteat!(V.nzchunks, stnext)
        set_lastused!(V, itc)
    elseif i - ilast == oneunit(Ti)  # append to left chunk
        appendnzrangesat!(V.nzranges, itc)
        push!(V.nzchunks[itc], val)
        set_lastused!(V, itc)
    elseif inextfirst - i == oneunit(Ti)  # prepend to right chunk
        prependnzrangesat!(V.nzranges, stnext)
        pushfirst!(V.nzchunks[stnext], val)
        set_lastused!(V, stnext)
    else  # insert single element chunk
        insert!(V.nzranges, stnext, UnitRange{Ti}(i,i))
        insert!(V.nzchunks, stnext, [val])
        set_lastused!(V, stnext)
    end

    V.nnz += 1
    return V

end

@inline Base.setindex!(V::DensedSparseVector{Tv}, value, i::Integer) where {Tv} = _setindex!(V, Tv(value), i)
@inline Base.setindex!(V::FixedDensedSparseVector{Tv}, value, i::Integer) where {Tv} = _setindex!(V, Tv(value), i)


@inline function Base.setindex!(V::DensedSVSparseVector{Tv,Ti,m}, vectorvalue::Union{AbstractVector,Tuple}, i::Integer) where {Tv,Ti,m}
    sv = basetype(eltype(eltype(V.nzchunks))){Tuple{m},Tv}(vectorvalue)
    _setindex!(V, sv, i)
end

@inline function Base.setindex!(V::DensedSVSparseVector{Tv}, value, i::Integer, j::Integer) where {Tv}
    sv = getindex(V, i)
    sv = @set sv[j] = Tv(value)
    _setindex!(V, sv, i)
end


function Base.setindex!(V::DensedVLSparseVector{Tv,Ti}, vectorvalue, i::Integer) where {Tv,Ti}

    # fast check for cached chunk index
    if i in get_cached_indices(V)
        itc = get_cached_itchunk(V)
        ifirst, chunk, offsets = first(V.nzranges[itc]), V.nzchunks[itc], V.offsets[itc]
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
        set_lastused!(V, itc)
        return V
    end

    itc = searchsortedlast_ranges(V, i)

    # check the index exist and update its data
    if itc != beforestartnzchunk_index(V)  # the index `i` is not before the first index
        ifirst, chunk, offsets = first(V.nzranges[itc]), V.nzchunks[itc], V.offsets[itc]
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
            set_lastused!(V, itc)
            return V
        end
    end


    if V.nnz == 0
        push!(V.nzranges, UnitRange{Ti}(i,i))
        push!(V.nzchunks, [Tv(v) for v in vectorvalue])
        push!(V.offsets, [1])
        append!(V.offsets[1], length(vectorvalue)+1)
        V.nnz += 1
        V.n = max(V.n, i)
        set_lastused!(V, 1)
        return V
    end

    if itc == beforestartnzchunk_index(V)  # the index `i` is before the first index
        inextfirst = first(V.nzranges[1])
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(V.nzranges, UnitRange{Ti}(i,1))
            pushfirst!(V.nzchunks, [Tv(v) for v in vectorvalue])
            pushfirst!(V.offsets, [1])
            append!(V.offsets[1], length(vectorvalue)+1)
        else
            # V.nzranges[1] -= 1
            prependnzrangesat!(V.nzranges, 1)
            prepend!(V.nzchunks[1], vectorvalue)
            @view(V.offsets[1][2:end]) .+= length(vectorvalue)
            insert!(V.offsets[1], 2, length(vectorvalue)+1)
        end
        V.nnz += 1
        set_lastused!(V, 1)
        return V
    end

    ids, chunk, offsets = V.nzranges[itc], V.nzchunks[itc], V.offsets[itc]

    if i >= first(V.nzranges[end])  # the index `i` is after the last key index
        if i > last(ids) + 1  # there is will be the gap in indices after inserting
            push!(V.nzranges, UnitRange{Ti}(i,i))
            push!(V.nzchunks, [Tv(v) for v in vectorvalue])
            push!(V.offsets, [1])
            push!(V.offsets[end], length(vectorvalue)+1)
        else  # just append to last chunk
            appendnzrangesat!(V.nzranges, itc)
            append!(V.nzchunks[itc], vectorvalue)
            push!(V.offsets[itc], V.offsets[itc][end]+length(vectorvalue))
        end
        V.nnz += 1
        V.n = max(V.n, i)
        set_lastused!(V, length(V.nzranges))
        return V
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(offsets)-1 - 1
    stnext = itc + 1
    inextfirst = first(V.nzranges[stnext])

    if inextfirst - ilast == 2  # join nzchunks
        append!(V.nzchunks[itc], vectorvalue, V.nzchunks[stnext])
        V.offsets[stnext] .+= V.offsets[itc][end]-1 + length(vectorvalue)
        append!(V.offsets[itc], V.offsets[stnext])
        appendnzrangesat!(V.nzranges, itc, length(V.nzranges[stnext])+1)
        deleteat!(V.nzranges, stnext)
        deleteat!(V.nzchunks, stnext)
        deleteat!(V.offsets, stnext)
        set_lastused!(V, itc)
    elseif i - ilast == 1  # append to left chunk
        appendnzrangesat!(V.nzranges, itc)
        append!(V.nzchunks[itc], vectorvalue)
        push!(V.offsets[itc], V.offsets[itc][end]+length(vectorvalue))
        set_lastused!(V, itc)
    elseif inextfirst - i == 1  # prepend to right chunk
        prependnzrangesat!(V.nzranges, stnext)
        prepend!(V.nzchunks[stnext], vectorvalue)
        @view(V.offsets[stnext][2:end]) .+= length(vectorvalue)
        insert!(V.offsets[stnext], 2, length(vectorvalue)+1)
        set_lastused!(V, stnext)
    else  # insert single element chunk
        insert!(V.nzranges, stnext, UnitRange{Ti}(i,i))
        insert!(V.nzchunks, stnext, [Tv(v) for v in vectorvalue])
        insert!(V.offsets, stnext, [1])
        push!(V.offsets[stnext], length(vectorvalue)+1)
        set_lastused!(V, stnext)
    end

    V.nnz += 1
    return V

end



@inline function Base.setindex!(V::DynamicDensedSparseVector{Tv,Ti}, value, idx::Integer) where {Tv,Ti}
    val = eltype(V)(value)
    i = Ti(idx)

    # fast check for cached chunk index
    if i in V.lastused.indices
        chunk = get_nzchunk(V, V.lastused.itchunk)
        chunk[i - first(V.lastused.indices) + 1] = val
        return V
    end

    itc = searchsortedlast_ranges(V, i)

    sstatus = status((V.nzchunks, itc))
    @boundscheck if sstatus == 0 # invalid semitoken
        throw(KeyError(i))
    end

    # check the index exist and update its data
    if V.nnz > 0 && sstatus != 2  # the index `i` is not before the first index
        ifirst, chunk = deref((V.nzchunks, itc))
        if ifirst + length(chunk) > i
            chunk[i - ifirst + 1] = val
            set_lastused!(V, itc)
            return V
        end
    end

    if V.nnz == 0
        V.nzchunks[i] = [val]
        V.nnz += 1
        V.n = max(V.n, i)
        set_lastused!(V, startof(V.nzchunks))
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
        set_lastused!(V, startof(V.nzchunks))
        return V
    end

    ifirst, chunk = deref((V.nzchunks, itc))

    if i >= deref_key((V.nzchunks, lastindex(V.nzchunks))) #lastkey(V) # the index `i` is after the last key index
        if ifirst + length(chunk) < i  # there is will be the gap in indices after inserting
            V.nzchunks[i] = [val]
        else  # just append to last chunk
            V.nzchunks[itc] = push!(chunk, val)
        end
        V.nnz += 1
        V.n = max(V.n, i)
        set_lastused!(V, lastindex(V.nzchunks))
        return V
    end

    set_lostused!(V)

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = advance((V.nzchunks, itc))
    inextfirst = deref_key((V.nzchunks, stnext))

    if inextfirst - ilast == 2  # join nzchunks
        V.nzchunks[itc] = append!(chunk, [val], deref_value((V.nzchunks, stnext)))
        delete!((V.nzchunks, stnext))
    elseif i - ilast == 1  # append to left chunk
        V.nzchunks[itc] = push!(chunk, val)
        set_lastused!(V, itc)
    elseif inextfirst - i == 1  # prepend to right chunk
        V.nzchunks[i] = pushfirst!(deref_value((V.nzchunks, stnext)), val)
        delete!((V.nzchunks, stnext))
    else  # insert single element chunk
        V.nzchunks[i] = [val]
    end

    V.nnz += 1
    return V

end

@inline function SparseArrays.dropstored!(V::AbstractCompressedDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

    V.nnz == 0 && return V

    itc = searchsortedlast_ranges(V, i)

    if itc == beforestartnzchunk_index(V)  # the index `i` is before first index
        return V
    end

    ifirst = first(V.nzranges[itc])
    ilast = last(V.nzranges[itc])
    lenchunk = length(V.nzranges[itc])

    if i > ilast  # the index `i` is outside of data chunk indices
        return V
    end

    if lenchunk == 1
        deleteat!(V.nzchunks[itc], 1)
        deleteat!(V.nzranges, itc)
        deleteat!(V.nzchunks, itc)
    elseif i == ilast  # last index in chunk
        popnzrangesat!(V.nzranges, itc)
        pop!(V.nzchunks[itc])
    elseif i == ifirst  # first element in chunk
        popfirstnzrangesat!(V.nzranges, itc)
        popfirst!(V.nzchunks[itc])
    else
        popnzrangesat!(V.nzranges, itc, ilast-i+1)
        insert!(V.nzranges, itc+1, UnitRange{Ti}(i+1,ilast))
        insert!(V.nzchunks, itc+1, V.nzchunks[itc][i-ifirst+1+1:end])
        resize!(V.nzchunks[itc], i-ifirst+1 - 1)
    end

    V.nnz -= 1
    # V.lastusedchunkindex = 0
    set_lostused!(V)

    return V
end

@inline function SparseArrays.dropstored!(V::DensedVLSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

    V.nnz == 0 && return V

    itc = searchsortedlast_ranges(V, i)

    if itc == beforestartnzchunk_index(V)  # the index `i` is before first index
        return V
    end

    ids = V.nzranges[itc]
    ifirst = first(ids)

    if !in(i, ids) # i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return V
    end

    if length(ids) == 1
        deleteat!(V.nzranges, itc)
        deleteat!(V.nzchunks, itc)
        deleteat!(V.offsets, itc)
    elseif i == last(ids)  # last index in chunk
        popnzrangesat!(V.nzranges, itc)
        len = V.offsets[itc][end] - V.offsets[itc][end-1]
        resize!(V.nzchunks[itc], length(V.nzchunks[itc]) - len)
        pop!(V.offsets[itc])
        @assert(length(V.nzchunks[itc]) == V.offsets[itc][end]-1)
    elseif i == ifirst  # first element in chunk
        popfirstnzrangesat!(V.nzranges, itc)
        len = V.offsets[itc][2] - V.offsets[itc][1]
        deleteat!(V.nzchunks[itc], 1:len)
        popfirst!(V.offsets[itc])
        V.offsets[itc] .-= V.offsets[itc][1] - 1
        @assert(length(V.nzchunks[itc]) == V.offsets[itc][end]-1)
    else # split chunk
        popnzrangesat!(V.nzranges, itc, last(ids)-i+1)
        insert!(V.nzranges, itc+1, UnitRange{Ti}(i+1,last(ids)))
        insert!(V.nzchunks, itc+1, V.nzchunks[itc][V.offsets[itc][i-ifirst+1+1]:end])
        resize!(V.nzchunks[itc], V.offsets[itc][i-ifirst+1] - 1)
        insert!(V.offsets, itc+1, V.offsets[itc][i-ifirst+1 + 1:end])
        resize!(V.offsets[itc], i-ifirst+1)
        V.offsets[itc+1] .-= V.offsets[itc+1][1] - 1
        @assert(length(V.nzchunks[itc]) == V.offsets[itc][end]-1)
        @assert(length(V.nzchunks[itc+1]) == V.offsets[itc+1][end]-1)
    end

    V.nnz -= 1
    set_lostused!(V)

    return V
end

@inline function SparseArrays.dropstored!(V::DynamicDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

    V.nnz == 0 && return V

    itc = searchsortedlast_ranges(V, i)

    if itc == beforestartnzchunk_index(V)  # the index `i` is before first index
        return V
    end

    ifirst, chunk = deref((V.nzchunks, itc))

    if i >= ifirst + length(chunk)  # the index `i` is outside of data chunk indices
        return V
    end

    if length(chunk) == 1
        deleteat!(chunk, 1)
        delete!(V.nzchunks, i)
    elseif i == ifirst + length(chunk) - 1  # last index in chunk
        pop!(chunk)
        V.nzchunks[itc] = chunk
    elseif i == ifirst
        popfirst!(chunk)
        V.nzchunks[i+1] = chunk
        delete!(V.nzchunks, i)
    else
        V.nzchunks[i+1] = chunk[i-ifirst+1+1:end]
        V.nzchunks[itc] = resize!(chunk, i-ifirst+1 - 1)
    end

    V.nnz -= 1
    set_lostused!(V)

    return V
end


# there is exist LinearAlgebra.fillstored!
# although fill! from sparsevector.jl:2300 fills only non-zeros.
# fill!(v::SparseVector, x) fill non-zeros in v with 0.0 if x == 0.0,
# else it fill to full Vector if x != 0.0


function _expand_full!(V::DensedSparseVector{Tv,Ti}) where {Tv,Ti}
    isempty(V) || empty!(V)
    resize!(V.nzranges, 1)
    fi = firstindex(V)
    V.nzranges[1] = UnitRange{Ti}(fi,fi)
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
    resize!(C.nzranges, nnzch)
    resize!(C.nzchunks, nnzch)
    for (itc, (indices,chunk)) in enumerate(nzchunkspairs(A))
        if isassigned(C.nzchunks, itc)
            resize!(C.nzchunks[itc], length(chunk))
        else
            C.nzchunks[itc] = Vector{Tv}(undef, length(chunk))
        end
        C.nzranges[itc] = UnitRange{Ti}(indices)
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
            deleteat!(C.nzranges, iC)
            deleteat!(C.nzchunks, iC)
            (resizes += 1) > nresizes && return _similar_resize!(C, A)
        else #if kC > kA
            insert!(C.nzranges, iC, kA)
            insert!(C.nzchunks, iC, Vector{Tv}(undef, length(chunkA)))
            (resizes += 1) > nresizes && return _similar_resize!(C, A)
            iC = advance(C, iC)
            iA = advance(A, iA)
        end
    end

    nnzch = nnzchunks(A)
    resize!(C.nzranges, nnzch)
    resize!(C.nzchunks, nnzch)

    while iA != pastendnzchunk_index(A)
        kA, chunkA = get_key_and_nzchunk(A, iA)
        C.nzchunks[kA] = Vector{Tv}(undef, length(chunkA))
        iA = advance(A, iA)
    end

    return C
end

_similar_sparse_indices!(C::FixedDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti} = (_check_same_sparse_indices(C, A); return C)

function _similar_sparse_indices!(C::AbstractDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    if C.zpbc == true
        return C
    else
        throw(ArgumentError("FIXME:"))
    end
end

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

@inline _copy_chunk_to!(C::DensedSparseVector{Tv,Ti}, itc, k, chunk) where {Tv,Ti} = (C.nzranges[itc] = UnitRange{Ti}(k,k+length(chunk)-1); C.nzchunks[itc] .= Tv.(chunk))
@inline _copy_chunk_to!(C::FixedDensedSparseVector{Tv,Ti}, itc, k, chunk) where {Tv,Ti} = @view(C.nzchunks[C.offsets[itc]:C.offsets[itc+1]-1]) .= Tv.(chunk)
@inline _copy_chunk_to!(C::DynamicDensedSparseVector{Tv,Ti}, itc, k, chunk) where {Tv,Ti} = C.nzchunks[Ti(k)] .= Tv.(chunk)

function Base.copyto!(C::AbstractDensedSparseVector{Tv,Ti}, A::AbstractDensedSparseVector) where {Tv,Ti}
    _similar_sparse_indices!(C, A)
    for (itc, (ids,chunk)) in enumerate(nzchunkspairs(A))
        _copy_chunk_to!(C, itc, first(ids), chunk)
    end
    return C
end



function Base.empty!(V::Union{DensedSparseVector,DensedSVSparseVector})
    empty!(V.nzranges); empty!(V.nzchunks);
    set_lostused!(V)
    V.nnz = 0
    V
end
function Base.empty!(V::DensedVLSparseVector)
    empty!(V.nzranges); empty!(V.nzchunks); empty!(V.offsets)
    set_lostused!(V)
    V.nnz = 0
    V
end
function Base.empty!(V::DynamicDensedSparseVector)
    empty!(V.nzchunks)
    set_lostused!(V)
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
