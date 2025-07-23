
"""
_CompressedChunk_ is an minimal continuous storage "block" in DensedSparseVector,
i.e. CompressedChunk is the continuous non-zeros data between "sparse holes".
"""
module CompressedChunks

export AbstractCompressedChunk, CompressedChunk0, CompressedChunk1, CompressedChunkN, CompressedChunkVL


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



"""
The _Compressed Chunk_ types are the structs like the `Vector`
with continuously stored blocks (or scalars as blocks with N=1),
but the `getindex` and another operations are for the blocks.

One-dimensional `getindex` will always return `view` on block.
Two-dimensional `getindex` will return an value of block at the
_second_index_. The behavior of `setindex` is similar.

Parameterized type storage:

`N = 0` -- In this case the CompressedChunk store only one block in chunk,
thus it can be imagine as N = length(cc.vls) (useless?);

`N = 1` -- scalar values stored in `vls`, i.e. blocks length N = 1.

`N = number` -- vector blocks with length N stored in `vls`;

`N = -1` -- variable length blocks stored in `vls`, in `ofs` stored the starts of blocks in `vls`.

`idx` is the UnitRange with the first and last indices of blocks in current chunk:
`firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`.
`idx` can be considered as offset axes.
Useful for fast access to the blocks indices without the math and the length evaluations.

`ofs` is the unified interface for all AbstractCompressedChunk to have the indices
for fast access to the start positions of blocks in the chunk.

`vls` is the Vector which continuously store all block/scalar values.
"""
abstract type AbstractCompressedChunk{Tv,N} <: AbstractVector{Tv} end


"""
$(TYPEDEF)
Struct fields:
$(TYPEDFIELDS)
"""
struct CompressedChunk0{Tv,Ti,N} <: AbstractCompressedChunk{Tv,0}
    "the indices of first block and last block in chunk: `firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`"
    idx::UnitRange{Ti}
    "the only one block is the whole `vls`"
    vls::Vector{Tv}
    "in this case the `ofs` refers to the 1 and the past last position in `vls`"
    ofs::UnitRange{Int}

    CompressedChunk0(i, vls) = CompressedChunk0{eltype(vls),eltype(i)}(i, vls)
    CompressedChunk0{Tv,Ti,N}(i, vls) where {Tv,N,Ti} = CompressedChunk0{Tv,Ti}(i, vls)
    CompressedChunk0{Tv,Ti}(i::Number, vls) where {Tv,Ti} = CompressedChunk0{Tv,Ti}(range(i, length=length(vls)), vls)
    CompressedChunk0{Tv,Ti}(r::UnitRange, vls) where {Tv,Ti} = CompressedChunk0{Tv,Ti}(UnitRange{Ti}(r), vls)
    function CompressedChunk0{Tv,Ti}(r::UnitRange{Ti}, vls) where {Tv,Ti}
        n = length(vls)
        @assert length(r) == n
        ur = range(1, n+1)
        new{Tv,0,Ti}(r, vls, ur)
    end
end

# TODO: Not the one-starting views for the CompressedChunk to have
# for DensedSparseVectors the ability to have this option.
# It is possible via the OffsetArrays.jl

"""
$(TYPEDEF)
Struct fields:
$(TYPEDFIELDS)
"""
struct CompressedChunk1{Tv,Ti,N} <: AbstractCompressedChunk{Tv,1}
    # TODO: FIXME redo from CompressedChunk0
    "the indices of first block and last block in chunk: `firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`"
    idx::UnitRange{Ti}
    "the blocks are stored continuously in `vls`"
    vls::Vector{Tv}
    "in this case the `ofs` refers to each position in `vls`, and the past last position in `vls`"
    ofs::UnitRange{Int}

    CompressedChunk1(i, vls) = CompressedChunk1{eltype(vls),eltype(i)}(i, vls)
    CompressedChunk1{Tv,Ti,N}(i, vls) where {Tv,N,Ti} = CompressedChunk1{Tv,Ti}(i, vls)
    CompressedChunk1{Tv,Ti}(i::Number, vls) where {Tv,Ti} = CompressedChunk1{Tv,Ti}(range(i, length=length(vls)), vls)
    CompressedChunk1{Tv,Ti}(r::UnitRange, vls) where {Tv,Ti} = CompressedChunk1{Tv,Ti}(UnitRange{Ti}(r), vls)
    function CompressedChunk1{Tv,Ti}(r::UnitRange{Ti}, vls) where {Tv,Ti}
        n = length(vls)
        @assert length(r) == n
        ur = range(1, n+1)
        new{Tv,0,Ti}(r, vls, ur)
    end
end

struct CompressedChunkN{Tv,Ti,N} <: AbstractCompressedChunk{Tv,N}
    "the indices of first block and last block in chunk: `firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`"
    idx::UnitRange{Ti}
    # May be resizable Matrix{Tv}(N,m)? https://github.com/JuliaArrays/ElasticArrays.jl
    "the blocks are stored continuously in `vls`"
    vls::Vector{Tv}
    "the `ofs` refers to the start positions of blocks in `vls`"
    ofs::StepRangeLen{Int,Int,Int,Int}

    CompressedChunkN{Tv,Ti,N}(i::Number, vls) where {Tv,Ti,N} = CompressedChunkN{Tv,Ti,N}(range(i, length=div(length(vls),N)), vls)
    function CompressedChunkN{Tv,Ti,N}(r::UnitRange, vls) where {Tv,Ti,N}
        @assert mod(length(vls), N) == 0
        lenv = length(vls)
        n = div(lenv, N)
        @assert length(r) == n
        srl = StepRangeLen{Int,Int,Int,Int}(1,N,n+1)
        new{Tv,Ti,N}(UnitRange{Ti}(r), vls, srl)
    end
end

struct CompressedChunkVL{Tv,Ti,N} <: AbstractCompressedChunk{Tv,-1}
    "the indices of first block and last block in chunk: `firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`"
    idx::UnitRange{Ti}
    "the blocks are stored continuously in `vls`"
    vls::Vector{Tv}
    "in `ofs` stored the start positions of blocks in `vls`. And in the last position store after the last index of `vls`"
    ofs::Vector{Int}

    CompressedChunkVL{Tv,Ti,N}(i, vls, ofs) where {Tv,Ti,N} = CompressedChunkVL{Tv,Ti}(i, vls, ofs)
    CompressedChunkVL{Tv,Ti}(i::Number, vls, ofs) where {Tv,Ti} = CompressedChunkVL{Tv,Ti}(range(i, length=length(ofs)-1), vls, ofs)
    function CompressedChunkVL{Tv,Ti}(r::UnitRange, vls, ofs) where {Tv,Ti}
        @assert first(ofs) == 1 && last(ofs) - 1 == length(vls)
        @assert issorted(ofs)
        n = length(ofs) - 1
        @assert length(r) == n
        new{Tv,Ti,-1}(UnitRange{Ti}(r), vls, ofs)
    end
end

const CompressedBlockChunk{Tv,N} = Union{CompressedChunkN{Tv,N}, CompressedChunkVL{Tv,-1}}


#
# TODO: Refactor all code below as the CompressedChunk0 and CompressedChunk1 are the distinct types
# and thus there is no reason for separate functions -- all functions should be AbstractCompressedChunk only!
#

# # size2(cc::CompressedChunk{Tv,0}) where {Tv}   = 1
# # size2(_::CompressedChunk{Tv,N}) where {Tv,N}  = N
# size2(cc::CompressedChunk0)                   = 1
# size2(_::CompressedChunkN{Tv,N}) where {Tv,N} = N
# function size2(cc::AbstractCompressedChunk)
#     l = 0
#     for i = 1:length(cc)
#         l = max(l, cc.ofs[i+1]-cc.ofs[i])
#     end
#     l
# end

@inline Base.in(i::Integer, cc::AbstractCompressedChunk) = in(i, cc.idx)
Base.@propagate_inbounds Base.length(cc::AbstractCompressedChunk) = length(cc.ofs) - 1
Base.@propagate_inbounds Base.size(cc::AbstractCompressedChunk) = (length(cc), )
Base.@propagate_inbounds Base.axes(cc::AbstractCompressedChunk) = (firstindex(cc):lastindex(cc),)

# Base.@propagate_inbounds Base.size(cc::CompressedChunk{Tv,N}) where {Tv,N}  = (length(cc), N)
# Base.@propagate_inbounds Base.size(cc::CompressedChunkN{Tv,N}) where {Tv,N} = (length(cc), N)
# Base.@propagate_inbounds Base.size(cc::CompressedChunk{Tv,-1}) where Tv     = (length(cc), size2(cc))
# Base.@propagate_inbounds Base.size(cc::CompressedChunkVL)                   = (length(cc), size2(cc))
# Base.@propagate_inbounds Base.axes(cc::AbstractCompressedChunk) = (Base.OneTo(length(cc)), Base.OneTo(size2(cc)))

# There should be only by blocks iterations.
Base.@propagate_inbounds Base.iterate(cc::CompressedChunk0) = length(cc) > 0 ? (cc[firstindex(cc),1], firstindex(cc)+1) : nothing
Base.@propagate_inbounds Base.iterate(cc::CompressedChunk0, state) = state <= lastindex(cc) ? (cc[state,1], state+1) : nothing
# Base.@propagate_inbounds Base.iterate(cc::CompressedChunk{Tv,0}) where Tv = length(cc) > 0 ? (cc[1,1], 2) : nothing
# Base.@propagate_inbounds Base.iterate(cc::CompressedChunk{Tv,0}, state) where Tv = state <= length(cc) ? (cc[state,1], state+1) : nothing

Base.@propagate_inbounds Base.iterate(cc::AbstractCompressedChunk) = length(cc) > 0 ? (cc[firstindex(cc)], firstindex(cc)+1) : nothing
Base.@propagate_inbounds Base.iterate(cc::AbstractCompressedChunk, state) = state <= lastindex(cc) ? (cc[state], state+1) : nothing

@inline Base.firstindex(cc::AbstractCompressedChunk) = first(cc.idx)
@inline Base.lastindex(cc::AbstractCompressedChunk) = last(cc.idx)
# @inline Base.eachindex(cc::AbstractCompressedChunk) = firstindex(cc):lastindex(cc)

Base.@propagate_inbounds function Base.getindex(cc::AbstractCompressedChunk, i::Integer, j::Integer)
    @boundscheck in(i, cc.idx)
    _getindex(cc, i, j)
end
Base.@propagate_inbounds function Base.getindex(cc::AbstractCompressedChunk, i::Integer)
    @boundscheck in(i, cc.idx)
    _getindex(cc, i)
end

# Base.@propagate_inbounds _getindex(cc::CompressedChunk0, i::Integer, j::Integer)                    = cc.vls[i]
# Base.@propagate_inbounds _getindex(cc::CompressedChunkN{Tv,N}, i::Integer, j::Integer) where {Tv,N} = cc.vls[(i-1)*N + j]
# Base.@propagate_inbounds _getindex(cc::CompressedChunkVL, i::Integer, j::Integer)                   = cc.vls[cc.ofs[i]+j-1]
#
# Base.@propagate_inbounds _getindex(cc::CompressedChunk0, i::Integer)                                = @view(cc.vls[i:i])
# Base.@propagate_inbounds _getindex(cc::CompressedChunkN{Tv,N}, i::Integer) where {Tv,N}             = @view(cc.vls[1+(i-1)*N:i*N])
# Base.@propagate_inbounds _getindex(cc::CompressedChunkVL, i::Integer)                               = @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1])

Base.@propagate_inbounds _getindex(cc::AbstractCompressedChunk, idx::Integer, j::Integer) = (i=idx-firstindex(cc)+1; cc.vls[cc.ofs[i]+j-1])
Base.@propagate_inbounds _getindex(cc::AbstractCompressedChunk, idx::Integer) = (i=idx-firstindex(cc)+1; @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]))

Base.@propagate_inbounds function Base.setindex!(cc::AbstractCompressedChunk, item, i::Integer, j::Integer)
    @boundscheck in(i, cc)
    _setindex!(cc, item, i, j)
end
Base.@propagate_inbounds function Base.setindex!(cc::AbstractCompressedChunk, item, i::Integer)
    # FIXME: What are to return in *BLOCK* assingment function?
    @boundscheck in(i, cc)
    _setindex!(cc, item, i)
end

Base.@propagate_inbounds function issetindex!(cc::AbstractCompressedChunk, item, i::Integer, j::Integer)
    if in(i, cc) && j <= blocklength(cc, i)
        _setindex!(cc, item, i, j)
        return true
    else
        return false
    end
end

Base.@propagate_inbounds function issetindex!(cc::CompressedChunk0, item::Number, i::Integer)
    if in(i, cc)
        _setindex!(cc, item, i)
        return true
    else
        return false
    end
end
Base.@propagate_inbounds function issetindex!(cc::AbstractCompressedChunk, item, i::Integer)
    if in(i, cc) && length(item) == blocklength(cc, i)
        _setindex!(cc, item, i)
        return true
    else
        return false
    end
end

@inline blocklength(cc::CompressedChunk0, _::Integer=1) = length(cc.ofs) - 1
@inline blocklength(::CompressedChunk1, _::Integer=1) = 1
@inline blocklength(::CompressedChunkN{Tv,N}, i::Integer=1) where {Tv,N} = N
@inline blocklength(cc::AbstractCompressedChunk, i::Integer) = (idx = i-firstindex(cc)+1; cc.ofs[idx+1] - cc.ofs[idx])

Base.@propagate_inbounds function _setindex!(cc::AbstractCompressedChunk{Tv}, item, idx::Integer, j::Integer) where Tv
    i = idx-firstindex(cc)+1
    cc.vls[cc.ofs[i]+j-1] = Tv(item)
    item
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk0{Tv}, item, idx::Integer) where Tv
    i=idx-firstindex(cc)+1
    @assert i == 1
    if length(item) == length(cc.vls)
        cc.vls .= Tv.(item)
    elseif length(item) == 1
        cc.vls .= Tv(item)
    else
        throw(ArgumentError("Arg item nor scalar, nor suitable length ($(length(item))) container"))
    end
    return Tv(item);
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk1{Tv}, item, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    cc.vls[i] = Tv(item)
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunkN, item, idx::Integer)
    i = idx-firstindex(cc)+1
    @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]) .= item
    item
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunkVL, item, idx::Integer)
    i = idx-firstindex(cc)+1
    if blocklength(cc, idx) == length(item)
        @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]) .= item
    else #if blocklength(cc, idx) != length(item)
        vls = cc.vls
        ofs = cc.ofs
        splice!(vls, ofs[i]:ofs[i+1]-1, item)
        dif = (ofs[i+1]-ofs[i]) - length(item)
        for k = i+1:length(ofs)
            ofs[k] -= dif
        end
    end
    return item
end

Base.push!(cc::T, item) where {T<:CompressedChunk0} = T(firstindex(cc), push!(cc.vls, item))
function Base.push!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,N,T<:CompressedChunkN{Tv,N}}
    @boundscheck length(items) == N
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    append!(vls, items)
    return T(firstindex(cc), vls)
end
function Base.push!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunkVL{Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    append!(vls, items)
    ofs = cc.ofs
    push!(ofs, last(ofs) + length(items))
    return T(firstindex(cc), vls, ofs)
end


function Base.pushfirst!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,N,T<:AbstractCompressedChunk{Tv,N}}
    @boundscheck if N != 0
        @boundscheck length(items) == N
    end
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    prepend!(vls, items)
    return T(firstindex(cc)-1, vls)
end
function Base.pushfirst!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunkVL{Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    ofs = cc.ofs
    prepend!(vls, items)
    pushfirst!(ofs, 1)
    len = length(items)
    for i = 2:length(ofs)
        ofs[i] += len
    end
    return T(firstindex(cc)-1, vls, ofs)
end

"Return a `CompressedChunk` consisting of all but the first component of `cc`."
function tail!(cc::T) where {T<:CompressedChunk0}
    length(cc) == 0 && throw(ArgumentError("Cannot call tail! on an empty tuple"))
    return T(firstindex(cc)+1, popfirst!(cc.vls))
end
function tail!(cc::T) where {Tv,N,T<:CompressedChunkN{Tv,N}}
    length(cc) == 0 && throw(ArgumentError("Cannot call tail! on an empty tuple"))
    vls = cc.vls
    deleteat!(vls, 1:N)
    return T(firstindex(cc)+1, vls)
end
function tail!(cc::T) where {Tv,T<:CompressedChunkVL{Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call tail! on an empty tuple"))
    vls = cc.vls
    ofs = cc.ofs
    N = ofs[2] - ofs[1]
    deleteat!(vls, 1:N)
    popfirst!(ofs)
    for i = 1:length(ofs)
        ofs[i] -= N
    end
    return T(firstindex(cc)+1, vls, ofs)
end

"Return a `CompressedChunk` consisting of all but the last component of `cc`."
function front!(cc::T) where {T<:CompressedChunk0}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    return T(firstindex(cc), pop!(cc.vls))
end
function front!(cc::T) where {Tv,N,T<:CompressedChunkN{Tv,N}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    vls = cc.vls
    ofs = cc.ofs
    len = length(vls) - N
    resize!(vls, len)
    pop!(ofs)
    return T(firstindex(cc), vls, ofs)
end
function front!(cc::T) where {Tv,T<:CompressedChunkVL{Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    vls = cc.vls
    ofs = cc.ofs
    N = ofs[end] - ofs[end-1]
    len = length(vls) - N
    resize!(vls, len)
    pop!(ofs)
    return T(firstindex(cc), vls, ofs)
end


# derived from base/array.jl
Base.append!(cc::AbstractCompressedChunk, iter...) = foldl(append!, iter, init=cc)
Base.push!(cc::AbstractCompressedChunk, iter...) = append!(cc, iter)

function Base.append!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunk0{Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    append!(vls, items)
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,N,T<:CompressedChunkN{Tv,N},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    for item in items
        @assert length(item) == N
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        append!(vls, item)
    end
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,T<:CompressedChunkVL{Tv},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    ofs = cc.ofs
    for item in items
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        append!(vls, item)
        push!(ofs, last(ofs) + length(item))
    end
    return T(firstindex(cc), vls, ofs)
end
function Base.append!(cc::T, items::AbstractCompressedChunk) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    append!(vls, items.vls)
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::AbstractCompressedChunk) where {T<:CompressedChunkVL}
    vls = cc.vls
    ofs = cc.ofs
    len = length(ofs) - 1
    append!(vls, items.vls)
    length(items.ofs) > 1 && append!(ofs, @view(items.ofs[2:end]))
    for i = len+1+1:length(ofs)
        ofs[i] += ofs[i-1] - 1
    end
    return T(firstindex(cc), vls, ofs)
end
function Base.append!(cc::T, iter) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    for item in iter
        push!(vls, item)
    end
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, iter) where {T<:CompressedChunkVL}
    vls = cc.vls
    ofs = cc.ofs
    n = 0
    for item in iter
        n += 1
        push!(vls, item)
    end
    push!(ofs, last(ofs) + n)
    return T(firstindex(cc), vls, ofs)
end


# derived from base/array.jl
Base.prepend!(cc::AbstractCompressedChunk, iter...) = foldr((v, cc) -> prepend!(cc, v), iter, init=cc)
Base.pushfirst!(cc::AbstractCompressedChunk, iter...) = prepend!(cc, iter)

function Base.prepend!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunk0{Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    prepend!(vls, items)
    return T(firstindex(cc)-length(items), vls)
end
function Base.prepend!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,N,T<:CompressedChunkN{Tv,N},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    len = 0
    for item in items
        @assert length(item) == N
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        len += length(item)
        prepend!(vls, item)
    end
    return T(firstindex(cc)-len, vls)
end
function Base.prepend!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,T<:CompressedChunkVL{Tv},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    ofs = cc.ofs
    for item in items
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        prepend!(vls, item)
        pushfirst!(ofs, 1)
        N = length(item)
        for i = 2:length(ofs)
            ofs[i] += N
        end
    end
    return T(firstindex(cc)-length(items), vls, ofs)
end
function Base.prepend!(cc::T, items::AbstractCompressedChunk) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    prepend!(vls, items.vls)
    return T(firstindex(cc)-length(items), vls)
end
function Base.prepend!(cc::T, items::AbstractCompressedChunk) where {T<:CompressedChunkVL}
    vls = cc.vls
    ofs = cc.ofs
    len = length(items.ofs) - 1
    prepend!(vls, items.vls)
    if len > 0
        popfirst!(ofs)
        prepend!(ofs, items.ofs)
    end
    for i = len+1+1:length(ofs)
        ofs[i] += ofs[i-1] - 1
    end
    return T(firstindex(cc)-len, vls, ofs)
end
function Base.prepend!(cc::T, iter) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    n = 0
    for item in iter
        n += 1
        pushfirst!(vls, item)
    end
    reverse!(vls, 1, n)
    return T(firstindex(cc)-n, vls)
end
function Base.prepend!(cc::T, iter) where {T<:CompressedChunkVL}
    vls = cc.vls
    ofs = cc.ofs
    n = 0
    for item in iter
        n += 1
        pushfirst!(vls, item)
    end
    reverse!(vls, 1, n)
    pushfirst!(ofs, 1)
    for i = 2:length(ofs)
        ofs[i] += n
    end
    return T(firstindex(cc)-n, vls, ofs)
end


"Delete specified element with index `idx` and thus split vector `cc` in two parts and retuns them in tuple."
function splitat!(cc::T, idx::Integer) where {Tv,T<:AbstractCompressedChunk{Tv}}
    vls = cc.vls
    ofs = cc.ofs
    pos = Int(idx - first(vls.idx) + 1)
    vls2 = vls[ofs[pos+1]:end]
    resize!(vls, ofs[pos]-1)
    return (T(firstindex(cc), vls), T(idx+1, vls2))
end
function splitat!(cc::T, idx::Integer) where {Tv,T<:AbstractCompressedChunk{Tv,-1}}
    vls = cc.vls
    ofs = cc.ofs
    pos = Int(idx - first(vls.idx) + 1)
    vls2 = vls[ofs[pos+1]:end]
    ofs2 = ofs[pos+1:end]
    i0 = first(ofs2) - 1
    ofs2 .-= i0
    resize!(vls, ofs[pos]-1)
    resize!(ofs, pos)
    return (T(firstindex(cc), vls, ofs), T(idx+1, vls2, ofs2))
end


# insert!
# deleteat! -- to multiple delete use range or some other collection: deleteat!(collection, inds)
# splice! -- to inset use `splice!(collection, n:n-1, replacement)`
# resize! -- not need


end  # of module CompressedChunks

