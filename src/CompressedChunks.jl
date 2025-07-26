
"""
_CompressedChunk_ is an minimal continuous storage "block" in DensedSparseVector,
i.e. CompressedChunk is the continuous non-zeros data between "sparse holes".
"""
module CompressedChunks

export AbstractCompressedChunk
export CompressedChunk, CompressedChunk0, CompressedChunk1, CompressedChunkL, CompressedChunkVL
export field_ofs_type


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
The _Compressed Chunk_ types are the structs like the `OffsetVector`
with continuously stored blocks (or scalars as blocks with L=1),
but the `getindex` and another operations are for the blocks.

I.e. CompressedChunk is the OffsetVector of Vectors structure.

One-dimensional `getindex` will always return `view` on block.
Two-dimensional `getindex` will return an value in the block at the
_second_index_ position. The behavior of `setindex!` is similar.

The iterator will return the view on blocks.

Parameterized type storage:

~~`L = ''` -- scalar values stored in `vls`, i.e. blocks length L = 1.
The iterators will returns scalars nor blocks.~~

~~`L = 0` -- In this case the CompressedChunk store only one block in chunk,
thus it can be imagine as L = length(cc.vls) (IS IT USELESS?);~~

`L = 0` -- scalar values stored in `vls`, i.e. blocks length L = 1.
Almost the same as `L = 1`, but iterators will returns scalars instead of blocks.

`L = 1` -- scalar values stored in `vls`, i.e. blocks length L = 1.
IS IT NEED separate `L = 1` if there is exist `L = 0` for scalars???
There is exist common CompressedChunk{1,Tv,Ti}!!!!!
The only small advantage over common CompressedChunk{L} is the some faster `ofs`
calculation because UnitRange instead StepRangeLen.

`L = number` -- vector blocks with length L stored in `vls`;

`L = -1` -- variable length blocks stored in `vls`, in `ofs` stored the starts of blocks in `vls`.

`idx` is the UnitRange{Ti} with the first and last indices of blocks in current chunk:
`firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`.
`idx` can be considered as offset axes.
It is useful for fast access to the blocks indices without the math and the length evaluations.

`ofs` is the unified interface for all AbstractCompressedChunk to have the `Int` indices
for fast access to the start positions of blocks in the storage `vls`.

`vls` is the Vector{Tv} which continuously stored all block/scalar values.
"""
abstract type AbstractCompressedChunk{L,Tv,Ti} <: AbstractVector{Tv} end


# TODO: Not the one-starting views for the CompressedChunk to have
# for DensedSparseVectors the ability to have this option.
# See https://docs.julialang.org/en/v1/devdocs/offset-arrays/
# It is possible via the OffsetArrays.jl



" Evaluate type for `ofs` field of `struct CompressedChunk` during compilation"
@inline function field_ofs_type(::Val{N}) where {N}
    if N == -1
        return Vector{Int}
    elseif N == 0
        return UnitRange{Int}
    elseif N == 1
        return UnitRange{Int}
    else
        return StepRangeLen{Int, Int, Int}
    end
end

"""
$(TYPEDEF)
Struct fields:
$(TYPEDFIELDS)
"""
struct CompressedChunk{L,Tv,Ti,TO} <: AbstractCompressedChunk{L,Tv,Ti}
    "the indices of first block and last block in chunk:
     `firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`"
    idx::UnitRange{Ti}
    "`ofs` refers to start positions of each block in `vls`. And in the last position store after the last index of `vls`"
    #ofs::UnitRange{Int}
    ofs::TO
    "the blocks are stored continuously in `vls`"
    vls::Vector{Tv}


    CompressedChunk(i::Number, vls) = CompressedChunk(range(i, length=length(vls)), vls)
    CompressedChunk(i::UnitRange, vls) = CompressedChunk{0,eltype(vls),eltype(i),field_ofs_type(Val(0))}(i, vls)

    CompressedChunk{L}(i, vls) where L = CompressedChunk{L,eltype(vls),eltype(i),field_ofs_type(Val(L))}(i, vls)

    function CompressedChunk{L}(i::Number, vls) where L
        if L == 0 || (L == -1 && length(vls) == 0)
            CompressedChunk{L,eltype(vls),eltype(i),field_ofs_type(Val(L))}(range(i, length=length(vls)), vls)
        elseif L > 0
            CompressedChunk{L,eltype(vls),eltype(i),field_ofs_type(Val(L))}(range(i, length=div(length(vls), L)), vls)
        else # if L == -1
            throw(MethodError(CompressedChunk{L}, (i, vls)))
        end
    end

    function CompressedChunk{L,Tv,Ti,TO}(r::UnitRange, vls) where {L,Tv,Ti,TO}
        if L < 0 || !isa(vls, Vector{Tv})
            throw(MethodError(CompressedChunk{L,Tv,Ti,TO}, (r, vls)))
        end
        if L == 0 || L == 1
            n = length(vls)
            @assert length(r) == n
            ur = range(1, n+1)
            return new{L,Tv,Ti,TO}(UnitRange{Ti}(r), ur, vls)
        elseif L > 0
            @assert mod(length(vls), L) == 0
            lenv = length(vls)
            n = div(lenv, L)
            @assert length(r) == n
            srl = StepRangeLen{Int,Int,Int,Int}(1,L,n+1)
            return new{L,Tv,Ti,TO}(UnitRange{Ti}(r), srl, vls)
        else#if L == -1
            # create empty CompressedChunk{-1}
            @assert length(r) == 0 && length(vls) == 0
            return new{L,Tv,Ti,TO}(UnitRange{Ti}(r), Int[1], vls)
        #else#if L == -1
        #    @assert first(ofs) == 1 && last(ofs) - 1 == length(vls)
        #    @assert issorted(ofs)
        #    n = length(ofs) - 1
        #    @assert length(r) == n
        #    if vls isa Vector{Tv} && ofs isa Vector{Int}
        #        new{-1,Tv,Ti}(UnitRange{Ti}(r), ofs, vls)
        #    else
        #        throw(MethodError(CompressedChunkVL{-1,Tv,Ti}, (UnitRange{Ti}(r), ofs, vls)))
        #    end
        end
    end

end



const CompressedChunk0{L,Tv,Ti} = CompressedChunk{0,Tv,Ti,UnitRange{Int}}
const CompressedChunk1{L,Tv,Ti} = CompressedChunk{1,Tv,Ti,UnitRange{Int}}
const CompressedChunkL{L,Tv,Ti} = CompressedChunk{L,Tv,Ti,StepRangeLen{Int, Int, Int}}
const CompressedChunkVL{L,Tv,Ti} = CompressedChunk{-1,Tv,Ti,Vector{Int}}

const CompressedScalarChunk{L,Tv,Ti} = Union{CompressedChunk0{Tv,Ti}}
const CompressedBlockChunk{L,Tv,Ti} = Union{CompressedChunk1{Tv,Ti}, CompressedChunkL{L,Tv,Ti}, CompressedChunkVL{Tv,Ti}}


#
# TODO: Refactor all code below as the CompressedChunk{0} and CompressedChunk1 are the distinct types
# and thus there is no reason for separate functions -- all functions should be AbstractCompressedChunk only!
#

# # size2(cc::CompressedChunk{Tv,0}) where {Tv}   = 1
# # size2(_::CompressedChunk{Tv,L}) where {Tv,L}  = L
# size2(cc::CompressedChunk{0})                   = 1
# size2(_::CompressedChunkL{Tv,L}) where {Tv,L} = L
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

# Base.@propagate_inbounds Base.size(cc::CompressedChunk{Tv,L}) where {Tv,L}  = (length(cc), L)
# Base.@propagate_inbounds Base.size(cc::CompressedChunkL{Tv,L}) where {Tv,L} = (length(cc), L)
# Base.@propagate_inbounds Base.size(cc::CompressedChunk{Tv,-1}) where Tv     = (length(cc), size2(cc))
# Base.@propagate_inbounds Base.size(cc::CompressedChunk{-1})                   = (length(cc), size2(cc))
# Base.@propagate_inbounds Base.axes(cc::AbstractCompressedChunk) = (Base.OneTo(length(cc)), Base.OneTo(size2(cc)))

# There should be only by blocks iterations except `CompressedChunk{0}`.
Base.@propagate_inbounds Base.iterate(cc::CompressedChunk{0}) = length(cc) > 0 ? (cc[firstindex(cc),1], firstindex(cc)+1) : nothing
Base.@propagate_inbounds Base.iterate(cc::CompressedChunk{0}, state) = state <= lastindex(cc) ? (cc[state,1], state+1) : nothing

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

# Base.@propagate_inbounds _getindex(cc::CompressedChunk{0}, i::Integer, j::Integer)                    = cc.vls[i]
# Base.@propagate_inbounds _getindex(cc::CompressedChunkL{Tv,L}, i::Integer, j::Integer) where {Tv,L} = cc.vls[(i-1)*L + j]
# Base.@propagate_inbounds _getindex(cc::CompressedChunk{-1}, i::Integer, j::Integer)                   = cc.vls[cc.ofs[i]+j-1]
#
# Base.@propagate_inbounds _getindex(cc::CompressedChunk{0}, i::Integer)                                = @view(cc.vls[i:i])
# Base.@propagate_inbounds _getindex(cc::CompressedChunkL{Tv,L}, i::Integer) where {Tv,L}             = @view(cc.vls[1+(i-1)*L:i*L])
# Base.@propagate_inbounds _getindex(cc::CompressedChunk{-1}, i::Integer)                               = @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1])

Base.@propagate_inbounds _getindex(cc::AbstractCompressedChunk, idx::Integer, j::Integer) = (i=idx-firstindex(cc)+1; cc.vls[cc.ofs[i]+j-1])
Base.@propagate_inbounds _getindex(cc::AbstractCompressedChunk, idx::Integer) = (i=idx-firstindex(cc)+1; @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]))

Base.@propagate_inbounds function Base.setindex!(cc::AbstractCompressedChunk, item, i::Integer, j::Integer)
    @boundscheck in(i, cc)
    _setindex!(cc, item, i, j)
end
Base.@propagate_inbounds function Base.setindex!(cc::AbstractCompressedChunk, item, i::Integer)
    # FIXME: What should be returned in the *BLOCK* assignment function?
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

Base.@propagate_inbounds function issetindex!(cc::CompressedChunk, item::Number, i::Integer)
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

@inline blocklength(cc::CompressedChunk{0}, _::Integer=1) = 1
@inline blocklength(::CompressedChunk{L}, i::Integer=1) where {L} = L
@inline blocklength(cc::AbstractCompressedChunk, i::Integer) = (idx = i-firstindex(cc)+1; cc.ofs[idx+1] - cc.ofs[idx])

Base.@propagate_inbounds function _setindex!(cc::AbstractCompressedChunk{Tv}, item, idx::Integer, j::Integer) where Tv
    i = idx-firstindex(cc)+1
    cc.vls[cc.ofs[i]+j-1] = Tv(item)
    item
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{0,Tv}, item, idx::Integer) where Tv
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

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{Tv}, item, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    cc.vls[i] = Tv(item)
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1,Tv}, item::Tv, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]) .= item
    item
end
Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1,Tv}, item::Union{AbstractVector{Tv},AbstractRange{Tv}}, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]) .= item
    item
end
Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1,Tv}, item, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    @view(cc.vls[cc.ofs[i]:cc.ofs[i+1]-1]) .= Tv.(item)
    item
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1}, item, idx::Integer)
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

Base.push!(cc::T, item) where {T<:CompressedChunk{0}} = T(firstindex(cc), push!(cc.vls, item))
function Base.push!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,L,T<:CompressedChunk{L,Tv}}
    @boundscheck length(items) == L
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    append!(vls, items)
    return T(firstindex(cc), vls)
end
function Base.push!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunk{-1,Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    append!(vls, items)
    ofs = cc.ofs
    push!(ofs, last(ofs) + length(items))
    return T(firstindex(cc), ofs, vls)
end


function Base.pushfirst!(cc::T, items::Union{AbstractVector,Tuple}) where {L,Tv,Ti,T<:AbstractCompressedChunk{L,Tv,Ti}}
    @boundscheck if L != 0
        @boundscheck length(items) == L
    end
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    prepend!(vls, items)
    return T(firstindex(cc)-1, vls)
end
function Base.pushfirst!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunk{-1,Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    ofs = cc.ofs
    prepend!(vls, items)
    pushfirst!(ofs, 1)
    len = length(items)
    for i = 2:length(ofs)
        ofs[i] += len
    end
    return T(firstindex(cc)-1, ofs, vls)
end

"Return a `CompressedChunk` consisting of all but the first component of `cc`."
function tail!(cc::T) where {T<:CompressedChunk{0}}
    length(cc) == 0 && throw(ArgumentError("Cannot call tail! on an empty tuple"))
    return T(firstindex(cc)+1, popfirst!(cc.vls))
end
function tail!(cc::T) where {L,Tv,T<:CompressedChunk{L,Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call tail! on an empty tuple"))
    vls = cc.vls
    deleteat!(vls, 1:L)
    return T(firstindex(cc)+1, vls)
end
function tail!(cc::T) where {Tv,T<:CompressedChunk{-1,Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call tail! on an empty tuple"))
    vls = cc.vls
    ofs = cc.ofs
    L = ofs[2] - ofs[1]
    deleteat!(vls, 1:L)
    popfirst!(ofs)
    for i = 1:length(ofs)
        ofs[i] -= L
    end
    return T(firstindex(cc)+1, ofs, vls)
end

"Return a `CompressedChunk` consisting of all but the last component of `cc`."
function front!(cc::T) where {T<:CompressedChunk{0}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    return T(firstindex(cc), pop!(cc.vls))
end
function front!(cc::T) where {L,Tv,T<:CompressedChunk{L,Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    vls = cc.vls
    ofs = cc.ofs
    len = length(vls) - L
    resize!(vls, len)
    pop!(ofs)
    return T(firstindex(cc), vls, ofs)
end
function front!(cc::T) where {Tv,T<:CompressedChunk{-1,Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    vls = cc.vls
    ofs = cc.ofs
    L = ofs[end] - ofs[end-1]
    len = length(vls) - L
    resize!(vls, len)
    pop!(ofs)
    return T(firstindex(cc), ofs, vls)
end


# derived from base/array.jl
Base.append!(cc::AbstractCompressedChunk, iter...) = foldl(append!, iter, init=cc)
Base.push!(cc::AbstractCompressedChunk, iter...) = append!(cc, iter)

function Base.append!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunk{0,Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    append!(vls, items)
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {L,Tv,T<:CompressedChunk{L,Tv},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    for item in items
        @assert length(item) == L
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        append!(vls, item)
    end
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,T<:CompressedChunk{-1,Tv},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    ofs = cc.ofs
    for item in items
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        append!(vls, item)
        push!(ofs, last(ofs) + length(item))
    end
    return T(firstindex(cc), ofs, vls)
end
function Base.append!(cc::T, items::AbstractCompressedChunk) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    append!(vls, items.vls)
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::AbstractCompressedChunk) where {T<:CompressedChunk{-1}}
    vls = cc.vls
    ofs = cc.ofs
    len = length(ofs) - 1
    append!(vls, items.vls)
    length(items.ofs) > 1 && append!(ofs, @view(items.ofs[2:end]))
    for i = len+1+1:length(ofs)
        ofs[i] += ofs[i-1] - 1
    end
    return T(firstindex(cc), ofs, vls)
end
function Base.append!(cc::T, iter) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    for item in iter
        push!(vls, item)
    end
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, iter) where {T<:CompressedChunk{-1}}
    vls = cc.vls
    ofs = cc.ofs
    n = 0
    for item in iter
        n += 1
        push!(vls, item)
    end
    push!(ofs, last(ofs) + n)
    return T(firstindex(cc), ofs, vls)
end


# derived from base/array.jl
Base.prepend!(cc::AbstractCompressedChunk, iter...) = foldr((v, cc) -> prepend!(cc, v), iter, init=cc)
Base.pushfirst!(cc::AbstractCompressedChunk, iter...) = prepend!(cc, iter)

function Base.prepend!(cc::T, items::Union{AbstractVector,Tuple}) where {Tv,T<:CompressedChunk{0,Tv}}
    items isa Tuple && (items = map(x -> convert(Tv, x), items))
    vls = cc.vls
    prepend!(vls, items)
    return T(firstindex(cc)-length(items), vls)
end
function Base.prepend!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,L,T<:CompressedChunkL{Tv,L},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    len = 0
    for item in items
        @assert length(item) == L
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        len += length(item)
        prepend!(vls, item)
    end
    return T(firstindex(cc)-len, vls)
end
function Base.prepend!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {Tv,T<:CompressedChunk{-1,Tv},C<:Union{AbstractVector,Tuple}}
    vls = cc.vls
    ofs = cc.ofs
    for item in items
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        prepend!(vls, item)
        pushfirst!(ofs, 1)
        L = length(item)
        for i = 2:length(ofs)
            ofs[i] += L
        end
    end
    return T(firstindex(cc)-length(items), ofs, vls)
end
function Base.prepend!(cc::T, items::AbstractCompressedChunk) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    prepend!(vls, items.vls)
    return T(firstindex(cc)-length(items), vls)
end
function Base.prepend!(cc::T, items::AbstractCompressedChunk) where {T<:CompressedChunk{-1}}
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
    return T(firstindex(cc)-len, ofs, vls)
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
function Base.prepend!(cc::T, iter) where {T<:CompressedChunk{-1}}
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
    return T(firstindex(cc)-n, ofs, vls)
end


"Delete specified element with index `idx` and thus split vector `cc` in two parts and retuns them in tuple."
function splitat!(cc::T, idx::Integer) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    ofs = cc.ofs
    pos = Int(idx - first(vls.idx) + 1)
    vls2 = vls[ofs[pos+1]:end]
    resize!(vls, ofs[pos]-1)
    return (T(firstindex(cc), vls), T(idx+1, vls2))
end
function splitat!(cc::T, idx::Integer) where {Tv,Ti,T<:AbstractCompressedChunk{Tv,Ti,-1}}
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

