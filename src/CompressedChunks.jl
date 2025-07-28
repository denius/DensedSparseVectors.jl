
"""
_CompressedChunk_ is an minimal continuous storage "block" in DensedSparseVector,
i.e. CompressedChunk is the continuous non-zeros data between "sparse holes".
"""
module CompressedChunks

export AbstractCompressedChunk
export CompressedChunk, CompressedChunk0, CompressedChunk1, CompressedChunkL, CompressedChunkVL
export compressedchunk, compressedchunk_type


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



abstract type AbstractCompressedChunk{L,Tv,Ti} <: AbstractVector{Tv} end

# TODO: Try https://github.com/JuliaArrays/HybridArrays.jl as the storage.


# TODO: Not the one-starting views for the CompressedChunk to have
# for DensedSparseVectors the ability to have this option.
# See https://docs.julialang.org/en/v1/devdocs/offset-arrays/
# It is possible via the OffsetArrays.jl



" Evaluate type for `ptr` field of `struct CompressedChunk` during compilation"
@inline function cc_field_ptr_type(::Val{L}) where {L}
    if L == -1
        return Vector{Int}
    elseif L == 0
        return UnitRange{Int}
    elseif L == 1
        return UnitRange{Int}
    else
        return StepRangeLen{Int, Int, Int, Int}
    end
end

" Evaluate type for `struct CompressedChunk`"
@inline function compressedchunk_type(L,Tv,Ti)
    return CompressedChunk{L,Tv,Ti,cc_field_ptr_type(Val(L))}
end


"""
The _Compressed Chunk_ types are the structs like the `OffsetVector`
with continuously stored blocks (or scalars as blocks with L=1),
but the `getindex` and another operations are for the values/blocks.

I.e. CompressedChunk is the OffsetVector of Vectors structure.

One-dimensional `getindex` will always return `view` on block.
Two-dimensional `getindex` will return an value in the block at the
_second_index_ position. The behavior of `setindex!` is similar.

The iterator will return the view on blocks.

`struct CompressedChunk{L,Tv,Ti,Tp} <: AbstractCompressedChunk{L,Tv,Ti}`
is an universal struct which properties are determined by type parameters.

## Parameterized type storage:

~~`L = ''` -- scalar values stored in `vls`, i.e. blocks length L = 1.
The iterators will returns scalars nor blocks.~~

~~`L = 0` -- In this case the CompressedChunk store only one block in chunk,
thus it can be imagine as L = length(cc.vls) (IS IT USELESS?);~~

`L = 0` -- scalar values stored in `vls`, i.e. blocks length L = 1.
Almost the same as `L = 1`, but iterators will returns scalars instead of blocks.

`L = 1` -- scalar values stored in `vls`, i.e. blocks length L = 1.
IS IT NEED separate `L = 1` if there is exist `L = 0` for scalars???
There is exist common CompressedChunk{1,Tv,Ti}!!!!!
The only small advantage over common CompressedChunk{L} is the some faster `ptr`
calculation because UnitRange instead StepRangeLen.

`L = number` -- vector blocks with length L stored in `vls`;

`L = -1` -- variable length blocks stored in `vls`,
in `ptr` stored the starts of blocks in `vls`.

## Other parameters

`Tv` and `Ti` are the type of stored values and type of its indices.

`Tp` is the type for internal storage for offsets `ptr`. It is different for different `L`,
and can't be evaluated at compilation time, thus it calculated by `cc_field_ptr_type(Val(L))`
at creating.

## Internals

`idx` is the UnitRange{Ti} with the first and last indices of values/blocks in current chunk:
`firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`.
`idx` can be considered as offset axes.
It is useful for fast access to the blocks indices without the math and the length evaluations.

`ptr` is the same as the `colptr` in `SparseMatrixCSC`. `ptr` is the unified interface for
all AbstractCompressedChunk to have the `Int` indices for fast access
to the start positions of blocks in the storage `vls`. 

`vls` is the Vector{Tv} which continuously stored all block/scalar values.

$(TYPEDEF)
Struct fields:
$(TYPEDFIELDS)
"""
struct CompressedChunk{L,Tv,Ti,Tp} <: AbstractCompressedChunk{L,Tv,Ti}
    "the indices of first value/block and last value/block in chunk:
     `firstindex(cc) = first(cc.idx)` and `lastindex(cc) = last(cc.idx)`"
    idx::UnitRange{Ti}
    "`ptr` refers to start positions of each value/block in `vls`. And in the last position store after the last index of `vls`"
    #ptr::UnitRange{Int}
    ptr::Tp
    "the values/blocks are stored continuously in `vls`"
    vls::Vector{Tv}

    function CompressedChunk{L,Tv,Ti,Tp}(idx::UnitRange{Ti}, ptr::Tp, vls::Vector{Tv}) where {L,Tv,Ti,Tp}
        # TODO: checks sizes via asserts. Check alignment of idx, ptr and vls.
        return new{L,Tv,Ti,Tp}(idx, ptr, vls)
    end

    CompressedChunk(i::Number, vls) = CompressedChunk(range(i, length=length(vls)), vls)
    CompressedChunk(i::UnitRange, vls) = CompressedChunk{0,eltype(vls),eltype(i),cc_field_ptr_type(Val(0))}(i, vls)

    CompressedChunk{L}(i, vls) where L = CompressedChunk{L,eltype(vls),eltype(i),cc_field_ptr_type(Val(L))}(i, vls)

    function CompressedChunk{L}(i::Number, vls) where L
        if L == 0 || (L == -1 && length(vls) == 0)
            CompressedChunk{L,eltype(vls),eltype(i),cc_field_ptr_type(Val(L))}(range(i, length=length(vls)), vls)
        elseif L > 0
            CompressedChunk{L,eltype(vls),eltype(i),cc_field_ptr_type(Val(L))}(range(i, length=div(length(vls), L)), vls)
        else # if L == -1
            throw(ArgumentError(LazyString("CompressedChunk{L}(i::Number, vls) where L: unreleased yet option of L = $(L)")))
        end
    end

    function CompressedChunk{L,Tv,Ti,Tp}(r::UnitRange, vls) where {L,Tv,Ti,Tp}
        if L < 0 || !isa(vls, Vector{Tv})
            throw(ArgumentError(LazyString("CompressedChunk{L}(i::Number, vls) where L: unreleased yet option of L = $(L) and vls is not an Vector")))
        end
        if L == 0 || L == 1
            n = length(vls)
            @assert length(r) == n
            ur = range(1, n+1)
            return new{L,Tv,Ti,Tp}(UnitRange{Ti}(r), ur, vls)
        elseif L > 0
            @assert mod(length(vls), L) == 0
            lenv = length(vls)
            n = div(lenv, L)
            @assert length(r) == n
            srl = StepRangeLen{Int,Int,Int,Int}(1,L,n+1)
            return new{L,Tv,Ti,Tp}(UnitRange{Ti}(r), srl, vls)
        else#if L == -1
            # create empty CompressedChunk{-1}
            @assert length(r) == 0 && length(vls) == 0
            return new{L,Tv,Ti,Tp}(UnitRange{Ti}(r), Int[1], vls)
        #else#if L == -1
        #    @assert first(ptr) == 1 && last(ptr) - 1 == length(vls)
        #    @assert issorted(ptr)
        #    n = length(ptr) - 1
        #    @assert length(r) == n
        #    if vls isa Vector{Tv} && ptr isa Vector{Int}
        #        new{-1,Tv,Ti}(UnitRange{Ti}(r), ptr, vls)
        #    else
        #        throw(MethodError(CompressedChunkVL{-1,Tv,Ti}, (UnitRange{Ti}(r), ptr, vls)))
        #    end
        end
    end

end



const CompressedChunk0{L,Tv,Ti} = CompressedChunk{0,Tv,Ti,UnitRange{Int}}
const CompressedChunk1{L,Tv,Ti} = CompressedChunk{1,Tv,Ti,UnitRange{Int}}
const CompressedChunkL{L,Tv,Ti} = CompressedChunk{L,Tv,Ti,StepRangeLen{Int,Int,Int,Int}}
const CompressedChunkVL{L,Tv,Ti} = CompressedChunk{-1,Tv,Ti,Vector{Int}}

const CompressedScalarChunk{L,Tv,Ti} = Union{CompressedChunk0{Tv,Ti}}
const CompressedBlockChunk{L,Tv,Ti} = Union{CompressedChunk1{Tv,Ti}, CompressedChunkL{L,Tv,Ti}, CompressedChunkVL{Tv,Ti}}

function compressedchunk(L, i::Integer, val)
    Ti = eltype(i)
    Tv = eltype(val)
    if L == 0
        return CompressedChunk{L}(UnitRange{Ti}(i:i), val)
    elseif L == 1
        if length(val) == 1
            return CompressedChunk{L}(UnitRange{Ti}(i:i), Vector{Tv}([val[1]]))
        else
            return CompressedChunk{L}(UnitRange{Ti}(i:i), Vector{Tv}(val))
        end
    elseif L == -1
        throw(ArgumentError(LazyString("compressedchunk(L, i::Integer, val): unreleased yet option of L = $(L)")))
    else # if L > 1
        if length(val) == L
            return CompressedChunk{L}(UnitRange{Ti}(i:i), Vector{Tv}(val))
        else
            throw(ArgumentError(LazyString("length(val)=$(length(val)) is not equal L=$(L)")))
        end
    end
end

function compressedchunk(::Type{T}, i::Integer, val) where {L,Tv,Ti, T<:AbstractCompressedChunk{L,Tv,Ti}}
    if L == 0
        return T(UnitRange{Ti}(i:i), val)
    elseif L == 1
        if length(val) == 1
            return T(UnitRange{Ti}(i:i), Vector{Tv}([val[1]]))
        else
            return T(UnitRange{Ti}(i:i), Vector{Tv}(val))
        end
    elseif L == -1
        throw(ArgumentError(LazyString("compressedchunk(L, i::Integer, val): unreleased yet option of L = $(L)")))
    else # if L > 1
        if length(val) == L
            return T(UnitRange{Ti}(i:i), Vector{Tv}(val))
        else
            throw(ArgumentError(LazyString("length(val)=$(length(val)) is not equal L=$(L)")))
        end
    end
end


function Base.similar(cc::CompressedChunk{L,Tv,Ti,Tp}) where {L,Tv,Ti,Tp}
    idx = copy(cc.idx)
    ptr = copy(cc.ptr)
    vls = similar(cc.vls)
    return CompressedChunk{L,Tv,Ti,Tp}(idx, ptr, vls)
end

function Base.copy(cc::CompressedChunk{L,Tv,Ti,Tp}) where {L,Tv,Ti,Tp}
    idx = copy(cc.idx)
    ptr = copy(cc.ptr)
    vls = copy(cc.vls)
    return CompressedChunk{L,Tv,Ti,Tp}(idx, ptr, vls)
end

#
# TODO: Refactor all code below as the CompressedChunk{0} and CompressedChunk1 are the distinct types
# and thus there is no reason for separate functions -- all functions should be AbstractCompressedChunk only!
#

# # size2(cc::CompressedChunk{0,Tv}) where {Tv}   = 1
# # size2(_::CompressedChunk{L,Tv}) where {L,Tv}  = L
# size2(cc::CompressedChunk{0})                   = 1
# size2(_::CompressedChunkL{L,Tv}) where {L,Tv} = L
# function size2(cc::AbstractCompressedChunk)
#     l = 0
#     for i = 1:length(cc)
#         l = max(l, cc.ptr[i+1]-cc.ptr[i])
#     end
#     l
# end

@inline Base.in(i::Integer, cc::AbstractCompressedChunk) = in(i, cc.idx)
Base.@propagate_inbounds Base.length(cc::AbstractCompressedChunk) = length(cc.ptr) - 1
Base.@propagate_inbounds Base.size(cc::AbstractCompressedChunk) = (length(cc), )
Base.@propagate_inbounds Base.axes(cc::AbstractCompressedChunk) = (firstindex(cc):lastindex(cc),)
Base.@propagate_inbounds function axes(cc::AbstractCompressedChunk, d)
    @inline
    d::Integer <= 1 ? (firstindex(cc):lastindex(cc)) : OneTo(1)
end
Base.@propagate_inbounds Base.values(cc::AbstractCompressedChunk) = cc.vls


SparseArrays.nnz(cc::AbstractCompressedChunk) = length(cc.vls)

# Base.@propagate_inbounds Base.size(cc::CompressedChunk{L,Tv}) where {L,Tv}  = (length(cc), L)
# Base.@propagate_inbounds Base.size(cc::CompressedChunkL{L,Tv}) where {L,Tv} = (length(cc), L)
# Base.@propagate_inbounds Base.size(cc::CompressedChunk{-1,Tv}) where Tv     = (length(cc), size2(cc))
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
# Base.@propagate_inbounds _getindex(cc::CompressedChunkL{L,Tv}, i::Integer, j::Integer) where {L,Tv} = cc.vls[(i-1)*L + j]
# Base.@propagate_inbounds _getindex(cc::CompressedChunk{-1}, i::Integer, j::Integer)                   = cc.vls[cc.ptr[i]+j-1]
#
# Base.@propagate_inbounds _getindex(cc::CompressedChunk{0}, i::Integer)                                = @view(cc.vls[i:i])
# Base.@propagate_inbounds _getindex(cc::CompressedChunkL{L,Tv}, i::Integer) where {L,Tv}             = @view(cc.vls[1+(i-1)*L:i*L])
# Base.@propagate_inbounds _getindex(cc::CompressedChunk{-1}, i::Integer)                               = @view(cc.vls[cc.ptr[i]:cc.ptr[i+1]-1])

Base.@propagate_inbounds _getindex(cc::AbstractCompressedChunk, idx::Integer, j::Integer) = (i=idx-firstindex(cc)+1; cc.vls[cc.ptr[i]+j-1])
Base.@propagate_inbounds _getindex(cc::AbstractCompressedChunk, idx::Integer) = (i=idx-firstindex(cc)+1; @view(cc.vls[cc.ptr[i]:cc.ptr[i+1]-1]))

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
@inline blocklength(cc::AbstractCompressedChunk, i::Integer) = (idx = i-firstindex(cc)+1; cc.ptr[idx+1] - cc.ptr[idx])

Base.@propagate_inbounds function _setindex!(cc::AbstractCompressedChunk{Tv}, item, idx::Integer, j::Integer) where Tv
    i = idx-firstindex(cc)+1
    cc.vls[cc.ptr[i]+j-1] = Tv(item)
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
    @view(cc.vls[cc.ptr[i]:cc.ptr[i+1]-1]) .= item
    item
end
Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1,Tv}, item::Union{AbstractVector{Tv},AbstractRange{Tv}}, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    @view(cc.vls[cc.ptr[i]:cc.ptr[i+1]-1]) .= item
    item
end
Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1,Tv}, item, idx::Integer) where Tv
    i = idx-firstindex(cc)+1
    @view(cc.vls[cc.ptr[i]:cc.ptr[i+1]-1]) .= Tv.(item)
    item
end

Base.@propagate_inbounds function _setindex!(cc::CompressedChunk{-1}, item, idx::Integer)
    i = idx-firstindex(cc)+1
    if blocklength(cc, idx) == length(item)
        @view(cc.vls[cc.ptr[i]:cc.ptr[i+1]-1]) .= item
    else #if blocklength(cc, idx) != length(item)
        vls = cc.vls
        ptr = cc.ptr
        splice!(vls, ptr[i]:ptr[i+1]-1, item)
        dif = (ptr[i+1]-ptr[i]) - length(item)
        for k = i+1:length(ptr)
            ptr[k] -= dif
        end
    end
    return item
end

Base.push!(cc::T, item) where {T<:CompressedChunk{0}} = T(firstindex(cc), push!(cc.vls, item))
function Base.push!(cc::T, items::Union{AbstractVector,Tuple}) where {L,Tv,T<:CompressedChunk{L,Tv}}
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
    ptr = cc.ptr
    push!(ptr, last(ptr) + length(items))
    return T(firstindex(cc), ptr, vls)
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
    ptr = cc.ptr
    prepend!(vls, items)
    pushfirst!(ptr, 1)
    len = length(items)
    for i = 2:length(ptr)
        ptr[i] += len
    end
    return T(firstindex(cc)-1, ptr, vls)
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
    ptr = cc.ptr
    L = ptr[2] - ptr[1]
    deleteat!(vls, 1:L)
    popfirst!(ptr)
    for i = 1:length(ptr)
        ptr[i] -= L
    end
    return T(firstindex(cc)+1, ptr, vls)
end

"Return a `CompressedChunk` consisting of all but the last component of `cc`."
function front!(cc::T) where {T<:CompressedChunk{0}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    return T(firstindex(cc), pop!(cc.vls))
end
function front!(cc::T) where {L,Tv,T<:CompressedChunk{L,Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    vls = cc.vls
    ptr = cc.ptr
    len = length(vls) - L
    resize!(vls, len)
    pop!(ptr)
    return T(firstindex(cc), vls, ptr)
end
function front!(cc::T) where {Tv,T<:CompressedChunk{-1,Tv}}
    length(cc) == 0 && throw(ArgumentError("Cannot call front! on an empty tuple"))
    vls = cc.vls
    ptr = cc.ptr
    L = ptr[end] - ptr[end-1]
    len = length(vls) - L
    resize!(vls, len)
    pop!(ptr)
    return T(firstindex(cc), ptr, vls)
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
    ptr = cc.ptr
    for item in items
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        append!(vls, item)
        push!(ptr, last(ptr) + length(item))
    end
    return T(firstindex(cc), ptr, vls)
end
function Base.append!(cc::T, items::AbstractCompressedChunk) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    append!(vls, items.vls)
    return T(firstindex(cc), vls)
end
function Base.append!(cc::T, items::AbstractCompressedChunk) where {T<:CompressedChunk{-1}}
    vls = cc.vls
    ptr = cc.ptr
    len = length(ptr) - 1
    append!(vls, items.vls)
    length(items.ptr) > 1 && append!(ptr, @view(items.ptr[2:end]))
    for i = len+1+1:length(ptr)
        ptr[i] += ptr[i-1] - 1
    end
    return T(firstindex(cc), ptr, vls)
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
    ptr = cc.ptr
    n = 0
    for item in iter
        n += 1
        push!(vls, item)
    end
    push!(ptr, last(ptr) + n)
    return T(firstindex(cc), ptr, vls)
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
function Base.prepend!(cc::T, items::Union{AbstractVector{C},Tuple{C}}) where {L,Tv,T<:CompressedChunkL{L,Tv},C<:Union{AbstractVector,Tuple}}
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
    ptr = cc.ptr
    for item in items
        item isa Tuple && (item = map(x -> convert(Tv, x), item))
        prepend!(vls, item)
        pushfirst!(ptr, 1)
        L = length(item)
        for i = 2:length(ptr)
            ptr[i] += L
        end
    end
    return T(firstindex(cc)-length(items), ptr, vls)
end
function Base.prepend!(cc::T, items::AbstractCompressedChunk) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    prepend!(vls, items.vls)
    return T(firstindex(cc)-length(items), vls)
end
function Base.prepend!(cc::T, items::AbstractCompressedChunk) where {T<:CompressedChunk{-1}}
    vls = cc.vls
    ptr = cc.ptr
    len = length(items.ptr) - 1
    prepend!(vls, items.vls)
    if len > 0
        popfirst!(ptr)
        prepend!(ptr, items.ptr)
    end
    for i = len+1+1:length(ptr)
        ptr[i] += ptr[i-1] - 1
    end
    return T(firstindex(cc)-len, ptr, vls)
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
    ptr = cc.ptr
    n = 0
    for item in iter
        n += 1
        pushfirst!(vls, item)
    end
    reverse!(vls, 1, n)
    pushfirst!(ptr, 1)
    for i = 2:length(ptr)
        ptr[i] += n
    end
    return T(firstindex(cc)-n, ptr, vls)
end


"Delete specified element with index `idx` and thus split vector `cc` in two parts and retuns them in tuple."
function splitat!(cc::T, idx::Integer) where {T<:AbstractCompressedChunk}
    vls = cc.vls
    ptr = cc.ptr
    pos = Int(idx - first(vls.idx) + 1)
    vls2 = vls[ptr[pos+1]:end]
    resize!(vls, ptr[pos]-1)
    return (T(firstindex(cc), vls), T(idx+1, vls2))
end
function splitat!(cc::T, idx::Integer) where {Tv,Ti,T<:AbstractCompressedChunk{-1,Tv,Ti}}
    vls = cc.vls
    ptr = cc.ptr
    pos = Int(idx - first(vls.idx) + 1)
    vls2 = vls[ptr[pos+1]:end]
    ptr2 = ptr[pos+1:end]
    i0 = first(ptr2) - 1
    ptr2 .-= i0
    resize!(vls, ptr[pos]-1)
    resize!(ptr, pos)
    return (T(firstindex(cc), vls, ptr), T(idx+1, vls2, ptr2))
end


# insert!
# deleteat! -- to multiple delete use range or some other collection: deleteat!(collection, inds)
# splice! -- to inset use `splice!(collection, n:n-1, replacement)`
# resize! -- not need


# function Base.show(io::IO, ::MIME"text/plain", x::Union{CompressedChunk0{Tv},CompressedChunk{Tv,0}}) where Tv
#     # print(io, length(x), "-element ", typeof(x))
#     print(io, length(x), "-element ", typeof(x), " with indices ", x.idx)
#     if length(x) != 0
#         println(io, ":")
#         show(IOContext(io, :typeinfo => eltype(x)), x)
#     end
# end
# function Base.show(io::IOContext, x::Union{CompressedChunk0{Tv},CompressedChunk{Tv,0}}) where Tv
#     if isempty(x)
#         return show(io, MIME("text/plain"), x)
#     end
#     limit = get(io, :limit, false)::Bool
#     half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
#     if !haskey(io, :compact)
#         io = IOContext(io, :compact => true)
#     end
#     for k = eachindex(x)
#         if k < half_screen_rows || k > length(x) - half_screen_rows
#             print(io, " ")
#             if isassigned(x, Int(k))
#                 show(io, x[k,1])
#             else
#                 print(io, Base.undef_ref_str)
#             end
#             k != lastindex(x) && println(io)
#         elseif k == half_screen_rows
#             # println(io, "   ", " "^pad, "   \u22ee")
#             println(io, " \u22ee")
#         end
#     end
# end


function Base.show(io::IO, ::MIME"text/plain", x::AbstractCompressedChunk)
    # print(io, length(x), "-element ", typeof(x))
    print(io, length(x), "-element ", typeof(x), " with indices ", x.idx)
    if length(x) != 0
        println(io, ":")
        # show(IOContext(io, :typeinfo => eltype(x)), x)
        show(IOContext(io, :typeinfo => Vector{eltype(x)}), x)
    end
end
function Base.show(io::IOContext, x::AbstractCompressedChunk)
    if isempty(x)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(x)
        if k < half_screen_rows || k > length(x) - half_screen_rows
            print(io, " ")
            if isassigned(x, Int(k))
                show(io, x[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != lastindex(x) && println(io)
        elseif k == half_screen_rows
            # println(io, "   ", " "^pad, "   \u22ee")
            println(io, " \u22ee")
        end
    end
end

end  # of module CompressedChunks

