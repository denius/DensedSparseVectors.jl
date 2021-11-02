#
#  SpacedVector
#  DensedSparseVector
#
# fast and slow realizations:
# 1. on two vectors: Vector{FirstIndex<:Int} and Vector{SomeVectorData}
# 2. on SortedDict{FirstIndex<:Int, SomeVectorData}.
# The first realization is for fast index access, the second one is for creating and rebuilding.
# Also on the SortedDict{Index<:Int, value} -- the simples and bug free.

#module DensedSparseVectors
#export DensedSparseVector

import Base.Broadcast: BroadcastStyle
using Base.Broadcast: AbstractArrayStyle, Broadcasted, DefaultArrayStyle
using DataStructures
using FillArrays
using SparseArrays
using Random

#import Base: getindex, setindex!, unsafe_load, unsafe_store!, nnz, length, isempty


abstract type AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractSpacedVector{Tv,Ti,Tx,Ts} <: AbstractSpacedDensedSparseVector{Tv,Ti,Tx,Ts} end
abstract type AbstractDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc} end

mutable struct SpacedVectorIndex{Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{Int}} <: AbstractSpacedVector{Bool,Ti,Tx,Ts}
    n::Int     # the vector length
    nnz::Int   # number of non-zero elements
    nzind::Tx  # Vector of chunk's first indices
    data::Ts   # Vector{Int} -- Vector of chunks lengths
end

#mutable struct SpacedVector{Tv,Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{<:AbstractVector{Tv}}} <: AbstractSpacedVector{Tv,Ti,Tx,Ts}
mutable struct SpacedVector{Tv,Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{<:AbstractVector}} <: AbstractSpacedVector{Tv,Ti,Tx,Ts}
    n::Int     # the vector length
    nnz::Int   # number of non-zero elements
    nzind::Tx  # Vector of chunk's first indices
    data::Ts   # Td{<:AbstractVector{Tv}} -- Vector of Vectors (chunks) with values
end

mutable struct DensedSparseVector{Tv,Ti,Td<:AbstractVector{Tv},Tc<:AbstractDict{Ti,Td}} <: AbstractDensedSparseVector{Tv,Ti,Td,Tc}
    n::Int       # the vector length
    nnz::Int     # number of non-zero elements
    lastkey::Ti  # the last node key in `data` tree
    data::Tc     # Tc{Ti,Td{Tv}} -- tree based (sorted) Dict data container
end

include("constructors.jl")

function Base.length(v::AbstractDensedSparseVector)
    if v.n != 0
        return v.n
    elseif !isempty(v.data)
        (ilastfirst, chunk) = last(v.data)
        return Int(ilastfirst) + length(chunk) - 1
    else
        return 0
    end
end
SparseArrays.nnz(v::AbstractSpacedDensedSparseVector) = v.nnz
Base.isempty(v::AbstractSpacedDensedSparseVector) = v.nnz == 0
Base.size(v::AbstractSpacedDensedSparseVector) = (v.n,)
Base.axes(v::AbstractSpacedDensedSparseVector) = (Base.OneTo(v.n),)
Base.ndims(::AbstractSpacedDensedSparseVector) = 1
Base.ndims(::Type{AbstractSpacedDensedSparseVector}) = 1
Base.strides(v::AbstractSpacedDensedSparseVector) = (1,)
Base.eltype(v::AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} = Pair{Ti,Tv}
Base.IndexStyle(::AbstractSpacedDensedSparseVector) = IndexLinear()

function Base.similar(v::SpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    nzind = similar(v.nzind)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(zip(v.nzind, v.data))
        nzind[i] = k
        data[i] = similar(d)
    end
    return SpacedVector{Tv,Ti,Tx,Ts}(v.n, v.nnz, nzind, data)
end
function Base.similar(v::SpacedVector{Tv,Ti,Tx,Ts}, ::Type{ElType}) where {Tv,Ti,Tx,Ts,ElType<:Pair{Tin,Tvn}} where {Tin,Tvn}
    nzind = similar(v.nzind, Tin)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(zip(v.nzind, v.data))
        nzind[i] = k
        data[i] = similar(d, Tvn)
    end
    return SpacedVector{Tvn,Tin,typeof(nzind),typeof(data)}(v.n, v.nnz, nzind, data)
end
function Base.similar(v::SpacedVector{Tv,Ti,Tx,Ts}, ::Type{ElType}) where {Tv,Ti,Tx,Ts,ElType}
    nzind = similar(v.nzind)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(zip(v.nzind, v.data))
        nzind[i] = k
        data[i] = similar(d, ElType)
    end
    #@debug "in similar(SpacedVector): ElType = $ElType"
    return SpacedVector{ElType,Ti,typeof(nzind),typeof(data)}(v.n, v.nnz, nzind, data)
end

@inline get_chunk_length(v::SpacedVectorIndex, chunk) = chunk
@inline get_chunk_length(v::SpacedVector, chunk) = size(chunk)[1]
@inline get_chunk_length(v::DensedSparseVector, chunk) = size(chunk)[1]
@inline get_chunk(v::SpacedVectorIndex, chunk) = trues(chunk)
@inline get_chunk(v::SpacedVector, chunk) = chunk
@inline get_chunk(v::DensedSparseVector, chunk) = chunk
@inline get_chunk_value(v::SpacedVectorIndex, chunk, i) = 1 <= i <= chunk
@inline get_chunk_value(v::SpacedVector, chunk, i) = chunk[i]
@inline get_chunk_value(v::DensedSparseVector, chunk, i) = chunk[i]
@inline pairschunks(v::AbstractSpacedVector) = zip(v.nzind, v.data)
@inline pairschunks(v::AbstractDensedSparseVector) = pairs(v.data)

function SparseArrays.nonzeroinds(v::AbstractSpacedDensedSparseVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    ret = Vector{Ti}()
    for (k,d) in pairschunks(v)
        append!(ret, (k:k+get_chunk_length(v, d)-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractSpacedDensedSparseVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    ret = Tv===Bool ? BitVector() : Vector{Tv}()
    for (k,d) in pairschunks(v)
        append!(ret, get_chunk(v, d))
    end
    return ret
end

"`iteratenzpairs(v::AbstractVector)` iterates over nonzeros and returns pair of index and value"
Base.@propagate_inbounds function iteratenzpairs(v::SparseVector, state = (1, length(v.nzind)))
    i, len = state
    if i <= len
        return ((@inbounds v.nzind[i], @inbounds v.nzval[i]), (i+1, len))
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzpairs(v::Vector, state = 1)
    if state-1 < length(v)
        return ((state, @inbounds v[state]), state + 1)
        #return (Pair(state, @inbounds v[state]), state + 1)
    else
        return nothing
    end
end

struct SVIIteratorState
    next::Int          #  index of current chunk
    nextpos::Int       #  index in the current chunk
    currentkey::Int
    chunklen::Int
end

function get_init_state(v::SpacedVectorIndex)
    if (ret = iteratenzchunks(v)) !== nothing
        i, next = ret
        return SVIIteratorState(next, 1, v.nzind[i], v.data[i])
    else
        return SVIIteratorState(1, 1, 1, Td[])
    end
end
Base.@propagate_inbounds function iteratenzpairs(v::SpacedVectorIndex{Ti,Tx,Ts}, state = get_init_state(v)) where {Ti,Tx,Ts}
    next, nextpos, key, chunklen = state.next, state.nextpos, state.currentkey, state.chunklen
    if nextpos <= chunklen
        return ((Ti(key+nextpos-1), true), SVIIteratorState(next, nextpos + 1, key, chunklen))
    elseif (ret = iteratenzchunks(v, next)) !== nothing
        i, next = ret
        key = v.nzind[i]; chunklen = v.data[i]
        return ((key, true), SVIIteratorState(next, 2, Int(key), chunklen))
    else
        return nothing
    end
end

struct SVIteratorState{Td}
    next::Int          #  index of next chunk
    nextpos::Int       #  index in the current chunk of item will be get
    currentkey::Int
    chunk::Td
end

function get_init_state(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td
    if (ret = iteratenzchunks(v)) !== nothing
        i, next = ret
        return SVIteratorState{Td}(next, 1, v.nzind[i], v.data[i])
    else
        return SVIteratorState{Td}(1, 1, 1, Td[])
    end
end
Base.@propagate_inbounds function iteratenzpairs(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}, state = get_init_state(v)) where {Ti,Tx,Ts<:AbstractVector{Td}} where {Td<:AbstractVector{Tv}} where Tv
    next, nextpos, key, chunk = state.next, state.nextpos, state.currentkey, state.chunk
    if nextpos <= length(chunk)
        return ((Ti(key+nextpos-1), chunk[nextpos]), SVIteratorState{Td}(next, nextpos + 1, key, chunk))
    elseif (ret = iteratenzchunks(v, next)) !== nothing
        i, next = ret
        key = v.nzind[i]; chunk = v.data[i]
        return ((key, chunk[1]), SVIteratorState{Td}(next, 2, Int(key), chunk))
    else
        return nothing
    end
end


"`iteratenzpairs(v::AbstractVector)` iterates over nonzeros and returns pair of index and `Ref` of value"
Base.@propagate_inbounds function iteratenzpairsRef(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}, state = get_init_state(v)) where {Ti,Tx,Ts<:AbstractVector{Td}} where {Td<:AbstractVector{Tv}} where Tv
    next, nextpos, key, chunk = state.next, state.nextpos, state.currentkey, state.chunk
    if nextpos <= length(chunk)
        return ((Ti(key+nextpos-1), Ref(chunk, nextpos)), SVIteratorState{Td}(next, nextpos + 1, key, chunk))
    elseif (ret = iteratenzchunks(v, next)) !== nothing
        i, next = ret
        key = v.nzind[i]; chunk = v.data[i]
        return ((key, Ref(chunk, 1)), SVIteratorState{Td}(next, 2, Int(key), chunk))
    else
        return nothing
    end
end

# TODO: replace iteratenzpairs(v::SubArray) with Iterator which call iteratenzpairs(v.parent)
struct NZIteratePairsSubArray{T}
    p::T
end

function get_init_state(v::SubArray{<:Any,<:Any,<:AbstractSpacedVector{Tv,Ti,Tx,Ts}}) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td
    if nnz(v.parent) == 0
        return SVIteratorState{Td}(1, 1, Ti(1), Td[])
    else
        i = first(v.indices[1])
        key = searchsortedlast(v.parent.nzind, i)
        if key == 0
            return SVIteratorState{Td}(1, 1, v.parent.nzind[1], v.parent.data[1])
        else
            return SVIteratorState{Td}(key, i - v.parent.nzind[key] + 1, v.parent.nzind[key], v.parent.data[key])
        end
    end
end
Base.@propagate_inbounds function iteratenzpairs(v::SubArray{<:Any,<:Any,<:AbstractSpacedVector{Tv,Ti,Tx,Ts},<:Tuple{UnitRange{<:Any}}}, state = get_init_state(v)) where {Ti,Tx,Ts<:AbstractVector{Td}} where {Td<:AbstractVector{Tv}} where Tv

    next, nextpos, key, chunk = state.next, state.nextpos, state.currentkey, state.chunk
    i = convert(Ti, key + nextpos-1)

    if i > last(v.indices[1])
        return nothing
    elseif nextpos < length(chunk)
        d = chunk[nextpos]
        return ((i, d), SVIteratorState{Td}(next, nextpos + 1, key, chunk))
    elseif nextpos == length(chunk)
        d = chunk[nextpos]
        if next < length(v.parent.nzind)
            return ((i, d), SVIteratorState{Td}(next + 1, 1, v.parent.nzind[next+1], v.parent.data[next+1]))
        elseif next == length(v.parent.nzind)
            return ((i, d), SVIteratorState{Td}(next, nextpos + 1, key, chunk))
        else
            return nothing
        end
    else
        return nothing
    end
end

struct DSVIteratorState{Td}
    semitoken::DataStructures.Tokens.IntSemiToken
    nextpos::Int       #  index in the current chunk
    currentkey::Int
    chunk::Td
end

function get_init_state(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc}
    if (ret = iteratenzchunks(v)) !== nothing
        i, next = ret
        return DSVIteratorState{Td}(next, 1, deref_key((v.data, i)), deref_value((v.data, i)))
    else
        return DSVIteratorState{Td}(startof(v.data), 1, 1, Td[])
    end
end
Base.@propagate_inbounds function iteratenzpairs(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}, state = get_init_state(v)) where {Tv,Ti,Td,Tc}
    next, nextpos, key, chunk = state.semitoken, state.nextpos, state.currentkey, state.chunk
    if nextpos <= length(chunk)
        return ((Ti(key+nextpos-1), chunk[nextpos]), DSVIteratorState{Td}(next, nextpos + 1, key, chunk))
    elseif (ret = iteratenzchunks(v, next)) !== nothing
        i, next = ret
        key, chunk = deref((v.data, i))
        return ((key, chunk[1]), DSVIteratorState{Td}(next, 2, Int(key), chunk))
    else
        return nothing
    end
end

Base.@propagate_inbounds function iteratenzpairsRef(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}, state = get_init_state(v)) where {Tv,Ti,Td,Tc}
    next, nextpos, key, chunk = state.semitoken, state.nextpos, state.currentkey, state.chunk
    if nextpos <= length(chunk)
        i = convert(Ti, key + nextpos-1)
        d = Ref(chunk, nextpos)
        return ((i, d), DSVIteratorState{Td}(next, nextpos + 1, key, chunk))
    else
        if next != pastendsemitoken(v.data)
            key, chunk = deref((v.data, next))
            d = Ref(chunk, 1)
            return ((key, d), DSVIteratorState{Td}(advance((v.data,next)), 2, key, chunk))
        else
            return nothing
        end
    end
end




"`iteratenzchunks(v::AbstractVector)` iterates over nonzero chunks and returns start index of chunk and chunk"
Base.@propagate_inbounds function iteratenzchunks(v::AbstractSpacedVector, state = 1)
    if state <= length(v.nzind)
        return (state, state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(v::AbstractDensedSparseVector, state = startof(v.data))
    if state != pastendsemitoken(v.data)
        stnext = advance((v.data, state))
        return (state, stnext)
    else
        return nothing
    end
end


struct NZInds{It}
    itr::It
end
"`nzinds(v::AbstractVector)` is the `Iterator` over nonzero indices of vector `v`"
nzinds(itr) = NZInds(itr)
@inline function Base.iterate(it::NZInds, state...)
    y = iteratenzpairs(it.itr, state...)
    if y !== nothing
        return (y[1][1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZInds{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZInds{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZInds}) = Base.SizeUnknown()
Base.reverse(it::NZInds) = NZInds(reverse(it.itr))
@inline Base.keys(v::AbstractSpacedDensedSparseVector) = nzinds(v)

struct NZVals{It}
    itr::It
end
"`nzvals(v::AbstractVector)` is the `Iterator` over nonzero values of `v`"
nzvals(itr) = NZVals(itr)
@inline function Base.iterate(it::NZVals, state...)
    y = iteratenzpairs(it.itr, state...)
    if y !== nothing
        return (y[1][2], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZVals{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZVals{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZVals}) = Base.SizeUnknown()
Base.reverse(it::NZVals) = NZVals(reverse(it.itr))

struct NZValsRef{It}
    itr::It
end
"""
`nzvalsRef(v::AbstractVector)` is the `Iterator` over nonzero values of `v`,
returns the reference `Ref` of iterated values
"""
nzvalsRef(itr) = NZValsRef(itr)
@inline function Base.iterate(it::NZValsRef, state...)
    y = iteratenzpairsRef(it.itr, state...)
    if y !== nothing
        return (y[1][2], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZValsRef{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZValsRef{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZValsRef}) = Base.SizeUnknown()
Base.reverse(it::NZValsRef) = NZValsRef(reverse(it.itr))

struct NZPairs{It}
    itr::It
end
"`nzpairs(v::AbstractVector)` is the `Iterator` over nonzeros of `v` and returns pair of index and value"
@inline nzpairs(itr) = NZPairs(itr)
@inline function Base.iterate(it::NZPairs, state...)
    y = iteratenzpairs(it.itr, state...)
    if y !== nothing
        #return (y[1], y[2])
        return (Pair(y[1]...), y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZPairs{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZPairs{It}}) where {It} = Base.EltypeUnknown()
#Base.IteratorEltype(::Type{NZPairs{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZPairs}) = Base.SizeUnknown()
Base.reverse(it::NZPairs) = NZPairs(reverse(it.itr))


SparseArrays.findnz(v::AbstractSpacedDensedSparseVector) = (nzinds(v), nzvals(v))
#SparseArrays.findnz(v::AbstractSpacedDensedSparseVector) = (SparseArrays.nonzeroinds(v), SparseArrays.nonzeros(v))


@inline function Base.isstored(v::AbstractSpacedVector, i::Integer)
    v.nnz == 0 && return false

    st = searchsortedlast(v.nzind, i)
    if st == 0  # the index `i` is before first index
        return false
    elseif i >= v.nzind[st] + get_chunk_length(v, v.data[st])
        # the index `i` is outside of data chunk indices
        return false
    end

    return true
end
@inline Base.haskey(v::AbstractSpacedVector, i) = isstored(v, i)

@inline function Base.isstored(v::AbstractDensedSparseVector, i::Integer)
    v.nnz == 0 && return false

    st = searchsortedlast(v.data, i)
    sstatus = status((v.data, st))
    if sstatus == 2 || sstatus == 0  # the index `i` is before first index or invalid
        return false
    elseif i >= deref_key((v.data, st)) + get_chunk_length(v, deref_value((v.data, st)))
        # the index `i` is outside of data chunk indices
        return false
    end

    return true
end
@inline Base.haskey(v::AbstractDensedSparseVector, i) = isstored(v, i)


@inline function Base.getindex(v::AbstractSpacedVector{Tv,Ti,Td,Tc}, i::Integer) where {Tv,Ti,Td,Tc}

    v.nnz == 0 && return zero(Tv)

    st = searchsortedlast(v.nzind, i)

    # the index `i` is before first index
    st == 0 && return zero(Tv)

    ifirst, chunk = v.nzind[st], v.data[st]

    # the index `i` is outside of data chunk indices
    i >= ifirst + get_chunk_length(v, chunk) && return zero(Tv)

    return get_chunk_value(v, chunk, i - ifirst + 1)
end

#@inline function Base.getindex(v::SpacedVector{Tv,Ti,Td,Tc}, i::Integer) where {Tv,Ti,Td,Tc}
#
#    v.nnz == 0 && return zero(Tv)
#
#    st = searchsortedlast(v.nzind, i)
#
#    # the index `i` is before first index
#    st == 0 && return zero(Tv)
#
#    ifirst, chunk = v.nzind[st], v.data[st]
#
#    # the index `i` is outside of data chunk indices
#    i >= ifirst + length(chunk) && return zero(Tv)
#
#    return chunk[i - ifirst + 1]
#end

@inline function Base.unsafe_load(v::SpacedVector, i::Integer)
    st = searchsortedlast(v.nzind, i)
    ifirst, chunk = v.nzind[st], v.data[st]
    return chunk[i - ifirst + 1]
end


@inline function Base.getindex(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}, i::Integer) where {Tv,Ti,Td,Tc}

    v.nnz == 0 && return zero(Tv)

    st = searchsortedlast(v.data, i)

    sstatus = status((v.data, st))
    if sstatus == 2 || sstatus == 0  # the index `i` is before first index or invalid
        return zero(Tv)
    end

    (ifirst, chunk) = deref((v.data, st))

    if i >= ifirst + get_chunk_length(v, chunk)  # the index `i` is outside of data chunk indices
        return zero(Tv)
    end

    return get_chunk_value(v, chunk, i - ifirst + 1)
end


function Base.setindex!(v::SpacedVectorIndex{Ti,Tx,Ts}, value, i::Integer) where {Ti,Tx,Ts}

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if v.nnz > 0 && st > 0  # the index `i` is not before the first index
        ifirst, chunklen = v.nzind[st], v.data[st]
        if i < ifirst + chunklen
            return v
        end
    end

    if v.nnz == 0
        v.nzind = push!(v.nzind, Ti(i))
        v.data = push!(v.data, 1)
        v.nnz += 1
        v.n = max(v.n, Int(i))
        return v
    end

    if st == 0  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, Td(Fill(val,1)))
        else
            v.nzind[1] -= 1
            pushfirst!(v.data[1], val)
        end
        v.nnz += 1
        return v
    end

    ifirst, chunklen = v.nzind[st], v.data[st]

    if i >= v.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + chunklen  # there is will be the gap in indices after inserting
            push!(v.nzind, i)
            push!(v.data, 1)
        else  # just append to last chunk
            v.data[st] += 1
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + chunklen - 1
    stnext = st + 1
    inextfirst = v.nzind[stnext]

    if inextfirst - ilast == 2  # join chunks
        v.data[st] += 1 + v.data[stnext]
        v.nzind = deleteat!(v.nzind, stnext)
        v.data  = deleteat!(v.data, stnext)
    elseif i - ilast == 1  # append to left chunk
        v.data[st] += 1
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        v.data[stnext] += 1
    else  # insert single element chunk
        v.nzind = insert!(v.nzind, stnext, Ti(i))
        v.data  = insert!(v.data, stnext, 1)
    end

    v.nnz += 1
    return v

end

function Base.setindex!(v::SpacedVector{Tv,Ti,Tx,Ts}, value, i::Integer) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td
    val = value

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if v.nnz > 0 && st > 0  # the index `i` is not before the first index
        ifirst, chunk = v.nzind[st], v.data[st]
        if i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            return v
        end
    end

    if v.nnz == 0
        v.nzind = push!(v.nzind, Ti(i))
        v.data = push!(v.data, Td(Fill(val,1)))
        v.nnz += 1
        v.n = max(v.n, Int(i))
        return v
    end

    if st == 0  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, Td(Fill(val,1)))
        else
            v.nzind[1] -= 1
            pushfirst!(v.data[1], val)
        end
        v.nnz += 1
        return v
    end

    ifirst, chunk = v.nzind[st], v.data[st]

    if i >= v.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(v.nzind, i)
            push!(v.data, Td(Fill(val,1)))
        else  # just append to last chunk
            v.data[st] = push!(chunk, val)
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = st + 1
    inextfirst = v.nzind[stnext]

    if inextfirst - ilast == 2  # join chunks
        v.data[st] = append!(chunk, Td(Fill(val,1)), v.data[stnext])
        v.nzind = deleteat!(v.nzind, stnext)
        v.data  = deleteat!(v.data, stnext)
    elseif i - ilast == 1  # append to left chunk
        v.data[st] = push!(chunk, val)
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        v.data[stnext] = pushfirst!(v.data[stnext], val)
    else  # insert single element chunk
        v.nzind = insert!(v.nzind, stnext, Ti(i))
        v.data  = insert!(v.data, stnext, Td(Fill(val,1)))
    end

    v.nnz += 1
    return v

end

@inline function Base.unsafe_store!(v::SpacedVector{Tv,Ti,Tx,Ts}, value, i::Integer) where {Tv,Ti,Tx,Ts}
    st = searchsortedlast(v.nzind, i)
    v.data[st][i-v.nzind[st]+1] = Tv(value)
    return v
end



function Base.setindex!(v::DensedSparseVector{Tv,Ti,Td,Tc}, value, i::Integer) where {Tv,Ti,Td,Tc}
    val = value

    st = searchsortedlast(v.data, i)

    sstatus = status((v.data, st))
    @boundscheck if sstatus == 0 # invalid semitoken
        trow(KeyError(i))
    end

    # check the index exist and update its data
    if v.nnz > 0 && sstatus != 2  # the index `i` is not before the first index
        (ifirst, chunk) = deref((v.data, st))
        if ifirst + length(chunk) > i
            chunk[i - ifirst + 1] = val
            return v
        end
    end

    if v.nnz == 0
        v.data[i] = Td(Fill(val,1))
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastkey = Ti(i)
        return v
    end

    if sstatus == 2  # the index `i` is before the first index
        stnext = startof(v.data)
        inextfirst = deref_key((v.data, stnext))
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            v.data[i] = Td(Fill(val,1))
        else
            v.data[i] = pushfirst!(deref_value((v.data, stnext)), val)
            delete!((v.data, stnext))
        end
        v.nnz += 1
        return v
    end

    (ifirst, chunk) = deref((v.data, st))

    #if sstatus == 3  # the index `i` is after the last index
    # Note: `searchsortedlast` isn't got `status((tree, semitoken)) == 3`, the 2 only.
    #       The `searchsortedlast` is the same.
    if i >= v.lastkey # the index `i` is after the last key index
        if ifirst + length(chunk) < i  # there is will be the gap in indices after inserting
            v.data[i] = Td(Fill(val,1))
            v.lastkey = Ti(i)
        else  # just append to last chunk
            v.data[st] = push!(chunk, val)
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = advance((v.data, st))
    inextfirst = deref_key((v.data, stnext))

    if inextfirst - ilast == 2  # join chunks
        v.data[st] = append!(chunk, Td(Fill(val,1)), deref_value((v.data, stnext)))
        delete!((v.data, stnext))
        v.lastkey == inextfirst && (v.lastkey = ifirst)
    elseif i - ilast == 1  # append to left chunk
        v.data[st] = push!(chunk, val)
    elseif inextfirst - i == 1  # prepend to right chunk
        v.data[i] = pushfirst!(deref_value((v.data, stnext)), val)
        delete!((v.data, stnext))
        v.lastkey == inextfirst && (v.lastkey -= 1)
    else  # insert single element chunk
        v.data[i] = Td(Fill(val,1))
    end

    v.nnz += 1
    return v

end

function Base.setindex!(v::AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc}, data::AbstractSpacedDensedSparseVector, index::Integer) where {Tv,Ti,Td,Tc}
    i0 = Ti(index-1)
    if v === data
        cdata = deepcopy(data)
        for (i,d) in nzpairs(cdata)
            v[i0+i] = Tv(d)
        end
    else
        for (i,d) in nzpairs(data)
            v[i0+i] = Tv(d)
        end
    end
    return v
end

@inline function Base.unsafe_store!(v::DensedSparseVector{Tv,Ti,Td,Tc}, value, i::Integer) where {Tv,Ti,Td,Tc}
    (ifirst, chunk) = deref((v.data, searchsortedlast(v.data, i)))
    chunk[i - ifirst + 1] = Tv(value)
    return v
end




@inline function Base.delete!(v::SpacedVectorIndex{Ti,Tx,Ts}, i::Integer) where {Tv,Ti,Tx,Ts}

    v.nnz == 0 && return v

    st = searchsortedlast(v.nzind, i)

    if st == 0  # the index `i` is before first index
        return v
    end

    ifirst = v.nzind[st]
    lenchunk = v.data[st]

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return v
    end

    if lenchunk == 1
        v.nzind = deleteat!(v.nzind, st)
        v.data = deleteat!(v.data, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        v.data[st] -= 1
    elseif i == ifirst  # first element in chunk
        v.nzind[st] += 1
        v.data[st] -= 1
    else
        v.nzind = insert!(v.nzind, st+1, Ti(i+1))
        v.data  = insert!(v.data, st+1, lenchunk - (i-ifirst+1))
        v.data[st] -= (lenchunk-(i-ifirst+1)) + 1
    end

    v.nnz -= 1

    return v
end

@inline function Base.delete!(v::SpacedVector{Tv,Ti,Tx,Ts}, i::Integer) where {Tv,Ti,Tx,Ts}

    v.nnz == 0 && return v

    st = searchsortedlast(v.nzind, i)

    if st == 0  # the index `i` is before first index
        return v
    end

    ifirst = v.nzind[st]
    lenchunk = length(v.data[st])

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return v
    end

    if lenchunk == 1
        deleteat!(v.data[st], 1)
        v.nzind = deleteat!(v.nzind, st)
        v.data = deleteat!(v.data, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        pop!(v.data[st])
    elseif i == ifirst  # first element in chunk
        v.nzind[st] += 1
        popfirst!(v.data[st])
    else
        v.nzind = insert!(v.nzind, st+1, Ti(i+1))
        v.data  = insert!(v.data, st+1, v.data[st][i-ifirst+1+1:end])
        resize!(v.data[st], i-ifirst+1 - 1)
    end

    v.nnz -= 1

    return v
end

@inline function Base.delete!(v::DensedSparseVector{Tv,Ti,Td,Tc}, i::Integer) where {Tv,Ti,Td,Tc}

    v.nnz == 0 && return v

    st = searchsortedlast(v.data, i)

    sstatus = status((v.data, st))
    if sstatus == 2 || sstatus == 0  # the index `i` is before first index or invalid
        return v
    end

    (ifirst, chunk) = deref((v.data, st))

    if i >= ifirst + length(chunk)  # the index `i` is outside of data chunk indices
        return v
    end

    if length(chunk) == 1
        deleteat!(chunk, 1)
        v.data = delete!(v.data, i)
        i == v.lastkey && (v.lastkey = v.nnz > 1 ? deref_key((v.data, lastindex(v.data))) : typemin(Ti))
    elseif i == ifirst + length(chunk) - 1  # last index in chunk
        pop!(chunk)
        v.data[st] = chunk
    elseif i == ifirst
        popfirst!(chunk)
        v.data[i+1] = chunk
        v.data = delete!(v.data, i)
        i == v.lastkey && (v.lastkey += 1)
    else
        v.data[i+1] = chunk[i-ifirst+1+1:end]
        v.data[st] = resize!(chunk, i-ifirst+1 - 1)
    end

    v.nnz -= 1

    return v
end

#
#  Broadcasting
#
struct DensedSparseVectorStyle <: AbstractArrayStyle{1} end

const DSpVecStyle = DensedSparseVectorStyle

DSpVecStyle(::Val{0}) = DSpVecStyle()
DSpVecStyle(::Val{1}) = DSpVecStyle()
DSpVecStyle(::Val{N}) where N = DefaultArrayStyle{N}()

Base.Broadcast.BroadcastStyle(s::DSpVecStyle, ::DefaultArrayStyle{0}) = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{0}, s::DSpVecStyle) = s
Base.Broadcast.BroadcastStyle(s::DSpVecStyle, ::DefaultArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{M}, s::DSpVecStyle) where {M} = s
Base.Broadcast.BroadcastStyle(s::DSpVecStyle, ::AbstractArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::AbstractArrayStyle{M}, s::DSpVecStyle) where {M} = s

Base.Broadcast.BroadcastStyle(::Type{<:AbstractSpacedDensedSparseVector}) = DSpVecStyle()

Base.similar(bc::Broadcasted{DSpVecStyle}) = similar(find_dsv(bc))
Base.similar(bc::Broadcasted{DSpVecStyle}, ::Type{ElType}) where ElType = similar(find_dsv(bc), ElType)

"`find_dsv(bc::Broadcasted)` returns the first of any `AbstractSpacedDensedSparseVector` in `bc`"
find_dsv(bc::Base.Broadcast.Broadcasted) = find_dsv(bc.args)
find_dsv(args::Tuple) = find_dsv(find_dsv(args[1]), Base.tail(args))
find_dsv(x::Base.Broadcast.Extruded) = x.x
find_dsv(x) = x
find_dsv(::Tuple{}) = nothing
find_dsv(v::AbstractSpacedDensedSparseVector, rest) = v
find_dsv(v::SpacedVector, rest) = v
find_dsv(::Any, rest) = find_dsv(rest)

function Base.copy(bc::Broadcasted{<:DSpVecStyle})
    dest = similar(bc)
    bcf = Broadcast.flatten(bc)
    @debug for (i,a) in enumerate(bcf.args) println("$i: $(typeof(a))") end
    nzbroadcast!(bcf.f, dest, bcf.args)
end

function Base.copyto!(dest::AbstractVector, bc::Broadcasted{<:DSpVecStyle})
    bcf = Broadcast.flatten(bc)
    nzbroadcast!(bcf.f, dest, bcf.args)
end

struct ItWrapper{T} x::T end
ItWrapper(v::T) where T = ItWrapper{T}(v)
@inline Base.getindex(v::ItWrapper, i::Integer) = v.x
@inline iteratenzpairs(v::ItWrapper, state = 1) = ((state, v.x), state + 1)
@inline Base.ndims(v::ItWrapper) = 1

function nzbroadcast!(f, dest, args)
    # replace scalars with iterable wrapper
    args = map(a -> ndims(a) == 0 ? ItWrapper(a) : a, args)
    # replace single-value DenseArray's with iterable wrapper
    args = map(a -> isa(a, DenseArray) && length(a) == 1 ? ItWrapper(a[1]) : a, args)
    @debug args

    # check indices are the same
    # issimilar()

    # create `nzvals` iterator for each item in args
    iters = map(nzvals, args)
    iters = (nzvalsRef(dest), iters...)

    for res in zip(iters...)
        @debug foreach(x->println(x), enumerate(res))
        first(res)[] = f(Base.tail(res)...)
    end
    return dest
end


#
#  Testing
#

function testfun_create(T::Type, n = 500_000)

    dsv = T(n)

    Random.seed!(1234)
    for i in rand(1:n, 4*n)
        dsv[i] = rand()
    end

    dsv
end

function testfun_create_dense(T::Type, n = 500_000, nchunks = 100)

    dsv = T(n)
    chunklen = max(1, floor(Int, n / nchunks))

    Random.seed!(1234)
    for i = 0:nchunks-1
        len = chunklen - rand(1:nchunks÷2)
        dsv[1+i*chunklen:len+i*chunklen] .= rand(len)
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


function testfun1(sv)
    I = 0
    S = 0.0
    for ic in axes(sv.nzind, 1)
        ind = sv.nzind[ic] - 1
        dat = sv.data[ic]
        for i in axes(dat,1)
            I += ind+i
            S += dat[i]
        end
    end
    (I, S)
end

function testfun_nzpairs(sv)
    I = 0
    S = 0.0
    for (k,v) in nzpairs(sv)
        I += k
        S += v
    end
    (I, S)
end

function testfun_nzinds(sv)
    I = 0
    for k in nzinds(sv)
        I += k
    end
    (I, 0.0)
end

function testfun_nzvals(sv)
    S = 0.0
    for v in nzvals(sv)
        S += v
    end
    (0, S)
end

function testfun_nzvalsRef(sv)
    S = 0.0
    for v in nzvalsRef(sv)
        S += v[]
    end
    (0, S)
end

function testfun_findnz(sv)
    I = 0
    S = 0.0
    for (k,v) in zip(SparseArrays.findnz(sv)...)
        I += k
        S += v
    end
    (I, S)
end


#end  # of module DensedSparseVectors
