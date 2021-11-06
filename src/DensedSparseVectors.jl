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
using DocStringExtensions
using DataStructures
using FillArrays
using SparseArrays
using Random

#import Base: getindex, setindex!, unsafe_load, unsafe_store!, nnz, length, isempty


abstract type AbstractAlmostSparseVector{Tv,Ti,Td,Tc} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractSpacedVector{Tv,Ti,Tx,Ts} <: AbstractAlmostSparseVector{Tv,Ti,Tx,Ts} end
abstract type AbstractDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractAlmostSparseVector{Tv,Ti,Td,Tc} end


"""The `SpacedVectorIndex` is for fast indices creating and saving for `SpacedVector`.
It is almost the same as the `SpacedVector` but without data storing.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct SpacedVectorIndex{Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{Int}} <: AbstractSpacedVector{Bool,Ti,Tx,Ts}
    "`n` is the vector length"
    n::Int     # the vector length
    "Number of stored non-zero elements"
    nnz::Int   # number of non-zero elements
    "Vector of chunk's first indices"
    nzind::Tx  # Vector of chunk's first indices
    "`Vector{Int}` -- Vector of chunks lengths"
    data::Ts   # Vector{Int} -- Vector of chunks lengths
end


"""
The `SpacedVector` is alike the `Vector` but have the omits in stored indices/data and,
It is the subtype of `AbstractSparseVector`. The speed of `Broadcasting` on `SpacedVector`
is almost the same as on the `Vector`, but the speed by direct index access is almost few times
slower then the for `Vector`'s one.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct SpacedVector{Tv,Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{<:AbstractVector}} <: AbstractSpacedVector{Tv,Ti,Tx,Ts}
    "`n` is the vector length"
    n::Int     # the vector length
    "Number of stored non-zero elements"
    nnz::Int   # number of non-zero elements
    "Vector of chunk's first indices"
    nzind::Tx  # Vector of chunk's first indices
    "`Td{<:AbstractVector{Tv}}` -- Vector of Vectors (chunks) with values"
    data::Ts   # Td{<:AbstractVector{Tv}} -- Vector of Vectors (chunks) with values
end


"""
The `DensedSparseVector` is alike the `SparseVector` but should have the almost all indices are consecuitive stored.
The speed of `Broadcasting` on `DensedSparseVector`
is almost the same as on the `Vector` excluding the cases where the indices are wide broaded and there is no consecuitive ranges of indices. The speed by direct index access is ten or more times slower then the for `Vector`'s one. The main purpose of this type is the construction of the `AbstractAlmostSparseVector` vectors with further convertion to `SpacedVector`.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSparseVector{Tv,Ti<:Integer,Td<:AbstractVector{Tv},Tc<:AbstractDict{Ti,Td}} <: AbstractDensedSparseVector{Tv,Ti,Td,Tc}
    "`n` is the vector length"
    n::Int
    "Number of stored non-zero elements"
    nnz::Int
    "The last node key`::{Ti}` in `data` tree"
    lastkey::Ti
    "`Tc{Ti,Td{Tv}}` -- Tree based (sorted) Dict data container"
    data::Tc
end




include("constructors.jl")





function Base.length(v::AbstractSpacedVector)
    if v.n != 0
        return v.n
    elseif !isempty(v.nzind)
        return Int(last(v.nzind)) + length(last(v.data)) - 1
    else
        return 0
    end
end
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
Base.@propagate_inbounds SparseArrays.nnz(v::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractAlmostSparseVector} = foldl((s,c)->(s+length(c)), nzchunks(v); init=0)
SparseArrays.nnz(v::AbstractAlmostSparseVector) = v.nnz
Base.isempty(v::AbstractAlmostSparseVector) = v.nnz == 0
Base.size(v::AbstractAlmostSparseVector) = (v.n,)
Base.axes(v::AbstractAlmostSparseVector) = (Base.OneTo(v.n),)
Base.ndims(::AbstractAlmostSparseVector) = 1
Base.ndims(::Type{AbstractAlmostSparseVector}) = 1
Base.strides(v::AbstractAlmostSparseVector) = (1,)
#Base.eltype(v::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} = Pair{Ti,Tv}
Base.eltype(v::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} = Tv
Base.IndexStyle(::AbstractAlmostSparseVector) = IndexLinear()

Base.similar(v::SpacedVectorIndex{Ti,Tx,Ts}) where {Ti,Tx,Ts} =
    return SpacedVectorIndex{Ti,Tx,Ts}(v.n, v.nnz, copy(v.nzind), copy(v.data))
Base.similar(v::SpacedVectorIndex{Ti,Tx,Ts}, ::Type{ElType}) where {Ti,Tx,Ts,ElType} = similar(v)

function Base.similar(v::SpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    nzind = similar(v.nzind)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = similar(d)
    end
    return SpacedVector{Tv,Ti,Tx,Ts}(v.n, v.nnz, nzind, data)
end
function Base.similar(v::SpacedVector{Tv,Ti,Tx,Ts}, ::Type{ElType}) where {Tv,Ti,Tx,Ts,ElType<:Pair{Tin,Tvn}} where {Tin,Tvn}
    nzind = similar(v.nzind, Tin)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = similar(d, Tvn)
    end
    return SpacedVector{Tvn,Tin,typeof(nzind),typeof(data)}(v.n, v.nnz, nzind, data)
end
function Base.similar(v::SpacedVector{Tv,Ti,Tx,Ts}, ::Type{ElType}) where {Tv,Ti,Tx,Ts,ElType}
    nzind = similar(v.nzind)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = similar(d, ElType)
    end
    return SpacedVector{ElType,Ti,typeof(nzind),typeof(data)}(v.n, v.nnz, nzind, data)
end

Base.@propagate_inbounds length_of_chunk(v::SpacedVectorIndex, chunk) = chunk
Base.@propagate_inbounds length_of_chunk(v::SpacedVector, chunk) = length(chunk)
Base.@propagate_inbounds length_of_chunk(v::DensedSparseVector, chunk) = length(chunk)
@inline get_chunk_length(v::SpacedVectorIndex, i) = v.data[i]
@inline get_chunk_length(v::SpacedVector, i) = size(v.data[i])[1]
@inline get_chunk_length(v::DensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = size(deref_value((v.data, i)))[1]
@inline get_chunk(v::Number, i) = v
@inline get_chunk(v::Vector, i) = v
@inline get_chunk(v::SparseVector, i) = view(v.nzval, i:i)
@inline get_chunk(v::SpacedVectorIndex, i) = Fill(true, v.data[i])
@inline get_chunk(v::SpacedVector, i) = v.data[i]
@inline get_chunk(v::DensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = deref_value((v.data, i))
@inline function get_chunk(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractSpacedVector}
    idx1 = first(v.indices[1])
    key = v.parent.nzind[i]
    len = length(v.parent.data[i])
    if key <= idx1 < key + len
        return @view(v.parent.data[i][idx1:end])
    elseif key <= last(v.indices[1]) < key + len
        return view(v.parent.data[i], 1:(last(v.indices[1])-key+1))
    else
        return @view(v.parent.data[i][1:end])
    end
end
@inline function get_chunk(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractDensedSparseVector}
    idx1 = first(v.indices[1])
    key, chunk = deref((v.parent.data, i))
    len = length(chunk)
    if key <= idx1 < key + len
        return @view(chunk[idx1:end])
    elseif key <= last(v.indices[1]) < key + len
        return view(chunk, 1:(last(v.indices[1])-key+1))
    else
        return @view(chunk[1:end])
    end
end
@inline get_chunk_key(v::Vector, i) = i
@inline get_chunk_key(v::SparseVector, i) = v.nzind[i]
@inline get_chunk_key(v::SpacedVectorIndex, i) = v.nzind[i]
@inline get_chunk_key(v::SpacedVector, i) = v.nzind[i]
@inline get_chunk_key(v::DensedSparseVector, i) = deref_key((v.data, i))
@inline function get_chunk_key(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractSpacedVector}
    if v.parent.nzind[i] <= first(v.indices[1]) < v.parent.nzind[i] + length(v.parent.data[i])
        return first(v.indices[1])
    else
        return v.parent.nzind[i]
    end
end
@inline function get_chunk_key(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractDensedSparseVector}
    key, chunk = deref((v.parent.data, i))
    len = length(chunk)
    if key <= first(v.indices[1]) < key + len
        return first(v.indices[1])
    else
        return key
    end
end
@inline get_key_and_chunk(v::Vector, i) = (i, v)
@inline get_key_and_chunk(v::SparseVector, i) = (v.nzind[i], view(v.data, i:i))
@inline get_key_and_chunk(v::SpacedVectorIndex, i) = (v.nzind[i], v.data[i])
@inline get_key_and_chunk(v::SpacedVector, i) = (v.nzind[i], v.data[i])
@inline get_key_and_chunk(v::DensedSparseVector, i) = deref((v.data, i))
@inline get_key_and_chunk(v::Vector) = (1, eltype(v)[])
@inline get_key_and_chunk(v::SparseVector, i) = (eltype(v.nzind)(1), view(v.data, 1:0))
@inline get_key_and_chunk(v::SpacedVectorIndex) = (valtype(v.nzind)(1), valtype(v.data)(0))
@inline get_key_and_chunk(v::SpacedVector) = (valtype(v.nzind)(1), valtype(v.data)())
@inline get_key_and_chunk(v::DensedSparseVector) = (keytype(v.data)(1), valtype(v.data)())
@inline getindex_chunk(v::SpacedVectorIndex, chunk, i) = 1 <= i <= chunk
@inline getindex_chunk(v::SpacedVector, chunk, i) = chunk[i]
@inline getindex_chunk(v::DensedSparseVector, chunk, i) = chunk[i]
@inline Base.firstindex(v::AbstractSpacedVector) = firstindex(v.nzind)
@inline Base.firstindex(v::AbstractDensedSparseVector) = startof(v.data)
@inline Base.lastindex(v::AbstractSpacedVector) = lastindex(v.nzind)
@inline Base.lastindex(v::AbstractDensedSparseVector) = lastindex(v.data)
@inline beforestartindex(v::AbstractSpacedVector) = firstindex(v) - 1
@inline beforestartindex(v::AbstractDensedSparseVector) = beforestartsemitoken(v.data)
@inline pastendindex(v::AbstractSpacedVector) = lastindex(v) + 1
@inline pastendindex(v::AbstractDensedSparseVector) = pastendsemitoken(v.data)
@inline DataStructures.advance(v::AbstractSpacedVector, state) = state + 1
@inline DataStructures.advance(v::AbstractDensedSparseVector, state) = advance((v.data, state))
#@inline DataStructures.advance(v::SubArray{<:Any,<:Any,<:T}, state) where {T<:AbstractDensedSparseVector} = advance((v.parent.data, state))
@inline searchsortedlastchunk(v::AbstractSpacedVector, i) = searchsortedlast(v.nzind, i)
@inline searchsortedlastchunk(v::AbstractDensedSparseVector, i) = searchsortedlast(v.data, i)

@inline function search_nzchunk(v::AbstractAlmostSparseVector, i::Integer)
    if i == 1
        return nnz(v) == 0 ? beforestartindex(v) : firstindex(v)
    else
        st = searchsortedlastchunk(v, i)
        if st === beforestartindex(v)
            return firstindex(v)
        else
            key = get_chunk_key(v, st)
            len = get_chunk_length(v, st)
            if key + len - 1 >= i
                return st
            else
                return advance(v, st)
            end
        end
    end
end

function SparseArrays.nonzeroinds(v::AbstractAlmostSparseVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    ret = Vector{Ti}()
    for (k,d) in nzchunkpairs(v)
        append!(ret, (k:k+length(d)-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractAlmostSparseVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    ret = Tv===Bool ? BitVector() : Vector{Tv}()
    for d in nzchunks(v)
        append!(ret, collect(d))
    end
    return ret
end

# TODO: Type piracy!!!
Base.@propagate_inbounds SparseArrays.nnz(v::DenseArray) = length(v)

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
#Base.@propagate_inbounds function iteratenzchunks(v::SubArray{<:Any,<:Any,<:T}, state = search_nzchunk(v.parent, first(v.indices[1]))) where {T<:AbstractSpacedVector}
#    if state <= length(v.parent.nzind)
#        if last(v.indices[1]) >= v.parent.nzind[state] + length(v.parent.data[state])
#            return (state, state + 1)
#        elseif v.parent.nzind[state] <= last(v.indices[1]) < v.parent.nzind[state] + length(v.parent.data[state])
#            return (state, state + 1)
#        else
#            return nothing
#        end
#    else
#        return nothing
#    end
#end
Base.@propagate_inbounds function iteratenzchunks(v::SubArray{<:Any,<:Any,<:T}, state = search_nzchunk(v.parent, first(v.indices[1]))) where {T<:AbstractAlmostSparseVector}
    if state !== pastendindex(v.parent)
        key = get_chunk_key(v.parent, state)
        len = get_chunk_length(v.parent, state)
        if last(v.indices[1]) >= key + len
            return (state, advance(v.parent, state))
        elseif key <= last(v.indices[1]) < key + len
            return (state, advance(v.parent, state))
        else
            return nothing
        end
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(v::SparseVector, state = (1, nnz(v)))
    i, len = state
    if i <= len
        return (i, (i+1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(v::Vector, state = 1)
    if state == 1
        return (state, state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzchunks(v::Number, state = 1) = (state, state)

"`iteratenzpairs(v::AbstractAlmostSparseVector)` iterates over nonzero elements of vector and returns pair of index and value"
function iteratenzpairs end
"`iteratenzpairsview(v::AbstractAlmostSparseVector)` iterates over nonzero elements of vector and returns pair of index and `view` of value"
function iteratenzpairsview end
"`iteratenzvals(v::AbstractAlmostSparseVector)` iterates over nonzero elements of vector and returns value"
function iteratenzvals end
"`iteratenzvalsview(v::AbstractAlmostSparseVector)` iterates over nonzero elements of vector and returns pair of index and `view` of value"
function iteratenzvalsview end
"`iteratenzinds(v::AbstractAlmostSparseVector)` iterates over nonzero elements of vector and returns indices"
function iteratenzinds end

#
# iteratenzSOMEs() for `Number`, `Vector` and `SparseVector`
#

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
Base.@propagate_inbounds iteratenzpairs(v::Number, state = 1) = ((state, v), state+1)

Base.@propagate_inbounds function iteratenzvals(v::SparseVector, state = (1, length(v.nzind)))
    i, len = state
    if i <= len
        return (@inbounds v.nzval[i], (i+1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzvals(v::Vector, state = 1)
    if state-1 < length(v)
        return (@inbounds v[state], state + 1)
    elseif length(v) == 1
        return (@inbounds v[1], state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzvals(v::Number, state = 1) = (v, state+1)


#
# `AbstractAlmostSparseVector` iterators
#


struct ASDSVIteratorState{Tn,Td}
    next::Tn         # index (Int or Semitoken) of next chunk
    nextpos::Int     # index in the current chunk of item will be get
    currentkey::Int  # the start index of current chunk
    chunk::Td        # current chunk
    chunklen::Int    # current chunk length
end

@inline ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where {T<:AbstractSpacedVector{Tv,Ti,Tx,Ts}} where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td =
    ASDSVIteratorState{Int, Td}(next, nextpos, currentkey, chunk, chunklen)
@inline ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where {T<:AbstractDensedSparseVector{Tv,Ti,Td,Tc}} where {Tv,Ti,Td,Tc} =
    ASDSVIteratorState{DataStructures.Tokens.IntSemiToken, Td}(next, nextpos, currentkey, chunk, chunklen)

function get_iterator_init_state(v::T, i::Integer = 1) where {T<:AbstractAlmostSparseVector}
    # start iterations from `i` index
    st = search_nzchunk(v, i)
    if (ret = iteratenzchunks(v, st)) !== nothing
        idxchunk, next = ret
        key, chunk = get_key_and_chunk(v, idxchunk)
        return ASDSVIteratorState{T}(next, max(1, i - key + 1), key, chunk, length_of_chunk(v, chunk))
    else
        key, chunk = get_key_and_chunk(v)
        return ASDSVIteratorState{T}(1, 1, key, chunk, 0)
    end
end

for (fn, ret1, ret2) in
        ((:iteratenzpairs    , :((Ti(key+nextpos-1), chunk[nextpos]))              , :((key, chunk[1]))         ),
         (:iteratenzpairsview, :((Ti(key+nextpos-1), view(chunk, nextpos:nextpos))), :((key, view(chunk, 1:1))) ),
         (:(Base.iterate)    , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvals     , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvals     , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvalsview , :(view(chunk, nextpos:nextpos))                     , :(view(chunk, 1:1))        ),
         (:iteratenzinds     , :(Ti(key+nextpos-1))                                , :(key)                     ))

    @eval Base.@propagate_inbounds function $fn(v::T, state = get_iterator_init_state(v)) where {T<:AbstractAlmostSparseVector{Tv,Ti,Tx,Ts}} where {Ti,Tv,Tx,Ts}
        next, nextpos, key, chunk, chunklen = state.next, state.nextpos, state.currentkey, state.chunk, state.chunklen
        if nextpos <= chunklen
            return ($ret1, ASDSVIteratorState{T}(next, nextpos + 1, key, chunk, chunklen))
        elseif (ret = iteratenzchunks(v, next)) !== nothing
            i, next = ret
            key, chunk = get_key_and_chunk(v, i)
            return ($ret2, ASDSVIteratorState{T}(next, 2, Int(key), chunk, length_of_chunk(v, chunk)))
        else
            return nothing
        end
    end
end


for (fn, ret1, ret2) in
        ((:iteratenzpairs    , :((Ti(key+nextpos-1), chunk[nextpos]))              , :((key, chunk[1]))         ),
         (:iteratenzpairsview, :((Ti(key+nextpos-1), view(chunk, nextpos:nextpos))), :((key, view(chunk, 1:1))) ),
         (:(Base.iterate)    , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvals     , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvalsview , :(view(chunk, nextpos:nextpos))                     , :(view(chunk, 1:1))        ),
         (:iteratenzinds     , :(Ti(key+nextpos-1))                                , :(key)                     ))

    @eval Base.@propagate_inbounds function $fn(v::SubArray{<:Any,<:Any,<:T}, state = get_iterator_init_state(v.parent, first(v.indices[1]))) where {T<:AbstractAlmostSparseVector{Tv,Ti,Tx,Ts}} where {Ti,Tv,Tx,Ts}
        next, nextpos, key, chunk, chunklen = state.next, state.nextpos, state.currentkey, state.chunk, state.chunklen
        if key+nextpos-1 > last(v.indices[1])
            return nothing
        elseif nextpos <= chunklen
            return ($ret1, ASDSVIteratorState{T}(next, nextpos + 1, key, chunk, chunklen))
        elseif (ret = iteratenzchunks(v.parent, next)) !== nothing
            i, next = ret
            key, chunk = get_key_and_chunk(v.parent, i)
            return ($ret2, ASDSVIteratorState{T}(next, 2, Int(key), chunk, length_of_chunk(v.parent, chunk)))
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
"`nzchunks(v::AbstractAlmostSparseVector)` is the `Iterator` over chunks of nonzeros and returns tuple of start index and chunk vector"
@inline nzchunks(itr) = NZChunks(itr)
@inline function Base.iterate(it::NZChunks, state...)
    y = iteratenzchunks(it.itr, state...)
    if y !== nothing
        return (get_chunk(it.itr, y[1]), y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZChunks{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZChunks{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZChunks}) = Base.SizeUnknown()
Base.reverse(it::NZChunks) = NZChunks(reverse(it.itr))


struct NZChunkPairs{It}
    itr::It
end
"`nzchunkpairs(v::AbstractAlmostSparseVector)` is the `Iterator` over chunks of nonzeros and returns tuple of start index and chunk vector"
@inline nzchunkpairs(itr) = NZChunkPairs(itr)
@inline function Base.iterate(it::NZChunkPairs, state...)
    y = iteratenzchunks(it.itr, state...)
    if y !== nothing
        return ((get_chunk_key(it.itr, y[1]), get_chunk(it.itr, y[1])), y[2])
        #return (get_key_and_chunk(it.itr, y[1]), y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZChunkPairs{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZChunkPairs{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZChunkPairs}) = Base.SizeUnknown()
Base.reverse(it::NZChunkPairs) = NZChunkPairs(reverse(it.itr))


struct NZInds{It}
    itr::It
end
"`nzinds(v::AbstractVector)` is the `Iterator` over nonzero indices of vector `v`"
nzinds(itr) = NZInds(itr)
@inline function Base.iterate(it::NZInds, state...)
    y = iteratenzinds(it.itr, state...)
    if y !== nothing
        return (y[1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZInds{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZInds{It}}) where {It} = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:NZInds}) = Base.SizeUnknown()
Base.reverse(it::NZInds) = NZInds(reverse(it.itr))
@inline Base.keys(v::AbstractAlmostSparseVector) = nzinds(v)


struct NZVals{It}
    itr::It
end
"`nzvals(v::AbstractVector)` is the `Iterator` over nonzero values of `v`"
nzvals(itr) = NZVals(itr)
@inline function Base.iterate(it::NZVals, state...)
    y = iteratenzvals(it.itr, state...)
    if y !== nothing
        return (y[1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZVals{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZVals{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZVals}) = Base.SizeUnknown()
Base.reverse(it::NZVals) = NZVals(reverse(it.itr))


struct NZValsView{It}
    itr::It
end
"""
`NZValsView(v::AbstractVector)` is the `Iterator` over nonzero values of `v`,
returns the `view(v, idx:idx)` of iterated values
"""
nzvalsview(itr) = NZValsView(itr)
@inline function Base.iterate(it::NZValsView, state...)
    y = iteratenzvalsview(it.itr, state...)
    if y !== nothing
        return (y[1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZValsView{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZValsView{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZValsView}) = Base.SizeUnknown()
Base.reverse(it::NZValsView) = NZValsView(reverse(it.itr))


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
Base.IteratorSize(::Type{<:NZPairs}) = Base.SizeUnknown()
Base.reverse(it::NZPairs) = NZPairs(reverse(it.itr))



SparseArrays.findnz(v::AbstractAlmostSparseVector) = (nzinds(v), nzvals(v))
#SparseArrays.findnz(v::AbstractAlmostSparseVector) = (SparseArrays.nonzeroinds(v), SparseArrays.nonzeros(v))


@inline function Base.isstored(v::AbstractAlmostSparseVector, i::Integer)
    st = searchsortedlastchunk(v, i)
    if st === beforestartindex(v)  # the index `i` is before first index
        return false
    elseif i >= get_chunk_key(v, st) + get_chunk_length(v, st)
        # the index `i` is outside of data chunk indices
        return false
    end
    return true
end

@inline Base.haskey(v::AbstractAlmostSparseVector, i) = isstored(v, i)


@inline function Base.getindex(v::AbstractSpacedVector, i::Integer)
    st = searchsortedlast(v.nzind, i)
    if st !== 0  # the index `i` is not before the first index
        (ifirst, chunk) = get_key_and_chunk(v, st)
        if i < ifirst + length_of_chunk(v, chunk)  # the index `i` is inside of data chunk indices range
            return getindex_chunk(v, chunk, i - ifirst + 1)
        end
    end
    return zero(eltype(v))
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
    #st = searchsortedlast(v.data, i)
    #if st !== beforestartsemitoken(v.data)  # the index `i` is not before first index
    #    (ifirst, chunk) = get_key_and_chunk(v, st)
    #    if i < ifirst + length_of_chunk(v, chunk)  # the index `i` is inside of data chunk indices range
    #        return getindex_chunk(v, chunk, i - ifirst + 1)
    #    end
    #end
    #return zero(Tv)
    return getindex_helper(v, i, v.data)
end

@inline function getindex_helper(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}, i::Integer, data) where {Tv,Ti,Td,Tc}
    st = searchsortedlast(data, i)
    if st !== beforestartsemitoken(data)  # the index `i` is not before first index
        (ifirst, chunk) = get_key_and_chunk(v, st)
        if i < ifirst + length_of_chunk(v, chunk)  # the index `i` is inside of data chunk indices range
            return getindex_chunk(v, chunk, i - ifirst + 1)
        end
    end
    return zero(eltype(v))
end


function Base.setindex!(v::SpacedVectorIndex{Ti,Tx,Ts}, value, i::Integer) where {Ti,Tx,Ts}

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if st > 0  # the index `i` is not before the first index
    #if v.nnz > 0 && st > 0  # the index `i` is not before the first index
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
    if st > 0  # the index `i` is not before the first index
    #if v.nnz > 0 && st > 0  # the index `i` is not before the first index
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

function Base.setindex!(v::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}, data::AbstractAlmostSparseVector, index::Integer) where {Tv,Ti,Td,Tc}
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


Base.@propagate_inbounds Base.fill!(v::AbstractAlmostSparseVector, value) = foreach(c -> fill!(c, value), nzchunks(v))
Base.@propagate_inbounds Base.fill!(v::SubArray{<:Any,<:Any,<:T}, value) where {T<:AbstractAlmostSparseVector} = foreach(c -> fill!(c, value), nzchunks(v))



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
struct AlmostSparseVectorStyle <: AbstractArrayStyle{1} end

const AlSpVecStyle = AlmostSparseVectorStyle

AlSpVecStyle(::Val{0}) = AlSpVecStyle()
AlSpVecStyle(::Val{1}) = AlSpVecStyle()
AlSpVecStyle(::Val{N}) where N = DefaultArrayStyle{N}()

Base.Broadcast.BroadcastStyle(s::AlSpVecStyle, ::DefaultArrayStyle{0}) = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{0}, s::AlSpVecStyle) = s
Base.Broadcast.BroadcastStyle(s::AlSpVecStyle, ::DefaultArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{M}, s::AlSpVecStyle) where {M} = s
Base.Broadcast.BroadcastStyle(s::AlSpVecStyle, ::AbstractArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::AbstractArrayStyle{M}, s::AlSpVecStyle) where {M} = s

Base.Broadcast.BroadcastStyle(::Type{<:AbstractAlmostSparseVector}) = AlSpVecStyle()

Base.similar(bc::Broadcasted{AlSpVecStyle}) = similar(find_asv(bc))
Base.similar(bc::Broadcasted{AlSpVecStyle}, ::Type{ElType}) where ElType = similar(find_asv(bc), ElType)

"`find_asv(bc::Broadcasted)` returns the first of any `AbstractAlmostSparseVector` in `bc`"
find_asv(bc::Base.Broadcast.Broadcasted) = find_asv(bc.args)
find_asv(args::Tuple) = find_asv(find_asv(args[1]), Base.tail(args))
find_asv(x::Base.Broadcast.Extruded) = x.x
find_asv(x) = x
find_asv(::Tuple{}) = nothing
find_asv(v::AbstractAlmostSparseVector, rest) = v
find_asv(v::SpacedVector, rest) = v
find_asv(::Any, rest) = find_asv(rest)

function Base.Broadcast.instantiate(bc::Broadcasted{AlSpVecStyle})
    if bc.axes isa Nothing
        v1 = find_asv(bc)
        bcf = Broadcast.flatten(bc)
        if !issamenzlengths(v1, bcf.args)
            throw(DimensionMismatch("Number of nonzeros of vectors must be equal, but have nnz's: $(map((a)->nnz(a), filter((a)->isa(a,AbstractVector), bcf.args)))"))
        end
        bcaxes = axes(v1)
        #bcaxes = Broadcast.combine_axes(bc.args...)
    else
        bcaxes = bc.axes
        # AlmostSparseVector is flexible in assignment in any direction thus any sizes are allowed
        #check_broadcast_axes(axes, bc.args...)
    end
    return Broadcasted{AlSpVecStyle}(bc.f, bc.args, bcaxes)
end

function Base.copy(bc::Broadcasted{<:AlSpVecStyle})
    dest = similar(bc, Broadcast.combine_eltypes(bc.f, bc.args))
    bcf = Broadcast.flatten(bc)
    nzbroadcast!(bcf.f, dest, bcf.args)
end

Base.@propagate_inbounds Base.copyto!(dest::AbstractAlmostSparseVector, bc::Broadcasted{<:AbstractArrayStyle{0}}) = nzcopyto!(dest, bc)
Base.@propagate_inbounds Base.copyto!(dest::AbstractAlmostSparseVector, bc::Broadcasted{<:AbstractArrayStyle{1}}) = nzcopyto!(dest, bc)
Base.@propagate_inbounds Base.copyto!(dest::AbstractAlmostSparseVector, bc::Broadcasted{<:AbstractArrayStyle{2}}) = nzcopyto!(dest, bc)
Base.@propagate_inbounds Base.copyto!(dest::AbstractVector, bc::Broadcasted{<:AlSpVecStyle}) = nzcopyto!(dest, bc)
Base.@propagate_inbounds Base.copyto!(dest::AbstractAlmostSparseVector, bc::Broadcasted{<:AlSpVecStyle}) = nzcopyto!(dest, bc)

Base.@propagate_inbounds function nzcopyto!(dest, bc)
    bcf = Broadcast.flatten(bc)
    @boundscheck if !issamenzlengths(dest, bcf.args)
        throw(DimensionMismatch("Number of nonzeros of vectors must be equal, but have nnz's: $(map((a)->nnz(a), filter((a)->isa(a,AbstractVector), (dest, bcf.args...))))"))
    end
    if length(bcf.args) == 1 && isa(bcf.args[1], Number)
        foreach(c -> fill!(c, bcf.args[1]), nzchunks(dest))
    elseif length(bcf.args) == 1 && isa(bcf.args[1], DenseArray) && length(bcf.args[1]) == 1
        foreach(c -> fill!(c, bcf.args[1][1]), nzchunks(dest))
    elseif isbynzchunks((dest, bcf.args...)) && issimilar_asv(dest, bcf.args)
        args1 = map(a -> isa_asv(a) ? a : ItWrapper(a[]), bcf.args)
        nzbroadcastchunks!(bcf.f, dest, args1)
    else
        nzbroadcast!(bcf.f, dest, bcf.args)
    end
    return dest
end


struct ItWrapper{T} x::T end
ItWrapper(v::T) where T = ItWrapper{T}(v)
@inline Base.getindex(v::ItWrapper, i::Integer) = v.x
@inline Base.iterate(v::ItWrapper, state = 1) = (v.x, state)
@inline iteratenzchunks(v::ItWrapper, state = 1) = (state, state)
@inline get_chunk(v::ItWrapper, i) = v
@inline Base.ndims(v::ItWrapper) = 1


@inline tuples_conv(i1, t1, i2, t2) =
    i1[1] < i2[1] ? (t1[1], tuples_conv(Iterators.tail(i1), Iterators.tail(t1), i2, t2)...) :
                    (t2[1], tuples_conv(i1, t1, Iterators.tail(i2), Iterators.tail(t2))...)
@inline tuples_conv(::Tuple{}, ::Tuple{}, i2, t2) = (t2[1], tuples_conv((), (), Iterators.tail(i2), Iterators.tail(t2))...)
@inline tuples_conv(i1, t1, ::Tuple{}, ::Tuple{}) = (t1[1], tuples_conv(Iterators.tail(i1), Iterators.tail(t1), (), ())...)
@inline tuples_conv(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()


@generated function nzbroadcastchunks!(f, dest, args)
    return quote
        # create `nzchunks()` iterator for each item in args
        nzchunksiters = map(nzchunks, args)
        nzchunksiters = (nzchunks(dest), nzchunksiters...)
        for (dst, rest...) in zip(nzchunksiters...)
            i = 0
            @inbounds for res in zip(rest...)
                dst[i+=1] = f(res...)
            end
        end
        return dest
    end
end

@generated function nzbroadcast!(f, dest, args)
    # This function is generated because the `f(rest...)` was created via `Broadcast.flatten(bc)`,
    # is allocates the heap memory when apply with splatted tuple.
    codeInit = quote
        # create `nzvals()` iterator for each item in args
        iters = map(nzvals, args)
        # for the result there is the `view` `nzvals` iterator
        iters = (nzvalsview(dest), iters...)
    end
    code = quote
        for (dst, rest...) in zip(iters...)
            dst[1] = f(rest...)
        end
    end
    return quote
        $codeInit
        @inbounds $code
        return dest
    end
end

@inline @generated function issamenzlengths(dest, args)
    return quote
        nz = nnz(dest)
        @inbounds for a in args
            if isa_asv(a)                               ||
               isa(a, AbstractSparseVector)             ||
               isa(a, DenseArray) && length(a) > 1      ||
               isa(a, SubArray) && length(a) > 1
                nnz(a) != nz && return false
            end
        end
        return true
    end
end

@inline isa_asv(a) = isa(a, AbstractAlmostSparseVector) ||
                     (isa(a, SubArray) && isa(a.parent, AbstractAlmostSparseVector))

"Is vectors are similar by non-zero chunks"
function issimilar_asv(dest, args)

    args1 = filter(a->isa_asv(a), args)

    iters = map(nzchunkpairs, (dest, args1...))
    for (dst, rest...) in zip(iters...)
        foldl((s,r)-> s && r[1]==dst[1], rest, init=true) || return false
        foldl((s,r)-> s && length(r[2])==length(dst[2]), rest, init=true) || return false
    end
    return true
end

"Is all vectors are iterable by non-zero chunks"
@inline function isbynzchunks(args)
    return !any(a -> !(isa_asv(a)                           ||
                       isa(a, Number)                       ||
                       isa(a, DenseArray) && length(a) == 1 ||
                       isa(a, SubArray)   && length(a) == 1    ), args)
end

#
#  Testing functions
#

function testfun_create(T::Type, n = 500_000)

    dsv = T(n)

    Random.seed!(1234)
    for i in rand(1:n, 4*n)
        dsv[i] = rand()
    end

    dsv
end

function testfun_create_dense(T::Type, n = 500_000, nchunks = 100, density = 0.95)

    dsv = T(n)
    chunklen = max(1, floor(Int, n / nchunks))

    Random.seed!(1234)
    for i = 0:nchunks-1
        len = floor(Int, chunklen*density + randn() * chunklen * min(0.1, (1.0-density), density))
        len = max(1, min(chunklen-2, len))
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


function testfun_nzchunks(sv)
    I = 0
    S = 0.0
    for (startindex,chunk) in nzchunkpairs(sv)
        startindex -= 1
        for i in axes(chunk,1)
            I += startindex + i
            S += chunk[i]
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
    for v in nzvalsview(sv)
        S += v[1]
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
