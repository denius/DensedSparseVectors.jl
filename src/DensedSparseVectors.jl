#
#  DensedSparseVector
#  SDictDensedSparseVector
#
# fast and slow realizations:
# 1. on two vectors: Vector{FirstIndex<:Int} and Vector{SomeVectorData}
# 2. on SortedDict{FirstIndex<:Int, SomeVectorData}.
# The first realization is for fast index access, the second one is for creating and rebuilding.
# Also on the SortedDict{Index<:Int, value} -- the simples and bug free.

#module DensedSparseVectors
#export DensedSparseVector

import Base: ForwardOrdering, Forward
const FOrd = ForwardOrdering

import Base.Broadcast: BroadcastStyle
using Base.Broadcast: AbstractArrayStyle, Broadcasted, DefaultArrayStyle
using DocStringExtensions
using DataStructures
using FillArrays
using IterTools
using SparseArrays
using Random

#import Base: getindex, setindex!, unsafe_load, unsafe_store!, nnz, length, isempty


abstract type AbstractDensedSparseVector{Tv,Ti} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractVectorDensedSparseVector{Tv,Ti} <: AbstractDensedSparseVector{Tv,Ti} end
abstract type AbstractSDictDensedSparseVector{Tv,Ti} <: AbstractDensedSparseVector{Tv,Ti} end


"""The `DensedSparseIndex` is for fast indices creating and saving for `DensedSparseVector`.
It is almost the same as the `DensedSparseVector` but without data storing.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSparseIndex{Ti} <: AbstractVectorDensedSparseVector{Bool,Ti}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Tx{Ti} -- Vector of chunk's first indices
    "Storage for lengths of chunks of non-zero values"
    data::Vector{Int}  # Tx{Int} -- Vector of chunks lengths
    "Vector length"
    n::Int     # the vector length
    "Number of stored non-zero elements"
    nnz::Int   # number of non-zero elements

    DensedSparseIndex{Ti}(n::Integer, nzind, data) where {Ti} = new{Ti}(0, nzind, data, n, foldl((s,c)->(s+c), data; init=0))
end

DensedSparseIndex(n::Integer, nzind, data) = DensedSparseIndex{eltype(nzind)}(n, nzind, data)
DensedSparseIndex{Ti}(n::Integer = 0) where {Ti} = DensedSparseIndex{Ti}(n, Vector{Ti}(), Vector{Int}())
DensedSparseIndex(n::Integer = 0) = DensedSparseIndex{Int}(n)

"""
The `DensedSparseVector` is alike the `Vector` but have the omits in stored indices/data and,
It is the subtype of `AbstractSparseVector`. The speed of `Broadcasting` on `DensedSparseVector`
is almost the same as on the `Vector`, but the speed by direct index access is almost few times
slower then the for `Vector`'s one.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSparseVector{Tv,Ti} <: AbstractVectorDensedSparseVector{Tv,Ti}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s"
    data::Vector{Vector{Tv}}   # Tx{<:AbstractVector{Tv}} -- Vector of Vectors (chunks) with values
    "Vector length"
    n::Int     # the vector length
    "Number of stored non-zero elements"
    nnz::Int   # number of non-zero elements

    DensedSparseVector{Tv,Ti}(n::Integer, nzind, data) where {Tv,Ti} =
        new{Tv,Ti}(0, nzind, data, n, foldl((s,c)->(s+length(c)), data; init=0))
    DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = new{Tv,Ti}(0, Vector{Ti}(), Vector{Vector{Tv}}(), n, 0)
end

DensedSparseVector(n::Integer = 0) = DensedSparseVector{Float64,Int}(n)

"""
The `SDictDensedSparseVector` is alike the `SparseVector` but should have the almost all indices are consecuitive stored.
The speed of `Broadcasting` on `SDictDensedSparseVector` is almost the same as
on the `Vector` excluding the cases where the indices are wide broaded and
there is no consecuitive ranges of indices. The speed by direct index access is ten or
more times slower then the for `Vector`'s one. The main purpose of this type is
the construction of the `AbstractDensedSparseVector` vectors with further conversion to `DensedSparseVector`.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct SDictDensedSparseVector{Tv,Ti} <: AbstractSDictDensedSparseVector{Tv,Ti}
    "Index of last used chunk"
    lastusedchunkindex::DataStructures.Tokens.IntSemiToken
    "Storage for indices of the first element of non-zero chunks and corresponding chunks as `SortedDict(Int=>Vector)`"
    data::SortedDict{Ti,Vector{Tv},FOrd}
    "Vector length"
    n::Int
    "Number of stored non-zero elements"
    nnz::Int

    function SDictDensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
        data = SortedDict{Ti,Vector{Tv},FOrd}(Forward)
        new{Tv,Ti}(beforestartsemitoken(data), data, n, 0)
    end
    SDictDensedSparseVector{Tv,Ti}(n::Integer, data::SortedDict{K,V}) where {Tv,Ti,K,V<:AbstractVector} =
        new{Tv,Ti}(beforestartsemitoken(data), data, n, foldl((s,c)->(s+length(c)), values(data); init=0))
end


SDictDensedSparseVector(n::Integer = 0) = SDictDensedSparseVector{Float64,Int}(n)


#
#  Converters
#

function DensedSparseIndex(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(v))
    data = Vector{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = length_of_that_nzchunk(v, d)
    end
    return DensedSparseIndex{Ti}(v.n, nzind, data)
end


function DensedSparseVector(v::AbstractSDictDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(v))
    data = Vector{Vector{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = Vector{Tv}(d)
    end
    return DensedSparseVector{Tv,Ti}(v.n, nzind, data)
end




Base.length(v::AbstractDensedSparseVector) = v.n
Base.@propagate_inbounds SparseArrays.nnz(v::SubArray{<:Any,<:Any,<:T}) where {T<:AbstractDensedSparseVector} = foldl((s,c)->(s+length(c)), nzchunks(v); init=0)
SparseArrays.nnz(v::AbstractDensedSparseVector) = v.nnz
Base.isempty(v::AbstractDensedSparseVector) = v.nnz == 0
Base.size(v::AbstractDensedSparseVector) = (v.n,)
Base.axes(v::AbstractDensedSparseVector) = (Base.OneTo(v.n),)
Base.ndims(::AbstractDensedSparseVector) = 1
Base.ndims(::Type{AbstractDensedSparseVector}) = 1
Base.strides(v::AbstractDensedSparseVector) = (1,)
Base.eltype(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti} = Tv
Base.IndexStyle(::AbstractDensedSparseVector) = IndexLinear()

Base.similar(v::DensedSparseIndex) = DensedSparseIndex(v.n, copy(v.nzind), copy(v.data))
Base.similar(v::DensedSparseIndex, ::Type{ElType}) where {ElType} = DensedSparseIndex{ElType}(v.n, copy(v.nzind), copy(v.data))

Base.similar(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti} = similar(v, Tv)
Base.similar(v::AbstractDensedSparseVector{Tv,Ti}, ::Type{ElType}) where {Tv,Ti,ElType} = similar(v, Pair{Ti,ElType})
function Base.similar(v::DensedSparseVector, ::Type{ElType}) where {ElType<:Pair{Tin,Tvn}} where {Tin,Tvn}
    nzind = similar(v.nzind, Tin)
    data = similar(v.data)
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = similar(d, Tvn)
    end
    return DensedSparseVector{Tvn,Tin}(v.n, nzind, data)
end
function Base.similar(v::SDictDensedSparseVector, ::Type{ElType}) where {ElType<:Pair{Tin,Tvn}} where {Tin,Tvn}
    data = SortedDict{Tin, Vector{Tvn}, FOrd}(Forward)
    for (k,d) in nzchunkpairs(v)
        data[k] = similar(d, Tvn)
    end
    return SDictDensedSparseVector{Tvn,Tin}(v.n, data)
end

function Base.collect(::Type{ElType}, v::AbstractDensedSparseVector) where ElType
    res = zeros(ElType, length(v))
    for (i,v) in nzpairs(v)
        res[i] = ElType(v)
    end
    return res
end
Base.collect(v::AbstractDensedSparseVector) = collect(eltype(v), v)

nnzchunks(v::AbstractDensedSparseVector) = length(v.data)
Base.@propagate_inbounds length_of_that_nzchunk(v::DensedSparseIndex, chunk) = chunk
Base.@propagate_inbounds length_of_that_nzchunk(v::DensedSparseVector, chunk) = length(chunk)
Base.@propagate_inbounds length_of_that_nzchunk(v::SDictDensedSparseVector, chunk) = length(chunk)
@inline get_nzchunk_length(v::DensedSparseIndex, i) = v.data[i]
@inline get_nzchunk_length(v::DensedSparseVector, i) = size(v.data[i])[1]
@inline get_nzchunk_length(v::SDictDensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = size(deref_value((v.data, i)))[1]
@inline get_nzchunk(v::Number, i) = v
@inline get_nzchunk(v::Vector, i) = v
@inline get_nzchunk(v::SparseVector, i) = view(v.nzval, i:i)
@inline get_nzchunk(v::DensedSparseIndex, i) = Fill(true, v.data[i])
@inline get_nzchunk(v::DensedSparseVector, i) = v.data[i]
@inline get_nzchunk(v::SDictDensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = deref_value((v.data, i))
@inline function get_nzchunk(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractVectorDensedSparseVector}
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
@inline function get_nzchunk(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractSDictDensedSparseVector}
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
@inline get_nzchunk_key(v::Vector, i) = i
@inline get_nzchunk_key(v::SparseVector, i) = v.nzind[i]
@inline get_nzchunk_key(v::DensedSparseIndex, i) = v.nzind[i]
@inline get_nzchunk_key(v::DensedSparseVector, i) = v.nzind[i]
@inline get_nzchunk_key(v::SDictDensedSparseVector, i) = deref_key((v.data, i))
@inline function get_nzchunk_key(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractVectorDensedSparseVector}
    if v.parent.nzind[i] <= first(v.indices[1]) < v.parent.nzind[i] + length(v.parent.data[i])
        return first(v.indices[1])
    else
        return v.parent.nzind[i]
    end
end
@inline function get_nzchunk_key(v::SubArray{<:Any,<:Any,<:T}, i) where {T<:AbstractSDictDensedSparseVector}
    key, chunk = deref((v.parent.data, i))
    len = length(chunk)
    if key <= first(v.indices[1]) < key + len
        return first(v.indices[1])
    else
        return key
    end
end
@inline get_key_and_nzchunk(v::Vector, i) = (i, v)
@inline get_key_and_nzchunk(v::SparseVector, i) = (v.nzind[i], view(v.data, i:i))
@inline get_key_and_nzchunk(v::DensedSparseIndex, i) = (v.nzind[i], v.data[i])
@inline get_key_and_nzchunk(v::DensedSparseVector, i) = (v.nzind[i], v.data[i])
@inline get_key_and_nzchunk(v::SDictDensedSparseVector, i) = deref((v.data, i))

@inline get_key_and_nzchunk(v::Vector) = (1, eltype(v)[])
@inline get_key_and_nzchunk(v::SparseVector) = (eltype(v.nzind)(1), view(v.data, 1:0))
@inline get_key_and_nzchunk(v::DensedSparseIndex) = (valtype(v.nzind)(1), valtype(v.data)(0))
@inline get_key_and_nzchunk(v::DensedSparseVector) = (valtype(v.nzind)(1), valtype(v.data)())
@inline get_key_and_nzchunk(v::SDictDensedSparseVector) = (keytype(v.data)(1), valtype(v.data)())

@inline getindex_nzchunk(v::DensedSparseIndex, chunk, i) = 1 <= i <= chunk
@inline getindex_nzchunk(v::DensedSparseVector, chunk, i) = chunk[i]
@inline getindex_nzchunk(v::SDictDensedSparseVector, chunk, i) = chunk[i]

@inline Base.firstindex(v::AbstractVectorDensedSparseVector) = firstindex(v.nzind)
@inline Base.firstindex(v::AbstractSDictDensedSparseVector) = startof(v.data)
@inline Base.lastindex(v::AbstractVectorDensedSparseVector) = lastindex(v.nzind)
@inline Base.lastindex(v::AbstractSDictDensedSparseVector) = lastindex(v.data)

@inline lastkey(v::AbstractVectorDensedSparseVector) = last(v.nzind)
"the index of first element in last chunk of non-zero values"
@inline lastkey(v::AbstractSDictDensedSparseVector) = deref_key((v.data, lastindex(v.data)))
@inline beforestartindex(v::AbstractVectorDensedSparseVector) = firstindex(v) - 1
@inline beforestartindex(v::AbstractSDictDensedSparseVector) = beforestartsemitoken(v.data)
@inline pastendindex(v::AbstractVectorDensedSparseVector) = lastindex(v) + 1
@inline pastendindex(v::AbstractSDictDensedSparseVector) = pastendsemitoken(v.data)

@inline DataStructures.advance(v::AbstractVectorDensedSparseVector, state) = state + 1
@inline DataStructures.advance(v::AbstractSDictDensedSparseVector, state) = advance((v.data, state))
@inline searchsortedlastchunk(v::AbstractVectorDensedSparseVector, i) = searchsortedlast(v.nzind, i)
@inline searchsortedlastchunk(v::AbstractSDictDensedSparseVector, i) = searchsortedlast(v.data, i)

@inline function search_nzchunk(v::AbstractDensedSparseVector, i::Integer)
    if i == 1 # the most of use cases
        return nnz(v) == 0 ? beforestartindex(v) : firstindex(v)
    else
        st = searchsortedlastchunk(v, i)
        if st === beforestartindex(v)
            return firstindex(v)
        else
            key = get_nzchunk_key(v, st)
            len = get_nzchunk_length(v, st)
            if key + len - 1 >= i
                return st
            else
                return advance(v, st)
            end
        end
    end
end

@inline SparseArrays.sparse(v::AbstractDensedSparseVector) =
    SparseVector(length(v), SparseArrays.nonzeroinds(v), SparseArrays.nonzeros(v))

function SparseArrays.nonzeroinds(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    ret = Vector{Ti}()
    for (k,d) in nzchunkpairs(v)
        append!(ret, (k:k+length_of_that_nzchunk(v,d)-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    ret = Vector{Tv}()
    for d in nzchunks(v)
        append!(ret, collect(d))
    end
    return ret
end
SparseArrays.findnz(v::AbstractDensedSparseVector) = (nzinds(v), nzvals(v))
#SparseArrays.findnz(v::AbstractDensedSparseVector) = (SparseArrays.nonzeroinds(v), SparseArrays.nonzeros(v))

# FIXME: Type piracy!!!
Base.@propagate_inbounds SparseArrays.nnz(v::DenseArray) = length(v)

"`iteratenzchunks(v::AbstractVector)` iterates over non-zero chunks and returns start index of chunk and chunk"
Base.@propagate_inbounds function iteratenzchunks(v::AbstractVectorDensedSparseVector, state = 1)
    if state <= length(v.nzind)
        return (state, state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(v::AbstractSDictDensedSparseVector, state = startof(v.data))
    if state != pastendsemitoken(v.data)
        stnext = advance((v.data, state))
        return (state, stnext)
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzchunks(v::SubArray{<:Any,<:Any,<:T}, state = search_nzchunk(v.parent, first(v.indices[1]))) where {T<:AbstractDensedSparseVector}
    if state !== pastendindex(v.parent)
        key = get_nzchunk_key(v.parent, state)
        len = get_nzchunk_length(v.parent, state)
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
Base.@propagate_inbounds function iteratenzchunks(v::Vector, state = (1, length(v)))
    i, len = state
    if len == 1
        return (1, state)
    elseif i == 1
        return (i, (i + 1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzchunks(v::Number, state = 1) = (1, state)

"`iteratenzpairs(v::AbstractDensedSparseVector)` iterates over non-zero elements of vector and returns pair of index and value"
function iteratenzpairs end
"`iteratenzpairsview(v::AbstractDensedSparseVector)` iterates over non-zero elements of vector and returns pair of index and `view` of value"
function iteratenzpairsview end
"`iteratenzvals(v::AbstractDensedSparseVector)` iterates over non-zero elements of vector and returns value"
function iteratenzvals end
"`iteratenzvalsview(v::AbstractDensedSparseVector)` iterates over non-zero elements of vector and returns pair of index and `view` of value"
function iteratenzvalsview end
"`iteratenzinds(v::AbstractDensedSparseVector)` iterates over non-zero elements of vector and returns indices"
function iteratenzinds end

#
# iteratenzSOMEs() iterators for `Number`, `Vector` and `SparseVector`
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

Base.@propagate_inbounds function iteratenzinds(v::SparseVector, state = (1, length(v.nzind)))
    i, len = state
    if i-1 < len
        return (@inbounds v.nzind[i], (i+1, len))
    else
        return nothing
    end
end
Base.@propagate_inbounds function iteratenzinds(v::Vector, state = 1)
    if state-1 < length(v)
        return (state, state + 1)
    else
        return nothing
    end
end
Base.@propagate_inbounds iteratenzinds(v::Number, state = 1) = (state, state+1)


#
# `AbstractDensedSparseVector` iteration functions
#


struct ASDSVIteratorState{Tn,Td}
    next::Tn         # index (Int or Semitoken) of next chunk
    nextpos::Int     # index in the current chunk of item will be get
    currentkey::Int  # the start index of current chunk
    chunk::Td        # current chunk
    chunklen::Int    # current chunk length
end

@inline ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where {T<:AbstractVectorDensedSparseVector{Tv,Ti}} where {Tv,Ti} =
    ASDSVIteratorState{Int, Vector{Tv}}(next, nextpos, currentkey, chunk, chunklen)
@inline ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where {T<:AbstractSDictDensedSparseVector{Tv,Ti}} where {Tv,Ti} =
    ASDSVIteratorState{DataStructures.Tokens.IntSemiToken, Vector{Tv}}(next, nextpos, currentkey, chunk, chunklen)

function get_iterator_init_state(v::T, i::Integer = 1) where {T<:AbstractDensedSparseVector}
    # start iterations from `i` index
    st = search_nzchunk(v, i)
    if (ret = iteratenzchunks(v, st)) !== nothing
        idxchunk, next = ret
        key, chunk = get_key_and_nzchunk(v, idxchunk)
        return ASDSVIteratorState{T}(next, max(1, i - key + 1), key, chunk, length_of_that_nzchunk(v, chunk))
    else
        key, chunk = get_key_and_nzchunk(v)
        return ASDSVIteratorState{T}(1, 1, key, chunk, 0)
    end
end

for (fn, ret1, ret2) in
        ((:iteratenzpairs    , :((Ti(key+nextpos-1), chunk[nextpos]))              , :((key, chunk[1]))         ),
         (:iteratenzpairsview, :((Ti(key+nextpos-1), view(chunk, nextpos:nextpos))), :((key, view(chunk, 1:1))) ),
         #(:(Base.iterate)    , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvals     , :(chunk[nextpos])                                   , :(chunk[1])                ),
         (:iteratenzvalsview , :(view(chunk, nextpos:nextpos))                     , :(view(chunk, 1:1))        ),
         (:iteratenzinds     , :(Ti(key+nextpos-1))                                , :(key)                     ))

    @eval Base.@propagate_inbounds function $fn(v::T, state = get_iterator_init_state(v)) where
                                                {T<:AbstractDensedSparseVector{Tv,Ti}} where {Ti,Tv}
        next, nextpos, key, chunk, chunklen = fieldvalues(state)
        if nextpos <= chunklen
            return ($ret1, ASDSVIteratorState{T}(next, nextpos + 1, key, chunk, chunklen))
        elseif (ret = iteratenzchunks(v, next)) !== nothing
            i, next = ret
            key, chunk = get_key_and_nzchunk(v, i)
            return ($ret2, ASDSVIteratorState{T}(next, 2, Int(key), chunk, length_of_that_nzchunk(v, chunk)))
        else
            return nothing
        end
    end
end


for (fn, ret1, ret2) in
        ((:iteratenzpairs    , :((Ti(key+nextpos-1-first(v.indices[1])+1), chunk[nextpos]))              ,
                               :((Ti(key-first(v.indices[1])+1), chunk[1]))                                 ),
         (:iteratenzpairsview, :((Ti(key+nextpos-1-first(v.indices[1])+1), view(chunk, nextpos:nextpos))),
                               :((Ti(key-first(v.indices[1])+1), view(chunk, 1:1)))                         ),
         #(:(Base.iterate)    , :(chunk[nextpos])                                                         ,
         #                      :(chunk[1])                                                                  ),
         (:iteratenzvals     , :(chunk[nextpos])                                                         ,
                               :(chunk[1])                                                                  ),
         (:iteratenzvalsview , :(view(chunk, nextpos:nextpos))                                           ,
                               :(view(chunk, 1:1))                                                          ),
         (:iteratenzinds     , :(Ti(key+nextpos-1)-first(v.indices[1])+1)                                ,
                               :(Ti(key-first(v.indices[1])+1))                                             ))

    @eval Base.@propagate_inbounds function $fn(v::SubArray{<:Any,<:Any,<:T},
                                                state = get_iterator_init_state(v.parent, first(v.indices[1]))) where
                                                {T<:AbstractDensedSparseVector{Tv,Ti}} where {Tv,Ti}
        next, nextpos, key, chunk, chunklen = fieldvalues(state)
        if key+nextpos-1 > last(v.indices[1])
            return nothing
        elseif nextpos <= chunklen
            return ($ret1, ASDSVIteratorState{T}(next, nextpos + 1, key, chunk, chunklen))
        elseif (ret = iteratenzchunks(v.parent, next)) !== nothing
            i, next = ret
            key, chunk = get_key_and_nzchunk(v.parent, i)
            return ($ret2, ASDSVIteratorState{T}(next, 2, Int(key), chunk, length_of_that_nzchunk(v.parent, chunk)))
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
"`nzchunks(v::AbstractDensedSparseVector)` is the `Iterator` over chunks of nonzeros and returns tuple of start index and chunk vector"
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
Base.IteratorSize(::Type{<:NZChunks}) = Base.SizeUnknown()
Base.reverse(it::NZChunks) = NZChunks(reverse(it.itr))


struct NZChunkPairs{It}
    itr::It
end
"`nzchunkpairs(v::AbstractDensedSparseVector)` is the `Iterator` over non-zero chunks,
 returns tuple of start index and vector of non-zero values."
@inline nzchunkpairs(itr) = NZChunkPairs(itr)
@inline function Base.iterate(it::NZChunkPairs, state...)
    y = iteratenzchunks(it.itr, state...)
    if y !== nothing
        #return ((get_nzchunk_key(it.itr, y[1]), get_nzchunk(it.itr, y[1])), y[2])
        return (get_key_and_nzchunk(it.itr, y[1]), y[2])
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
"`nzinds(v::AbstractVector)` is the `Iterator` over non-zero indices of vector `v`."
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
@inline Base.keys(v::AbstractDensedSparseVector) = nzinds(v)


struct NZVals{It}
    itr::It
end
"`nzvals(v::AbstractVector)` is the `Iterator` over non-zero values of `v`."
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
`NZValsView(v::AbstractVector)` is the `Iterator` over non-zero values of `v`,
returns the `view(v, idx:idx)` of iterated values.
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
"`nzpairs(v::AbstractVector)` is the `Iterator` over nonzeros of `v` and returns pair of index and value."
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





@inline function Base.isstored(v::AbstractDensedSparseVector, i::Integer)
    st = searchsortedlastchunk(v, i)
    if st === beforestartindex(v)  # the index `i` is before first index
        return false
    elseif i >= get_nzchunk_key(v, st) + get_nzchunk_length(v, st)
        # the index `i` is outside of data chunk indices
        return false
    end
    return true
end

@inline Base.haskey(v::AbstractDensedSparseVector, i) = Base.isstored(v, i)


@inline function Base.getindex(v::AbstractVectorDensedSparseVector, i::Integer)
    if (st = v.lastusedchunkindex) !== beforestartindex(v)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length_of_that_nzchunk(v, chunk)
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    st = searchsortedlast(v.nzind, i)
    if st !== beforestartindex(v)  # the index `i` is not before the first index
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if i < ifirst + length_of_that_nzchunk(v, chunk)  # is the index `i` inside of data chunk indices range
            v.lastusedchunkindex = st
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    v.lastusedchunkindex = beforestartindex(v)
    return zero(eltype(v))
end


@inline function Base.getindex(v::AbstractSDictDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}
    if (st = v.lastusedchunkindex) !== beforestartsemitoken(v.data)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length_of_that_nzchunk(v, chunk)
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    st = searchsortedlast(v.data, i)
    if st !== beforestartsemitoken(v.data)  # the index `i` is not before first index
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if i < ifirst + length_of_that_nzchunk(v, chunk)  # is the index `i` inside of data chunk indices range
            v.lastusedchunkindex = st
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    v.lastusedchunkindex = beforestartsemitoken(v.data)
    return zero(Tv)
end


# FIXME: complete me
function Base.setindex!(v::DensedSparseIndex{Ti}, value, i::Integer) where {Ti}

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
            pushfirst!(v.data, [val])
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

@inline function Base.setindex!(v::DensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = Tv(value)

    if (st = v.lastusedchunkindex) != beforestartindex(v)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            return v
        end
    end

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if st != beforestartindex(v)  # the index `i` is not before the first index
        ifirst, chunk = v.nzind[st], v.data[st]
        if i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            v.lastusedchunkindex = st
            return v
        end
    end

    if v.nnz == 0
        v.nzind = push!(v.nzind, Ti(i))
        v.data = push!(v.data, [val])
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = 1
        return v
    end

    if st == beforestartindex(v)  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, [val])
        else
            v.nzind[1] -= 1
            pushfirst!(v.data[1], val)
        end
        v.nnz += 1
        v.lastusedchunkindex = 1
        return v
    end

    ifirst, chunk = v.nzind[st], v.data[st]

    if i >= v.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(v.nzind, i)
            push!(v.data, [val])
        else  # just append to last chunk
            v.data[st] = push!(chunk, val)
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = length(v.nzind)
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = st + 1
    inextfirst = v.nzind[stnext]

    if inextfirst - ilast == 2  # join chunks
        v.data[st] = append!(chunk, [val], v.data[stnext])
        v.nzind = deleteat!(v.nzind, stnext)
        v.data  = deleteat!(v.data, stnext)
        v.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        v.data[st] = push!(chunk, val)
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        v.data[stnext] = pushfirst!(v.data[stnext], val)
        v.lastusedchunkindex = stnext
    else  # insert single element chunk
        v.nzind = insert!(v.nzind, stnext, Ti(i))
        v.data  = insert!(v.data, stnext, [val])
        v.lastusedchunkindex = stnext
    end

    v.nnz += 1
    return v

end



@inline function Base.setindex!(v::SDictDensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = eltype(v)(value)

    if (st = v.lastusedchunkindex) !== beforestartsemitoken(v.data)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            return v
        end
    end

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
            v.lastusedchunkindex = st
            return v
        end
    end

    if v.nnz == 0
        v.data[i] = [val]
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = startof(v.data)  # firstindex(v.data)
        return v
    end

    if sstatus == 2  # the index `i` is before the first index
        stnext = startof(v.data)
        inextfirst = deref_key((v.data, stnext))
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            v.data[i] = [val]
        else
            v.data[i] = pushfirst!(deref_value((v.data, stnext)), val)
            delete!((v.data, stnext))
        end
        v.nnz += 1
        v.lastusedchunkindex = startof(v.data)
        return v
    end

    (ifirst, chunk) = deref((v.data, st))

    if i >= lastkey(v) # the index `i` is after the last key index
        if ifirst + length(chunk) < i  # there is will be the gap in indices after inserting
            v.data[i] = [val]
        else  # just append to last chunk
            v.data[st] = push!(chunk, val)
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = lastindex(v.data)
        return v
    end

    v.lastusedchunkindex = beforestartsemitoken(v.data)

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = advance((v.data, st))
    inextfirst = deref_key((v.data, stnext))

    if inextfirst - ilast == 2  # join chunks
        v.data[st] = append!(chunk, [val], deref_value((v.data, stnext)))
        delete!((v.data, stnext))
    elseif i - ilast == 1  # append to left chunk
        v.data[st] = push!(chunk, val)
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.data[i] = pushfirst!(deref_value((v.data, stnext)), val)
        delete!((v.data, stnext))
    else  # insert single element chunk
        v.data[i] = [val]
    end

    v.nnz += 1
    return v

end

function Base.setindex!(v::AbstractDensedSparseVector{Tv,Ti}, data::AbstractDensedSparseVector, index::Integer) where {Tv,Ti}
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



Base.@propagate_inbounds Base.fill!(v::AbstractDensedSparseVector, value) = foreach(c -> fill!(c, value), nzchunks(v))
Base.@propagate_inbounds Base.fill!(v::SubArray{<:Any,<:Any,<:T}, value) where {T<:AbstractDensedSparseVector} = foreach(c -> fill!(c, value), nzchunks(v))



@inline function Base.delete!(v::DensedSparseIndex{Ti}, i::Integer) where {Ti}

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
    v.lastusedchunkindex = 0

    return v
end

@inline function Base.delete!(v::DensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

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
    v.lastusedchunkindex = 0

    return v
end

@inline function Base.delete!(v::SDictDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

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
    elseif i == ifirst + length(chunk) - 1  # last index in chunk
        pop!(chunk)
        v.data[st] = chunk
    elseif i == ifirst
        popfirst!(chunk)
        v.data[i+1] = chunk
        v.data = delete!(v.data, i)
    else
        v.data[i+1] = chunk[i-ifirst+1+1:end]
        v.data[st] = resize!(chunk, i-ifirst+1 - 1)
    end

    v.nnz -= 1
    v.lastusedchunkindex = beforestartsemitoken(v.data)

    return v
end

#
#  Broadcasting
#
struct DensedSparseVectorStyle <: AbstractArrayStyle{1} end

const AlSpVecStyle = DensedSparseVectorStyle

AlSpVecStyle(::Val{0}) = AlSpVecStyle()
AlSpVecStyle(::Val{1}) = AlSpVecStyle()
AlSpVecStyle(::Val{N}) where N = DefaultArrayStyle{N}()

Base.Broadcast.BroadcastStyle(s::AlSpVecStyle, ::DefaultArrayStyle{0}) = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{0}, s::AlSpVecStyle) = s
Base.Broadcast.BroadcastStyle(s::AlSpVecStyle, ::DefaultArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{M}, s::AlSpVecStyle) where {M} = s
Base.Broadcast.BroadcastStyle(s::AlSpVecStyle, ::AbstractArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::AbstractArrayStyle{M}, s::AlSpVecStyle) where {M} = s

Base.Broadcast.BroadcastStyle(::Type{<:AbstractDensedSparseVector}) = AlSpVecStyle()
Base.Broadcast.BroadcastStyle(::Type{<:SubArray{<:Any,<:Any,<:T}}) where {T<:AbstractDensedSparseVector} = AlSpVecStyle()

Base.similar(bc::Broadcasted{AlSpVecStyle}) = similar(find_AASV(bc))
Base.similar(bc::Broadcasted{AlSpVecStyle}, ::Type{ElType}) where ElType = similar(find_AASV(bc), ElType)

"`find_AASV(bc::Broadcasted)` returns the first of any `AbstractDensedSparseVector` in `bc`"
find_AASV(bc::Base.Broadcast.Broadcasted) = find_AASV(bc.args)
find_AASV(args::Tuple) = find_AASV(find_AASV(args[1]), Base.tail(args))
find_AASV(x::Base.Broadcast.Extruded) = x.x  # expose internals of Broadcast but else don't work
find_AASV(x) = x
find_AASV(::Tuple{}) = nothing
find_AASV(v::AbstractDensedSparseVector, rest) = v
find_AASV(::Any, rest) = find_AASV(rest)

nzDimensionMismatchMsg(args)::String = "Number of nonzeros of vectors must be equal, but have nnz's:" *
                                       "$(map((a)->nnz(a), filter((a)->(isa(a,AbstractVector)&&!ismathscalar(a)), args)))"

function Base.Broadcast.instantiate(bc::Broadcasted{AlSpVecStyle})
    if bc.axes isa Nothing
        v1 = find_AASV(bc)
        bcf = Broadcast.flatten(bc)
        if !similarlength(nnz(v1), args)
            throw(DimensionMismatch(nzDimensionMismatchMsg(bcf.args)))
        end
        bcaxes = axes(v1)
        #bcaxes = Broadcast.combine_axes(bc.args...)
    else
        bcaxes = bc.axes
        # AbstractDensedSparseVector is flexible in assignment in any direction thus any sizes are allowed
        #check_broadcast_axes(axes, bc.args...)
    end
    return Broadcasted{AlSpVecStyle}(bc.f, bc.args, bcaxes)
end

function Base.copy(bc::Broadcasted{<:AlSpVecStyle})
    dest = similar(bc, Broadcast.combine_eltypes(bc.f, bc.args))
    bcf = Broadcast.flatten(bc)
    @boundscheck similarlength(nnz(dest), bcf.args) || throw(DimensionMismatch(nzDimensionMismatchMsg((dest, bcf.args...))))
    nzcopyto_flatten!(bcf.f, dest, bcf.args)
end

Base.copyto!(dest::AbstractDensedSparseVector, bc::Broadcasted{<:AbstractArrayStyle{0}}) = nzcopyto!(dest, bc)
Base.copyto!(dest::AbstractDensedSparseVector, bc::Broadcasted{<:AbstractArrayStyle{1}}) = nzcopyto!(dest, bc)
#Base.copyto!(dest::AbstractDensedSparseVector, bc::Broadcasted{<:AbstractArrayStyle{2}}) = nzcopyto!(dest, bc)
Base.copyto!(dest::SubArray{<:Any,<:Any,<:AbstractDensedSparseVector}, bc::Broadcasted{<:AbstractArrayStyle{0}}) = nzcopyto!(dest, bc)
Base.copyto!(dest::SubArray{<:Any,<:Any,<:AbstractDensedSparseVector}, bc::Broadcasted{<:AbstractArrayStyle{1}}) = nzcopyto!(dest, bc)
#Base.copyto!(dest::SubArray{<:Any,<:Any,<:AbstractDensedSparseVector}, bc::Broadcasted{<:AbstractArrayStyle{2}}) = nzcopyto!(dest, bc)
Base.copyto!(dest::AbstractVector, bc::Broadcasted{<:AlSpVecStyle}) = nzcopyto!(dest, bc)
Base.copyto!(dest::AbstractDensedSparseVector, bc::Broadcasted{<:AlSpVecStyle}) = nzcopyto!(dest, bc)
Base.copyto!(dest::SubArray{<:Any,<:Any,<:AbstractDensedSparseVector}, bc::Broadcasted{<:AlSpVecStyle}) = nzcopyto!(dest, bc)

function nzcopyto!(dest, bc)
    bcf = Broadcast.flatten(bc)
    @boundscheck similarlength(nnz(dest), bcf.args) || throw(DimensionMismatch(nzDimensionMismatchMsg((dest, bcf.args...))))
    nzcopyto_flatten!(bcf.f, dest, bcf.args)
end

function nzcopyto_flatten!(f, dest, args)
    if iterablenzchunks(dest, args) && issimilar_AASV(dest, args)
        nzbroadcastchunks!(f, dest, args)
    else
        nzbroadcast!(f, dest, args)
    end
    return dest
end

#struct ItWrapper{T}
#    x::T
#end
#ItWrapper(v) = ItWrapper{typeof(v[])}(v[])
#@inline Base.getindex(v::ItWrapper, i::Integer) = v.x
#@inline Base.iterate(v::ItWrapper, state = 1) = (v.x, state)
#@inline iteratenzchunks(v::ItWrapper, state = 1) = (state, state)
#@inline get_nzchunk(v::ItWrapper, i) = v
#@inline Base.ndims(v::ItWrapper) = 1
#@inline Base.length(v::ItWrapper) = 1
#
#@inline iteratenzchunks(v::Base.RefValue, state = 1) = (state, state)
#@inline get_nzchunk(v::Base.RefValue, i) = v[]

@generated function nzbroadcastchunks!(f, dest, args)
    codeInit = quote
        # create `nzchunks()` iterator for each item in args
        nzchunksiters = map(nzchunks, args)
        nzchunksiters = (nzchunks(dest), nzchunksiters...)
    end
    code = quote
        for (dst, rest...) in zip(nzchunksiters...)
            dst .= f.(rest...)
        end
    end
    return quote
        $codeInit
        @inbounds $code
        return dest
    end
end

"`nzbroadcast!(f, dest, args)` performs broadcasting over non-zero values of vectors in `args`.
Note 1: `f` and `args` should be `flatten` `bc.f` and `bc.args` respectively.
Note 2: The coincidence of vectors indices should be checked and provided by the user."
@generated function nzbroadcast!(f, dest, args)
    return quote
        # create `nzvals()` iterator for each item in args
        iters = map(nzvals, args)
        # for the result there is the `view` `nzvals` iterator
        iters = (nzvalsview(dest), iters...)
        for (dst, rest...) in zip(iters...)
            dst[] = f(rest...)
        end
        return dest
    end
end


similarlength(n, args::Tuple) = (ismathscalar(first(args)) || n == nnz(first(args))) && similarlength(n, Iterators.tail(args))
similarlength(n, a) = ismathscalar(a) || n == nnz(a)
similarlength(n, a::Tuple{}) = true


@inline isa_AASV(a) = isa(a, AbstractDensedSparseVector) ||
                     (isa(a, SubArray) && isa(a.parent, AbstractDensedSparseVector))

"Are the vectors the similar in every non-zero chunk"
function issimilar_AASV(dest, args::Tuple)

    args1 = filter(a->isa_AASV(a), args)

    iters = map(nzchunkpairs, (dest, args1...))
    for (dst, rest...) in zip(iters...)
        idx = dst[1]
        len = length(dst[2])
        foldl((s,r)-> s && r[1]==idx, rest, init=true) || return false
        foldl((s,r)-> s && length(r[2])==len, rest, init=true) || return false
    end
    return true
end
issimilar_AASV(dest, args) = issimilar_AASV(dest, (args,))

"Are all vectors iterable by non-zero chunks"
iterablenzchunks(a, args...) = isa_AASV_or_scalar(a) || iterablenzchunks(a, iterablenzchunks(args...))
iterablenzchunks(a, b) = isa_AASV_or_scalar(a) || isa_AASV_or_scalar(b)
iterablenzchunks(a) = isa_AASV_or_scalar(a)

isa_AASV_or_scalar(a) = isa_AASV(a) || ismathscalar(a)

@inline function ismathscalar(a)
    return (isa(a, Number)                       ||
            isa(a, DenseArray) && length(a) == 1 ||
            isa(a, SubArray)   && length(a) == 1    )
end

#
#  Aux functions
#
# derived from stdlib/SparseArrays/src/sparsevector.jl
#
function Base.show(io::IOContext, x::DensedSparseIndex)
    nzind = [v[1] for v in nzchunkpairs(x)]
    nzval = [v[2] for v in nzchunkpairs(x)]
    n = length(nzind)
    if isempty(nzind)
        return Base.show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    pad = ndigits(n)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                Base.show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end

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

function testfun_nzgetindex(sv)
    S = 0.0
    for i in nzinds(sv)
        S += sv[i]
    end
    (0, S)
end

function testfun_setindex(sv)
    for i in nzinds(sv)
        sv[i] = 0.0
    end
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
