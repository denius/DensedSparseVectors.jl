
module DensedSparseVectors
export AbstractDensedSparseVector, DensedSparseIndex, DensedSparseVector, DensedSVSparseVector, DensedVLSparseVector
export SDictDensedSparseVector, SDictDensedSparseIndex
export nzpairs, nzvals, nzvalsview, nzinds, nzchunks, nzchunkpairs
export findfirstnz, findlastnz, findfirstindexnz, findlastindexnz
export iteratenzpairs, iteratenzpairsview, iteratenzvals, iteratenzvalsview, iteratenzinds
export testfun_create, testfun_createSV, testfun_createVL, testfun_create_seq, testfun_create_dense, testfun_delete!, testfun_getindex, testfun_nzgetindex, testfun_setindex, testfun_nzchunks, testfun_nzpairs, testfun_nzinds, testfun_nzvals, testfun_nzvalsRef, testfun_findnz


import Base: ForwardOrdering, Forward
const FOrd = ForwardOrdering

import Base.Broadcast: BroadcastStyle
using Base.Broadcast: AbstractArrayStyle, Broadcasted, DefaultArrayStyle
using DocStringExtensions
using DataStructures
using FillArrays
using IterTools
using Setfield
using SparseArrays
using StaticArrays
import SparseArrays: nonzeroinds, nonzeros
using Random


abstract type AbstractDensedSparseVector{Tv,Ti} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractVectorDensedSparseVector{Tv,Ti} <: AbstractDensedSparseVector{Tv,Ti} end
abstract type AbstractSDictDensedSparseVector{Tv,Ti} <: AbstractDensedSparseVector{Tv,Ti} end


# TODO: It needs `iterator` which only non-zeros, Set alike behaviour.
"""The `DensedSparseIndex` is for fast indices creating and saving for `DensedSparseVector`.
It is almost the same as the `DensedSparseVector` but without data storing.
In the case of not big vector length it is better to use `Set`.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSparseIndex{Ti} <: AbstractVectorDensedSparseVector{Bool,Ti}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}
    "Storage for lengths of chunks of non-zero values"
    data::Vector{Int}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    DensedSparseIndex{Ti}(n::Integer, nzind, data) where {Ti} = new{Ti}(0, nzind, data, n, foldl((s,c)->(s+c), data; init=0))
end

DensedSparseIndex(n::Integer, nzind, data) = DensedSparseIndex{eltype(nzind)}(n, nzind, data)
DensedSparseIndex{Ti}(n::Integer = 0) where {Ti} = DensedSparseIndex{Ti}(n, Vector{Ti}(), Vector{Int}())
DensedSparseIndex(n::Integer = 0) = DensedSparseIndex{Int}(n)

function DensedSparseIndex(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(v))
    data = Vector{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = length_of_that_nzchunk(v, d)
    end
    return DensedSparseIndex{Ti}(v.n, nzind, data)
end
function DensedSparseIndex(v::AbstractSparseVector{Tv,Ti}) where {Tv,Ti}
    sv = DensedSparseIndex{Ti}(length(v))
    for i in nonzeroinds(v)
        sv[i] = true
    end
    return sv
end



"""
The `DensedSparseVector` is alike the `Vector` but have the omits in stored indices/data.
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
    data::Vector{Vector{Tv}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    DensedSparseVector{Tv,Ti}(n::Integer, nzind, data) where {Tv,Ti} =
        new{Tv,Ti}(0, nzind, data, n, foldl((s,c)->(s+length(c)), data; init=0))
    DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = new{Tv,Ti}(0, Vector{Ti}(), Vector{Vector{Tv}}(), n, 0)
end

DensedSparseVector(n::Integer = 0) = DensedSparseVector{Float64,Int}(n)

function DensedSparseVector(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(v))
    data = Vector{Vector{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = Vector{Tv}(d)
    end
    return DensedSparseVector{Tv,Ti}(v.n, nzind, data)
end

#"View for DensedSparseVector"
#struct DensedSparseVectorView{Tv,Ti,T,Tc} <: AbstractVectorDensedSparseVector{Tv,Ti}
#    "Index of first chunk in `view` v"
#    firstchunkindex::Int
#    "Index of last chunk in `view` v"
#    lastchunkindex::Int
#    "View on DensedSparseVector"
#    v::Tc
#end



"""
The `DensedSVSparseVector` is the version of `DensedSparseVector` with `SVector` as elements
and alike `Matrix` with sparse first dimension and with dense `SVector` in second dimension.
See `DensedSparseVector` for details.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct DensedSVSparseVector{Tv,Ti,m} <: AbstractVectorDensedSparseVector{Tv,Ti}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    data::Vector{Vector{SVector{m,Tv}}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    DensedSVSparseVector{Tv,Ti,m}(n::Integer, nzind, data) where {Tv,Ti,m} =
        new{Tv,Ti,m}(0, nzind, data, n, foldl((s,c)->(s+length(c)), data; init=0))
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
mutable struct DensedVLSparseVector{Tv,Ti} <: AbstractVectorDensedSparseVector{Tv,Ti}
    "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    data::Vector{Vector{Tv}}
    "Offsets of starts of variable lengtn vestors in `data`"
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





"""The `SDictDensedSparseIndex` is for fast indices creating and saving for `SDictDensedSparseVector`.
It is almost the same as the `SDictDensedSparseVector` but without data storing.
In the case of not big vector length it is better to use `Set`.
$(TYPEDEF)
Mutable struct fields:
$(TYPEDFIELDS)
"""
mutable struct SDictDensedSparseIndex{Ti} <: AbstractSDictDensedSparseVector{Bool,Ti}
    "Index of last used chunk"
    lastusedchunkindex::DataStructures.Tokens.IntSemiToken
    "Storage for indices of the first element of non-zero chunks and corresponding chunks lengths
     as `SortedDict(Int=>Vector)`"
    data::SortedDict{Ti,Int,FOrd}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int

    function SDictDensedSparseIndex{Ti}(n::Integer = 0) where {Ti}
        data = SortedDict{Ti,Int,FOrd}(Forward)
        new{Ti}(beforestartsemitoken(data), data, n, 0)
    end
    SDictDensedSparseIndex{Ti}(n::Integer, data::SortedDict{K,V}) where {Ti,K,V} =
        new{Ti}(beforestartsemitoken(data), data, n, foldl((s,c)->(s+c), values(data); init=0))
end

SDictDensedSparseIndex(n::Integer, data) = SDictDensedSparseIndex{keytype(data)}(n, data)
#SDictDensedSparseIndex{Ti}(n::Integer = 0) where {Ti} = SDictDensedSparseIndex{Ti}(n, SortedDict{Ti,Int,FOrd}(Forward))
SDictDensedSparseIndex(n::Integer = 0) = SDictDensedSparseIndex{Int}(n)

function SDictDensedSparseIndex(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    data = SortedDict{Ti,Int,FOrd}(Forward)
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        data[i] = length_of_that_nzchunk(v, d)
    end
    return SDictDensedSparseIndex{Ti}(v.n, data)
end

function SDictDensedSparseIndex(v::AbstractSparseVector{Tv,Ti}) where {Tv,Ti}
    sv = SDictDensedSparseIndex{Ti}(length(v))
    for i in nonzeroinds(v)
        sv[i] = true
    end
    return sv
end



"""
The `SDictDensedSparseVector` is alike the `SparseVector` but should have the almost all indices are consecuitive stored.
The speed of `Broadcasting` on `SDictDensedSparseVector` is almost the same as
on the `Vector` excluding the cases where the indices are wide broaded and
there is no consecuitive ranges of indices. The speed by direct index access is ten or
more times slower then the for `Vector`'s one. The main purpose of this type is
the construction of the `SDictDensedSparseVector` vectors with further conversion to `DensedSparseVector`.
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
    n::Ti
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

function SDictDensedSparseVector(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    data = SortedDict{Ti, Vector{Tv}, FOrd}(Forward)
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = Vector{Tv}(d)
    end
    return SDictDensedSparseVector{Tv,Ti}(v.n, data)
end

"Convert any `AbstractSparseVector`s to particular `AbstractDensedSparseVector`"
function (::Type{T})(v::AbstractSparseVector{Tv,Ti}) where {T<:AbstractDensedSparseVector,Tv,Ti}
    sv = T{Tv,Ti}(length(v))
    for (i,d) in zip(nonzeroinds(v), nonzeros(v))
        sv[i] = d
    end
    return sv
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
Base.similar(v::SDictDensedSparseIndex) = SDictDensedSparseIndex(v.n, deepcopy(v.data))
Base.similar(v::SDictDensedSparseIndex, ::Type{ElType}) where {ElType} = SDictDensedSparseIndex{ElType}(v.n, deepcopy(v.data))

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
Base.@propagate_inbounds length_of_that_nzchunk(v::SDictDensedSparseIndex, chunk) = chunk
Base.@propagate_inbounds length_of_that_nzchunk(v::AbstractVectorDensedSparseVector, chunk) = length(chunk)
Base.@propagate_inbounds length_of_that_nzchunk(v::SDictDensedSparseVector, chunk) = length(chunk)
@inline get_nzchunk_length(v::DensedSparseIndex, i) = v.data[i]
@inline get_nzchunk_length(v::SDictDensedSparseIndex, i) = deref_value((v.data, i))
@inline get_nzchunk_length(v::DensedSparseVector, i) = size(v.data[i])[1]
@inline get_nzchunk_length(v::SDictDensedSparseVector, i::DataStructures.Tokens.IntSemiToken) = size(deref_value((v.data, i)))[1]
@inline get_nzchunk(v::Number, i) = v
@inline get_nzchunk(v::Vector, i) = v
@inline get_nzchunk(v::SparseVector, i) = view(v.nzval, i:i)
@inline get_nzchunk(v::DensedSparseIndex, i) = Fill(true, v.data[i])
@inline get_nzchunk(v::SDictDensedSparseIndex, i) = Fill(true, deref_value((v.data, i)))
@inline get_nzchunk(v::AbstractVectorDensedSparseVector, i) = v.data[i]
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
@inline get_nzchunk_key(v::SDictDensedSparseIndex, i) = deref_key((v.data, i))
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
@inline get_key_and_nzchunk(v::SDictDensedSparseIndex, i) = deref((v.data, i))
@inline get_key_and_nzchunk(v::AbstractVectorDensedSparseVector, i) = (v.nzind[i], v.data[i])
@inline get_key_and_nzchunk(v::SDictDensedSparseVector, i) = deref((v.data, i))

@inline get_key_and_nzchunk(v::Vector) = (1, eltype(v)[])
@inline get_key_and_nzchunk(v::SparseVector) = (eltype(v.nzind)(1), view(v.data, 1:0))
@inline get_key_and_nzchunk(v::DensedSparseIndex) = (valtype(v.nzind)(1), valtype(v.data)(0))
@inline get_key_and_nzchunk(v::SDictDensedSparseIndex) = (keytype(v.data)(1), valtype(v.data)(0))
@inline get_key_and_nzchunk(v::AbstractVectorDensedSparseVector) = (valtype(v.nzind)(1), valtype(v.data)())
@inline get_key_and_nzchunk(v::SDictDensedSparseVector) = (keytype(v.data)(1), valtype(v.data)())

@inline getindex_nzchunk(v::DensedSparseIndex, chunk, i) = 1 <= i <= chunk
@inline getindex_nzchunk(v::SDictDensedSparseIndex, chunk, i) = 1 <= i <= chunk
@inline getindex_nzchunk(v::AbstractVectorDensedSparseVector, chunk, i) = chunk[i]
@inline getindex_nzchunk(v::SDictDensedSparseVector, chunk, i) = chunk[i]

#@inline Base.firstindex(v::AbstractVectorDensedSparseVector) = firstindex(v.nzind)
#@inline Base.firstindex(v::AbstractSDictDensedSparseVector) = startof(v.data)
#@inline Base.lastindex(v::AbstractVectorDensedSparseVector) = lastindex(v.nzind)
#@inline Base.lastindex(v::AbstractSDictDensedSparseVector) = lastindex(v.data)
@inline Base.firstindex(v::AbstractVectorDensedSparseVector{Tv,Ti}) where {Tv,Ti} = Ti(firstindex(v.nzind))
@inline Base.firstindex(v::AbstractSDictDensedSparseVector) = startof(v.data)
@inline Base.lastindex(v::AbstractVectorDensedSparseVector{Tv,Ti}) where {Tv,Ti} = v.n
@inline Base.lastindex(v::AbstractSDictDensedSparseVector) = lastindex(v.data)

@inline returnzero(v::DensedSparseIndex) = false
@inline returnzero(v::SDictDensedSparseIndex) = false
@inline returnzero(v::DensedSVSparseVector) = zero(eltype(eltype(v.data)))
@inline returnzero(v::AbstractDensedSparseVector) = zero(eltype(v))

"the index of first element in last chunk of non-zero values"
@inline lastkey(v::AbstractVectorDensedSparseVector) = last(v.nzind)
@inline lastkey(v::AbstractSDictDensedSparseVector) = deref_key((v.data, lastindex(v.data)))
@inline beforestartindex(v::AbstractVectorDensedSparseVector) = firstindex(v) - 1
@inline beforestartindex(v::AbstractSDictDensedSparseVector) = beforestartsemitoken(v.data)
@inline pastendindex(v::AbstractVectorDensedSparseVector) = lastindex(v) + 1
@inline pastendindex(v::AbstractSDictDensedSparseVector) = pastendsemitoken(v.data)

@inline DataStructures.advance(v::AbstractVectorDensedSparseVector, state) = state + 1
@inline DataStructures.advance(v::AbstractSDictDensedSparseVector, state) = advance((v.data, state))
@inline searchsortedlastchunk(v::AbstractVectorDensedSparseVector, i) = searchsortedlast(v.nzind, i)
@inline searchsortedlastchunk(v::AbstractSDictDensedSparseVector, i) = searchsortedlast(v.data, i)

"Returns nzchunk is on `i`, or after `i`"
@inline function searchsortedlast_nzchunk(v::AbstractDensedSparseVector, i::Integer)
    if i == 1 # most of use cases
        return nnz(v) == 0 ? pastendindex(v) : firstindex(v)
    elseif nnz(v) != 0
        st = searchsortedlastchunk(v, i)
        if st != beforestartindex(v)
            key = get_nzchunk_key(v, st)
            len = get_nzchunk_length(v, st)
            if i < key + len
                return st
            else
                return advance(v, st)
            end
        else
            return firstindex(v)
        end
    else
        return beforestartindex(v)
    end
end

"Returns nzchunk is on `i`, or before `i`"
@inline function searchsortedfirst_nzchunk(v::AbstractDensedSparseVector, i::Integer)
    if nnz(v) != 0
        return searchsortedlastchunk(v, i)
    else
        return beforestartindex(v)
    end
end

@inline SparseArrays.sparse(v::AbstractDensedSparseVector) =
    SparseVector(length(v), nonzeroinds(v), nonzeros(v))

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
function SparseArrays.nonzeroinds(v::DensedVLSparseVector{Tv,Ti}) where {Tv,Ti}
    ret = Vector{Ti}()
    for (k,d) in zip(v.nzind, v.offsets)
        append!(ret, (k:k+length(d)-1-1))
    end
    return ret
end
#SparseArrays.findnz(v::AbstractDensedSparseVector) = (nzinds(v), nzvals(v))
SparseArrays.findnz(v::AbstractDensedSparseVector) = (nonzeroinds(v), nonzeros(v))



"Returns the index of first non-zero element in sparse vector."
@inline findfirstindexnz(v::SparseVector) = nnz(v) > 0 ? v.nzind[1] : nothing
@inline findfirstindexnz(v::AbstractVectorDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    nnz(v) > 0 ? Ti(v.nzind[1]) : nothing
@inline findfirstindexnz(v::AbstractSDictDensedSparseVector{Tv,Ti}) where {Tv,Ti} =
    nnz(v) > 0 ? Ti(deref_key((v.data, startof(v.data)))) : nothing
function findfirstindexnz(v::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(v.parent) == 0 && return nothing
    ifirst, ilast = first(v.indices[1]), last(v.indices[1])
    st = searchsortedlast_nzchunk(v.parent, ifirst)
    st == pastendindex(v.parent) && return nothing
    key = get_nzchunk_key(v.parent, st)
    len = get_nzchunk_length(v.parent, st)
    if key <= ifirst < key + len  # ifirst index within nzchunk range
        return Ti(1)
    elseif ifirst <= key <= ilast  # nzchunk[1] somewhere in ifirst:ilast range
        return Ti(key-ifirst+1)
    else
        return nothing
    end
end

"Returns the index of last non-zero element in sparse vector."
@inline findlastindexnz(v::SparseVector) = nnz(v) > 0 ? v.nzind[end] : nothing
@inline findlastindexnz(v::AbstractVectorDensedSparseVector) =
    nnz(v) > 0 ? v.nzind[end] + length_of_that_nzchunk(v, v.data[end]) - 1 : nothing
@inline function findlastindexnz(v::AbstractSDictDensedSparseVector)
    if nnz(v) > 0
        lasttoken = lastindex(v.data)
        return deref_key((v.data, lasttoken)) + length_of_that_nzchunk(v, deref_value((v.data, lasttoken))) - 1
    else
        return nothing
    end
end
function findlastindexnz(v::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(v.parent) == 0 && return nothing
    ifirst, ilast = first(v.indices[1]), last(v.indices[1])
    st = searchsortedfirst_nzchunk(v.parent, ilast)
    st == beforestartindex(v.parent) && return nothing
    key = get_nzchunk_key(v.parent, st)
    len = get_nzchunk_length(v.parent, st)
    if key <= ilast < key + len  # ilast index within nzchunk range
        return Ti(ilast - ifirst + 1)
    elseif ifirst <= key+len-1 <= ilast  # nzchunk[end] somewhere in ifirst:ilast range
        return Ti(key+len-1 - ifirst+1)
    else
        return nothing
    end
end

"Returns value of first non-zero element in the sparse vector."
@inline findfirstnz(v::AbstractSparseVector) = nnz(v) > 0 ? v[findfirstindexnz(v)] : nothing
function findfirstnz(v::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(v.parent) == 0 && return nothing
    ifirst, ilast = first(v.indices[1]), last(v.indices[1])
    st = searchsortedlast_nzchunk(v.parent, ifirst)
    st == pastendindex(v.parent) && return nothing
    key, chunk = get_key_and_nzchunk(v.parent, st)
    len = length_of_that_nzchunk(v.parent, chunk)
    if key <= ifirst < key + len  # ifirst index within nzchunk range
        return chunk[ifirst-key+1]
    elseif ifirst <= key <= ilast  # nzchunk[1] somewhere in ifirst:ilast range
        return chunk[1]
    else
        return nothing
    end
end

"Returns value of last non-zero element in the sparse vector."
@inline findlastnz(v::AbstractSparseVector) = nnz(v) > 0 ? v[findlastindexnz(v)] : nothing
function findlastnz(v::SubArray{<:Any,<:Any,<:T})  where {T<:AbstractDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    nnz(v.parent) == 0 && return nothing
    ifirst, ilast = first(v.indices[1]), last(v.indices[1])
    st = searchsortedfirst_nzchunk(v.parent, ilast)
    st == beforestartindex(v.parent) && return nothing
    key, chunk = get_key_and_nzchunk(v.parent, st)
    len = length_of_that_nzchunk(v.parent, chunk)
    if key <= ilast < key + len  # ilast index within nzchunk range
        return chunk[ilast-key+1]
    elseif ifirst <= key+len-1 <= ilast  # nzchunk[end] somewhere in ifirst:ilast range
        return chunk[end]
    else
        return nothing
    end
end


@inline function Base.findfirst(testf::Function, v::AbstractDensedSparseVector)
    for p in nzpairs(v)
        testf(last(p)) && return first(p)
    end
    return nothing
end

@inline Base.findall(testf::Function, v::AbstractDensedSparseVector) = collect(first(p) for p in nzpairs(v) if testf(last(p)))



# FIXME: Type piracy!!!
Base.@propagate_inbounds SparseArrays.nnz(v::DenseArray) = length(v)

"`iteratenzchunks(v::AbstractVector)` iterates over non-zero chunks and returns start index of elements in chunk and chunk"
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
Base.@propagate_inbounds function iteratenzchunks(v::SubArray{<:Any,<:Any,<:T}, state = searchsortedlast_nzchunk(v.parent, first(v.indices[1]))) where {T<:AbstractDensedSparseVector}
    if state != pastendindex(v.parent)
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

"`iteratenzpairs(v::AbstractDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and value"
function iteratenzpairs end
"`iteratenzpairsview(v::AbstractDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and `view` of value"
function iteratenzpairsview end
"`iteratenzvals(v::AbstractDensedSparseVector)` iterates over non-zero elements of vector and returns value"
function iteratenzvals end
"`iteratenzvalsview(v::AbstractDensedSparseVector)` iterates over non-zero elements
 of vector and returns pair of index and `view` of value"
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

@inline function ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where
                                                           {T<:DensedSparseIndex{Ti}} where {Ti}
    ASDSVIteratorState{Int, Int}(next, nextpos, currentkey, chunk, chunklen)
end
@inline function ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where
                                                           {T<:SDictDensedSparseIndex{Ti}} where {Ti}
    ASDSVIteratorState{DataStructures.Tokens.IntSemiToken, Int}(next, nextpos, currentkey, chunk, chunklen)
end
@inline function ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where
                                                           {T<:AbstractVectorDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    ASDSVIteratorState{Int, Vector{Tv}}(next, nextpos, currentkey, chunk, chunklen)
end
@inline function ASDSVIteratorState{T}(next, nextpos, currentkey, chunk, chunklen) where
                                                           {T<:AbstractSDictDensedSparseVector{Tv,Ti}} where {Tv,Ti}
    ASDSVIteratorState{DataStructures.Tokens.IntSemiToken, Vector{Tv}}(next, nextpos, currentkey, chunk, chunklen)
end

# start iterations from `i` index
function get_iterator_init_state(v::T, i::Integer = 1) where {T<:AbstractDensedSparseVector}
    st = searchsortedlast_nzchunk(v, i)
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
"`nzchunks(v::AbstractDensedSparseVector)` is the `Iterator` over chunks of nonzeros and
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


#
# Assignments
#


@inline function Base.isstored(v::AbstractDensedSparseVector, i::Integer)
    st = searchsortedlastchunk(v, i)
    if st == beforestartindex(v)  # the index `i` is before first index
        return false
    elseif i >= get_nzchunk_key(v, st) + get_nzchunk_length(v, st)
        # the index `i` is outside of data chunk indices
        return false
    end
    return true
end

@inline Base.haskey(v::AbstractDensedSparseVector, i) = Base.isstored(v, i)
@inline Base.in(i, v::AbstractDensedSparseVector) = Base.isstored(v, i)


@inline function Base.getindex(v::AbstractVectorDensedSparseVector, i::Integer)
    if (st = v.lastusedchunkindex) != beforestartindex(v)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length_of_that_nzchunk(v, chunk)
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    st = searchsortedlast(v.nzind, i)
    if st != beforestartindex(v)  # the index `i` is not before the first index
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if i < ifirst + length_of_that_nzchunk(v, chunk)  # is the index `i` inside of data chunk indices range
            v.lastusedchunkindex = st
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    v.lastusedchunkindex = beforestartindex(v)
    return returnzero(v)
end

@inline Base.getindex(v::DensedSVSparseVector, i::Integer, j::Integer) = getindex(v, i)[j]

@inline function Base.getindex(v::DensedVLSparseVector, i::Integer)
    if (st = v.lastusedchunkindex) != beforestartindex(v)
        (ifirst, chunk, offsets) = v.nzind[st], v.data[st], v.offsets[st]
        if ifirst <= i < ifirst + length(offsets)-1
            offs = offsets[i-ifirst+1]
            len = offsets[i-ifirst+1+1] - offsets[i-ifirst+1]
            return @view(chunk[offs:offs+len-1])
        end
    end
    st = searchsortedlast(v.nzind, i)
    if st != beforestartindex(v)  # the index `i` is not before the first index
        (ifirst, chunk, offsets) = v.nzind[st], v.data[st], v.offsets[st]
        if i < ifirst + length(offsets)-1  # is the index `i` inside of data chunk indices range
            v.lastusedchunkindex = st
            offs = offsets[i-ifirst+1]
            len = offsets[i-ifirst+1+1] - offsets[i-ifirst+1]
            return @view(chunk[offs:offs+len-1])
        end
    end
    v.lastusedchunkindex = beforestartindex(v)
    return view(v.dummy, 1:0)
end
@inline function Base.getindex(v::DensedVLSparseVector, i::Integer, j::Integer)
    vv = getindex(v, i)
    if j <= length(vv)
        return vv[j]
    else
        return returnzero(v)
    end
end

@inline function Base.getindex(v::AbstractSDictDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}
    if (st = v.lastusedchunkindex) != beforestartsemitoken(v.data)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length_of_that_nzchunk(v, chunk)
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    st = searchsortedlast(v.data, i)
    if st != beforestartsemitoken(v.data)  # the index `i` is not before first index
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if i < ifirst + length_of_that_nzchunk(v, chunk)  # is the index `i` inside of data chunk indices range
            v.lastusedchunkindex = st
            return getindex_nzchunk(v, chunk, i - ifirst + 1)
        end
    end
    v.lastusedchunkindex = beforestartsemitoken(v.data)
    return zero(Tv)
end


function Base.setindex!(v::DensedSparseIndex{Ti}, value, i::Integer) where {Ti}

    if (st = v.lastusedchunkindex) != beforestartindex(v)
        (ifirst, chunklen) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + chunklen
            return v
        end
    end

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if st != beforestartindex(v)  # the index `i` is not before the first index
        ifirst, chunklen = v.nzind[st], v.data[st]
        if i < ifirst + chunklen
            v.lastusedchunkindex = st
            return v
        end
    end

    if v.nnz == 0
        v.nzind = push!(v.nzind, Ti(i))
        v.data = push!(v.data, 1)
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = 1
        return v
    end

    if st == beforestartindex(v)  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, 1)
        else
            v.nzind[1] -= 1
            v.data[1] += 1
        end
        v.nnz += 1
        v.lastusedchunkindex = 1
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
        v.lastusedchunkindex = length(v.nzind)
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + chunklen - 1
    stnext = st + 1
    inextfirst = v.nzind[stnext]

    if inextfirst - ilast == 2  # join chunks
        v.data[st] += 1 + v.data[stnext]
        deleteat!(v.nzind, stnext)
        deleteat!(v.data, stnext)
        v.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        v.data[st] += 1
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        v.data[stnext] += 1
        v.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(v.nzind, stnext, Ti(i))
        insert!(v.data, stnext, 1)
        v.lastusedchunkindex = stnext
    end

    v.nnz += 1
    return v

end

@inline Base.push!(v::DensedSparseIndex, i::Integer) = setindex!(v, true, i)


@inline function Base.setindex!(v::SDictDensedSparseIndex{Ti}, value, i::Integer) where {Ti}

    if (st = v.lastusedchunkindex) != beforestartsemitoken(v.data)
        (ifirst, chunklen) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + chunklen
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
        (ifirst, chunklen) = deref((v.data, st))
        if ifirst + chunklen > i
            v.lastusedchunkindex = st
            return v
        end
    end

    if v.nnz == 0
        v.data[i] = 1
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = startof(v.data)  # firstindex(v.data)
        return v
    end

    if sstatus == 2  # the index `i` is before the first index
        stnext = startof(v.data)
        inextfirst = deref_key((v.data, stnext))
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            v.data[i] = 1
        else
            v.data[i] = deref_value((v.data, stnext)) + 1
            delete!((v.data, stnext))
        end
        v.nnz += 1
        v.lastusedchunkindex = startof(v.data)
        return v
    end

    (ifirst, chunklen) = deref((v.data, st))

    if i >= lastkey(v) # the index `i` is after the last key index
        if ifirst + chunklen < i  # there is will be the gap in indices after inserting
            v.data[i] = 1
        else  # just append to last chunklen
            v.data[st] += 1
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = lastindex(v.data)
        return v
    end

    v.lastusedchunkindex = beforestartsemitoken(v.data)

    # the index `i` is somewhere between indices
    ilast = ifirst + chunklen - 1
    stnext = advance((v.data, st))
    inextfirst = deref_key((v.data, stnext))

    if inextfirst - ilast == 2  # join chunks
        v.data[st] = chunklen + 1 + deref_value((v.data, stnext))
        delete!((v.data, stnext))
    elseif i - ilast == 1  # append to left chunk
        v.data[st] += 1
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.data[i] = deref_value((v.data, stnext)) + 1
        delete!((v.data, stnext))
    else  # insert single element chunk
        v.data[i] = 1
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
        push!(v.nzind, Ti(i))
        push!(v.data, [val])
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
            push!(v.data[st], val)
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
        append!(v.data[st], [val], v.data[stnext])
        deleteat!(v.nzind, stnext)
        deleteat!(v.data, stnext)
        v.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        push!(v.data[st], val)
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        pushfirst!(v.data[stnext], val)
        v.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(v.nzind, stnext, Ti(i))
        insert!(v.data, stnext, [val])
        v.lastusedchunkindex = stnext
    end

    v.nnz += 1
    return v

end



@inline function Base.setindex!(v::DensedSVSparseVector{Tv,Ti,m}, value, i::Integer) where {Tv,Ti,m}
    sv = eltype(eltype(v.data))(value)

    if (st = v.lastusedchunkindex) != beforestartindex(v)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = sv
            return v
        end
    end

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if st != beforestartindex(v)  # the index `i` is not before the first index
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        #ifirst, chunk = v.nzind[st], v.data[st]
        if i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = sv
            v.lastusedchunkindex = st
            return v
        end
    end

    if v.nnz == 0
        push!(v.nzind, Ti(i))
        push!(v.data, [sv])
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = 1
        return v
    end

    if st == beforestartindex(v)  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, [sv])
        else
            v.nzind[1] -= 1
            pushfirst!(v.data[1], sv)
        end
        v.nnz += 1
        v.lastusedchunkindex = 1
        return v
    end

    ifirst, chunk = v.nzind[st], v.data[st]

    if i >= v.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(v.nzind, i)
            push!(v.data, [sv])
        else  # just append to last chunk
            push!(v.data[st], sv)
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
        append!(v.data[st], [sv], v.data[stnext])
        deleteat!(v.nzind, stnext)
        deleteat!(v.data, stnext)
        v.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        push!(v.data[st], sv)
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        pushfirst!(v.data[stnext], sv)
        v.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(v.nzind, stnext, Ti(i))
        insert!(v.data, stnext, [sv])
        v.lastusedchunkindex = stnext
    end

    v.nnz += 1
    return v

end

@inline function Base.setindex!(v::DensedSVSparseVector{Tv,Ti,m}, value, i::Integer, j::Integer) where {Tv,Ti,m}
    val = Tv(value)

    if (st = v.lastusedchunkindex) != beforestartindex(v)
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        if ifirst <= i < ifirst + length(chunk)
            sv = chunk[i - ifirst + 1]
            chunk[i - ifirst + 1] = @set sv[j] = val
            return v
        end
    end

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if st != beforestartindex(v)  # the index `i` is not before the first index
        (ifirst, chunk) = get_key_and_nzchunk(v, st)
        #ifirst, chunk = v.nzind[st], v.data[st]
        if i < ifirst + length(chunk)
            sv = chunk[i - ifirst + 1]
            chunk[i - ifirst + 1] = @set sv[j] = val
            v.lastusedchunkindex = st
            return v
        end
    end

    sv = zeros(eltype(eltype(v.data)))
    sv = @set sv[j] = val

    if v.nnz == 0
        push!(v.nzind, Ti(i))
        push!(v.data, [sv])
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = 1
        return v
    end

    if st == beforestartindex(v)  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, [sv])
        else
            v.nzind[1] -= 1
            pushfirst!(v.data[1], sv)
        end
        v.nnz += 1
        v.lastusedchunkindex = 1
        return v
    end

    ifirst, chunk = v.nzind[st], v.data[st]

    if i >= v.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(chunk)  # there is will be the gap in indices after inserting
            push!(v.nzind, i)
            push!(v.data, [sv])
        else  # just append to last chunk
            push!(v.data[st], sv)
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
        append!(v.data[st], [sv], v.data[stnext])
        deleteat!(v.nzind, stnext)
        deleteat!(v.data, stnext)
        v.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        push!(v.data[st], sv)
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        pushfirst!(v.data[stnext], sv)
        v.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(v.nzind, stnext, Ti(i))
        insert!(v.data, stnext, [sv])
        v.lastusedchunkindex = stnext
    end

    v.nnz += 1
    return v

end

@inline function Base.setindex!(v::DensedVLSparseVector{Tv,Ti}, value::AbstractVector, i::Integer) where {Tv,Ti}

    if (st = v.lastusedchunkindex) != beforestartindex(v)
        ifirst, chunk, offsets = v.nzind[st], v.data[st], v.offsets[st]
        if ifirst <= i < ifirst + length(offsets)-1
            lenvalue = length(value)
            pos1 = i-ifirst+1
            offs1 = offsets[pos1]
            offs2 = offsets[pos1+1] - 1
            len = offs2 - offs1 + 1
            if lenvalue == len
                chunk[offs1:offs2] .= value
            elseif lenvalue < len
                deleteat!(chunk, offs1+lenvalue-1:offs2-1)
                offsets[pos1+1:end] .-= len - lenvalue
                offs2 = offsets[pos1+1] - 1
                chunk[offs1:offs2] .= value
            else
                resize!(chunk, length(chunk) + lenvalue - len)
                offsets[pos1+1:end] .+= lenvalue - len
                offs2 = offsets[pos1+1] - 1
                for i = length(chunk):-1:offs2 + 1
                    chunk[i] = chunk[i - (lenvalue - len)]
                end
                chunk[offs1:offs2] .= value
            end
            return v
        end
    end

    st = searchsortedlast(v.nzind, i)

    # check the index exist and update its data
    if st != beforestartindex(v)  # the index `i` is not before the first index
        ifirst, chunk, offsets = v.nzind[st], v.data[st], v.offsets[st]
        if ifirst <= i < ifirst + length(offsets)-1
            lenvalue = length(value)
            pos1 = i-ifirst+1
            offs1 = offsets[pos1]
            offs2 = offsets[pos1+1] - 1
            len = offs2 - offs1 + 1
            if lenvalue == len
                chunk[offs1:offs2] .= value
            elseif lenvalue < len
                deleteat!(chunk, offs1+lenvalue-1:offs2-1)
                offsets[pos1+1:end] .-= len - lenvalue
                offs2 = offsets[pos1+1] - 1
                chunk[offs1:offs2] .= value
            else
                resize!(chunk, length(chunk) + lenvalue - len)
                offsets[pos1+1:end] .+= lenvalue - len
                offs2 = offsets[pos1+1] - 1
                for i = length(chunk):-1:offs2 + 1
                    chunk[i] = chunk[i - (lenvalue - len)]
                end
                chunk[offs1:offs2] .= value
            end
            return v
        end
    end

    #if length(value) == 0
    #    return v
    #end

    if v.nnz == 0
        push!(v.nzind, Ti(i))
        push!(v.data, Vector(value))
        push!(v.offsets, [1])
        append!(v.offsets[1], length(value)+1)
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = 1
        return v
    end

    if st == beforestartindex(v)  # the index `i` is before the first index
        inextfirst = v.nzind[1]
        if inextfirst - i > 1  # there is will be gap in indices after inserting
            pushfirst!(v.nzind, i)
            pushfirst!(v.data, Vector(value))
            pushfirst!(v.offsets, [1])
            append!(v.offsets[1], length(value)+1)
        else
            v.nzind[1] -= 1
            prepend!(v.data[1], value)
            v.offsets[1][2:end] .+= length(value)
            insert!(v.offsets[1], 2, length(value)+1)
        end
        v.nnz += 1
        v.lastusedchunkindex = 1
        return v
    end

    ifirst, chunk, offsets = v.nzind[st], v.data[st], v.offsets[st]

    if i >= v.nzind[end]  # the index `i` is after the last key index
        if i > ifirst + length(offsets)-1  # there is will be the gap in indices after inserting
            push!(v.nzind, i)
            push!(v.data, Vector(value))
            push!(v.offsets, [1])
            push!(v.offsets[end], length(value)+1)
        else  # just append to last chunk
            append!(v.data[st], value)
            push!(v.offsets[st], v.offsets[st][end]+length(value))
        end
        v.nnz += 1
        v.n = max(v.n, Int(i))
        v.lastusedchunkindex = length(v.nzind)
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(offsets)-1 - 1
    stnext = st + 1
    inextfirst = v.nzind[stnext]

    if inextfirst - ilast == 2  # join chunks
        append!(v.data[st], value, v.data[stnext])
        v.offsets[stnext] .+= v.offsets[st][end]-1 + length(value)
        append!(v.offsets[st], v.offsets[stnext])
        #append!(v.offsets[st], [v.offsets[st][end]+length(value)], v.offsets[stnext])
        deleteat!(v.nzind, stnext)
        deleteat!(v.data, stnext)
        deleteat!(v.offsets, stnext)
        v.lastusedchunkindex = st
    elseif i - ilast == 1  # append to left chunk
        append!(v.data[st], value)
        push!(v.offsets[st], v.offsets[st][end]+length(value))
        v.lastusedchunkindex = st
    elseif inextfirst - i == 1  # prepend to right chunk
        v.nzind[stnext] -= 1
        prepend!(v.data[stnext], value)
        v.offsets[stnext][2:end] .+= length(value)
        insert!(v.offsets[stnext], 2, length(value)+1)
        v.lastusedchunkindex = stnext
    else  # insert single element chunk
        insert!(v.nzind, stnext, Ti(i))
        insert!(v.data, stnext, Vector(value))
        #insert!(v.data, stnext, [Vector(value)])
        insert!(v.offsets, stnext, [1])
        push!(v.offsets[stnext], length(value)+1)
        v.lastusedchunkindex = stnext
    end

    v.nnz += 1
    return v

end



@inline function Base.setindex!(v::SDictDensedSparseVector{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    val = eltype(v)(value)

    if (st = v.lastusedchunkindex) != beforestartsemitoken(v.data)
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

    if st == beforestartindex(v)  # the index `i` is before first index
        return v
    end

    ifirst = v.nzind[st]
    lenchunk = v.data[st]

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return v
    end

    if lenchunk == 1
        deleteat!(v.nzind, st)
        deleteat!(v.data, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        v.data[st] -= 1
    elseif i == ifirst  # first element in chunk
        v.nzind[st] += 1
        v.data[st] -= 1
    else
        insert!(v.nzind, st+1, Ti(i+1))
        insert!(v.data, st+1, lenchunk - (i-ifirst+1))
        v.data[st] -= (lenchunk-(i-ifirst+1)) + 1
    end

    v.nnz -= 1
    v.lastusedchunkindex = 0

    return v
end

@inline function Base.delete!(v::SDictDensedSparseIndex{Ti}, i::Integer) where {Ti}

    v.nnz == 0 && return v

    st = searchsortedlast(v.data, i)

    if st == beforestartindex(v)  # the index `i` is before first index
        return v
    end

    (ifirst, chunklen) = deref((v.data, st))

    if i >= ifirst + chunklen  # the index `i` is outside of data chunk indices
        return v
    end

    if chunklen == 1
        delete!(v.data, i)
    elseif i == ifirst + chunklen - 1  # last index in chunk
        v.data[st] -= 1
    elseif i == ifirst
        v.data[i+1] = deref_value((v.data, st)) - 1
        delete!(v.data, i)
    else
        v.data[i+1] = chunklen - (i-ifirst+1)
        v.data[st] -= (chunklen-(i-ifirst+1)) + 1
    end

    v.nnz -= 1
    v.lastusedchunkindex = beforestartsemitoken(v.data)

    return v
end


@inline function Base.delete!(v::AbstractVectorDensedSparseVector, i::Integer)

    v.nnz == 0 && return v

    st = searchsortedlast(v.nzind, i)

    if st == beforestartindex(v)  # the index `i` is before first index
        return v
    end

    ifirst = v.nzind[st]
    lenchunk = length(v.data[st])

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return v
    end

    if lenchunk == 1
        deleteat!(v.data[st], 1)
        deleteat!(v.nzind, st)
        deleteat!(v.data, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        pop!(v.data[st])
    elseif i == ifirst  # first element in chunk
        v.nzind[st] += 1
        popfirst!(v.data[st])
    else
        insert!(v.nzind, st+1, i+1)
        insert!(v.data, st+1, v.data[st][i-ifirst+1+1:end])
        resize!(v.data[st], i-ifirst+1 - 1)
    end

    v.nnz -= 1
    v.lastusedchunkindex = 0

    return v
end

@inline function Base.delete!(v::DensedVLSparseVector, i::Integer)

    v.nnz == 0 && return v

    st = searchsortedlast(v.nzind, i)

    if st == beforestartindex(v)  # the index `i` is before first index
        return v
    end

    ifirst = v.nzind[st]
    lenchunk = length(v.offsets[st]) - 1

    if i >= ifirst + lenchunk  # the index `i` is outside of data chunk indices
        return v
    end

    if lenchunk == 1
        deleteat!(v.data[st], 1)
        deleteat!(v.nzind, st)
        deleteat!(v.data, st)
        deleteat!(v.offsets, st)
    elseif i == ifirst + lenchunk - 1  # last index in chunk
        len = v.offsets[st][end] - v.offsets[st][end-1]
        resize!(v.data[st], length(v.data[st]) - len)
        pop!(v.offsets[st])
        @assert(length(v.data[st]) == v.offsets[st][end]-1)
    elseif i == ifirst  # first element in chunk
        v.nzind[st] += 1
        len = v.offsets[st][2] - v.offsets[st][1]
        deleteat!(v.data[st], 1:len)
        popfirst!(v.offsets[st])
        v.offsets[st] .-= v.offsets[st][1] - 1
        @assert(length(v.data[st]) == v.offsets[st][end]-1)
    else
        insert!(v.nzind, st+1, i+1)
        insert!(v.data, st+1, v.data[st][v.offsets[st][i-ifirst+1+1]:end])
        resize!(v.data[st], v.offsets[st][i-ifirst+1] - 1)
        insert!(v.offsets, st+1, v.offsets[st][i-ifirst+1 + 1:end])
        resize!(v.offsets[st], i-ifirst+1)
        v.offsets[st+1] .-= v.offsets[st+1][1] - 1
        @assert(length(v.data[st]) == v.offsets[st][end]-1)
        @assert(length(v.data[st+1]) == v.offsets[st+1][end]-1)
    end

    v.nnz -= 1
    v.lastusedchunkindex = 0

    return v
end

@inline function Base.delete!(v::SDictDensedSparseVector{Tv,Ti}, i::Integer) where {Tv,Ti}

    v.nnz == 0 && return v

    st = searchsortedlast(v.data, i)

    if st == beforestartindex(v)  # the index `i` is before first index
        return v
    end

    (ifirst, chunk) = deref((v.data, st))

    if i >= ifirst + length(chunk)  # the index `i` is outside of data chunk indices
        return v
    end

    if length(chunk) == 1
        deleteat!(chunk, 1)
        delete!(v.data, i)
    elseif i == ifirst + length(chunk) - 1  # last index in chunk
        pop!(chunk)
        v.data[st] = chunk
    elseif i == ifirst
        popfirst!(chunk)
        v.data[i+1] = chunk
        delete!(v.data, i)
    else
        v.data[i+1] = chunk[i-ifirst+1+1:end]
        v.data[st] = resize!(chunk, i-ifirst+1 - 1)
    end

    v.nnz -= 1
    v.lastusedchunkindex = beforestartsemitoken(v.data)

    return v
end

function Base.empty!(v::Union{DensedSparseIndex,DensedSparseVector,DensedSVSparseVector})
    empty!(v.nzind)
    empty!(v.data)
    v.lastusedchunkindex = beforestartindex(v)
    v.nnz = 0
    v
end
function Base.empty!(v::DensedVLSparseVector)
    empty!(v.nzind)
    empty!(v.data)
    empty!(v.offsets)
    v.lastusedchunkindex = beforestartindex(v)
    v.nnz = 0
    v
end
function Base.empty!(v::Union{SDictDensedSparseVector,SDictDensedSparseVector})
    empty!(v.data)
    v.lastusedchunkindex = beforestartindex(v)
    v.nnz = 0
    v
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

## TODO: integrate `ItWrapper` instead of direct iterating over `Number` and `[Number]`,
## and may be and on `Vector` and `SparseVector`
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
function Base.show(io::IOContext, x::T) where {T<:Union{DensedSparseIndex,SDictDensedSparseIndex}}
    nzind = [v[1] for v in nzchunkpairs(x)]
    nzval = [v[2] for v in nzchunkpairs(x)]
    n = length(nzind)
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end

function quick_get_max_pad(v::AbstractDensedSparseVector)
    pad = 0
    for (key, chunk) in nzchunkpairs(v)
        pad = max(pad, ndigits(key), ndigits(key+length_of_that_nzchunk(v, chunk)-1))
    end
    pad
end

function Base.show(io::IOContext, x::AbstractDensedSparseVector)
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end


function Base.show(io::IO, ::MIME"text/plain", x::DensedSVSparseVector)
    xnnz = 0
    for v in x.data
        xnnz += length(v)
    end
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::DensedSVSparseVector)
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = Vector{eltype(eltype(x.data))}()
    for v in x.data
        for u in v
            push!(nzval, u)
        end
    end
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::DensedVLSparseVector)
    xnnz = 0
    for v in x.offsets
        xnnz += length(v)-1
    end
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::DensedVLSparseVector)
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = Vector{Vector{eltype(eltype(x.data))}}()
    for (offs,v) in zip(x.offsets, x.data)
        for i = 1:length(offs)-1
            push!(nzval, v[offs[i]:offs[i+1]-1])
        end
    end
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
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
    v = T(n)
    Random.seed!(1234)
    for i in shuffle(randsubseq(1:n, density))
        v[i] = rand()
    end
    v
end
function testfun_createSV(T::Type, n = 500_000, density = 0.9)
    v = T(1,n)
    Random.seed!(1234)
    for i in shuffle(randsubseq(1:n, density))
        for j = 1:1
            v[i,j] = rand()
        end
    end
    v
end
function testfun_createVL(T::Type, n = 500_000, density = 0.9)
    v = T(n)
    Random.seed!(1234)
    for i in shuffle(randsubseq(1:n, density))
        v[i] = rand(rand(0:7))
    end
    v
end

function testfun_create_seq(T::Type, n = 500_000, density = 0.9)
    v = T(n)
    Random.seed!(1234)
    for i in randsubseq(1:n, density)
        v[i] = rand()
    end
    v
end

function testfun_create_dense(T::Type, n = 500_000, nchunks = 800, density = 0.95)
    v = T(n)
    chunklen = max(1, floor(Int, n / nchunks))
    Random.seed!(1234)
    for i = 0:nchunks-1
        len = floor(Int, chunklen*density + randn() * chunklen * min(0.1, (1.0-density), density))
        len = max(1, min(chunklen-2, len))
        for j = 1:len
            v[i*chunklen + j] = rand()
        end
    end
    v
end


function testfun_delete!(v)
    Random.seed!(1234)
    indices = shuffle(nonzeroinds(v))
    for i in indices
        delete!(v, i)
    end
    v
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


end  # of module DensedSparseVectors
