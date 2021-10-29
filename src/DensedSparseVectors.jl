#
#  SpacedVector
#  DensedSparseVector
#
# fast and slow realizations:
# on couple of vectors: Vector{FirstIndex<:Int} and Vectror{SomeVectorData}
# and on SortedDict{FirstIndex<:Int, SomeVectorData} respectively.
# The first one for fast index access, the second one for creating and rebuilding.
# Also on the SortedDict{Index<:Int, value} -- the simples and bug free.

#module DensedSparseVectors
#export DensedSparseVector

using DataStructures
using FillArrays
using SparseArrays
using Random

#import Base: getindex, setindex!, unsafe_load, unsafe_store!, nnz, length, isempty


abstract type AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractSpacedVector{Tv,Ti,Tx,Ts} <: AbstractSpacedDensedSparseVector{Tv,Ti,Tx,Ts} end
abstract type AbstractDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc} end

mutable struct SpacedVectorIndex{Ti,Tx<:AbstractVector,Ts<:AbstractVector{Int}} <: AbstractSpacedVector{Bool,Ti,Tx,Ts}
    n::Int     # the vector length
    nnz::Int   # number of non-zero elements
    nzind::Tx  # Vector of chunk's first indices
    data::Ts   # Vector{Int} -- Vector of chunks lengths
end

mutable struct SpacedVector{Tv,Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{<:AbstractVector{Tv}}} <: AbstractSpacedVector{Tv,Ti,Tx,Ts}
    n::Int     # the vector length
    nnz::Int   # number of non-zero elements
    nzind::Tx  # Vector of chunk's first indices
    data::Ts   # Td{<:AbstractVector{Tv}} -- Vector of Vectors (chunks) with values
end

#const SpacedVectorIndex{Ti,Tx} = SpacedVector{Int,Ti,Tx,Tx}

mutable struct DensedSparseVector{Tv,Ti,Td<:Union{Tv,AbstractVector{Tv}},Tc<:AbstractDict{Ti,Td}} <: AbstractDensedSparseVector{Tv,Ti,Td,Tc}
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
Base.eltype(v::AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} = Pair{Ti,Tv}

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


struct SVIIteratorState
    next::Int          #  index of current chunk
    nextpos::Int       #  index in the current chunk
    currentkey::Int
    chunklen::Int
end

function get_init_state(v::SpacedVectorIndex)
    if nnz(v) == 0
        return SVIIteratorState(1, 1, Ti(1), 0)
    else
        return SVIIteratorState(1, 1, v.nzind[1], v.data[1])
    end
end
Base.@propagate_inbounds @inline function nziterate(v::SpacedVectorIndex{Ti,Tx,Ts}, state = get_init_state(v)) where {Ti,Tx,Ts}

    next, nextpos, key, chunklen = state.next, state.nextpos, state.currentkey, state.chunklen
    i = convert(Ti, key + nextpos-1)

    if nextpos < chunklen
        return ((i, true), SVIIteratorState(next, nextpos + 1, key, chunklen))
    elseif nextpos == chunklen
        if next < length(v.nzind)
            return ((i, true), SVIIteratorState(next + 1, 1, v.nzind[next+1], v.data[next+1]))
        elseif next == length(v.nzind)
            return ((i, true), SVIIteratorState(next, nextpos + 1, key, chunklen))
        else
            return nothing
        end
    else
        return nothing
    end
end

struct SVIteratorState{Td}
    next::Int          #  index of current chunk
    nextpos::Int       #  index in the current chunk
    currentkey::Int
    chunk::Td
end

function get_init_state(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td
    if nnz(v) == 0
        return SVIteratorState{Td}(1, 1, Ti(1), Td[])
    else
        return SVIteratorState{Td}(1, 1, v.nzind[1], v.data[1])
    end
end
Base.@propagate_inbounds @inline function nziterate(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}, state = get_init_state(v)) where {Ti,Tx,Ts<:AbstractVector{Td}} where {Td<:AbstractVector{Tv}} where Tv

    next, nextpos, key, chunk = state.next, state.nextpos, state.currentkey, state.chunk
    i = convert(Ti, key + nextpos-1)

    if nextpos < length(chunk)
        d = chunk[nextpos]
        return ((i, d), SVIteratorState{Td}(next, nextpos + 1, key, chunk))
    elseif nextpos == length(chunk)
        d = chunk[nextpos]
        if next < length(v.nzind)
            return ((i, d), SVIteratorState{Td}(next + 1, 1, v.nzind[next+1], v.data[next+1]))
        elseif next == length(v.nzind)
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
    st = startof(v.data)
    if nnz(v) == 0
        return DSVIteratorState{Td}(st, 1, 1, Td[])
    else
        return DSVIteratorState{Td}(st, 1, deref_key((v.data, st)), deref_value((v.data, st)))
    end
end
Base.@propagate_inbounds @inline function nziterate(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}, state = get_init_state(v)) where {Tv,Ti,Td,Tc}

    st, nextpos, key, chunk = state.semitoken, state.nextpos, state.currentkey, state.chunk
    i = convert(Ti, key + nextpos-1)

    if nextpos < length(chunk)
        d = chunk[nextpos]
        return ((i, d), DSVIteratorState{Td}(st, nextpos + 1, key, chunk))
    elseif nextpos == length(chunk)
        d = chunk[nextpos]
        if key < Int(v.lastkey)
            stnext = advance((v.data,st))
            return ((i, d), DSVIteratorState{Td}(stnext, 1, deref_key((v.data, stnext)), deref_value((v.data, stnext))))
        elseif key == Int(v.lastkey)
            return ((i, d), DSVIteratorState{Td}(st, nextpos + 1, key, chunk))
        else
            return nothing
        end
    else
        return nothing
    end
end



struct NZInds{It}
    itr::It
end
nzinds(itr) = NZInds(itr)
@inline function Base.iterate(f::NZInds, state...)
    y = nziterate(f.itr, state...)
    if y !== nothing
        return (y[1][1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZInds{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZInds{It}}) where {It} = Base.EltypeUnknown()
#!!! Base.IteratorEltype(::Type{NZInds{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZInds}) = Base.SizeUnknown()
Base.reverse(f::NZInds) = NZInds(reverse(f.itr))
@inline Base.keys(v::AbstractSpacedDensedSparseVector) = nzinds(v)

struct NZVals{It}
    itr::It
end
nzvals(itr) = NZVals(itr)
@inline function Base.iterate(f::NZVals, state...)
    y = nziterate(f.itr, state...)
    if y !== nothing
        return (y[1][2], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZVals{It}}) where {It} = eltype(It)
Base.IteratorEltype(::Type{NZVals{It}}) where {It} = Base.IteratorEltype(It)
Base.IteratorSize(::Type{<:NZVals}) = Base.SizeUnknown()
Base.reverse(f::NZVals) = NZVals(reverse(f.itr))

# TODO:
# julia> collect(nzpairs(dsv))
# ERROR: MethodError: no method matching length(::NZPairs{DensedSparseVector{Float64, Int64, Vector{Float64}, SortedDict{Int64, Vector{Float64}}}})

struct NZPairs{It}
    itr::It
end
nzpairs(itr) = NZPairs(itr)
@inline function Base.iterate(f::NZPairs, state...)
    y = nziterate(f.itr, state...)
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
Base.reverse(f::NZPairs) = NZPairs(reverse(f.itr))


SparseArrays.findnz(v::AbstractSpacedDensedSparseVector) = (SparseArrays.nonzeroinds(v), SparseArrays.nonzeros(v))


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
## https://docs.julialang.org/en/v1/manual/interfaces/#Selecting-an-appropriate-output-array
#
#Base.getindex(A::ArrayAndChar{T,N}, inds::Vararg{Int,N}) where {T,N} = A.data[inds...]
#Base.setindex!(A::ArrayAndChar{T,N}, val, inds::Vararg{Int,N}) where {T,N} = A.data[inds...] = val

Base.Broadcast.BroadcastStyle(::Type{<:AbstractSpacedDensedSparseVector}) = SparseArrays.HigherOrderFns.SparseVecStyle()

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


#end  # of module DensedSparseVectors
