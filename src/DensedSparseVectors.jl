#
#SpacedVector
#DensedSparseVector
#
# быстрая и медленная реализации:
# на Vector{Tuple|Pair{FirstIndex::Int, SomeVectorData}}
# и на SortedDict{FirstIndex::Int, SomeVectorData}
# первая для быстрого доступа, вторая для по-/перестроения,
# а также, на SortedDict{Index::Int, Value} -- как самый простой и багоустойчивый

#module DensedSparseVectors
#export DensedSparseVector

using DataStructures
using FillArrays
using SparseArrays

#import Base: getindex, setindex!, unsafe_load, unsafe_store!, nnz, length, isempty


abstract type AbstractSpacedDensedSparseVector{Tv,Ti,Tc,Td} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractSpacedVector{Tv,Ti,Tx,Td} <: AbstractSpacedDensedSparseVector{Tv,Ti,Tx,Td} end
abstract type AbstractDensedSparseVector{Tv,Ti,Tc,Td} <: AbstractSpacedDensedSparseVector{Tv,Ti,Tc,Td} end

mutable struct SpacedVector{Tv,Ti,Tx<:AbstractVector,Td<:AbstractVector} <: AbstractSpacedVector{Tv,Ti,Tx,Td}
    n::Int     # the vector length
    nnz::Int   # number of non-zero elements
    nzind::Tx  # Vector of chunk's first indices
    data::Tx   # Tx{Td{Tv}} -- Vector of Vectors (chunks) with values
end

mutable struct DensedSparseVector{Tv,Ti,Tc<:AbstractDict,Td<:AbstractVector} <: AbstractDensedSparseVector{Tv,Ti,Tc,Td}
    n::Int       # the vector length
    nnz::Int     # number of non-zero elements
    lastkey::Ti  # the last node key in `data` tree
    data::Tc     # Tc{Ti,Td{Tv}} -- tree based Dict data container
end


@inline function SpacedVector(n::Integer = 0)
    return SpacedVector{Float64,Int,Vector,Vector}(n, 0, Vector{Int}(), Vector{Float64}())
end
@inline function SpacedVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
    return SpacedVector{Tv,Ti,Vector,Vector}(n, 0, Vector{Ti}(), Vector{Tv}())
end
@inline function SpacedVector{Tv,Ti,Tx,Td}(n::Integer = 0) where {Tv,Ti,Tx,Td}
    return SpacedVector{Tv,Ti,Tx,Td}(n, 0, Tx{Ti}(), Td{Tv}())
end

@inline function DensedSparseVector(n::Integer = 0)
    return DensedSparseVector{Float64,Int,SortedDict,Vector}(n, 0, typemin(Int), SortedDict{Int,Vector{Float64}}())
end
@inline function DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
    return DensedSparseVector{Tv,Ti,SortedDict,Vector}(n, 0, typemin(Ti), SortedDict{Ti,Vector{Tv}}())
end
@inline function DensedSparseVector{Tv,Ti,Tc,Td}(n::Integer = 0) where {Tv,Ti,Tc,Td}
    return DensedSparseVector{Tv,Ti,Tc,Td}(n, 0, typemin(Ti), Tc{Ti,Td{Tv}}())
end

function SpacedVector(dsv::AbstractDensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td}
    nzind = Vector{Ti}(undef, length(dsv.data))
    data = Vector{Td}(undef, length(nzind))
    i = 1
    for (k,d) in dsv.data
        nzind[i] = k
        data[i] = d
        i += 1
    end
    return SpacedVector{Tv,Ti,Tx,Td}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVector{Tv,Ti,Tx,Td}(dsv::AbstractDensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tx,Tc,Td}
    nzind = Tx{Ti}(undef, length(dsv.data))
    data = Tx{Td}(undef, length(nzind))
    i = 1
    for (k,d) in dsv.data
        nzind[i] = k
        data[i] = d
        i += 1
    end
    return SpacedVector{Tv,Ti,Tx,Td}(dsv.n, dsv.nnz, nzind, data)
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
SparseArrays.nnz(v::AbstractSpacedDensedSparseVector) = v.nnz
Base.isempty(v::AbstractSpacedDensedSparseVector) = v.nnz == 0
Base.size(v::AbstractSpacedDensedSparseVector) = (v.n,)

struct SVIteratorState{Ti}
    next::Int          #  index of current chunk
    nextpos::Int       #  index in current chunk
    nextindex::Ti      #  global index of current position
    ref::Base.RefValue #  ref to chunk
end

function get_init_state(v::AbstractSpacedVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td<:AbstractVector}
    if nnz(v) > 0
        return SVIteratorState{Ti}(1, 1, v.nzind[1], Ref(v.data[1]))
    else
        return SVIteratorState{Ti}(1, 1, Ti(1), Ref(Td[]))
    end
end
function Base.iterate(v::AbstractSpacedVector{Tv,Ti,Tc,Td}, state = get_init_state(v)) where {Tv,Ti,Tc,Td<:AbstractVector}
    if state.nextpos < length(state.ref[])
        return ((state.nextindex, state.ref[][state.nextpos]),
                SVIteratorState{Ti}(state.next, state.nextpos + 1, state.nextindex + 1, state.ref))
    elseif state.nextpos == length(state.ref[])
        if state.next < length(v.nzind)
            next = state.next + 1
            return ((state.nextindex, state.ref[][state.nextpos]),
                    SVIteratorState{Ti}(next, 1, v.nzind[next], Ref(v.data[next])))
        elseif state.next == length(v.nzind)
            return ((state.nextindex, state.ref[][state.nextpos]),
                    SVIteratorState{Ti}(state.next + 1, state.nextpos + 1, state.nextindex + 1, state.ref))
        else
            return nothing
        end
    else
        return nothing
    end
end




struct NZInds{I}
    itr::I
end

nzinds(itr) = NZInds(itr)

@inline function Base.iterate(f::NZInds, state...)
    y = iterate(f.itr, state...)
    return y !== nothing ? (y[1][1], y[2]) : nothing
end
#@inline Base.iterate(f::NZInds, state...) = iterate(f.itr, state...)[1]

Base.eltype(::Type{NZInds{I}}) where {I} = eltype(I)
Base.IteratorEltype(::Type{NZInds{I}}) where {I} = IteratorEltype(I)
Base.IteratorSize(::Type{<:NZInds}) = SizeUnknown()

reverse(f::NZInds) = NZInds(reverse(f.itr))


function SparseArrays.nonzeroinds(v::AbstractSpacedVector{Tv,Ti,Tx,Td}) where {Tv,Ti,Tx,Td}
end

struct DSVIteratorState{Ti}
    next::Ti
    nextpos::Int
    nextindex::Ti
    ref::Base.RefValue
end

#function get_init_state(v::AbstractDensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td<:AbstractVector}
#    (k, d) = deref((v, startof(v)))
#    return DSVIteratorState{Ti,Td}(k, d, 1)
#end
#function Base.iterate(v::AbstractDensedSparseVector{Tv,Ti,Tc,Td}, state = get_init_state(v)) where {Tv,Ti,Tc,Td<:AbstractVector}
#    if state.pos != length(state.v)
#    else
#        st = advance((v.data, state.st))
#        (k, d) = deref((v, st))
#        return d[1], DSVIteratorState{Ti,Td}(k, d, 1)
#    end
#end

function SparseArrays.nonzeroinds(v::AbstractDensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td}
    ret = Vector{Ti}()
    for (k,d) in v.data
        append!(ret, (k:k+size(d)[1]-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractDensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td}
    ret = Vector{Tv}()
    for d in values(v.data)
        append!(ret, d)
    end
    return ret
end

function SparseArrays.nonzeroinds(v::AbstractSpacedVector{Tv,Ti,Tx,Td}) where {Tv,Ti,Tx,Td}
    ret = Vector{Ti}()
    for (k,d) in zip(v.nzind, v.data)
        append!(ret, (k:k+size(d)[1]-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractSpacedVector{Tv,Ti,Tx,Td}) where {Tv,Ti,Tx,Td}
    ret = Vector{Tv}()
    for d in v.data
        append!(ret, d)
    end
    return ret
end

@inline function Base.isstored(v::DensedSparseVector, i::Integer)

    v.nnz > 0 || return false

    st = searchsortedlast(v.data, i)
    sstatus = status((v.data, st))
    if sstatus == 2 || sstatus == 0  # the index `i` is before first index or invalid
        return false
    elseif i >= deref_key((v.data, st)) + length(deref_value((v.data, st)))
        # the index `i` is outside of data chunk indices
        return false
    end

    return true
end
@inline Base.haskey(v::DensedSparseVector, i) = isstored(v, i)


@inline function Base.getindex(v::DensedSparseVector{Tv,Ti,Tc,Td}, i::Integer) where {Tv,Ti,Tc,Td}

    v.nnz > 0 || return Tv(0)

    st = searchsortedlast(v.data, i)

    sstatus = status((v.data, st))
    if sstatus == 2 || sstatus == 0  # the index `i` is before first index or invalid
        return Tv(0)
    end

    (ifirst, chunk) = deref((v.data, st))

    if i >= ifirst + length(chunk)  # the index `i` is outside of data chunk indices
        return Tv(0)
    end

    return chunk[i - ifirst + 1]
end

@inline function Base.unsafe_load(v::DensedSparseVector, i::Integer)
    (ifirst, chunk) = deref((v.data, searchsortedlast(v.data, i)))
    return chunk[i - ifirst + 1]
end


@inline function Base.setindex!(v::DensedSparseVector{Tv,Ti,Tc,Td}, value, i::Integer) where {Tv,Ti,Tc,Td}
    val = convert(Tv, value)

    st = searchsortedlast(v.data, i)

    sstatus = status((v.data, st))
    @boundscheck if sstatus == 0 # invalid semitoken
        trow(KeyError(i))
    end

    # check the index exist and update its data
    if v.nnz > 0 && sstatus != 2  # the index `i` is not before the first index
        (ifirst, chunk) = deref((v.data, st))
        if i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            return v
        end
    end

    if v.nnz == 0
        v.data[i] = Td(Fill(val,1))
        v.nnz += 1
        v.n = Int(i)
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
        if ifirst + length(chunk) != i  # there is will be the gap in indices after inserting
            v.data[i] = Td(Fill(val,1))
            v.lastkey = Ti(i)
        else  # just append to last chunk
            v.data[st] = push!(chunk, val)
        end
        v.nnz += 1
        v.n = Int(i)
        return v
    end

    # the index `i` is somewhere between indices
    ilast = ifirst + length(chunk) - 1
    stnext = advance((v.data, st))
    inextfirst = deref_key((v.data, stnext))

    if inextfirst - ilast == 1  # join chunks
        v.data[st] = append!(chunk, [val], deref_value((v.data, stnext)))
        delete!((v.data, stnext))
    elseif i - ilast == 1  # append to left chunk
        v.data[st] = push!(chunk, val)
    elseif inextfirst - i == 1  # prepend to right chunk
        v.data[i] = pushfirst!(deref_value((v.data, stnext)), val)
        delete!((v.data, stnext))
    else  # insert single element chunk
        v.data[i] = Td(Fill(val,1))
    end

    v.nnz += 1
    return v

end

@inline function Base.unsafe_store!(v::DensedSparseVector{Tv,Ti,Tc,Td}, value, i) where {Tv,Ti,Tc,Td}
    (ifirst, chunk) = deref((v.data, searchsortedlast(v.data, i)))
    chunk[i - ifirst + 1] = convert(Tv, value)
    return v
end

@inline function Base.delete!(v::DensedSparseVector, i::Integer)

    v.nnz > 0 || return v

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
        v.data = delete!(v.data, i)
        i == v.lastkey && (v.lastkey = v.nnz > 0 ? lastindex(v.data) : typemin(Ti))
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
        #v.data[st] = chunk[1:i-ifirst+1-1]
        v.data[st] = resize!(chunk, i-ifirst+1 - 1)
    end

    v.nnz -= 1

    return v
end
@inline Base.deleteat!(v::DensedSparseVector, i::Integer) = delete!(v, i)


function merge!(v::DensedSparseVector, ifirst::Integer, app::AbstractVector)
    i = ifirst
    for d in app
        v[i] = d
        i += 1
    end
    return v
end

#end  # of module DensedSparseVectors
