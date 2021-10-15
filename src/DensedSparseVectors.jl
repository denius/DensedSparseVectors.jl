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


abstract type AbstractDensedSparseVector{Tv,Ti,Tc} <: AbstractSparseVector{Tv,Ti} end

#mutable struct SpacedVector{Tv,Ti,Tc<:AbstractVector,Td<:AbstractVector} <: AbstractDensedSparseVector{Tv,Ti,Tc}
#    n::Int            # the vector length
#    nnz::Int          # number of non-zero elements
#    nzind::Tc{Tv}     # Vector of chunks first indices
#    data::Tc{Td{Tv}}  # Vector of Vectors (chunks) with values
#end

mutable struct DensedSparseVector{Tv,Ti,Tc<:AbstractDict,Td<:AbstractVector} <: AbstractDensedSparseVector{Tv,Ti,Tc}
    n::Int               # the vector length
    nnz::Int             # number of non-zero elements
    data::Tc  # tree based Dict data container
    #data::Tc{Ti,Td{Tv}}  # tree based Dict data container
end


function DensedSparseVector{Tv,Ti,Tc,Td}() where {Tv,Ti,Tc,Td}
    return DensedSparseVector(0, 0, Tc{Ti,Td{Tv}}())
end


function Base.length(v::DensedSparseVector)
    if v.nnz == 0
        return 0
    else
        (ilastfirst, chunk) = last(v.data)
        return ilastfirst + length(chunk) - 1
    end
end
SparseArrays.nnz(v::AbstractDensedSparseVector) = v.nnz
Base.isempty(v::AbstractDensedSparseVector) = v.nnz == 0
Base.size(v::AbstractDensedSparseVector) = v.n

function SparseArrays.nonzeroinds(v::DensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td}
    ret = Vector{Ti}()
    for (k,d) in v.data
        append!(ret, (k:k+length(d)-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::DensedSparseVector{Tv,Ti,Tc,Td}) where {Tv,Ti,Tc,Td}
    ret = Vector{Tv}()
    for d in values(v.data)
        append!(ret, d)
    end
    return ret
end

#@inline function Base.haskey(v::DensedSparseVector{Tv,Ti,Tc,Td}, i) where {Tv,Ti,Tc<:AbstractDict,Td}
@inline function Base.haskey(v::DensedSparseVector, i)

    #isempty(v.data) && return false
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
@inline hasindex(v::DensedSparseVector, i) = haskey(v, i)


#@inline function Base.getindex(v::DensedSparseVector{Tv,Ti,Tc,Td}, i) where {Tv,Ti,Tc<:AbstractDict,Td}
@inline function Base.getindex(v::DensedSparseVector, i)

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

#@inline function Base.unsafe_load(v::DensedSparseVector{Tv,Ti,Tc,Td}, i) where {Tv,Ti,Tc<:AbstractDict,Td}
@inline function Base.unsafe_load(v::DensedSparseVector, i)
    (ifirst, chunk) = deref((v.data, searchsortedlast(v.data, i)))
    return chunk[i - ifirst + 1]
end


@inline function Base.setindex!(v::DensedSparseVector{Tv,Ti,Tc,Td}, value, i) where {Tv,Ti,Tc,Td}
    val = convert(Tv, value)

    st = searchsortedlast(v.data, i)

    sstatus = status((v.data, st))
    @show sstatus, deref((v.data, st))
    @boundscheck if sstatus == 0 # invalid semitoken
        trow(KeyError(i))
    end

    if v.nnz > 0 && (sstatus == 1 || sstatus == 3)  # the index `i` is not before the first index
        (ifirst, chunk) = deref((v.data, st))
        if i < ifirst + length(chunk)
            chunk[i - ifirst + 1] = val
            return v
        end
    end

    if isempty(v.data)
        v.data[i] = Td(Fill(val,1))
        v.nnz += 1
        v.n = i
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

    if sstatus == 3  # the index `i` is after the last index
        stlast = endof(v.data)
        (ilastfirst, lastchunk) = deref((v.data, stlast))
        if ilastfirst + length(lastchunk) != i  # there is will be gap in indices after inserting
            v.data[i] = Td(Fill(val,1))
        else  # just append to last chunk
            v.data[stlast] = push!(lastchunk, val)
        end
        v.nnz += 1
        v.n = i
        return v
    end

    # the index `i` is somewhere between indices
    (ifirst, chunk) = deref((v.data, st))
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

#end  # of module DensedSparseVectors
