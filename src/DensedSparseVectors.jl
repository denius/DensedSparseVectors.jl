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
using Random

#import Base: getindex, setindex!, unsafe_load, unsafe_store!, nnz, length, isempty


abstract type AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractSparseVector{Tv,Ti} end

abstract type AbstractSpacedVector{Tv,Ti,Tx,Ts} <: AbstractSpacedDensedSparseVector{Tv,Ti,Tx,Ts} end
abstract type AbstractDensedSparseVector{Tv,Ti,Td,Tc} <: AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc} end

mutable struct SpacedVector{Tv,Ti,Tx<:AbstractVector{Ti},Ts<:AbstractVector{<:AbstractVector{Tv}}} <: AbstractSpacedVector{Tv,Ti,Tx,Ts}
    n::Int     # the vector length
    nnz::Int   # number of non-zero elements
    nzind::Tx  # Vector of chunk's first indices
    data::Ts   # Td{<:AbstractVector{Tv}} -- Vector of Vectors (chunks) with values
end

mutable struct DensedSparseVector{Tv,Ti,Td<:AbstractVector{Tv},Tc<:AbstractDict{Ti,Td}} <: AbstractDensedSparseVector{Tv,Ti,Td,Tc}
    n::Int       # the vector length
    nnz::Int     # number of non-zero elements
    lastkey::Ti  # the last node key in `data` tree
    data::Tc     # Tc{Ti,Td{Tv}} -- tree based Dict data container
end


@inline function SpacedVector(n::Integer = 0)
    return SpacedVector{Float64,Int,Vector{Int},Vector{Vector{Float64}}}(n, 0, Vector{Int}(), Vector{Float64}())
end
@inline function SpacedVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
    return SpacedVector{Tv,Ti,Vector{Ti},Vector{Vector{Tv}}}(n, 0, Vector{Ti}(), Vector{Tv}())
end
@inline function SpacedVector{Tv,Ti,Tx,Ts}(n::Integer = 0) where {Tv,Ti,Tx,Ts}
    return SpacedVector{Tv,Ti,Tx,Ts}(n, 0, Tx(), Td())
end

@inline function DensedSparseVector(n::Integer = 0)
    return DensedSparseVector{Float64,Int,Vector{Float64},SortedDict{Int,Vector{Float64}}}(n, 0, typemin(Int), SortedDict{Int,Vector{Float64}}())
end
@inline function DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti}
    return DensedSparseVector{Tv,Ti,Vector,SortedDict}(n, 0, typemin(Ti), SortedDict{Ti,Vector{Tv}}())
end
@inline function DensedSparseVector{Tv,Ti,Td,Tc}(n::Integer = 0) where {Tv,Ti,Td,Tc}
    return DensedSparseVector{Tv,Ti,Td,Tc}(n, 0, typemin(Ti), Tc{Ti,Td{Tv}}())
end

function SpacedVector{Tdata}(dsv::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tdata,Tv,Ti,Td,Tc}
    nzind = Vector{Ti}(undef, length(dsv.data))
    data = Vector{Td}(undef, length(nzind))
    i = 1
    for (k,d) in dsv.data
        nzind[i] = k
        data[i] = d
        i += 1
    end
    return SpacedVector{Tv,Ti,Tdata{Ti},Tdata{Td}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVector{Tv,Ti,Tx,Ts}(dsv::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Tx,Ts,Td,Tc}
    nzind = Tx(undef, length(dsv.data))
    data = Ts(undef, length(nzind))
    i = 1
    for (k,d) in dsv.data
        nzind[i] = k
        data[i] = d
        i += 1
    end
    return SpacedVector{Tv,Ti,Tx,Ts}(dsv.n, dsv.nnz, nzind, data)
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

struct SVIteratorState{Td}
    next::Int          #  index of current chunk
    nextpos::Int       #  index in the current chunk
    currentkey::Int
    chunk::Td
end

@inline function get_init_state(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td
    if nnz(v) == 0
        return SVIteratorState{Td}(1, 1, Ti(1), Td[])
    else
        return SVIteratorState{Td}(1, 1, v.nzind[1], v.data[1])
    end
end
@inline function Base.iterate(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}, state = get_init_state(v)) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td

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
@inline function Base.iterate(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}, state = get_init_state(v)) where {Tv,Ti,Td,Tc}

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




struct NZInds{I}
    itr::I
end
nzinds(itr) = NZInds(itr)
@inline function Base.iterate(f::NZInds, state...)
    y = iterate(f.itr, state...)
    if y !== nothing
        return (y[1][1], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZInds{I}}) where {I} = eltype(I)
Base.IteratorEltype(::Type{NZInds{I}}) where {I} = IteratorEltype(I)
Base.IteratorSize(::Type{<:NZInds}) = SizeUnknown()
Base.reverse(f::NZInds) = NZInds(reverse(f.itr))
@inline Base.keys(v::AbstractSpacedDensedSparseVector) = nzinds(v)

struct NZVals{I}
    itr::I
end
nzvals(itr) = NZVals(itr)
@inline function Base.iterate(f::NZVals, state...)
    y = iterate(f.itr, state...)
    if y !== nothing
        return (y[1][2], y[2])
    else
        return nothing
    end
end
Base.eltype(::Type{NZVals{I}}) where {I} = eltype(I)
Base.IteratorEltype(::Type{NZVals{I}}) where {I} = IteratorEltype(I)
Base.IteratorSize(::Type{<:NZVals}) = SizeUnknown()
Base.reverse(f::NZVals) = NZVals(reverse(f.itr))



function SparseArrays.nonzeroinds(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc}
    ret = Vector{Ti}()
    for (k,d) in v.data
        append!(ret, (k:k+size(d)[1]-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc}
    ret = Vector{Tv}()
    for d in values(v.data)
        append!(ret, d)
    end
    return ret
end

function SparseArrays.nonzeroinds(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    ret = Vector{Ti}()
    for (k,d) in zip(v.nzind, v.data)
        append!(ret, (k:k+size(d)[1]-1))
    end
    return ret
end
function SparseArrays.nonzeros(v::AbstractSpacedVector{Tv,Ti,Tx,Ts}) where {Tv,Ti,Tx,Ts}
    ret = Vector{Tv}()
    for d in v.data
        append!(ret, d)
    end
    return ret
end

@inline function Base.isstored(v::DensedSparseVector, i::Integer)

    v.nnz == 0 && return false

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


@inline function Base.getindex(v::SpacedVector{Tv,Ti,Td,Tc}, i::Integer) where {Tv,Ti,Td,Tc}

    v.nnz == 0 && return Tv(0)

    st = searchsortedlast(v.nzind, i)

    # the index `i` is before first index
    st == 0 && return Tv(0)

    ifirst, chunk = v.nzind[st], v.data[st]

    # the index `i` is outside of data chunk indices
    i >= ifirst + length(chunk) && return Tv(0)

    return chunk[i - ifirst + 1]
end

@inline function Base.unsafe_load(v::SpacedVector, i::Integer)
    st = searchsortedlast(v.nzind, i)
    ifirst, chunk = v.nzind[st], v.data[st]
    return chunk[i - ifirst + 1]
end


@inline function Base.getindex(v::DensedSparseVector{Tv,Ti,Td,Tc}, i::Integer) where {Tv,Ti,Td,Tc}

    v.nnz == 0 && return Tv(0)

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


function Base.setindex!(v::SpacedVector{Tv,Ti,Tx,Ts}, value::Number, i::Integer) where {Tv,Ti,Tx,Ts<:AbstractVector{Td}} where Td
    val = Tv(value)

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





function Base.setindex!(v::DensedSparseVector{Tv,Ti,Td,Tc}, value::Number, i::Integer) where {Tv,Ti,Td,Tc}
    val = Tv(value)

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
        if ifirst + length(chunk) != i  # there is will be the gap in indices after inserting
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
    index = Ti(index-1)
    if v === data
        cdata = deepcopy(data)
        for (i,d) in cdata
            v[index+i] = Tv(d)
        end
    else
        for (i,d) in data
            v[index+i] = Tv(d)
        end
    end
    return v
end
function Base.setindex!(v::AbstractSpacedDensedSparseVector{Tv,Ti,Td,Tc}, data::AbstractVector, i::Integer) where {Tv,Ti,Td,Tc}
    for d in data
        v[i] = Tv(d)
        i += 1
    end
    return v
end

@inline function Base.unsafe_store!(v::DensedSparseVector{Tv,Ti,Td,Tc}, value, i::Integer) where {Tv,Ti,Td,Tc}
    (ifirst, chunk) = deref((v.data, searchsortedlast(v.data, i)))
    chunk[i - ifirst + 1] = Tv(value)
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
@inline Base.deleteat!(v::AbstractSpacedDensedSparseVector, i::Integer) = delete!(v, i)


function testfun_create(n = 500_000)

    dsv = DensedSparseVector(n)

    Random.seed!(1234)
    for i in rand(1:n, 4*n)
        dsv[i] = rand()
    end

    sv = SpacedVector(n)

    Random.seed!(1234)
    for i in rand(1:n, 4*n)
        sv[i] = rand()
    end

    (dsv, sv)
end

function testfun_create_dense()

    dsv = DensedSparseVector(500_000)

    Random.seed!(1234)
    dsv[1001]    = rand(100_000)
    dsv[200_001] = rand(50_000)
    dsv[400_001] = rand(20_000)

    sv = SpacedVector(500_000)

    Random.seed!(1234)
    sv[1001]    = rand(100_000)
    sv[200_001] = rand(50_000)
    sv[400_001] = rand(20_000)

    (dsv, sv)
end

function testfun_delete(dsv, sv)

    Random.seed!(1234)
    indices = shuffle(SparseArrays.nonzeroinds(dsv))
    for i in indices
        deleteat!(dsv, i)
    end

    Random.seed!(1234)
    indices = shuffle(SparseArrays.nonzeroinds(sv))
    for i in indices
        deleteat!(sv, i)
    end

    (dsv, sv)
end

function testfun1(sv)
    I = 0
    S = 0.0
    for ic in axes(sv.nzind, 1)
        ind = sv.nzind[ic]-1
        dat = sv.data[ic]
        for i in axes(dat,1)
            I += ind+i
            S += dat[i]
        end
    end
    (I, S)
end

function testfun2(sv)
    I=0
    S=0.0
    for (k,v) in sv
        I += k
        S += v
    end
    (I, S)
end

function testfun3(sv)
    I=0
    for k in nzinds(sv)
        I += k
    end
    (I, 0.0)
end

function testfun4(sv)
    S = 0.0
    for v in nzvals(sv)
        S += v
    end
    (0, S)
end


#end  # of module DensedSparseVectors
