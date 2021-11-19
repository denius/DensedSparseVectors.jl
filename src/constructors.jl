

@inline SpacedIndex(n::Integer = 0) = SpacedIndex{Int,Vector,Vector}(n)
@inline SpacedIndex{Ti}(n::Integer = 0) where {Ti} = SpacedIndex{Ti,Vector}(n)
@inline SpacedIndex{Ti,Tx}(n::Integer = 0) where {Ti,Tx} =
    SpacedIndex{Ti,Tx{Int},Tx{Ti}}(n, 0, Tx{Ti}(), Tx{Int}(), 0)

@inline SpacedVector(n::Integer = 0) = SpacedVector{Float64,Int,Vector,Vector}(n)
@inline SpacedVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = SpacedVector{Tv,Ti,Vector,Vector}(n)
@inline SpacedVector{Tv,Ti,Ts,Tx}(n::Integer = 0) where {Tv,Ti,Ts,Tx} =
    SpacedVector{Tv,Ti,Tx{Ts{Tv}},Tx{Ti}}(n, 0, Tx{Ti}(), Tx{Ts{Tv}}(), 0)

@inline DensedSparseVector(n::Integer = 0) = DensedSparseVector{Float64,Int,Vector,SortedDict}(n)
@inline DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} = DensedSparseVector{Tv,Ti,Vector,SortedDict}(n)
@inline function DensedSparseVector{Tv,Ti,Td,Tc}(n::Integer = 0) where {Tv,Ti,Td,Tc}
    data = Tc{Ti,Td{Tv}}()
    DensedSparseVector{Tv,Ti,Td{Tv},Tc{Ti,Td{Tv},FOrd}}(n, 0, data, beforestartsemitoken(data))
end


SpacedIndex(v::AbstractAlmostSparseVector) = SpacedIndex{Vector}(v)

function SpacedIndex{Tdata}(v::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tdata<:AbstractVector,Tv,Ti,Td,Tc}
    nzind = Tdata{Ti}(undef, nnzchunks(v))
    data = Tdata{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = length_of_that_nzchunk(v, d)
    end
    return SpacedIndex{Ti,Tdata{Int},Tdata{Ti}}(v.n, v.nnz, nzind, data, 0)
end


SpacedVector(v::AbstractDensedSparseVector) = SpacedVector{Vector}(v)
SpacedVector{Tdata}(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tdata,Tv,Ti,Td,Tc} = SpacedVector{Tv,Ti,Td,Tdata}(v)

function SpacedVector{Tv,Ti,Ts,Tx}(v::AbstractDensedSparseVector) where {Tv,Ti,Ts,Tx}
    nzind = Tx{Ti}(undef, nnzchunks(v))
    data = Tx{Ts}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = Ts(d)
    end
    return SpacedVector{Tv,Ti,Tx{Ts},Tx{Ti}}(v.n, v.nnz, nzind, data, 0)
end

