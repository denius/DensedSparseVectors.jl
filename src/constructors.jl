import Base: ForwardOrdering, Forward
const FOrd = ForwardOrdering

@inline SparseArrays.sparse(sv::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} =
    SparseVector(length(sv), SparseArrays.nonzeroinds(sv), SparseArrays.nonzeros(sv))
@inline SparseArrays.SparseVector(sv::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} =
    SparseVector(length(sv), SparseArrays.nonzeroinds(sv), SparseArrays.nonzeros(sv))

@inline SpacedIndex(n::Integer = 0) =
    SpacedIndex{Int,Vector{Int},Vector{Int}}(n, 0, Vector{Int}(), Vector{Int}())
@inline SpacedIndex{Ti}(n::Integer = 0) where {Ti} =
    SpacedIndex{Ti,Vector{Ti},Vector{Int}}(n, 0, Vector{Ti}(), Vector{Int}())
@inline SpacedIndex{Ti,Tx}(n::Integer = 0) where {Ti,Tx} =
    SpacedIndex{Ti,Tx{Ti},Tx{Int}}(n, 0, Tx{Ti}(), Tx{Int}())

@inline SpacedVector(n::Integer = 0) =
    SpacedVector{Float64,Int,Vector{Int},Vector{Vector{Float64}}}(n, 0, Vector{Int}(), Vector{Vector{Float64}}())
@inline SpacedVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
    SpacedVector{Tv,Ti,Vector{Ti},Vector{Vector{Tv}}}(n, 0, Vector{Ti}(), Vector{Vector{Tv}}())
@inline SpacedVector{Tv,Ti,Tx,Ts}(n::Integer = 0) where {Tv,Ti,Tx,Ts} =
    SpacedVector{Tv,Ti,Tx{Ti},Tx{Ts{Tv}}}(n, 0, Tx{Ti}(), Tx{Ts{Tv}}())

@inline DensedSparseVector(n::Integer = 0) =
    DensedSparseVector{Float64,Int,Vector{Float64},SortedDict{Int,Vector{Float64},FOrd}}(n, 0, typemin(Int), SortedDict{Int,Vector{Float64},FOrd}(Forward))
@inline DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
    DensedSparseVector{Tv,Ti,Vector{Float64},SortedDict{Int,Vector{Float64},FOrd}}(n, 0, typemin(Ti), SortedDict{Ti,Vector{Tv},FOrd}(Forward))
@inline DensedSparseVector{Tv,Ti,Td,Tc}(n::Integer = 0) where {Tv,Ti,Td,Tc} =
    DensedSparseVector{Tv,Ti,Td{Tv},Tc{Ti,Td{Tv}}}(n, 0, typemin(Ti), Tc{Ti,Td{Tv}}())


SpacedIndex(v::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} = SpacedIndex{Vector}(v)

function SpacedIndex{Tdata}(v::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tdata<:AbstractVector,Tv,Ti,Td,Tc}
    nzind = Tdata{Ti}(undef, nnzchunks(v))
    data = Tdata{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = length_of_that_chunk(v, d)
    end
    return SpacedIndex{Ti,Tdata{Ti},Tdata{Int}}(dsv.n, dsv.nnz, nzind, data)
end


SpacedVector(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} = SpacedVector{Vector}(v)

function SpacedVector{Tdata}(v::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tdata,Tv,Ti,Td,Tc}
    nzind = Tdata{Ti}(undef, nnzchunks(v))
    data = Tdata{Td}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = d
    end
    return SpacedVector{Tv,Ti,Tdata{Ti},Tdata{Td}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVector{Tv,Ti,Tx,Ts}(dsv::AbstractDensedSparseVector) where {Tv,Ti,Tx,Ts}
    nzind = Tx{Ti}(undef, nnzchunks(v))
    data = Tx{Ts{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = Ts{Tv}(d)
    end
    return SpacedVector{Tv,Ti,Tx{Ti},Tx{Ts{Tv}}}(dsv.n, dsv.nnz, nzind, data)
end

