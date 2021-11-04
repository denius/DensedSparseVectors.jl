
@inline SparseArrays.sparse(sv::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} =
    SparseVector(length(sv), SparseArrays.nonzeroinds(sv), SparseArrays.nonzeros(sv))
@inline SparseArrays.SparseVector(sv::AbstractAlmostSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc} =
    SparseVector(length(sv), SparseArrays.nonzeroinds(sv), SparseArrays.nonzeros(sv))

@inline SpacedVectorIndex(n::Integer = 0) =
    SpacedVectorIndex{Int,Vector{Int},Vector{Int}}(n, 0, Vector{Int}(), Vector{Int}())
@inline SpacedVectorIndex{Ti}(n::Integer = 0) where {Ti} =
    SpacedVectorIndex{Ti,Vector{Ti},Vector{Int}}(n, 0, Vector{Ti}(), Vector{Int}())
@inline SpacedVectorIndex{Ti,Tx}(n::Integer = 0) where {Ti,Tx} =
    SpacedVectorIndex{Ti,Tx{Ti},Tx{Int}}(n, 0, Tx{Ti}(), Tx{Int}())

@inline SpacedVector(n::Integer = 0) =
    SpacedVector{Float64,Int,Vector{Int},Vector{Vector{Float64}}}(n, 0, Vector{Int}(), Vector{Vector{Float64}}())
@inline SpacedVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
    SpacedVector{Tv,Ti,Vector{Ti},Vector{Vector{Tv}}}(n, 0, Vector{Ti}(), Vector{Vector{Tv}}())
@inline SpacedVector{Tv,Ti,Tx,Ts}(n::Integer = 0) where {Tv,Ti,Tx,Ts} =
    SpacedVector{Tv,Ti,Tx{Ti},Tx{Ts{Tv}}}(n, 0, Tx{Ti}(), Tx{Ts{Tv}}())

@inline DensedSparseVector(n::Integer = 0) =
    DensedSparseVector{Float64,Int,Vector{Float64},SortedDict{Int,Vector{Float64}}}(n, 0, typemin(Int), SortedDict{Int,Vector{Float64}}())
@inline DensedSparseVector{Tv,Ti}(n::Integer = 0) where {Tv,Ti} =
    DensedSparseVector{Tv,Ti,Vector{Float64},SortedDict{Int,Vector{Float64}}}(n, 0, typemin(Ti), SortedDict{Ti,Vector{Tv}}())
@inline DensedSparseVector{Tv,Ti,Td,Tc}(n::Integer = 0) where {Tv,Ti,Td,Tc} =
    DensedSparseVector{Tv,Ti,Td{Tv},SortedDict{Ti,Td{Tv}}}(n, 0, typemin(Ti), Tc{Ti,Td{Tv}}())


function SpacedVectorIndex(sv::AbstractSpacedVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc}
    nzind = Vector{Ti}(undef, length(sv.nzind))
    data = Vector{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(zip(sv.nzind, sv.data))
        nzind[i] = k
        data[i] = length(d)
    end
    return SpacedVectorIndex{Ti,Vector{Ti},Vector{Int}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVectorIndex{Tdata}(sv::AbstractSpacedVector{Tv,Ti,Td,Tc}) where {Tdata<:AbstractVector,Tv,Ti,Td,Tc}
    nzind = Tdata{Ti}(undef, length(sv.data))
    data = Tdata{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(zip(sv.nzind, sv.data))
        nzind[i] = k
        data[i] = length(d)
    end
    return SpacedVectorIndex{Ti,Tdata{Ti},Tdata{Int}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVectorIndex(dsv::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc}
    nzind = Vector{Ti}(undef, length(dsv.data))
    data = Vector{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(dsv.data)
        nzind[i] = k
        data[i] = length(d)
    end
    return SpacedVectorIndex{Ti,Vector{Ti},Vector{Int}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVectorIndex{Tdata}(dsv::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tdata<:AbstractVector,Tv,Ti,Td,Tc}
    nzind = Tdata{Ti}(undef, length(dsv.data))
    data = Tdata{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(dsv.data)
        nzind[i] = k
        data[i] = length(d)
    end
    return SpacedVectorIndex{Ti,Tdata{Ti},Tdata{Int}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVector(dsv::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tv,Ti,Td,Tc}
    nzind = Vector{Ti}(undef, length(dsv.data))
    data = Vector{Td}(undef, length(nzind))
    for (i, (k,d)) in enumerate(dsv.data)
        nzind[i] = k
        data[i] = d
    end
    return SpacedVector{Tv,Ti,Vector{Ti},Vector{Td}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVector{Tdata}(dsv::AbstractDensedSparseVector{Tv,Ti,Td,Tc}) where {Tdata,Tv,Ti,Td,Tc}
    nzind = Vector{Ti}(undef, length(dsv.data))
    data = Vector{Td}(undef, length(nzind))
    for (i, (k,d)) in enumerate(dsv.data)
        nzind[i] = k
        data[i] = d
    end
    return SpacedVector{Tv,Ti,Tdata{Ti},Tdata{Td}}(dsv.n, dsv.nnz, nzind, data)
end

function SpacedVector{Tv,Ti,Tx,Ts}(dsv::AbstractDensedSparseVector) where {Tv,Ti,Tx,Ts}
    nzind = Tx{Ti}(undef, length(dsv.data))
    data = Tx{Ts{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(dsv.data)
        nzind[i] = k
        data[i] = Ts{Tv}(d)
    end
    return SpacedVector{Tv,Ti,Tx{Ti},Tx{Ts{Tv}}}(dsv.n, dsv.nnz, nzind, data)
end

