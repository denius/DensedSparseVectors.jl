





function SpacedIndex(v::AbstractAlmostSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(v))
    data = Vector{Int}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = length_of_that_nzchunk(v, d)
    end
    return SpacedIndex{Ti}(v.n, nzind, data)
end


function SpacedVector(v::AbstractDensedSparseVector{Tv,Ti}) where {Tv,Ti}
    nzind = Vector{Ti}(undef, nnzchunks(v))
    data = Vector{Vector{Tv}}(undef, length(nzind))
    for (i, (k,d)) in enumerate(nzchunkpairs(v))
        nzind[i] = k
        data[i] = Vector{Tv}(d)
    end
    return SpacedVector{Tv,Ti}(v.n, nzind, data)
end

