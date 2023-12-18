
#using BenchmarkTools
using DensedSparseVectors
import DensedSparseVectors as DSV
using Random

#
#  Testing functions
#

function testfun_create(T::Type, n = 1_000_000, density = 0.9)
    sv = T(n)
    Random.seed!(1234)
    randseq = randsubseq(1:n, density)
    Random.seed!(1234)
    ss = shuffle(randseq)
    Random.seed!(1234)
    for i in ss
        sv[i] = rand()
    end

    sdf = symdiff(SparseArrays.nonzeroinds(sv), randseq)
    length(sdf) > 0 && @show sdf

    sv
end
function testfun_createSV(T::Type, n = 1_000_000, m = 5, density = 0.9)
    sv = T(m,n)
    Random.seed!(1234)
    randseq = randsubseq(1:n, density)
    Random.seed!(1234)
    ss = shuffle(randseq)
    Random.seed!(1234)
    for i in ss
        for j = 1:m
            sv[i,j] = rand()
        end
    end
    sv
end
function testfun_createVL(T::Type, n = 1_000_000, density = 0.9)
    sv = T(n)
    Random.seed!(1234)
    randseq = randsubseq(1:n, density)
    Random.seed!(1234)
    ss = shuffle(randseq)
    Random.seed!(1234)
    for i in ss
        sv[i] = rand(rand(0:7))
    end
    sv
end

function testfun_create_cons(T::Type, n = 1_000_000, density = 0.9; m = 3)
    T <: DensedVLSparseVector && return testfun_createVL_cons(T, n, density)
    T <: DensedSVSparseVector && return testfun_createSV_cons(T, n, m, density)
    sv = T(n)
    Random.seed!(1234)
    ss = randsubseq(1:n, density)
    Random.seed!(1234)
    for i in ss
        sv[i] = rand()
    end
    sv
end
function testfun_createSV_cons(T::Type, n = 1_000_000, m = 3, density = 0.9)
    sv = T(m,n)
    Random.seed!(1234)
    ss = randsubseq(1:n, density)
    Random.seed!(1234)
    for i in ss
        for j = 1:m
            sv[i,j] = rand()
        end
    end
    sv
end
function testfun_createVL_cons(T::Type, n = 1_000_000, density = 0.9)
    sv = T(n)
    Random.seed!(1234)
    ss = randsubseq(1:n, density)
    Random.seed!(1234)
    for i in ss
        sv[i] = rand(rand(0:7))
    end
    sv
end

function testfun_create_dense(T::Type, n = 1_000_000, nchunks = 800, density = 0.95)
    sv = T(n)
    chunklen = max(1, floor(Int, n / nchunks))
    Random.seed!(1234)
    for i = 0:nchunks-1
        len = floor(Int, chunklen*density + randn() * chunklen * min(0.1, (1.0-density), density))
        len = max(1, min(chunklen-2, len))
        for j = 1:len
            sv[i*chunklen + j] = rand()
        end
    end
    sv
end


function testfun_dropstored!(sv)
    Random.seed!(1234)
    indices = shuffle(SparseArrays.nonzeroinds(sv))
    for i in indices
         SparseArrays.dropstored!(sv, i)
    end
    sv
end


function testfun_getindex_cons(sv)
    S = 0.0
    for i in eachindex(sv)
        S += sv[i]
    end
    (0, S)
end
function testfun_getindex(sv)
    S = 0.0
    Random.seed!(1234)
    indices = shuffle(eachindex(sv))
    for i in indices
        S += sv[i]
    end
    (0, S)
end
function testfun_getindex_cons(sv::DensedSparseVectors.AbstractDensedBlockSparseVector)
    S = 0.0
    for i in eachindex(sv)
        S += sum(sv[i])
    end
    S2 = 0.0
    for i in eachindex(sv)
        for j in eachindex(sv[i])
            S2 += sv[i][j]
        end
    end
    (0, S, S2)
end
function testfun_getindex(sv::DensedSparseVectors.AbstractDensedBlockSparseVector)
    Random.seed!(1234)
    indices = shuffle(eachindex(sv))
    S = 0.0
    for i in indices
        S += sum(sv[i])
    end
    S2 = 0.0
    for i in indices
        for j in shuffle(eachindex(sv[i]))
            S2 += sv[i][j]
        end
    end
    (0, S, S2)
end

function testfun_getindex_outer(sv, indices)
    S = 0.0
    for i in indices
        S += sv[i]
    end
    (0, S)
end

function testfun_nzgetindex(sv)
    S = 0.0
    for i in nzindices(sv)
        S += sv[i]
    end
    (0, S)
end

function testfun_setindex!_cons(sv)
    for i in nzindices(sv)
        sv[i] = 0.0
    end
end
function testfun_setindex!(sv)
    Random.seed!(1234)
    indices = shuffle(SparseArrays.nonzeroinds(sv))
    for i in indices
        sv[i] = 0.0
    end
end


function testfun_nzchunks(sv)
    I = 0
    S = 0.0
    for (ids,chunk) in nzchunkspairs(sv)
        I += length(ids)
        for i in axes(chunk,1)
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

function testfun_nzindices(sv)
    I = 0
    for k in nzindices(sv)
        I += k
    end
    (I, 0.0)
end

function testfun_nzblocks(sv)
    S = 0.0
    # for v in nzvalues(sv)
    for (i,v) in enumerate(nzblocks(sv))
        S += sum(v)
    end
    (0, S)
end

function testfun_nzvalues(sv)
    S = 0.0
    # for v in nzvalues(sv)
    for (i,v) in enumerate(nzvalues(sv))
        S += sum(v)
    end
    (0, S)
end

function testfun_nzvaluesview(sv)
    S = 0.0
    for v in nzvaluesview(sv)
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


