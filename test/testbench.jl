
#using BenchmarkTools
using DensedSparseVectors
import DensedSparseVectors as DSV
using Random

#
#  Testing functions
#

function testfun_create(T::Type, n = 1_000_000, density = 0.9)
    V = T(n)
    Random.seed!(1234)
    randseq = randsubseq(1:n, density)
    for i in shuffle(randseq)
        V[i] = rand()
    end

    sdf = symdiff(SparseArrays.nonzeroinds(V), randseq)
    length(sdf) > 0 && @show sdf

    V
end
function testfun_createSV(T::Type, n = 1_000_000, m = 5, density = 0.9)
    V = T(m,n)
    Random.seed!(1234)
    for i in shuffle(randsubseq(1:n, density))
        for j = 1:m
            V[i,j] = rand()
        end
    end
    V
end
function testfun_createVL(T::Type, n = 1_000_000, density = 0.9)
    V = T(n)
    Random.seed!(1234)
    for i in shuffle(randsubseq(1:n, density))
        V[i] = rand(rand(0:7))
    end
    V
end

function testfun_create_cons(T::Type, n = 1_000_000, density = 0.9)
    V = T(n)
    Random.seed!(1234)
    for i in randsubseq(1:n, density)
        V[i] = rand()
    end
    V
end
function testfun_createSV_cons(T::Type, n = 1_000_000, m = 5, density = 0.9)
    V = T(m,n)
    Random.seed!(1234)
    for i in randsubseq(1:n, density)
        for j = 1:m
            V[i,j] = rand()
        end
    end
    V
end
function testfun_createVL_cons(T::Type, n = 1_000_000, density = 0.9)
    V = T(n)
    Random.seed!(1234)
    for i in randsubseq(1:n, density)
        V[i] = rand(rand(0:7))
    end
    V
end

function testfun_create_dense(T::Type, n = 1_000_000, nchunks = 800, density = 0.95)
    V = T(n)
    chunklen = max(1, floor(Int, n / nchunks))
    Random.seed!(1234)
    for i = 0:nchunks-1
        len = floor(Int, chunklen*density + randn() * chunklen * min(0.1, (1.0-density), density))
        len = max(1, min(chunklen-2, len))
        for j = 1:len
            V[i*chunklen + j] = rand()
        end
    end
    V
end


function testfun_dropstored!(V)
    Random.seed!(1234)
    indices = shuffle(nonzeroinds(V))
    for i in indices
         SparseArrays.dropstored!(V, i)
    end
    V
end


function testfun_getindex(sv)
    S = 0.0
    for i in eachindex(sv)
        S += sv[i]
    end
    (0, S)
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

function testfun_setindex!(sv)
    for i in nzindices(sv)
        sv[i] = 0.0
    end
end


function testfun_nzchunks(sv)
    I = 0
    S = 0.0
    for (startindex,chunk) in nzchunkspairs(sv)
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

function testfun_nzindices(sv)
    I = 0
    for k in nzindices(sv)
        I += k
    end
    (I, 0.0)
end

function testfun_nzvalues(sv)
    S = 0.0
    for v in nzvalues(sv)
        S += v
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


