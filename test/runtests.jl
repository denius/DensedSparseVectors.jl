using BenchmarkTools
using DensedSparseVectors
using OffsetArrays
using SparseArrays
using Test

# https://github.com/JuliaLang/julia/issues/39952
basetype(::Type{T}) where T = Base.typename(T).wrapper


const list_of_Ti_to_test = (UInt32, Int64)
const list_of_Tv_to_test = (Int, Float64)
const list_of_containers_types_to_test = (DensedSparseVector, DynamicDensedSparseVector)
#const list_of_containers_types_to_test = (DensedSparseVector, DensedSVSparseVector, DensedVLSparseVector, DynamicDensedSparseVector)


@testset "Broadcast" begin
    for Ti in list_of_Ti_to_test
        for Tv in list_of_Tv_to_test
            for TypeDSV in list_of_containers_types_to_test

                @eval sv = SparseVector{$Tv,$Ti}(10, $Ti[1,2,3,6,7], $Tv[2,4,6,8,10])
                @eval v = Vector{$Tv}(sv)
                v1 = [2]
                @eval dsv1 = $TypeDSV(sv)
                @eval dsv2 = $TypeDSV(sv)
                @eval dsv3 = @inferred $TypeDSV(2.0 .* sv)

                @test dsv1 .+ sv == dsv3
                @test dsv1 .+ dsv2 == dsv3
                @test dsv1 .* [2] == dsv3
                @test [2] .* dsv1 == Int.(dsv3)
                @test 2 .* dsv1 == Int.(dsv3)
                @test dsv2 .* 2 == dsv3
                @test dsv2 .+ 2 == sv .+ 2
                @test (dsv3 .= dsv2 .* 2; dsv3 == sv .* 2)
                @test (dsv3 .= dsv2 .+ 2; dsv3 == sv .+ 2)
                @test (dsv3 .= dsv2 .+ 2 .* dsv1; dsv3 == sv .* 3)
                @test (dsv2 .= 2; dsv2 == fill!(copy(dsv3), 2))
                #@test dsv1 .+ [2] == dsv3 .+ 2


                @test (@ballocated $dsv3 .= $dsv1 .+ $dsv2 samples=100 evals=100) == 0
                @test (@ballocated $dsv3 .= $dsv1 .+ $dsv2 samples=100 evals=100) == 0
                @test (@ballocated $dsv3 .= $sv .+ $dsv2 samples=100 evals=100) == 0
                @test (@ballocated $dsv3 .= $dsv1 .+ $sv samples=100 evals=100) == 0
                @test (@ballocated $dsv3 .= $dsv1 .* $v1 samples=100 evals=100) == 0
                @test (@ballocated $dsv3 .= $v1 .* $dsv2 samples=100 evals=100) == 0

            end
        end
    end
end


@testset "Creating" begin
    for Ti in list_of_Ti_to_test
        for Tv in list_of_Tv_to_test
            for TypeDSV in list_of_containers_types_to_test
                @eval begin

                    dsv = @inferred $TypeDSV{$Tv,$Ti}()
                    @test length(dsv) == 0
                    @test nnz(dsv) == 0
                    @test length(nzchunks(dsv)) == 0

                    dsv = @inferred $TypeDSV{$Tv,$Ti}(1)
                    @test length(dsv) == 1
                    @test nnz(dsv) == 0
                    @test length(nzchunks(dsv)) == 0

                    sv = SparseVector{$Tv,$Ti}(10, $Ti[1,2,3,6,7], $Tv[2,4,6,8,10])
                    dsv = @inferred $TypeDSV(sv)
                    dsv = @inferred $TypeDSV{$Tv,$Ti}(sv)
                    @test length(dsv) == 10
                    @test nnz(dsv) == 5
                    @test length(nzchunks(dsv)) == 2
                    @test dsv == sv

                    #osv = OffsetArray(sv, 2:11)
                    #odsv = OffsetArray(dsv, 2:11)
                    #@test firstindex(odsv) == firstindex(osv) == 2
                    #@test lastindex(odsv) == lastindex(osv) == 11
                    #@test length(odsv) == 10
                    #@test nnz(odsv) == 5
                    #@test length(nzchunks(odsv)) == 2
                    #@test odsv == osv

                    @inferred setindex!(dsv, 14, 9)
                    sv[9] = 14
                    @test length(dsv) == 10
                    @test nnz(dsv) == 6
                    @test length(nzchunks(dsv)) == 3
                    @test dsv == sv

                    @inferred setindex!(dsv, 12, 8)
                    sv[8] = 12
                    @test length(dsv) == 10
                    @test nnz(dsv) == 7
                    @test length(nzchunks(dsv)) == 2
                    @test dsv == sv

                    @inferred setindex!(dsv, 12, 4)
                    sv[4] = 12
                    @inferred setindex!(dsv, 12, 5)
                    sv[5] = 12
                    @test length(dsv) == 10
                    @test nnz(dsv) == 9
                    @test length(nzchunks(dsv)) == 1
                    @test dsv == sv

                    @inferred SparseArrays.dropstored!(dsv, 4)
                    SparseArrays.dropstored!(sv, 4)
                    for i = 1:8
                        @inferred SparseArrays.dropstored!(dsv, i)
                        SparseArrays.dropstored!(sv, i)
                    end
                    @test length(dsv) == 10
                    @test nnz(dsv) == 1
                    @test length(nzchunks(dsv)) == 1
                    @test dsv == sv

                    @inferred SparseArrays.dropstored!(dsv, 4)
                    SparseArrays.dropstored!(sv, 4)
                    @test length(dsv) == 10
                    @test nnz(dsv) == 1
                    @test length(nzchunks(dsv)) == 1
                    @test dsv == sv

                    @inferred SparseArrays.dropstored!(dsv, 9)
                    SparseArrays.dropstored!(sv, 9)
                    @test length(dsv) == 10
                    @test nnz(dsv) == 0
                    @test length(nzchunks(dsv)) == 0
                    @test dsv == sv

                    @inferred setindex!(dsv, 12, 4)
                    sv[4] = 12
                    @inferred setindex!(dsv, 12, 5)
                    sv[5] = 12
                    @test length(dsv) == 10
                    @test nnz(dsv) == 2
                    @test length(nzchunks(dsv)) == 1
                    @test dsv == sv

                    @inferred SparseArrays.dropstored!(dsv, 5)
                    @inferred SparseArrays.dropstored!(dsv, 5)
                    @inferred SparseArrays.dropstored!(dsv, 4)
                    @test length(dsv) == 10
                    @test nnz(dsv) == 0
                    @test length(nzchunks(dsv)) == 0

                    @test isempty(dsv)

                end
            end
        end
    end
end


using Aqua
Aqua.test_all(DensedSparseVectors)

