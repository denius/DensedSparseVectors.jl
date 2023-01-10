using DensedSparseVectors
using SparseArrays
using Test

# https://github.com/JuliaLang/julia/issues/39952
basetype(::Type{T}) where T = Base.typename(T).wrapper


const list_of_Ti_to_test = (Int32, Int64)
const list_of_Tv_to_test = (Int, Float64)
const list_of_containers_types_to_test = (DensedSparseVector, DynamicDensedSparseVector)
#const list_of_containers_types_to_test = (DensedSparseVector, DensedSVSparseVector, DensedVLSparseVector, DynamicDensedSparseVector)


@testset "Creating" begin
    for Ti in list_of_Ti_to_test
        for Tv in list_of_Tv_to_test
            for TypeDSV in list_of_containers_types_to_test
                @eval begin

                    @inferred $TypeDSV{$Tv,$Ti}()
                    @test length($TypeDSV{$Tv,$Ti}()) == 0
                    @test nnz($TypeDSV{$Tv,$Ti}()) == 0

                    @inferred $TypeDSV{$Tv,$Ti}(1)
                    @test length($TypeDSV{$Tv,$Ti}(1)) == 1
                    @test nnz($TypeDSV{$Tv,$Ti}(1)) == 0

                    sv = SparseVector{$Tv,$Ti}(10, $Ti.([0,1,2,5,6]), $Tv.([2,4,6,8,10]))
                    @inferred $TypeDSV{$Tv,$Ti}(sv)
                    @test length($TypeDSV{$Tv,$Ti}(sv)) == 10
                    @test nnz($TypeDSV{$Tv,$Ti}(sv)) == 5

                end
            end
        end
    end
end


#using Aqua
#Aqua.test_all(DensedSparseVectors)

