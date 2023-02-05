
# derived from SparseArrays.jl's higherorderfns.jl

module HigherOrderFns

# This module provides higher order functions specialized for sparse arrays,
# particularly map[!]/broadcast[!] for DensedSparseVectors at present.
import Base: map, map!, broadcast, copy, copyto!

using Base: front, tail, to_shape
using SparseArrays: SparseVector, SparseMatrixCSC, AbstractSparseVector, AbstractSparseMatrixCSC,
                      AbstractSparseMatrix, AbstractSparseArray, indtype, nnz, nzrange, spzeros,
                      SparseVectorUnion, AdjOrTransSparseVectorUnion, nonzeroinds, nonzeros,
                      rowvals, getcolptr, widelength

#using SparseArrays: SparseVector, SparseMatrixCSC,
#                      AbstractSparseVector, AbstractSparseVector, #AbstractBlockDensedSparseVector,
#                      AbstractSparseMatrix, AbstractSparseArray,
#                      SparseVectorUnion, AdjOrTransSparseVectorUnion,
#                      indtype, nnz, nzrange, spzeros,
#                      nonzeroinds, nonzeros, rowvals, getcolptr, widelength,
#                      _iszero, _isnotzero
using Base.Broadcast: BroadcastStyle, Broadcasted, flatten
using LinearAlgebra

using StaticArrays

using ..DensedSparseVectors
using ..DensedSparseVectors: AbstractDensedSparseVector, AbstractVectorDensedSparseVector, AbstractBlockDensedSparseVector, AbstractSDictDensedSparseVector

# Some Unions definitions

#abstract type AbstractBlockDensedSparseVector{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti} end

const SparseVector2 = DensedSparseVector

mutable struct SparseMatrixCSC2{Tv,Ti,m} <: AbstractBlockDensedSparseVector{Tv,Ti}
   "Index of last used chunk"
    lastusedchunkindex::Int
    "Storage for indices of the first element of non-zero chunks"
    nzind::Vector{Ti}  # Vector of chunk's first indices
    "Storage for chunks of non-zero values as `Vector` of `Vector`s
     The resulting matrix size is m by n"
    nzchunks::Vector{Vector{SVector{m,Tv}}}
    "Vector length"
    n::Ti
    "Number of stored non-zero elements"
    nnz::Int
end


const DensedSparseVectorView{Tv,Ti}  = SubArray{Tv,1,<:AbstractDensedSparseVector{Tv,Ti},Tuple{Base.Slice{Base.OneTo{Int}}},false}
const DensedSparseVectorUnion{Tv,Ti} = Union{AbstractDensedSparseVector{Tv,Ti}, DensedSparseVectorView{Tv,Ti}}
const AdjOrTransDensedSparseVectorUnion{Tv,Ti} = LinearAlgebra.AdjOrTrans{Tv, <:DensedSparseVectorUnion{Tv,Ti}}

###const SparseColumnView{Tv,Ti}  = SubArray{Tv,1,<:AbstractSparseMatrixCSC{Tv,Ti},Tuple{Base.Slice{Base.OneTo{Int}},Int},false}
###const SparseVectorView{Tv,Ti}  = SubArray{Tv,1,<:AbstractSparseVector{Tv,Ti},Tuple{Base.Slice{Base.OneTo{Int}}},false}
###const SparseVectorUnion{Tv,Ti} = Union{AbstractCompressedVector{Tv,Ti}, SparseColumnView{Tv,Ti}, SparseVectorView{Tv,Ti}}
###const AdjOrTransSparseVectorUnion{Tv,Ti} = LinearAlgebra.AdjOrTrans{Tv, <:SparseVectorUnion{Tv,Ti}}
###const SVorFSV{Tv,Ti} = Union{SparseVector{Tv,Ti},FixedSparseVector{Tv,Ti}}



# This module is organized as follows:
# (0) Define BroadcastStyle rules and convenience types for dispatch
# (1) Define a common interface to SparseVectors and SparseMatrixCSCs sufficient for
#       map[!]/broadcast[!]'s purposes. The methods below are written against this interface.
# (2) Define entry points for map[!] (short children of _map_[not]zeropres!).
# (3) Define entry points for broadcast[!] (short children of _broadcast_[not]zeropres!).
# (4) Define _map_[not]zeropres! specialized for a single (input) sparse vector/matrix.
# (5) Define _map_[not]zeropres! specialized for a pair of (input) sparse vectors/matrices.
# (6) Define general _map_[not]zeropres! capable of handling >2 (input) sparse vectors/matrices.
# (7) Define _broadcast_[not]zeropres! specialized for a single (input) sparse vector/matrix.
# (8) Define _broadcast_[not]zeropres! specialized for a pair of (input) sparse vectors/matrices.
# (9) Define general _broadcast_[not]zeropres! capable of handling >2 (input) sparse vectors/matrices.
# (10) Define broadcast methods handling combinations of broadcast scalars and sparse vectors/matrices.
# (11) Define broadcast[!] methods handling combinations of scalars, sparse vectors/matrices,
#       structured matrices, and one- and two-dimensional Arrays.
# (12) Define map[!] methods handling combinations of sparse and structured matrices.


# (0) BroadcastStyle rules and convenience types for dispatch

###AbstractDensedSparseVector = Union{AbstractDensedSparseVector,AbstractBlockDensedSparseVector}
#DensedSparseVecOrMat = Union{AbstractVectorDensedSparseVector,AbstractSDictDensedSparseVector,AbstractBlockDensedSparseVector}
DensedSparseVecOrMat = Union{AbstractVectorDensedSparseVector,AbstractBlockDensedSparseVector}
SparseVecOrMat2 = Union{SparseVector2,SparseMatrixCSC2}

# broadcast container type promotion for combinations of sparse arrays and other types
struct DnsSparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
struct DnsSparseMatStyle <: Broadcast.AbstractArrayStyle{2} end
###Broadcast.BroadcastStyle(::Type{<:SparseVector2}) = DnsSparseVecStyle()
Broadcast.BroadcastStyle(::Type{<:AbstractVectorDensedSparseVector}) = DnsSparseVecStyle()
Broadcast.BroadcastStyle(::Type{<:AbstractBlockDensedSparseVector}) = DnsSparseMatStyle()
const DSPVM = Union{DnsSparseVecStyle,DnsSparseMatStyle}

# DnsSparseVecStyle handles 0-1 dimensions, DnsSparseMatStyle 0-2 dimensions.
# DnsSparseVecStyle promotes to DnsSparseMatStyle for 2 dimensions.
# Fall back to DefaultArrayStyle for higher dimensionality.
DnsSparseVecStyle(::Val{0}) = DnsSparseVecStyle()
DnsSparseVecStyle(::Val{1}) = DnsSparseVecStyle()
DnsSparseVecStyle(::Val{2}) = DnsSparseMatStyle()
DnsSparseVecStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()
DnsSparseMatStyle(::Val{0}) = DnsSparseMatStyle()
DnsSparseMatStyle(::Val{1}) = DnsSparseMatStyle()
DnsSparseMatStyle(::Val{2}) = DnsSparseMatStyle()
DnsSparseMatStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()

Broadcast.BroadcastStyle(::DnsSparseMatStyle, ::DnsSparseVecStyle) = DnsSparseMatStyle()

# Tuples promote to dense
# TODO: FIXME: Dense * View(DSV) should be DnsSparseVecStyle. Isn't it?
Broadcast.BroadcastStyle(::DnsSparseVecStyle, ::Broadcast.Style{Tuple}) = Broadcast.DefaultArrayStyle{1}()
Broadcast.BroadcastStyle(::DnsSparseMatStyle, ::Broadcast.Style{Tuple}) = Broadcast.DefaultArrayStyle{2}()

struct PromoteToSparse <: Broadcast.AbstractArrayStyle{2} end
PromoteToSparse(::Val{0}) = PromoteToSparse()
PromoteToSparse(::Val{1}) = PromoteToSparse()
PromoteToSparse(::Val{2}) = PromoteToSparse()
PromoteToSparse(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()

const StructuredMatrix = Union{Diagonal,Bidiagonal,Tridiagonal,SymTridiagonal}
Broadcast.BroadcastStyle(::Type{<:Adjoint{T,<:Union{SparseVector2,SparseMatrixCSC}} where T}) = PromoteToSparse()
Broadcast.BroadcastStyle(::Type{<:Transpose{T,<:Union{SparseVector2,SparseMatrixCSC}} where T}) = PromoteToSparse()

Broadcast.BroadcastStyle(s::DSPVM, ::Broadcast.AbstractArrayStyle{0}) = s
Broadcast.BroadcastStyle(s::DSPVM, ::Broadcast.DefaultArrayStyle{0}) = s
Broadcast.BroadcastStyle(::DSPVM, ::Broadcast.DefaultArrayStyle{1}) = PromoteToSparse()
Broadcast.BroadcastStyle(::DSPVM, ::Broadcast.DefaultArrayStyle{2}) = PromoteToSparse()

Broadcast.BroadcastStyle(::DSPVM, ::LinearAlgebra.StructuredMatrixStyle{<:StructuredMatrix}) = PromoteToSparse()
Broadcast.BroadcastStyle(::PromoteToSparse, ::LinearAlgebra.StructuredMatrixStyle{<:StructuredMatrix}) = PromoteToSparse()

Broadcast.BroadcastStyle(::PromoteToSparse, ::DSPVM) = PromoteToSparse()
Broadcast.BroadcastStyle(::PromoteToSparse, ::Broadcast.Style{Tuple}) = Broadcast.DefaultArrayStyle{2}()

# FIXME: currently sparse broadcasts are only well-tested on known array types, while any AbstractArray
# could report itself as a DefaultArrayStyle().
# See https://github.com/JuliaLang/julia/pull/23939#pullrequestreview-72075382 for more details
is_supported_sparse_broadcast() = true
is_supported_sparse_broadcast(::AbstractArray, rest...) = false
is_supported_sparse_broadcast(::AbstractSparseArray, rest...) = is_supported_sparse_broadcast(rest...)
is_supported_sparse_broadcast(::StructuredMatrix, rest...) = is_supported_sparse_broadcast(rest...)
is_supported_sparse_broadcast(::Array, rest...) = is_supported_sparse_broadcast(rest...)
is_supported_sparse_broadcast(t::Union{Transpose, Adjoint}, rest...) = is_supported_sparse_broadcast(t.parent, rest...)
is_supported_sparse_broadcast(x, rest...) = axes(x) === () && is_supported_sparse_broadcast(rest...)
is_supported_sparse_broadcast(x::Ref, rest...) = is_supported_sparse_broadcast(rest...)

can_skip_sparsification(f, rest...) = false
can_skip_sparsification(::typeof(*), ::DensedSparseVectorUnion, ::AdjOrTransDensedSparseVectorUnion) = true

# Dispatch on broadcast operations by number of arguments
const Broadcasted0{Style<:Union{Nothing,BroadcastStyle},Axes,F} =
    Broadcasted{Style,Axes,F,Tuple{}}
const SpBroadcasted1{Style<:DSPVM,Axes,F,Args<:Tuple{DensedSparseVecOrMat}} =
    Broadcasted{Style,Axes,F,Args}
const SpBroadcasted2{Style<:DSPVM,Axes,F,Args<:Tuple{DensedSparseVecOrMat,DensedSparseVecOrMat}} =
    Broadcasted{Style,Axes,F,Args}

# (1) The definitions below provide a common interface to sparse vectors and matrices
# sufficient for the purposes of map[!]/broadcast[!]. This interface treats sparse vectors
# as n-by-one sparse matrices which, though technically incorrect, is how broacast[!] views
# sparse vectors in practice.
@inline numrows(A::AbstractVectorDensedSparseVector) = length(A)
@inline numrows(A::AbstractBlockDensedSparseVector) = size(A, 1)
@inline numcols(A::AbstractVectorDensedSparseVector) = 1
@inline numcols(A::AbstractBlockDensedSparseVector) = size(A, 2)
# numrows and numcols respectively yield size(A, 1) and size(A, 2), but avoid a branch
@inline columns(A::AbstractVectorDensedSparseVector) = 1
@inline columns(A::AbstractBlockDensedSparseVector) = 1:size(A, 2)
@inline colrange(A::AbstractVectorDensedSparseVector, j) = 1:length(nonzeroinds(A))
@inline colrange(A::AbstractBlockDensedSparseVector, j) = nzrange(A, j)
@inline colstartind(A::AbstractVectorDensedSparseVector, j) = one(indtype(A))
@inline colboundind(A::AbstractVectorDensedSparseVector, j) = convert(indtype(A), length(nonzeroinds(A)) + 1)
@inline colstartind(A::AbstractBlockDensedSparseVector, j) = getcolptr(A)[j]
@inline colboundind(A::AbstractBlockDensedSparseVector, j) = getcolptr(A)[j + 1]
@inline storedinds(A::AbstractVectorDensedSparseVector) = nonzeroinds(A)
@inline storedinds(A::AbstractBlockDensedSparseVector) = rowvals(A)
@inline storedvals(A::DensedSparseVecOrMat) = nonzeros(A)
@inline setcolptr!(A::AbstractVectorDensedSparseVector, j, val) = val
@inline setcolptr!(A::AbstractBlockDensedSparseVector, j, val) = getcolptr(A)[j] = val
function trimstorage!(A::DensedSparseVecOrMat, maxstored)
    resize!(storedinds(A), maxstored)
    resize!(storedvals(A), maxstored)
    return maxstored
end

function expandstorage!(A::DensedSparseVecOrMat, maxstored)
    if length(storedinds(A)) < maxstored
        resize!(storedinds(A), maxstored)
        resize!(storedvals(A), maxstored)
    end
    return maxstored
end

_checkbuffers(S::SparseMatrixCSC2) = (@assert length(getcolptr(S)) == size(S, 2) + 1 && getcolptr(S)[end] - 1 == length(rowvals(S)) == length(nonzeros(S)); S)
_checkbuffers(S::SparseVector2) = (@assert length(storedvals(S)) == length(storedinds(S)); S)

# (2) map[!] entry points
map(f::Tf, A::AbstractVectorDensedSparseVector) where {Tf} = _noshapecheck_map(f, A)
map(f::Tf, A::AbstractBlockDensedSparseVector) where {Tf} = _noshapecheck_map(f, A)
map(f::Tf, A::AbstractBlockDensedSparseVector, Bs::Vararg{SparseMatrixCSC2,N}) where {Tf,N} =
    (_checksameshape(A, Bs...); _noshapecheck_map(f, A, Bs...))
map(f::Tf, A::DensedSparseVecOrMat, Bs::Vararg{DensedSparseVecOrMat,N}) where {Tf,N} =
    (_checksameshape(A, Bs...); _noshapecheck_map(f, A, Bs...))
map!(f::Tf, C::AbstractBlockDensedSparseVector, A::AbstractBlockDensedSparseVector, Bs::Vararg{SparseMatrixCSC2,N}) where {Tf,N} =
    (_checksameshape(C, A, Bs...); _noshapecheck_map!(f, C, A, Bs...))
map!(f::Tf, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat, Bs::Vararg{DensedSparseVecOrMat,N}) where {Tf,N} =
    (_checksameshape(C, A, Bs...); _noshapecheck_map!(f, C, A, Bs...))

_noshapecheck_map!(f::Tf, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat, Bs::Vararg{DensedSparseVecOrMat,N}) where {Tf,N} =
    # Avoid calculating f(zero) unless necessary as it may fail.
    if _haszeros(A) && all(_haszeros, Bs)
        fofzeros = f(_zeros_eltypes(A, Bs...)...)
        if _iszero(fofzeros)
            _map_zeropres!(f, C, A, Bs...)
        else
            _map_notzeropres!(f, fofzeros, C, A, Bs...)
        end
    else
        _map_zeropres!(f, C, A, Bs...)
    end


function _noshapecheck_map(f::Tf, A::DensedSparseVecOrMat, Bs::Vararg{DensedSparseVecOrMat,N}) where {Tf,N}
    # Avoid calculating f(zero) unless necessary as it may fail.
    entrytypeC = Base.Broadcast.combine_eltypes(f, (A, Bs...))
    indextypeC = _promote_indtype(A, Bs...)
    if _haszeros(A) && all(_haszeros, Bs)
        fofzeros = f(_zeros_eltypes(A, Bs...)...)
        fpreszeros = _iszero(fofzeros)
        maxnnzC = Int(fpreszeros ? min(widelength(A), _sumnnzs(A, Bs...)) : widelength(A))
        C = _allocres(size(A), indextypeC, entrytypeC, maxnnzC)
        return fpreszeros ? _map_zeropres!(f, C, A, Bs...) :
                        _map_notzeropres!(f, fofzeros, C, A, Bs...)
    else
        maxnnzC = Int(widelength(A))
        C = _allocres(size(A), indextypeC, entrytypeC, maxnnzC)
        return _map_zeropres!(f, C, A, Bs...)
    end
end

# (3) broadcast[!] entry points
copy(bc::SpBroadcasted1) = _noshapecheck_map(bc.f, bc.args[1])

@inline function copyto!(C::DensedSparseVecOrMat, bc::Broadcasted0{Nothing})
    isempty(C) && return _finishempty!(C)
    f = bc.f
    fofnoargs = f()
    if _iszero(fofnoargs) # f() is zero, so empty C
        trimstorage!(C, 0)
        _finishempty!(C)
    else # f() is nonzero, so densify C and fill with independent calls to f()
        _densestructure!(C)
        storedvals(C)[1] = fofnoargs
        broadcast!(f, view(storedvals(C), 2:length(storedvals(C))))
    end
    return _checkbuffers(C)
end


function _diffshape_broadcast(f::Tf, A::DensedSparseVecOrMat, Bs::Vararg{DensedSparseVecOrMat,N}) where {Tf,N}
    fofzeros = f(_zeros_eltypes(A, Bs...)...)
    fpreszeros = _iszero(fofzeros)
    indextypeC = _promote_indtype(A, Bs...)
    entrytypeC = Base.Broadcast.combine_eltypes(f, (A, Bs...))
    axesC = Base.Broadcast.combine_axes(A, Bs...)
    shapeC = to_shape(axesC)
    ###maxnnzC = fpreszeros ? _checked_maxnnzbcres(shapeC, A, Bs...) : _densennz(shapeC)
    ###C = _allocres(shapeC, indextypeC, entrytypeC, maxnnzC)
    if fpreszeros
        C = similar(_promote_dest_arg((A, Bs...)), entrytypeC, indextypeC)
    else
        C = basetype(typeof(_promote_dest_arg((A, Bs...)))){entrytypeC, indextypeC}(axesC)
    end
    return fpreszeros ? _broadcast_zeropres!(f, C, A, Bs...) :
                        _broadcast_notzeropres!(f, fofzeros, C, A, Bs...)
end
# helper functions for map[!]/broadcast[!] entry points (and related methods below)
@inline _haszeros(A) = nnz(A) ≠ length(A)
@inline _sumnnzs(A) = nnz(A)
@inline _sumnnzs(A, Bs...) = nnz(A) + _sumnnzs(Bs...)
@inline _iszero(x) = x == 0
@inline _iszero(x::Number) = Base.iszero(x)
@inline _iszero(x::AbstractArray) = Base.iszero(x)
@inline _isnotzero(x) = iszero(x) !== true # like `!iszero(x)`, but handles `x::Missing`
@inline _isnotzero(x::Number) = !iszero(x)
@inline _isnotzero(x::AbstractArray) = !iszero(x)
@inline _zeros_eltypes(A) = (zero(eltype(A)),)
@inline _zeros_eltypes(A, Bs...) = (zero(eltype(A)), _zeros_eltypes(Bs...)...)
@inline _promote_indtype(A) = indtype(A)
@inline _promote_indtype(A, Bs...) = promote_type(indtype(A), _promote_indtype(Bs...))
@inline _aresameshape(A) = true
@inline _aresameshape(A, B) = size(A) == size(B)
@inline _aresameshape(A, B, Cs...) = _aresameshape(A, B) ? _aresameshape(B, Cs...) : false
@inline _checksameshape(As...) = _aresameshape(As...) || throw(DimensionMismatch("argument shapes must match"))
@inline _all_args_isa(t::Tuple{Any}, ::Type{T}) where T = isa(t[1], T)
@inline _all_args_isa(t::Tuple{Any,Vararg{Any}}, ::Type{T}) where T = isa(t[1], T) & _all_args_isa(tail(t), T)
@inline _all_args_isa(t::Tuple{Broadcasted}, ::Type{T}) where T = _all_args_isa(t[1].args, T)
@inline _all_args_isa(t::Tuple{Broadcasted,Vararg{Any}}, ::Type{T}) where T = _all_args_isa(t[1].args, T) & _all_args_isa(tail(t), T)
@inline _densennz(shape::NTuple{1}) = shape[1]
@inline _densennz(shape::NTuple{2}) = shape[1] * shape[2]
_maxnnzfrom(shape::NTuple{1}, A::SparseVector2) = nnz(A) * div(shape[1], length(A))
_maxnnzfrom(shape::NTuple{2}, A::SparseVector2) = nnz(A) * div(shape[1], length(A)) * shape[2]
_maxnnzfrom(shape::NTuple{2}, A::AbstractBlockDensedSparseVector) = nnz(A) * div(shape[1], size(A, 1)) * div(shape[2], size(A, 2))
@inline _maxnnzfrom_each(shape, ::Tuple{}) = ()
@inline _maxnnzfrom_each(shape, As) = (_maxnnzfrom(shape, first(As)), _maxnnzfrom_each(shape, tail(As))...)
@inline _unchecked_maxnnzbcres(shape, As::Tuple) = min(_densennz(shape), sum(_maxnnzfrom_each(shape, As)))
@inline _unchecked_maxnnzbcres(shape, As...) = _unchecked_maxnnzbcres(shape, As)
@inline _checked_maxnnzbcres(shape::NTuple{1}, As...) = shape[1] != 0 ? _unchecked_maxnnzbcres(shape, As) : 0
@inline _checked_maxnnzbcres(shape::NTuple{2}, As...) = shape[1] != 0 && shape[2] != 0 ? _unchecked_maxnnzbcres(shape, As) : 0
@inline function _allocres(shape::Union{NTuple{1},NTuple{2}}, indextype, entrytype, maxnnz)
    X = spzeros(entrytype, indextype, shape)
    resize!(storedinds(X), maxnnz)
    resize!(storedvals(X), maxnnz)
    return X
end
# https://github.com/JuliaLang/julia/issues/39952
basetype(::Type{T}) where T = Base.typename(T).wrapper
@inline _promote_dest_arg(As::Tuple{Any}) = first(As)
@inline _promote_dest_arg(As::Tuple{Any,Any}) = first(As)
@inline _promote_dest_arg(As::Tuple{AbstractDensedSparseVector,AbstractDensedSparseVector}) = first(As)
@inline _promote_dest_arg(As::Tuple{AbstractDensedSparseVector,Any}) = first(As)
@inline _promote_dest_arg(As::Tuple{Any,AbstractDensedSparseVector}) = last(As)
@inline _promote_dest_arg(As::Tuple{Any,Vararg{Any}}) = _promote_dest_arg((_promote_dest_arg((first(As), first(last(As)))), last(last(As))...))

# (4) _map_zeropres!/_map_notzeropres! specialized for a single sparse vector/matrix
"Stores only the nonzero entries of `map(f, Array(A))` in `C`."
function _map_zeropres!(f::Tf, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat) where Tf
    spaceC::Int = length(nonzeros(C))
    Ck = 1
    @inbounds for j in columns(C)
        setcolptr!(C, j, Ck)
        for Ak in colrange(A, j)
            Cx = f(storedvals(A)[Ak])
            if _isnotzero(Cx)
                Ck > spaceC && (spaceC = expandstorage!(C, Ck + nnz(A) - (Ak - 1)))
                storedinds(C)[Ck] = storedinds(A)[Ak]
                storedvals(C)[Ck] = Cx
                Ck += 1
            end
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return _checkbuffers(C)
end

"""
Densifies `C`, storing `fillvalue` in place of each unstored entry in `A` and
`f(A[i])`/`f(A[i,j])` in place of each stored entry `A[i]`/`A[i,j]` in `A`.
"""
function _map_notzeropres!(f::Tf, fillvalue, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat) where Tf
    # Build dense matrix structure in C, expanding storage if necessary
    _densestructure!(C)
    # Populate values
    fill!(storedvals(C), fillvalue)
    @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
        for Ak in colrange(A, j)
            Cx = f(storedvals(A)[Ak])
            Cx != fillvalue && (storedvals(C)[jo + storedinds(A)[Ak]] = Cx)
        end
    end
    # NOTE: Combining the fill! above into the loop above to avoid multiple sweeps over /
    # nonsequential access of storedvals(C) does not appear to improve performance.
    return _checkbuffers(C)
end
# helper functions for these methods and some of those below
@inline _densecoloffsets(A::AbstractVectorDensedSparseVector) = 0
@inline _densecoloffsets(A::AbstractBlockDensedSparseVector) = 0:size(A, 1):(size(A, 1)*(size(A, 2) - 1))
function _densestructure!(A::AbstractVectorDensedSparseVector)
    expandstorage!(A, length(A))
    copyto!(nonzeroinds(A), 1:length(A))
    return A
end
function _densestructure!(A::AbstractBlockDensedSparseVector)
    nnzA = size(A, 1) * size(A, 2)
    expandstorage!(A, nnzA)
    copyto!(getcolptr(A), 1:size(A, 1):(nnzA + 1))
    for k in _densecoloffsets(A)
        copyto!(rowvals(A), k + 1, 1:size(A, 1))
    end
    return A
end


# (5) _map_zeropres!/_map_notzeropres! specialized for a pair of sparse vectors/matrices
function _map_zeropres!(f::Tf, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat, B::DensedSparseVecOrMat) where Tf
    spaceC::Int = length(nonzeros(C))
    rowsentinelA = convert(indtype(A), numrows(C) + 1)
    rowsentinelB = convert(indtype(B), numrows(C) + 1)
    Ck = 1
    @inbounds for j in columns(C)
        setcolptr!(C, j, Ck)
        Ak, stopAk = colstartind(A, j), colboundind(A, j)
        Bk, stopBk = colstartind(B, j), colboundind(B, j)
        Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
        Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
        while true
            if Ai == Bi
                Ai == rowsentinelA && break # column complete
                Cx, Ci::indtype(C) = f(storedvals(A)[Ak], storedvals(B)[Bk]), Ai
                Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
            elseif Ai < Bi
                Cx, Ci = f(storedvals(A)[Ak], zero(eltype(B))), Ai
                Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
            else # Bi < Ai
                Cx, Ci = f(zero(eltype(A)), storedvals(B)[Bk]), Bi
                Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
            end
            # NOTE: The ordering of the conditional chain above impacts which matrices this
            # method performs best for. In the map situation (arguments have same shape, and
            # likely same or similar stored entry pattern), the Ai == Bi and termination
            # cases are equally or more likely than the Ai < Bi and Bi < Ai cases. Hence
            # the ordering of the conditional chain above differs from that in the
            # corresponding broadcast code (below).
            if _isnotzero(Cx)
                Ck > spaceC && (spaceC = expandstorage!(C, Ck + (nnz(A) - (Ak - 1)) + (nnz(B) - (Bk - 1))))
                storedinds(C)[Ck] = Ci
                storedvals(C)[Ck] = Cx
                Ck += 1
            end
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return _checkbuffers(C)
end
function _map_notzeropres!(f::Tf, fillvalue, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat, B::DensedSparseVecOrMat) where Tf
    # Build dense matrix structure in C, expanding storage if necessary
    _densestructure!(C)
    # Populate values
    fill!(storedvals(C), fillvalue)
    # NOTE: Combining this fill! into the loop below to avoid multiple sweeps over /
    # nonsequential access of storedvals(C) does not appear to improve performance.
    rowsentinelA = convert(indtype(A), numrows(A) + 1)
    rowsentinelB = convert(indtype(B), numrows(B) + 1)
    @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
        Ak, stopAk = colstartind(A, j), colboundind(A, j)
        Bk, stopBk = colstartind(B, j), colboundind(B, j)
        Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
        Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
        while true
            if Ai == Bi
                Ai == rowsentinelA && break # column complete
                Cx, Ci::indtype(C) = f(storedvals(A)[Ak], storedvals(B)[Bk]), Ai
                Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
            elseif Ai < Bi
                Cx, Ci = f(storedvals(A)[Ak], zero(eltype(B))), Ai
                Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
            else # Bi < Ai
                Cx, Ci = f(zero(eltype(A)), storedvals(B)[Bk]), Bi
                Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
            end
            Cx != fillvalue && (storedvals(C)[jo + Ci] = Cx)
        end
    end
    return _checkbuffers(C)
end


# (6) _map_zeropres!/_map_notzeropres! for more than two sparse matrices / vectors
function _map_zeropres!(f::Tf, C::DensedSparseVecOrMat, As::Vararg{DensedSparseVecOrMat,N}) where {Tf,N}
    spaceC::Int = length(nonzeros(C))
    rowsentinel = numrows(C) + 1
    Ck = 1
    stopks = _colstartind_all(1, As)
    @inbounds for j in columns(C)
        setcolptr!(C, j, Ck)
        ks = stopks
        stopks = _colboundind_all(j, As)
        rows = _rowforind_all(rowsentinel, ks, stopks, As)
        activerow = min(rows...)
        while activerow < rowsentinel
            vals, ks, rows = _fusedupdate_all(rowsentinel, activerow, rows, ks, stopks, As)
            Cx = f(vals...)
            if _isnotzero(Cx)
                Ck > spaceC && (spaceC = expandstorage!(C, Int(min(widelength(C), Ck + _sumnnzs(As...) - (sum(ks) - N)))))
                storedinds(C)[Ck] = activerow
                storedvals(C)[Ck] = Cx
                Ck += 1
            end
            activerow = min(rows...)
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return _checkbuffers(C)
end
function _map_notzeropres!(f::Tf, fillvalue, C::DensedSparseVecOrMat, As::Vararg{DensedSparseVecOrMat,N}) where {Tf,N}
    # Build dense matrix structure in C, expanding storage if necessary
    _densestructure!(C)
    # Populate values
    fill!(storedvals(C), fillvalue)
    # NOTE: Combining this fill! into the loop below to avoid multiple sweeps over /
    # nonsequential access of nonzeros(C) does not appear to improve performance.
    rowsentinel = numrows(C) + 1
    stopks = _colstartind_all(1, As)
    @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
        ks = stopks
        stopks = _colboundind_all(j, As)
        rows = _rowforind_all(rowsentinel, ks, stopks, As)
        activerow = min(rows...)
        while activerow < rowsentinel
            vals, ks, rows = _fusedupdate_all(rowsentinel, activerow, rows, ks, stopks, As)
            Cx = f(vals...)
            Cx != fillvalue && (storedvals(C)[jo + activerow] = Cx)
            activerow = min(rows...)
        end
    end
    return _checkbuffers(C)
end

# helper methods for map/map! methods just above
@inline _colstartind(j, A) = colstartind(A, j)
@inline _colstartind_all(j, ::Tuple{}) = ()
@inline _colstartind_all(j, As) = (
    _colstartind(j, first(As)),
    _colstartind_all(j, tail(As))...)
@inline _colboundind(j, A) = colboundind(A, j)
@inline _colboundind_all(j, ::Tuple{}) = ()
@inline _colboundind_all(j, As) = (
    _colboundind(j, first(As)),
    _colboundind_all(j, tail(As))...)
@inline _rowforind(rowsentinel, k, stopk, A) =
    k < stopk ? storedinds(A)[k] : convert(indtype(A), rowsentinel)
@inline _rowforind_all(rowsentinel, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _rowforind_all(rowsentinel, ks, stopks, As) = (
    _rowforind(rowsentinel, first(ks), first(stopks), first(As)),
    _rowforind_all(rowsentinel, tail(ks), tail(stopks), tail(As))...)

@inline function _fusedupdate(rowsentinel, activerow, row, k, stopk, A)
    # returns (val, nextk, nextrow)
    if row == activerow
        nextk = k + oneunit(k)
        (storedvals(A)[k], nextk, (nextk < stopk ? storedinds(A)[nextk] : oftype(row, rowsentinel)))
    else
        (zero(eltype(A)), k, row)
    end
end
@inline _fusedupdate_all(rowsentinel, activerow, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ((#=vals=#), (#=nextks=#), (#=nextrows=#))
@inline function _fusedupdate_all(rowsentinel, activerow, rows, ks, stopks, As)
    val, nextk, nextrow = _fusedupdate(rowsentinel, activerow, first(rows), first(ks), first(stopks), first(As))
    vals, nextks, nextrows = _fusedupdate_all(rowsentinel, activerow, tail(rows), tail(ks), tail(stopks), tail(As))
    return ((val, vals...), (nextk, nextks...), (nextrow, nextrows...))
end


# (7) _broadcast_zeropres!/_broadcast_notzeropres! specialized for a single (input) sparse vector/matrix
function _broadcast_zeropres!(f::Tf, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat) where Tf
    isempty(C) && return _finishempty!(C)
    spaceC::Int = length(nonzeros(C))

    # C and A cannot have the same shape, as we directed that case to map in broadcast's
    # entry point; here we need efficiently handle only heterogeneous C-A combinations where
    # one or both of C and A has at least one singleton dimension.
    #
    # We first divide the cases into two groups: those in which the input argument does not
    # expand vertically, and those in which the input argument expands vertically.
    #
    # Cases without vertical expansion
    Ck = 1
    if numrows(A) == numrows(C)
        @inbounds for j in columns(C)
            setcolptr!(C, j, Ck)
            bccolrangejA = numcols(A) == 1 ? colrange(A, 1) : colrange(A, j)
            for Ak in bccolrangejA
                Cx = f(storedvals(A)[Ak])
                if _isnotzero(Cx)
                    Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A)))
                    storedinds(C)[Ck] = storedinds(A)[Ak]
                    storedvals(C)[Ck] = Cx
                    Ck += 1
                end
            end
        end
    # Cases with vertical expansion
    else # numrows(A) != numrows(C) (=> numrows(A) == 1)
        @inbounds for j in columns(C)
            setcolptr!(C, j, Ck)
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Ax = Ak < stopAk ? storedvals(A)[Ak] : zero(eltype(A))
            fofAx = f(Ax)
            # if fofAx is zero, then either A's jth column is empty, or A's jth column
            # contains a nonzero value x but f(Ax) is nonetheless zero, so we need store
            # nothing in C's jth column. if to the contrary fofAx is nonzero, then we must
            # densely populate C's jth column with fofAx.
            if _isnotzero(fofAx)
                for Ci::indtype(C) in 1:numrows(C)
                    Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A)))
                    storedinds(C)[Ck] = Ci
                    storedvals(C)[Ck] = fofAx
                    Ck += 1
                end
            end
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return _checkbuffers(C)
end
function _broadcast_notzeropres!(f::Tf, fillvalue, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat) where Tf
    # For information on this code, see comments in similar code in _broadcast_zeropres! above
    # Build dense matrix structure in C, expanding storage if necessary
    _densestructure!(C)
    # Populate values
    fill!(storedvals(C), fillvalue)
    # Cases without vertical expansion
    if numrows(A) == numrows(C)
        @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
            bccolrangejA = numcols(A) == 1 ? colrange(A, 1) : colrange(A, j)
            for Ak in bccolrangejA
                Cx, Ci = f(storedvals(A)[Ak]), storedinds(A)[Ak]
                Cx != fillvalue && (storedvals(C)[jo + Ci] = Cx)
            end
        end
    # Cases with vertical expansion
    else # numrows(A) != numrows(C) (=> numrows(A) == 1)
        svA, svC = storedvals(A), storedvals(C)
        @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Ax = Ak < stopAk ? svA[Ak] : zero(eltype(A))
            fofAx = f(Ax)
            if fofAx != fillvalue
                for i in (jo + 1):(jo + numrows(C))
                    svC[i] = fofAx
                end
            end
        end
    end
    return _checkbuffers(C)
end


# (8) _broadcast_zeropres!/_broadcast_notzeropres! specialized for a pair of (input) sparse vectors/matrices
function _broadcast_zeropres!(f::Tf, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat, B::DensedSparseVecOrMat) where Tf
    isempty(C) && return _finishempty!(C)
    spaceC::Int = length(nonzeros(C))
    rowsentinelA = convert(indtype(A), numrows(C) + 1)
    rowsentinelB = convert(indtype(B), numrows(C) + 1)
    # C, A, and B cannot all have the same shape, as we directed that case to map in broadcast's
    # entry point; here we need efficiently handle only heterogeneous combinations of mats/vecs
    # with no singleton dimensions, one singleton dimension, and two singleton dimensions.
    # Cases involving objects with two singleton dimensions should be rare and optimizing
    # that case complicates the code appreciably, so we largely ignore that case's
    # performance below.
    #
    # We first divide the cases into two groups: those in which neither input argument
    # expands vertically, and those in which at least one argument expands vertically.
    #
    # NOTE: Placing the loops over columns outside the conditional chain segregating
    # argument shape combinations eliminates some code replication but unfortunately
    # hurts performance appreciably in some cases.
    #
    # Cases without vertical expansion
    Ck = 1
    if numrows(A) == numrows(B) == numrows(C)
        @inbounds for j in columns(C)
            setcolptr!(C, j, Ck)
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            # Restructuring this k/stopk code to avoid unnecessary colptr retrievals does
            # not improve performance signicantly. Leave in this less complex form.
            Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
            Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
            while true
                if Ai != Bi
                    if Ai < Bi
                        Cx, Ci = f(storedvals(A)[Ak], zero(eltype(B))), Ai
                        Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                    else # Ai > Bi
                        Cx, Ci = f(zero(eltype(A)), storedvals(B)[Bk]), Bi
                        Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                    end
                elseif #= Ai == Bi && =# Ai == rowsentinelA
                    break # column complete
                else #= Ai == Bi != rowsentinel =#
                    Cx, Ci::indtype(C) = f(storedvals(A)[Ak], storedvals(B)[Bk]), Ai
                    Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                    Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                end
                # NOTE: The ordering of the conditional chain above impacts which matrices
                # this method perform best for. In contrast to the map situation (arguments
                # have same shape, and likely same or similar stored entry pattern), where
                # the Ai == Bi and termination cases are equally or more likely than the
                # Ai < Bi and Bi < Ai cases, in the broadcast situation (arguments have
                # different shape, and likely largely disjoint expanded stored entry
                # pattern) the Ai < Bi and Bi < Ai cases are equally or more likely than the
                # Ai == Bi and termination cases. Hence the ordering of the conditional
                # chain above differs from that in the corresponding map code.
                if _isnotzero(Cx)
                    Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A, B)))
                    storedinds(C)[Ck] = Ci
                    storedvals(C)[Ck] = Cx
                    Ck += 1
                end
            end
        end
    # Cases with vertical expansion
    elseif numrows(A) == numrows(B) == 1 # && numrows(C) != 1, vertically expand both A and B
        @inbounds for j in columns(C)
            setcolptr!(C, j, Ck)
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Ax = Ak < stopAk ? storedvals(A)[Ak] : zero(eltype(A))
            Bx = Bk < stopBk ? storedvals(B)[Bk] : zero(eltype(B))
            Cx = f(Ax, Bx)
            if _isnotzero(Cx)
                for Ci::indtype(C) in 1:numrows(C)
                    Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A, B)))
                    storedinds(C)[Ck] = Ci
                    storedvals(C)[Ck] = Cx
                    Ck += 1
                end
            end
        end
    elseif numrows(A) == 1 # && numrows(B) == numrows(C) != 1 , vertically expand only A
        @inbounds for j in columns(C)
            setcolptr!(C, j, Ck)
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Ax = Ak < stopAk ? storedvals(A)[Ak] : zero(eltype(A))
            fvAzB = f(Ax, zero(eltype(B)))
            if _iszero(fvAzB)
                # either A's jth column is empty, or A's jth column contains a nonzero value
                # Ax but f(Ax, zero(eltype(B))) is nonetheless zero, so we can scan through
                # B's jth column without storing every entry in C's jth column
                while Bk < stopBk
                    Cx = f(Ax, storedvals(B)[Bk])
                    if _isnotzero(Cx)
                        Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A, B)))
                        storedinds(C)[Ck] = storedinds(B)[Bk]
                        storedvals(C)[Ck] = Cx
                        Ck += 1
                    end
                    Bk += oneunit(Bk)
                end
            else
                # A's jth column is nonempty and f(Ax, zero(eltype(B))) is not zero, so
                # we must store (likely) every entry in C's jth column
                Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                for Ci::indtype(C) in 1:numrows(C)
                    if Bi == Ci
                        Cx = f(Ax, storedvals(B)[Bk])
                        Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                    else
                        Cx = fvAzB
                    end
                    if _isnotzero(Cx)
                        Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A, B)))
                        storedinds(C)[Ck] = Ci
                        storedvals(C)[Ck] = Cx
                        Ck += 1
                    end
                end
            end
        end
    else # numrows(B) == 1 && numrows(A) == numrows(C) != 1, vertically expand only B
        @inbounds for j in columns(C)
            setcolptr!(C, j, Ck)
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Bx = Bk < stopBk ? storedvals(B)[Bk] : zero(eltype(B))
            fzAvB = f(zero(eltype(A)), Bx)
            if _iszero(fzAvB)
                # either B's jth column is empty, or B's jth column contains a nonzero value
                # Bx but f(zero(eltype(A)), Bx) is nonetheless zero, so we can scan through
                # A's jth column without storing every entry in C's jth column
                while Ak < stopAk
                    Cx = f(storedvals(A)[Ak], Bx)
                    if _isnotzero(Cx)
                        Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A, B)))
                        storedinds(C)[Ck] = storedinds(A)[Ak]
                        storedvals(C)[Ck] = Cx
                        Ck += 1
                    end
                    Ak += oneunit(Ak)
                end
            else
                # B's jth column is nonempty and f(zero(eltype(A)), Bx) is not zero, so
                # we must store (likely) every entry in C's jth column
                Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                for Ci::indtype(C) in 1:numrows(C)
                    if Ai == Ci
                        Cx = f(storedvals(A)[Ak], Bx)
                        Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                    else
                        Cx = fzAvB
                    end
                    if _isnotzero(Cx)
                        Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), A, B)))
                        storedinds(C)[Ck] = Ci
                        storedvals(C)[Ck] = Cx
                        Ck += 1
                    end
                end
            end
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return _checkbuffers(C)
end
function _broadcast_notzeropres!(f::Tf, fillvalue, C::DensedSparseVecOrMat, A::DensedSparseVecOrMat, B::DensedSparseVecOrMat) where Tf
    # For information on this code, see comments in similar code in _broadcast_zeropres! above
    # Build dense matrix structure in C, expanding storage if necessary
    _densestructure!(C)
    # Populate values
    fill!(storedvals(C), fillvalue)
    rowsentinelA = convert(indtype(A), numrows(C) + 1)
    rowsentinelB = convert(indtype(B), numrows(C) + 1)
    # Cases without vertical expansion
    if numrows(A) == numrows(B) == numrows(C)
        @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
            Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
            while true
                if Ai < Bi
                    Cx, Ci = f(storedvals(A)[Ak], zero(eltype(B))), Ai
                    Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                elseif Ai > Bi
                    Cx, Ci = f(zero(eltype(A)), storedvals(B)[Bk]), Bi
                    Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                elseif #= Ai == Bi && =# Ai == rowsentinelA
                    break # column complete
                else #= Ai == Bi != rowsentinel =#
                    Cx, Ci::indtype(C) = f(storedvals(A)[Ak], storedvals(B)[Bk]), Ai
                    Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                    Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                end
                Cx != fillvalue && (storedvals(C)[jo + Ci] = Cx)
            end
        end
    # Cases with vertical expansion
    elseif numrows(A) == numrows(B) == 1 # && numrows(C) != 1, vertically expand both A and B
        @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Ax = Ak < stopAk ? storedvals(A)[Ak] : zero(eltype(A))
            Bx = Bk < stopBk ? storedvals(B)[Bk] : zero(eltype(B))
            Cx = f(Ax, Bx)
            if Cx != fillvalue
                for Ck::Int in (jo + 1):(jo + numrows(C))
                    storedvals(C)[Ck] = Cx
                end
            end
        end
    elseif numrows(A) == 1 # && numrows(B) == numrows(C) != 1, vertically expand only A
        @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Ax = Ak < stopAk ? storedvals(A)[Ak] : zero(eltype(A))
            fvAzB = f(Ax, zero(eltype(B)))
            if fvAzB == fillvalue
                while Bk < stopBk
                    Cx = f(Ax, storedvals(B)[Bk])
                    Cx != fillvalue && (storedvals(C)[jo + storedinds(B)[Bk]] = Cx)
                    Bk += oneunit(Bk)
                end
            else
                Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                for Ci::indtype(C) in 1:numrows(C)
                    if Bi == Ci
                        Cx = f(Ax, storedvals(B)[Bk])
                        Bk += oneunit(Bk); Bi = Bk < stopBk ? storedinds(B)[Bk] : rowsentinelB
                    else
                        Cx = fvAzB
                    end
                    Cx != fillvalue && (storedvals(C)[jo + Ci] = Cx)
                end
            end
        end
    else # numrows(B) == 1 && numrows(A) == numrows(C) != 1, vertically expand only B
        @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
            Ak, stopAk = numcols(A) == 1 ? (colstartind(A, 1), colboundind(A, 1)) : (colstartind(A, j), colboundind(A, j))
            Bk, stopBk = numcols(B) == 1 ? (colstartind(B, 1), colboundind(B, 1)) : (colstartind(B, j), colboundind(B, j))
            Bx = Bk < stopBk ? storedvals(B)[Bk] : zero(eltype(B))
            fzAvB = f(zero(eltype(A)), Bx)
            if fzAvB == fillvalue
                while Ak < stopAk
                    Cx = f(storedvals(A)[Ak], Bx)
                    Cx != fillvalue && (storedvals(C)[jo + storedinds(A)[Ak]] = Cx)
                    Ak += oneunit(Ak)
                end
            else
                Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                for Ci::indtype(C) in 1:numrows(C)
                    if Ai == Ci
                        Cx = f(storedvals(A)[Ak], Bx)
                        Ak += oneunit(Ak); Ai = Ak < stopAk ? storedinds(A)[Ak] : rowsentinelA
                    else
                        Cx = fzAvB
                    end
                    Cx != fillvalue && (storedvals(C)[jo + Ci] = Cx)
                end
            end
        end
    end
    return _checkbuffers(C)
end
_finishempty!(C::AbstractVectorDensedSparseVector) = C
###_finishempty!(C::AbstractBlockDensedSparseVector) = (fill!(getcolptr(C), 1); C)
_finishempty!(C::AbstractBlockDensedSparseVector) = C

# special case - vector outer product
_copy(f::typeof(*), x::DensedSparseVectorUnion, y::AdjOrTransDensedSparseVectorUnion) = _outer(x, y)
@inline _outer(x::DensedSparseVectorUnion, y::Adjoint) = return _outer(conj, x, parent(y))
@inline _outer(x::DensedSparseVectorUnion, y::Transpose) = return _outer(identity, x, parent(y))
function _outer(trans::Tf, x, y) where Tf
    nx = length(x)
    ny = length(y)
    rowvalx = nonzeroinds(x)
    rowvaly = nonzeroinds(y)
    nzvalsx = nonzeros(x)
    nzvalsy = nonzeros(y)
    nnzx = length(nzvalsx)
    nnzy = length(nzvalsy)

    nnzC = nnzx * nnzy
    Tv = typeof(oneunit(eltype(x)) * oneunit(eltype(y)))
    Ti = promote_type(indtype(x), indtype(y))
    colptrC = zeros(Ti, ny + 1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalsC = Vector{Tv}(undef, nnzC)

    idx = 0
    @inbounds colptrC[1] = 1
    @inbounds for jj = 1:nnzy
        yval = nzvalsy[jj]
        if iszero(yval)
            nnzC -= nnzx
            continue
        end
        col = rowvaly[jj]
        yval = trans(yval)

        for ii = 1:nnzx
            xval = nzvalsx[ii]
            if iszero(xval)
                nnzC -= 1
                continue
            end
            idx += 1
            colptrC[col+1] += 1
            rowvalC[idx] = rowvalx[ii]
            nzvalsC[idx] = xval * yval
        end
    end
    cumsum!(colptrC, colptrC)
    resize!(rowvalC, nnzC)
    resize!(nzvalsC, nnzC)

    return SparseMatrixCSC2(nx, ny, colptrC, rowvalC, nzvalsC)
end

# (9) _broadcast_zeropres!/_broadcast_notzeropres! for more than two (input) sparse vectors/matrices
function _broadcast_zeropres!(f::Tf, C::DensedSparseVecOrMat, As::Vararg{DensedSparseVecOrMat,N}) where {Tf,N}
    isempty(C) && return _finishempty!(C)
    spaceC::Int = length(nonzeros(C))
    expandsverts = _expandsvert_all(C, As)
    expandshorzs = _expandshorz_all(C, As)
    rowsentinel = numrows(C) + 1
    Ck = 1
    @inbounds for j in columns(C)
        setcolptr!(C, j, Ck)
        ks = _startindforbccol_all(j, expandshorzs, As)
        stopks = _stopindforbccol_all(j, expandshorzs, As)
        # Neither fusing ks and stopks construction, nor restructuring them to avoid repeated
        # colptr lookups, improves performance significantly. So keep the less complex approach here.
        isemptys = _isemptycol_all(ks, stopks)
        defargs = _defargforcol_all(j, isemptys, expandsverts, ks, As)
        rows = _initrowforcol_all(j, rowsentinel, isemptys, expandsverts, ks, As)
        defaultCx = f(defargs...)
        activerow = min(rows...)
        if _iszero(defaultCx) # zero-preserving column scan
            while activerow < rowsentinel
                args, ks, rows = _fusedupdatebc_all(rowsentinel, activerow, rows, defargs, ks, stopks, As)
                Cx = f(args...)
                if _isnotzero(Cx)
                    Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), As)))
                    storedinds(C)[Ck] = activerow
                    storedvals(C)[Ck] = Cx
                    Ck += 1
                end
                activerow = min(rows...)
            end
        else # zero-non-preserving column scan
            for Ci in 1:numrows(C)
                if Ci == activerow
                    args, ks, rows = _fusedupdatebc_all(rowsentinel, activerow, rows, defargs, ks, stopks, As)
                    Cx = f(args...)
                    activerow = min(rows...)
                else
                    Cx = defaultCx
                end
                if _isnotzero(Cx)
                    Ck > spaceC && (spaceC = expandstorage!(C, _unchecked_maxnnzbcres(size(C), As)))
                    storedinds(C)[Ck] = Ci
                    storedvals(C)[Ck] = Cx
                    Ck += 1
                end
            end
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return _checkbuffers(C)
end
function _broadcast_notzeropres!(f::Tf, fillvalue, C::DensedSparseVecOrMat, As::Vararg{DensedSparseVecOrMat,N}) where {Tf,N}
    isempty(C) && return _finishempty!(C)
    # Build dense matrix structure in C, expanding storage if necessary
    _densestructure!(C)
    # Populate values
    fill!(storedvals(C), fillvalue)
    expandsverts = _expandsvert_all(C, As)
    expandshorzs = _expandshorz_all(C, As)
    rowsentinel = numrows(C) + 1
    @inbounds for (j, jo) in zip(columns(C), _densecoloffsets(C))
        ks = _startindforbccol_all(j, expandshorzs, As)
        stopks = _stopindforbccol_all(j, expandshorzs, As)
        # Neither fusing ks and stopks construction, nor restructuring them to avoid repeated
        # colptr lookups, improves performance significantly. So keep the less complex approach here.
        isemptys = _isemptycol_all(ks, stopks)
        defargs = _defargforcol_all(j, isemptys, expandsverts, ks, As)
        rows = _initrowforcol_all(j, rowsentinel, isemptys, expandsverts, ks, As)
        defaultCx = f(defargs...)
        activerow = min(rows...)
        if defaultCx == fillvalue # fillvalue-preserving column scan
            while activerow < rowsentinel
                args, ks, rows = _fusedupdatebc_all(rowsentinel, activerow, rows, defargs, ks, stopks, As)
                Cx = f(args...)
                Cx != fillvalue && (storedvals(C)[jo + activerow] = Cx)
                activerow = min(rows...)
            end
        else # fillvalue-non-preserving column scan
            for Ci in 1:numrows(C)
                if Ci == activerow
                    args, ks, rows = _fusedupdatebc_all(rowsentinel, activerow, rows, defargs, ks, stopks, As)
                    Cx = f(args...)
                    activerow = min(rows...)
                else
                    Cx = defaultCx
                end
                Cx != fillvalue && (storedvals(C)[jo + Ci] = Cx)
            end
        end
    end
    return _checkbuffers(C)
end

# helper method for broadcast/broadcast! methods just above
@inline _expandsvert(C, A) = numrows(A) != numrows(C)
@inline _expandsvert_all(C, ::Tuple{}) = ()
@inline _expandsvert_all(C, As) = (_expandsvert(C, first(As)), _expandsvert_all(C, tail(As))...)
@inline _expandshorz(C, A) = numcols(A) != numcols(C)
@inline _expandshorz_all(C, ::Tuple{}) = ()
@inline _expandshorz_all(C, As) = (_expandshorz(C, first(As)), _expandshorz_all(C, tail(As))...)
@inline _startindforbccol(j, expandshorz, A) = expandshorz ? colstartind(A, 1) : colstartind(A, j)
@inline _startindforbccol_all(j, ::Tuple{}, ::Tuple{}) = ()
@inline _startindforbccol_all(j, expandshorzs, As) = (
    _startindforbccol(j, first(expandshorzs), first(As)),
    _startindforbccol_all(j, tail(expandshorzs), tail(As))...)
@inline _stopindforbccol(j, expandshorz, A) = expandshorz ? colboundind(A, 1) : colboundind(A, j)
@inline _stopindforbccol_all(j, ::Tuple{}, ::Tuple{}) = ()
@inline _stopindforbccol_all(j, expandshorzs, As) = (
    _stopindforbccol(j, first(expandshorzs), first(As)),
    _stopindforbccol_all(j, tail(expandshorzs), tail(As))...)
@inline _isemptycol(k, stopk) = k == stopk
@inline _isemptycol_all(::Tuple{}, ::Tuple{}) = ()
@inline _isemptycol_all(ks, stopks) = (
    _isemptycol(first(ks), first(stopks)),
    _isemptycol_all(tail(ks), tail(stopks))...)
@inline _initrowforcol(j, rowsentinel, isempty, expandsvert, k, A) =
    expandsvert || isempty ? convert(indtype(A), rowsentinel) : storedinds(A)[k]
@inline _initrowforcol_all(j, rowsentinel, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _initrowforcol_all(j, rowsentinel, isemptys, expandsverts, ks, As) = (
    _initrowforcol(j, rowsentinel, first(isemptys), first(expandsverts), first(ks), first(As)),
    _initrowforcol_all(j, rowsentinel, tail(isemptys), tail(expandsverts), tail(ks), tail(As))...)
@inline _defargforcol(j, isempty, expandsvert, k, A) =
    expandsvert && !isempty ? storedvals(A)[k] : zero(eltype(A))
@inline _defargforcol_all(j, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _defargforcol_all(j, isemptys, expandsverts, ks, As) = (
    _defargforcol(j, first(isemptys), first(expandsverts), first(ks), first(As)),
    _defargforcol_all(j, tail(isemptys), tail(expandsverts), tail(ks), tail(As))...)
@inline function _fusedupdatebc(rowsentinel, activerow, row, defarg, k, stopk, A)
    # returns (val, nextk, nextrow)
    if row == activerow
        nextk = k + oneunit(k)
        (storedvals(A)[k], nextk, (nextk < stopk ? storedinds(A)[nextk] : oftype(row, rowsentinel)))
    else
        (defarg, k, row)
    end
end
@inline _fusedupdatebc_all(rowsent, activerow, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ((#=vals=#), (#=nextks=#), (#=nextrows=#))
@inline function _fusedupdatebc_all(rowsentinel, activerow, rows, defargs, ks, stopks, As)
    val, nextk, nextrow = _fusedupdatebc(rowsentinel, activerow, first(rows), first(defargs), first(ks), first(stopks), first(As))
    vals, nextks, nextrows = _fusedupdatebc_all(rowsentinel, activerow, tail(rows), tail(defargs), tail(ks), tail(stopks), tail(As))
    return ((val, vals...), (nextk, nextks...), (nextrow, nextrows...))
end


# (10) broadcast over combinations of broadcast scalars and sparse vectors/matrices

# broadcast entry points for combinations of sparse arrays and other (scalar) types
@inline function copy(bc::Broadcasted{<:DSPVM})
    bcf = flatten(bc)
    return _copy(bcf.f, bcf.args...)
end

_copy(f, args::AbstractVectorDensedSparseVector...) = _shapecheckbc(f, args...)
_copy(f, args::AbstractBlockDensedSparseVector...) = _shapecheckbc(f, args...)
_copy(f, args::DensedSparseVecOrMat...) = _diffshape_broadcast(f, args...)
# Otherwise, we incorporate scalars into the function and re-dispatch
function _copy(f, args...)
    parevalf, passedargstup = capturescalars(f, args)
    return _copy(parevalf, passedargstup...)
end
_copy(f) = throw(MethodError(_copy, (f,)))  # avoid method ambiguity

function _shapecheckbc(f, args...)
    _aresameshape(args...) ? _noshapecheck_map(f, args...) : _diffshape_broadcast(f, args...)
end


@inline function copyto!(dest::DensedSparseVecOrMat, bc::Broadcasted{<:DSPVM})
    if bc.f === identity && bc isa SpBroadcasted1 && Base.axes(dest) == (A = bc.args[1]; Base.axes(A))
        return copyto!(dest, A)
    end
    bcf = flatten(bc)
    As = map(arg->Base.unalias(dest, arg), bcf.args)
    return _copyto!(bcf.f, dest, As...)
end

@inline function _copyto!(f, dest, As::DensedSparseVecOrMat...)
    _aresameshape(dest, As...) && return _noshapecheck_map!(f, dest, As...)
    Base.Broadcast.check_broadcast_axes(axes(dest), As...)
    fofzeros = f(_zeros_eltypes(As...)...)
    if _iszero(fofzeros)
        return _broadcast_zeropres!(f, dest, As...)
    else
        return _broadcast_notzeropres!(f, fofzeros, dest, As...)
    end
end

@inline function _copyto!(f, dest, args...)
    # args contains nothing but DensedSparseVecOrMat and scalars
    # See below for capturescalars
    parevalf, passedsrcargstup = capturescalars(f, args)
    _copyto!(parevalf, dest, passedsrcargstup...)
end

# capturescalars takes a function (f) and a tuple of mixed sparse vectors/matrices and
# broadcast scalar arguments (mixedargs), and returns a function (parevalf, i.e. partially
# evaluated f) and a reduced argument tuple (passedargstup) containing only the sparse
# vectors/matrices in mixedargs in their original order, and such that the result of
# broadcast(parevalf, passedargstup...) is broadcast(f, mixedargs...)
@inline function capturescalars(f, mixedargs)
    let (passedsrcargstup, makeargs) = _capturescalars(mixedargs...)
        parevalf = (passed...) -> f(makeargs(passed...)...)
        return (parevalf, passedsrcargstup)
    end
end
# Work around losing Type{T}s as DataTypes within the tuple that makeargs creates
@inline capturescalars(f, mixedargs::Tuple{Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((args...)->f(T, args...), Base.tail(mixedargs))
@inline capturescalars(f, mixedargs::Tuple{Ref{Type{T}}, Ref{Type{S}}, Vararg{Any}}) where {T, S} =
    # This definition is identical to the one above and necessary only for
    # avoiding method ambiguity.
    capturescalars((args...)->f(T, args...), Base.tail(mixedargs))
@inline capturescalars(f, mixedargs::Tuple{DensedSparseVecOrMat, Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((a1, args...)->f(a1, T, args...), (mixedargs[1], Base.tail(Base.tail(mixedargs))...))
@inline capturescalars(f, mixedargs::Tuple{Union{Ref,AbstractArray{<:Any,0}}, Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((args...)->f(mixedargs[1], T, args...), Base.tail(Base.tail(mixedargs)))

nonscalararg(::DensedSparseVecOrMat) = true
nonscalararg(::Any) = false
scalarwrappedarg(::Union{AbstractArray{<:Any,0},Ref}) = true
scalarwrappedarg(::Any) = false

@inline function _capturescalars()
    return (), () -> ()
end
@inline function _capturescalars(arg, mixedargs...)
    let (rest, f) = _capturescalars(mixedargs...)
        if nonscalararg(arg)
            return (arg, rest...), @inline function(head, tail...)
                (head, f(tail...)...)
            end # pass-through to broadcast
        elseif scalarwrappedarg(arg)
            return rest, @inline function(tail...)
                (arg[], f(tail...)...) # TODO: This can put a Type{T} in a tuple
            end # unwrap and add back scalararg after (in makeargs)
        else
            return rest, @inline function(tail...)
                (arg, f(tail...)...)
            end # add back scalararg after (in makeargs)
        end
    end
end
@inline function _capturescalars(arg) # this definition is just an optimization (to bottom out the recursion slightly sooner)
    if nonscalararg(arg)
        return (arg,), (head,) -> (head,) # pass-through
    elseif scalarwrappedarg(arg)
        return (), () -> (arg[],) # unwrap
    else
        return (), () -> (arg,) # add scalararg
    end
end

# NOTE: The following two method definitions work around #19096.
broadcast(f::Tf, ::Type{T}, A::AbstractBlockDensedSparseVector) where {Tf,T} = broadcast(y -> f(T, y), A)
broadcast(f::Tf, A::AbstractBlockDensedSparseVector, ::Type{T}) where {Tf,T} = broadcast(x -> f(x, T), A)


# (11) broadcast[!] over combinations of scalars, sparse vectors/matrices, structured matrices,
# and one- and two-dimensional Arrays (via promotion of structured matrices and Arrays)
#
# for combinations involving only scalars, sparse arrays, structured matrices, and dense
# vectors/matrices, promote all structured matrices and dense vectors/matrices to sparse
# and rebroadcast. otherwise, divert to generic AbstractArray broadcast code.

function copy(bc::Broadcasted{PromoteToSparse})
    bcf = flatten(bc)
    if can_skip_sparsification(bcf.f, bcf.args...)
        return _copy(bcf.f, bcf.args...)
    elseif is_supported_sparse_broadcast(bcf.args...)
        return _copy(bcf.f, map(_sparsifystructured, bcf.args)...)
    else
        return copy(convert(Broadcasted{Broadcast.DefaultArrayStyle{length(axes(bc))}}, bc))
    end
end

@inline function copyto!(dest::DensedSparseVecOrMat, bc::Broadcasted{PromoteToSparse})
    bcf = flatten(bc)
    broadcast!(bcf.f, dest, map(_sparsifystructured, bcf.args)...)
end

_sparsifystructured(M::AbstractMatrix) = DensedSVSparseVector(M) # SparseMatrixCSC(M)
_sparsifystructured(V::AbstractVector) = DensedSparseVector(V) # SparseVector2(V)
#= # is it need?
_sparsifystructured(M::AbstractSparseMatrix) = DensedSVSparseVector(M) # SparseMatrixCSC2(M)
_sparsifystructured(V::AbstractSparseVector) = DensedSparseVector(V) # SparseVector2(V)
=#
_sparsifystructured(S::DensedSparseVecOrMat) = S
_sparsifystructured(x) = x


# (12) map[!] over combinations of sparse and structured matrices
###SparseOrStructuredMatrix = Union{SparseMatrixCSC,LinearAlgebra.StructuredMatrix}
SparseOrStructuredMatrix = Union{AbstractBlockDensedSparseVector,SparseMatrixCSC,LinearAlgebra.StructuredMatrix}
map(f::Tf, A::SparseOrStructuredMatrix, Bs::Vararg{SparseOrStructuredMatrix,N}) where {Tf,N} =
    (_checksameshape(A, Bs...); _noshapecheck_map(f, _sparsifystructured(A), map(_sparsifystructured, Bs)...))
map!(f::Tf, C::AbstractBlockDensedSparseVector, A::SparseOrStructuredMatrix, Bs::Vararg{SparseOrStructuredMatrix,N}) where {Tf,N} =
    (_checksameshape(C, A, Bs...); _noshapecheck_map!(f, C, _sparsifystructured(A), map(_sparsifystructured, Bs)...))

end
