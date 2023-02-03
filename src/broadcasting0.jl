
struct DensedSparseVectorStyle <: AbstractArrayStyle{1} end

const DnsSparseVecStyle = DensedSparseVectorStyle

DnsSparseVecStyle(::Val{0}) = DnsSparseVecStyle()
DnsSparseVecStyle(::Val{1}) = DnsSparseVecStyle()
DnsSparseVecStyle(::Val{N}) where N = DefaultArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::DnsSparseVecStyle, ::DnsSparseVecStyle) = DnsSparseVecStyle
Base.Broadcast.BroadcastStyle(s::DnsSparseVecStyle, ::DefaultArrayStyle{0}) = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{0}, s::DnsSparseVecStyle) = s
Base.Broadcast.BroadcastStyle(s::DnsSparseVecStyle, ::DefaultArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle{M}, s::DnsSparseVecStyle) where {M} = s
Base.Broadcast.BroadcastStyle(s::DnsSparseVecStyle, ::AbstractArrayStyle{M}) where {M} = s
Base.Broadcast.BroadcastStyle(::AbstractArrayStyle{M}, s::DnsSparseVecStyle) where {M} = s

Base.Broadcast.BroadcastStyle(::Type{<:AbstractDensedSparseVector}) = DnsSparseVecStyle()
Base.Broadcast.BroadcastStyle(::Type{<:SubArray{<:Any,<:Any,<:T}}) where {T<:AbstractDensedSparseVector} = DnsSparseVecStyle()

Base.similar(bc::Broadcasted{DnsSparseVecStyle}) = similar(find_ADSV(bc))
Base.similar(bc::Broadcasted{DnsSparseVecStyle}, ::Type{ElType}) where ElType = similar(find_ADSV(bc), ElType)

"`find_ADSV(bc::Broadcasted)` returns the first of any `AbstractDensedSparseVector` in `bc`"
find_ADSV(bc::Base.Broadcast.Broadcasted) = find_ADSV(bc.args)
find_ADSV(args::Tuple) = find_ADSV(find_ADSV(args[1]), Base.tail(args))
find_ADSV(x::Base.Broadcast.Extruded) = x.x  # expose internals of Broadcast but else don't work
find_ADSV(x) = x
find_ADSV(::Tuple{}) = nothing
find_ADSV(V::AbstractDensedSparseVector, rest) = V
find_ADSV(::Any, rest) = find_ADSV(rest)

nzDimensionMismatchMsg(args)::String = "Number of nonzeros of vectors must be equal, but have nnz's:" *
                                       "$(map((a)->nnz(a), filter((a)->(isa(a,AbstractVector)&&!ismathscalar(a)), args)))"
throwDimensionMismatch(args) = throw(DimensionMismatch(nzDimensionMismatchMsg(args)))

function Base.Broadcast.instantiate(bc::Broadcasted{DnsSparseVecStyle})
    if bc.axes isa Nothing
        v1 = find_ADSV(bc)
        bcf = Broadcast.flatten(bc)
        # FIXME: TODO: see https://github.com/JuliaLang/julia/issues/37558 to have an some performance penalty
        @boundscheck similarlength(nnz(v1), bcf.args) || throwDimensionMismatch(bcf.args)
        bcaxes = axes(v1)
        #bcaxes = Broadcast.combine_axes(bc.args...)
    else
        bcaxes = bc.axes
        # AbstractDensedSparseVector is flexible in assignment in any direction thus any sizes are allowed
        #check_broadcast_axes(axes, bc.args...)
    end
    return Broadcasted{DnsSparseVecStyle}(bc.f, bc.args, bcaxes)
end

function Base.copy(bc::Broadcasted{<:DnsSparseVecStyle})
    dest = similar(bc, Broadcast.combine_eltypes(bc.f, bc.args))
    bcf = Broadcast.flatten(bc)
    @boundscheck similarlength(nnz(dest), bcf.args) || throwDimensionMismatch((dest, bcf.args...))
    nzcopyto_flatten!(bcf.f, dest, bcf.args)
end


Base.copyto!(dest::AbstractVector, bc::Broadcasted{<:DnsSparseVecStyle}) = nzcopyto!(dest, bc)

function nzcopyto!(dest, bc)
    bcf = Broadcast.flatten(bc)
    # TODO: fix for `dsv1 .+ v::Vector`
    @boundscheck similarlength(nnz(dest), bcf.args) || throwDimensionMismatch((dest, bcf.args...))
    nzcopyto_flatten!(bcf.f, dest, bcf.args)
end

function nzcopyto_flatten!(f, dest, args)
    if iterablenzchunks(dest, args) && issimilar_ADSV(dest, args)
        nzbroadcastchunks!(f, dest, args)
    else
        nzbroadcast!(f, dest, args)
    end
    return dest
end

## TODO: integrate `ItWrapper` instead of direct iterating over `Number` and `[Number]`,
## and may be and on `Vector` and `SparseVector`
#struct ItWrapper{T}
#    x::T
#end
#ItWrapper(V) = ItWrapper{typeof(V[])}(V[])
#@inline Base.getindex(V::ItWrapper, i::Integer) = V.x
#@inline Base.iterate(V::ItWrapper, state = 1) = (V.x, state)
#@inline iteratenzchunks(V::ItWrapper, state = 1) = (state, state)
#@inline get_nzchunk(V::ItWrapper, i) = V
#@inline Base.ndims(V::ItWrapper) = 1
#@inline Base.length(V::ItWrapper) = 1
#
#@inline iteratenzchunks(V::Base.RefValue, state = 1) = (state, state)
#@inline get_nzchunk(V::Base.RefValue, i) = V[]

@generated function nzbroadcastchunks!(f, dest, args)
    codeInit = quote
        # create `nzchunks()` iterator for each item in args
        nzchunksiters = map(nzchunks, args)
        nzchunksiters = (nzchunks(dest), nzchunksiters...)
    end
    code = quote
        for (dst, rest...) in zip(nzchunksiters...)
            dst .= f.(rest...)
        end
    end
    return quote
        $codeInit
        @inbounds $code
        return dest
    end
end

"`nzbroadcast!(f, dest, args)` performs broadcasting over non-zero values of vectors in `args`.
Note 1: `f` and `args` should be `flatten` `bc.f` and `bc.args` respectively.
Note 2: The coincidence of vectors indices should be checked and provided by the user."
@generated function nzbroadcast!(f, dest, args)
    return quote
        # create `nzvalues()` iterator for each item in args
        iters = map(nzvalues, args)
        # for the result there is the `view` `nzvalues` iterator
        iters = (nzvaluesview(dest), iters...)
        for (dst, rest...) in zip(iters...)
            dst[] = f(rest...)
        end
        return dest
    end
end


similarlength(n, args::Tuple) = (ismathscalar(first(args)) || n == nnz(first(args))) && similarlength(n, Base.tail(args))
similarlength(n, a) = ismathscalar(a) || n == nnz(a)
similarlength(n, a::Tuple{}) = true


@inline isa_ADSV(a) = isa(a, AbstractDensedSparseVector) ||
                     (isa(a, SubArray) && isa(a.parent, AbstractDensedSparseVector))

"Are the vectors the similar in every non-zero chunk"
function issimilar_ADSV(dest, args::Tuple)

    args1 = filter(a->isa_ADSV(a), args)

    iters = map(nzchunkspairs, (dest, args1...))
    for (dst, rest...) in zip(iters...)
        idx = dst[1]
        len = length(dst[2])
        foldl((s,r)-> s && r[1]==idx, rest, init=true) || return false
        foldl((s,r)-> s && length(r[2])==len, rest, init=true) || return false
    end
    return true
end
issimilar_ADSV(dest, args) = issimilar_ADSV(dest, (args,))

"Are all vectors iterable by non-zero nzchunks"
iterablenzchunks(a, args...) = isa_ADSV_or_scalar(a) || iterablenzchunks(a, iterablenzchunks(args...))
iterablenzchunks(a, b) = isa_ADSV_or_scalar(a) || isa_ADSV_or_scalar(b)
iterablenzchunks(a) = isa_ADSV_or_scalar(a)

isa_ADSV_or_scalar(a) = isa_ADSV(a) || ismathscalar(a)

@inline function ismathscalar(a)
    return (isa(a, Number)                       ||
            isa(a, DenseArray) && length(a) == 1 ||
            isa(a, SubArray)   && length(a) == 1    )
end

