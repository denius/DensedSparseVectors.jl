
#
#  show functions
#
# derived from stdlib/SparseArrays/src/sparsevector.jl
#

function quick_get_max_pad(V::AbstractAllDensedSparseVector)
    pad = 0
    for (indices, _) in nzchunkspairs(V)
        pad = max(pad, ndigits(first(indices)), ndigits(last(indices)))
    end
    pad
end

function Base.show(io::IOContext, x::AbstractDensedSparseVector)
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end


function Base.show(io::IO, ::MIME"text/plain", x::DensedSVSparseVector)
    xnnz = 0
    for v in x.nzchunks
        xnnz += length(v)
    end
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::DensedSVSparseVector)
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = Vector{eltype(eltype(x.nzchunks))}()
    for v in x.nzchunks
        for u in v
            push!(nzval, u)
        end
    end
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::DensedVLSparseVector)
    xnnz = 0
    for v in x.offsets
        xnnz += length(v)-1
    end
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::DensedVLSparseVector)
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = Vector{Vector{eltype(eltype(x.nzchunks))}}()
    for (offs,v) in zip(x.offsets, x.nzchunks)
        for i = 1:length(offs)-1
            push!(nzval, v[offs[i]:offs[i+1]-1])
        end
    end
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    #pad = ndigits(n)
    pad = quick_get_max_pad(x)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(nzind)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            if isassigned(nzval, Int(k))
                show(io, nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end


function Base.show(io::IO, ::MIME"text/plain", x::Union{CompressedChunk0{Tv},CompressedChunk{Tv,0}}) where Tv
    print(io, length(x), "-element ", typeof(x))
    if length(x) != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::Union{CompressedChunk0{Tv},CompressedChunk{Tv,0}}) where Tv
    if isempty(x)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(x)
        if k < half_screen_rows || k > length(x) - half_screen_rows
            print(io, "  ")
            if isassigned(x, Int(k))
                show(io, x[k,1])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(x) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end


function Base.show(io::IO, ::MIME"text/plain", x::AbstractCompressedChunk)
    print(io, length(x), "-element ", typeof(x))
    if length(x) != 0
        println(io, ":")
        # show(IOContext(io, :typeinfo => eltype(x)), x)
        show(IOContext(io, :typeinfo => Vector{eltype(x)}), x)
    end
end
function Base.show(io::IOContext, x::AbstractCompressedChunk)
    if isempty(x)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = eachindex(x)
        if k < half_screen_rows || k > length(x) - half_screen_rows
            print(io, "  ")
            if isassigned(x, Int(k))
                show(io, x[k])
            else
                print(io, Base.undef_ref_str)
            end
            k != length(x) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end
