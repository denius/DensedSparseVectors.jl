
#
#  show functions
#
# derived from stdlib/SparseArrays/src/sparsevector.jl
#

function quick_get_max_pad(V::AbstractDensedCompressedVector)
    pad = 0
    for (indices, _) in nzchunkspairs(V)
        pad = max(pad, ndigits(first(indices)), ndigits(last(indices)))
    end
    pad
end

function Base.show(io::IO, ::MIME"text/plain", x::AbstractDensedCompressedVector)
    xnnz = 0
    for cchunk in x.nzchunks
        xnnz += length(cchunk)
    end
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::AbstractDensedCompressedVector)
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


# function Base.show(io::IO, ::MIME"text/plain", x::DensedSVSparseVector)
function Base.show(io::IO, ::MIME"text/plain", x::Union{DSV_0{L}, DSV_1{L}, DSV_L{L}}) where L
    xnnz = 0
    for cchunk in x.nzchunks
        xnnz += length(cchunk)
    end
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
function Base.show(io::IOContext, x::Union{DSV_0{L}, DSV_1{L}, DSV_L{L}}) where L
    # n = length(x)
    nzind = nonzeroinds(x)
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end

    if L == 0

        nzval = Vector{eltype(eltype(x.nzchunks))}()
        for cchunk in x.nzchunks
            append!(nzval, values(cchunk))
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

    else # L > 0

        nzval = Vector{eltype(x)}()
        for cchunk in x.nzchunks
            append!(nzval, values(cchunk))
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
                print(io, "  ", '[', rpad(nzind[k], pad), "]  =  [")
                for cchunk in x.nzchunks
                    nzval = values(cchunk)
                    ptr = cchunk.ptr
                    for r in map(i -> i:i+step(ptr)-1, ptr[1:end-1])
                        println(@view(nzval[r]))
                    end
                    for l in axes(nzval,1)
                        if isassigned(nzval, Int(l))
                            show(io, nzval[l])
                        else
                            print(io, Base.undef_ref_str)
                        end
                    end
                end
                k != length(nzind) && println(io,"]")
            elseif k == half_screen_rows
                println(io, "   ", " "^pad, "   \u22ee")
            end
        end
    end
end
# function Base.show(io::IOContext, x::DensedSVSparseVector)
#function Base.show(io::IOContext, x::DSV_L)
#    n = length(x)
#    nzind = nonzeroinds(x)
#    if isempty(nzind)
#        return show(io, MIME("text/plain"), x)
#    end
#    nzval = Vector{eltype(eltype(x.nzchunks))}()
#    for cchunk in x.nzchunks
#        append!(nzval, values(cchunk))
#    end
#    limit = get(io, :limit, false)::Bool
#    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
#    #pad = ndigits(n)
#    pad = quick_get_max_pad(x)
#    if !haskey(io, :compact)
#        io = IOContext(io, :compact => true)
#    end
#    for k = eachindex(nzind)
#        if k < half_screen_rows || k > length(nzind) - half_screen_rows
#            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
#            if isassigned(nzval, Int(k))
#                show(io, nzval[k])
#            else
#                print(io, Base.undef_ref_str)
#            end
#            k != length(nzind) && println(io)
#        elseif k == half_screen_rows
#            println(io, "   ", " "^pad, "   \u22ee")
#        end
#    end
#end


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


