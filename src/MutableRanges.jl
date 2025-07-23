
"""
    MutableRange(r::AbstractRange)

Wraps a parent range making it mutable.
Derived from https://github.com/Tokazama/StaticRanges.jl
"""
mutable struct MutableRange{T,P<:AbstractRange{T}} <: AbstractRange{T}
    parent::P
end


mrange(; kwargs...) = MutableRange(range(; kwargs...))
mrange(start; kwargs...) = MutableRange(range(start; kwargs...))
mrange(start, stop; kwargs...) = isempty(kwargs) ? MutableRange(start:stop) : MutableRange(range(start, stop; kwargs...))


Base.parent(mr::MutableRange)  = getfield(mr, :parent)
Base.length(mr::MutableRange)  = length(parent(mr))
Base.first(mr::MutableRange)   = first(parent(mr))
Base.last(mr::MutableRange)    = last(parent(mr))
Base.step(mr::MutableRange)    = step(parent(mr))
Base.isempty(mr::MutableRange) = isempty(parent(mr))
Base.map(mr::MutableRange)     = map(parent(mr))
Base.getindex(mr::MutableRange, i::Integer) = getindex(parent(mr), i)
Base.intersect(mr1::MutableRange, mr2::MutableRange) = MutableRange(intersect(parent(mr1), parent(mr2)))
Base.show(io::IO, mr::MutableRange)     = show(io::IO, parent(mr))


function Base.push!(mr::MutableRange{Ti,P}, x) where {Ti,P<:UnitRange}
    r = mr.parent
    @assert x == last(r) + step(r)
    mr.parent = UnitRange{Ti}(first(r), x)
    return mr
end
function Base.push!(mr::MutableRange{Ti,P}, x) where {Ti,P<:StepRange}
    r = mr.parent
    @assert x == last(r) + step(r)
    mr.parent = StepRange{Ti,Ti}(first(r), step(r), x)
    return mr
end
function Base.push!(mr::MutableRange{Ti,P}, x) where {Ti,P<:LinRange}
    r = mr.parent
    @assert x == last(r) + Int(step(r))
    mr.parent = LinRange{Ti,Ti}(first(r), x, length(r)+1)
    return mr
end
function Base.pushfirst!(mr::MutableRange{Ti,P}, x) where {Ti,P<:UnitRange}
    r = mr.parent
    @assert x == first(r)
    mr.parent = UnitRange{Ti}(x, last(r)+step(r))
    return mr
end
function Base.pushfirst!(mr::MutableRange{Ti,P}, x) where {Ti,P<:StepRange}
    r = mr.parent
    @assert x == first(r)
    mr.parent = StepRange{Ti,Ti}(x, step(r), last(r)+step(r))
    return mr
end
function Base.pushfirst!(mr::MutableRange{Ti,P}, x) where {Ti,P<:LinRange}
    r = mr.parent
    @assert x == first(r)
    mr.parent = LinRange{Ti,Ti}(x, last(r)+Int(step(r)), length(r)+1)
    return mr
end



