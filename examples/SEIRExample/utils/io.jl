"""
Write the state of a MTBP to binary file.
"""
function writeparticle(io, particle::Vector{Int64}, t::Int64)
    write(io, t)
    l = length(particle)
    write(io, Int64(l))
    nwrite = write(io, particle)
    @assert nwrite == l*sizeof(Int64) "bad file write - unexpected particle length"
end

"""
Read simulation in the format as output by simulate.jl.
"""
function readparticles(fn)
    path, t = open(fn, "r") do io 
        nsteps = read(io, Int64)
        path = Vector{Int64}[]
        t = Int64[]
        for step in 1:nsteps
            tstamp = read(io, Int64)
            push!(t, tstamp)
            particle_length = read(io, Int64)
            particle = Vector{Int64}(undef, particle_length)
            for eltidx in 1:particle_length
                particle[eltidx] = read(io, Int64)
            end
            push!(path, particle)
        end
        path, t
    end
    return path, t
end
