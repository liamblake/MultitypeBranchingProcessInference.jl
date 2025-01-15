using Random
using StatsPlots
using YAML

using MultitypeBranchingProcessInference

function writeparticles(io, particles::Vector{Vector{Int64}}, t::Int64)
    write(io, t)
    write(io, Int64(length(particles)))
    l = length(first(particles))
    write(io, Int64(l))
    for particle in particles
        nwrite = write(io, particle)
        @assert nwrite == l*sizeof(Int64) "bad file write - unexpected particle length"
    end
end

function main(
    infection_rate::Float64, exposed_stage_chage_rate::Float64, infectious_stage_chage_rate::Float64, 
    observation_probablity::Float64, 
    E_immigration_rate::Float64, I_immigration_rate::Float64,
    initial_E::Int64, initial_I::Int64, initial_O::Int64,
    seed::Int, n_particles::Int64, tstep::Int64, nsteps::Int64, writefilename, 
)
    rng = Random.Xoshiro()
    Random.seed!(rng, seed)

    mtbp = SEIR(1, 1,
        infection_rate, exposed_stage_chage_rate, infectious_stage_chage_rate, 
        observation_probablity, 
        [E_immigration_rate, I_immigration_rate], 
        Int64[initial_E, initial_I, initial_O, 1],
    )

    particles = ParticleStore(Float64, mtbp.state, n_particles)
    initstate!(particles, mtbp.initial_state)

    t = zero(tstep)

    open(joinpath(@__DIR__, writefilename), "w") do io
        write(io, Int64(nsteps+1))
        writeparticles(io, particles.store, t)
        for _ in 1:nsteps
            t += tstep
            for particle in particles.store
                simulate!(rng, particle, mtbp, tstep)
            end
            writeparticles(io, particles.store, t)
        end
    end
    return 0
end

params = YAML.load_file(joinpath(@__DIR__, ARGS[1]))
@time main(
    params["infection_rate"], params["exposed_stage_chage_rate"], params["infectious_stage_chage_rate"], 
    params["observation_probability"], 
    params["E_immigration_rate"], params["I_immigration_rate"],
    params["initial_E"], params["initial_I"], params["initial_O"],
    params["seed"], params["n_particles"], params["tstep"], params["nsteps"], params["outfilename"], 
)