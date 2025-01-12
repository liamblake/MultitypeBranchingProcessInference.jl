using ParticleFilter
using Test

using Random
using Distributions
using StableRNGs

mutable struct MockRNG <: AbstractRNG
    counter
end
mutable struct MockDistribution 
    counter::Int
end
mutable struct MockStateSpaceModel
    counter::Int
end
mutable struct MockParticles
    counter::Int
end

include("test_ParticleFilters.jl")
include("test_particlestore.jl")
