module KalmanFilters

using LinearAlgebra
using StaticArrays
import Distributions: mean, mean!

include("kalmanfilter.jl")

export KalmanFilter,
    update!,
    predict!,
    reset!,
    loglikelihood!,
    kalmanfilter!,
    mean,
    mean!

end