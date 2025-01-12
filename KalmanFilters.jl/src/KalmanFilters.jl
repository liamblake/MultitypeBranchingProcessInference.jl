module KalmanFilters

using LinearAlgebra
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