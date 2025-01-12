#!/bin/sh
julia --project=../../. simulate.jl params.yaml
julia --project=../../. dataprocessing.jl params.yaml