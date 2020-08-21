#= 
FastAI.jl:

Author: Peter Wolf (opus111@gmail.com) =#

module FastAI

using Random
using StatsBase
using Statistics
using Flux
using Flux: update!
using Flux.Data
using Zygote
using Infiltrator
using Base: length, getindex
using Random: randperm

export AbstractLearner
export AbstractCallback
export AbstractMetric
export IterableDataset
export MapDataset

export DataBunch
export train
export valid
export n_inp

export DummyCallback
export ProgressCallback
export Recorder

export Learner
export model
export data_bunch
export loss
export loss!
export opt
export opt!
export fit!
export add_cb!

export AvgMetric
export AvgLoss
export AvgSmoothLoss
export reset
export accumulate
export value
export name

export Callback


include("data/dataset.jl")
include("data/databunch.jl")
include("callbacks/callback.jl")
include("learner.jl")


include("metric.jl")
# include("callbacks/recorder.jl")
include("exercise.jl")
include("train.jl")

end
