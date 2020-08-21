#= 
learner.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI V2 Learner API to Julia

Methods for handling the training loop

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/learner.html =#

# using ..Callback



"""
    AbstractLearner

An `AbstractLearner` groups together a model, train and validate data,
  optimizer, loss function and callbacks.

[Callbacks](@ref) are used for every tweak of the training loop.
Callbacks receive epoch, batch and loss information which they may pass on to [Metrics](@ref).

# Require interface
- `data_bunch(l::AbstractLearner)`
- `data_bunch!(l::AbstractLearner, data_bunch)`
- `model(l::AbstractLearner)`
- `model!(l::AbstractLearner, model)`
- `loss(l::AbstractLearner)`
- `loss!(l::AbstractLearner, loss)`
- `opt(l::AbstractLearner)`
- `opt!(l::AbstractLearner, opt)`
- `add_cb!(learner::AbstractLearner, cb::AbstractCallback)`
- `cbs(learner::AbstractLearner)`
- `fit!(learner::AbstractLearner, epoch_count)`
"""
abstract type AbstractLearner end

using .Callback: AbstractCallback

"""
    Learner <: AbstractLearner
    Learner(data_bunch, model; opt = Flux.ADAM(), loss = Flux.mse)

A `Learner` is the standard grouping of a data bunch, model, optimizer, and loss.
"""
mutable struct Learner <: AbstractLearner
    cbs::Array{AbstractCallback}
    db::DataBunch
    model
    opt
    loss
end
Learner(data_bunch, model; opt=Flux.ADAM(), loss=Flux.mse) = Learner([], data_bunch, model, opt, loss)

"""
    data_bunch(l::Learner)

Get the data bunch for `l`.
"""
data_bunch(l::Learner) = l.db
"""
    data_bunch!(l::Learner, data_bunch)

Set the data bunch for `l` to `data_bunch`.
"""
data_bunch!(l::Learner, data_bunch) = l.db = data_bunch

"""
    model(l::Learner)

Get the model for `l`.
"""
model(l::Learner) = l.model
"""
    model!(l::Learner, model)

Set the model for `l` to `model`.
"""
model!(l::Learner, model) = l.model = model

"""
    loss(l::Learner)

Get the loss for `l`.
"""
loss(l::Learner) = l.loss
"""
    loss!(l::Learner, loss)

Set the loss for `l` to `loss`.
"""
loss!(l::Learner,loss) = l.loss = loss

"""
    opt(l::Learner)

Get the optimizer for `l`.
"""
opt(l::Learner) = l.opt
"""
    opt!(l::Learner, opt)

Set the optimizer for `l` to `opt`.
"""
opt!(l::Learner,opt) = l.opt = opt

"""
    add_cb!(learner::Learner, cb::AbstractCallback)

Add `cb` to the list of callbacks for `learner`.
"""
add_cb!(learner::Learner,cb::AbstractCallback) = push!(learner.cbs, cb)
"""
    cbs(learner::Learner)

Get the list of callbacks for `learner`.
"""
cbs(learner::Learner) = learner.cbs





"""
    implements_learner(T::DataType)

Test if a type implements the [`AbstractLearner`](@ref) interface.
"""
function implements_learner(T::DataType)
    return hasmethod(model, (T,)) &&
    hasmethod(model!, (T, Any)) &&
    hasmethod(loss, (T,)) &&
    hasmethod(loss!, (T, Any)) &&
    hasmethod(opt, (T,)) &&
    hasmethod(opt!, (T, Any)) &&
    hasmethod(cbs, (T,)) &&
    hasmethod(add_cb!, (T, AbstractCallback))
end

@assert implements_learner(Learner)
