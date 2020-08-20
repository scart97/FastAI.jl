#=
learner.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI V2 Learner API to Julia

Methods for handling the training loop

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/learner.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/learner.html

=#

"""
Basic class handling tweaks of the training loop by changing a [Learner](@ref) in various events

The training loop is defined in [Learner](@ref) a bit below and consists in a minimal set of instructions: looping through the data we:

compute the output of the model from the input
calculate a loss between this output and the desired target
compute the gradients of this loss with respect to all the model parameters
update the parameters accordingly

Any tweak of this training loop is defined in a Callback to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks).

A callback can implement the following methods:

before_fit
after_fit
after_cancel_fit

before_epoch
after_epoch
after_cancel_epoch

before_epoch_train
after_epoch_train
after_cancel_epoch_train

before_batch_train
batch_train_loss
after_batch_train
after_cancel_batch_train

before_epoch_validate
after_epoch_validate
after_cancel_epoch_validate

before_batch_validate
after_batch_validate
batch_validate_loss
after_cancel_batch_validate

By default handling of these events do nothing.  Special behavior is implemented by overriding these methods

"""
abstract type AbstractCallback end

struct CancelFitException <: Exception end
struct CancelEpochTrainException <: Exception end
struct CancelBatchTrainException <: Exception end
struct CancelEpochValidateException <: Exception end
struct CancelBatchValidateException <: Exception end

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

"""
    Learner <: AbstractLearner
    Learner(data_bunch, model; opt = Flux.ADAM(), loss = Flux.mse)

A `Learner` is the standard grouping of a data bunch, model, optimizer, and loss.
"""
mutable struct Learner <: AbstractLearner
    cbs:: Array{AbstractCallback}
    db::DataBunch
    model
    opt
    loss
end
Learner(data_bunch, model; opt=Flux.ADAM(), loss=Flux.mse) = Learner([],data_bunch,model,opt,loss)

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
opt!(l::Learner,opt) = l.opt=opt

"""
    add_cb!(learner::Learner, cb::AbstractCallback)

Add `cb` to the list of callbacks for `learner`.
"""
add_cb!(learner::Learner,cb::AbstractCallback) = push!(learner.cbs,cb)
"""
    cbs(learner::Learner)

Get the list of callbacks for `learner`.
"""
cbs(learner::Learner) = learner.cbs


function split_batch(batch, n_inp::Integer=1)
    # To handle models with multiple inputs/outputs
    # As the batch is a tuple, it returns two tuples
    x = batch[begin:n_inp]
    y = batch[n_inp+1:end]
    return x, y
end

function batch_step(learner::AbstractLearner, batch, batch_idx::Integer, is_training::Bool)
    x, y = split_batch(batch, learner |> data_bunch |> n_inp)
    # callback here
    y_hat = model(learner)(x...)
    # callback here
    l = loss(learner)(y_hat, y...)
    # callback here
    return l
end

function train_epoch!(learner::AbstractLearner, parameters, epoch_idx::Integer)
    trainmode!(learner) # TODO: move to TrainEvalCallback ?
    # callback here
    for (batch_idx, batch) in enumerate(learner |> data_bunch |> train)
        gradients = gradient(parameters) do
            batch_step(learner, batch, batch_idx, true)
        end
        # callback here
        # TODO: multiple optimizers and parameter groups
        # TODO: Also have schedulers and update them
        update!(opt(learner), parameters, gradients) 
    end
    # callback here
end


function valid_epoch(learner::AbstractLearner, epoch_idx::Integer)
    testmode!(learner) # TODO: move to TrainEvalCallback ?
    # callback here
    for (batch_idx, batch) in enumerate(learner |> data_bunch |> valid)
        batch_step(learner, batch, batch_idx, false)
    end
    # callback here
end

function fit!(learner::AbstractLearner, epochs::Integer)
    # TODO: the user can be able to specify multiple parameter
    # groups for differential learning rates
    
    # callback here
    parameters = Flux.params(model(learner))
    for epoch_idx in 1:epochs
        train_epoch!(learner, parameters, epoch_idx)
        valid_epoch(learner, epoch_idx)
    end
    # callback here
end


"""
    implements_learner(T::DataType)

Test if a type implements the [`AbstractLearner`](@ref) interface.
"""
function implements_learner(T::DataType)
    return hasmethod(model,(T,)) &&
        hasmethod(model!,(T,Any)) &&
        hasmethod(loss,(T,)) &&
        hasmethod(loss!,(T,Any)) &&
        hasmethod(opt,(T,)) &&
        hasmethod(opt!,(T,Any)) &&
        hasmethod(cbs,(T,)) &&
        hasmethod(add_cb!,(T,AbstractCallback))
end

@assert implements_learner(Learner)
