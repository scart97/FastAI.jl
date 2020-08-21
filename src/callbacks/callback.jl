#= 
callback.jl:

Author: Peter Wolf (opus111@gmail.com)

Port of the FastAI V2 Callback API to Julia

This code is inspired by FastAI, but differs from it in important ways

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/callback/core.py

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/callback.core.html

Extra influence:
https://github.com/lorenzoh/FluxTraining.jl/blob/master/src/callbacks/callback.jl =#

module Callback

module Phases

# Those go into the callback handler ?
struct PhaseCounter
    total_epochs::Union{Nothing,Integer}
    epoch_idx::Union{Nothing,Integer}
    total_batches::Union{Nothing,Integer}
    batch_idx::Union{Nothing,Integer}
end

abstract type AbstractPhase end

mutable struct TrainPhase <: AbstractPhase
    parameters
    epoch_idx::Integer
end

struct ValidationPhase <: AbstractPhase 
    epoch_idx::Integer
end

struct TestPhase <: AbstractPhase
    epoch_idx::Integer
end
struct InitializationPhase <: AbstractPhase end
struct CleanupPhase <: AbstractPhase end

export AbstractPhase,
    TrainPhase,
    ValidationPhase,
    TestPhase,
    InitializationPhase,
    CleanupPhase,
    PhaseCounter
end

using .Phases

module Events
    
# Training events
abstract type AbstractEvent end

# BeforeFitEvent
    # BeforeEpochEvent
        # BeforePredictEvent - x, y
        # BeforeLossEvent - y, y_hat 
        # AfterLossEvent - loss_val
        # BeforeUpdateEvent - gradients
        # CancelBatchEvent
    # AfterEpochEvent
    # CancelEpochEvent
# AfterFitEvent
# CancelFitEvent 

mutable struct BeforeFitEvent <: AbstractEvent
    epochs::Integer
end
struct AfterFitEvent <: AbstractEvent end
struct CancelFitEvent  <: AbstractEvent end

struct BeforeEpochEvent <: AbstractEvent end
struct AfterEpochEvent <: AbstractEvent end
struct CancelEpochEvent <: AbstractEvent end

mutable struct BeforePredictEvent <: AbstractEvent 
    x
    y
end
mutable struct BeforeLossEvent <: AbstractEvent 
    y
    y_hat 
end
mutable struct AfterLossEvent <: AbstractEvent 
    loss_val
end
mutable struct BeforeUpdateEvent <: AbstractEvent 
    gradients
end
struct CancelBatchEvent <: AbstractEvent end

export AbstractEvent,
    BeforeFitEvent
    BeforeEpochEvent
    BeforePredictEvent,
    BeforeLossEvent,
    AfterLossEvent,
    BeforeUpdateEvent,
    CancelBatchEvent
    AfterEpochEvent
    CancelEpochEvent
    AfterFitEvent
    CancelFitEvent

end

using .Events

struct CancelFitException <: Exception end
struct CancelBatchException <: Exception end
struct CancelEpochException <: Exception end


"""
Basic class handling tweaks of the training loop by changing a [Learner](@ref) in various events

The training loop is defined in [Learner](@ref) a bit below and consists in a minimal set of instructions: looping through the data we:

compute the output of the model from the input
calculate a loss between this output and the desired target
compute the gradients of this loss with respect to all the model parameters
update the parameters accordingly

Any tweak of this training loop is defined in a Callback to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks).

By default handling of these events do nothing.  Special behavior is implemented by overriding these methods

"""
abstract type AbstractCallback end

end