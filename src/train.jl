using ..Callback.Phases: AbstractPhase
using ..Callback.Events: AbstractEvent
using ..Callback: Phases, Events

mutable struct CallbackHandler
    learner::AbstractLearner
end

# Handling the callbacks
function (handler::CallbackHandler)(phase::AbstractPhase, event::AbstractEvent)
    for cb in cbs(handler.learner)
        # This is what the user will implement
        handle_callback!(cb, handler.learner, phase, event)
    end
    # This control what modified state will be returned
    return_params(phase, event)
end

# This is what the user will implement
function handle_callback!(::AbstractCallback, ::AbstractLearner, ::AbstractPhase, ::AbstractEvent)
    nothing
end

# This control what modified state will be returned
return_params(::AbstractPhase, ::AbstractEvent) = nothing
return_params(::Phases.InitializationPhase, e::Events.BeforeFitEvent) = e.epochs
return_params(p::Phases.TrainPhase, ::Events.BeforeEpochEvent) = p.parameters
return_params(::AbstractPhase, e::Events.BeforePredictEvent) = e.x, e.y
return_params(::AbstractPhase, e::Events.BeforeLossEvent) = e.y, e.y_hat
return_params(::AbstractPhase, e::Events.AfterLossEvent) = e.loss_val
return_params(::AbstractPhase, e::Events.BeforeUpdateEvent) = e.gradients


function split_batch(batch, n_inp::Integer=1)
    # To handle models with multiple inputs/outputs
    # As the batch is a tuple, it returns two tuples
    x = batch[begin:n_inp]
    y = batch[n_inp + 1:end]
    return x, y
end

function batch_step(learner::AbstractLearner, batch, phase::AbstractPhase, handler::CallbackHandler)
    x, y = split_batch(batch, learner |> data_bunch |> n_inp)
    x, y = handler(phase, Events.BeforePredictEvent(x, y))
    y_hat = model(learner)(x...)
    y, y_hat = handler(phase, Events.BeforeLossEvent(y, y_hat))
    loss_val = loss(learner)(y_hat, y...)
    loss_val = handler(phase, Events.AfterLossEvent(loss_val))
    return loss_val
end

function train_epoch!(learner::AbstractLearner, handler::CallbackHandler, epoch_idx::Integer)
    parameters = Flux.params(model(learner))
    train_phase = Phases.TrainPhase(parameters, epoch_idx)
    parameters = handler(train_phase, Events.BeforeEpochEvent())

    for (batch_idx, batch) in enumerate(learner |> data_bunch |> train)
        gradients = gradient(parameters) do 
            batch_step(learner, batch, train_phase, handler)
        end
        gradients = handler(train_phase, Events.BeforeUpdateEvent(gradients)) 
        # TODO: multiple optimizers and parameter groups
        # TODO: Also have schedulers and update them
        update!(opt(learner), parameters, gradients)
        # callback here - CancelBatchEvent 
    end
    handler(train_phase, Events.AfterEpochEvent())
    # callback here - CancelEpochEvent
end


function valid_epoch(learner::AbstractLearner, handler::CallbackHandler, epoch_idx::Integer)
    testmode!(learner) # TODO: move to TrainEvalCallback ?
    # callback here - BeforeEpochEvent
    val_phase = Phases.ValidationPhase(epoch_idx)
    handler(val_phase, Events.BeforeEpochEvent())
    for (batch_idx, batch) in enumerate(learner |> data_bunch |> valid)
        batch_step(learner, batch, val_phase, handler)
    end
    handler(val_phase, Events.AfterEpochEvent())
    # callback here - AfterEpochEvent

    # callback here - CancelEpochEvent
end




function fit!(learner::AbstractLearner, epochs::Integer)
    # TODO: the user can be able to specify multiple parameter
    # groups for differential learning rates
    handler = CallbackHandler(learner)
    epochs = handler(Phases.InitializationPhase(), Events.BeforeFitEvent(epochs))
    for epoch_idx in 1:epochs
        train_epoch!(learner, handler, epoch_idx)
        valid_epoch(learner, handler, epoch_idx)
    end
    # callback here - AfterFitEvent
    # callback here - CancelFitEvent
end