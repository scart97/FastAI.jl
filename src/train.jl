mutable struct CallbackHandler
    learner::AbstractLearner
end

# Handling the callbacks
function (handler::CallbackHandler)(phase::Phases.AbstractPhase, event::Events.AbstractEvent)
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

# This control what modified state will be changed
return_params(::AbstractPhase, ::AbstractEvent) = nothing
return_params(p::TrainPhase, ::BeforeEpochEvent) = p.parameters
return_params(::AbstractPhase, e::BeforePredictEvent) = e.x, e.y
return_params(::AbstractPhase, e::BeforeLossEvent) = e.y, e.y_hat
return_params(::AbstractPhase, e::AfterLossEvent) = e.loss_val
return_params(::AbstractPhase, e::BeforeUpdateEvent) = e.gradients


function split_batch(batch, n_inp::Integer=1)
    # To handle models with multiple inputs/outputs
    # As the batch is a tuple, it returns two tuples
    x = batch[begin:n_inp]
    y = batch[n_inp + 1:end]
    return x, y
end

function batch_step(learner::AbstractLearner, batch, batch_idx::Integer, is_training::Bool)
    x, y = split_batch(batch, learner |> data_bunch |> n_inp)
    # callback here - BeforePredictEvent
    y_hat = model(learner)(x...)
    # callback here - BeforeLossEvent
    loss_val = loss(learner)(y_hat, y...)
    # callback here - AfterLossEvent
    return loss_val
end

function train_epoch!(learner::AbstractLearner, epoch_idx::Integer)
    trainmode!(learner) # TODO: move to TrainEvalCallback ?
    # callback here - BeforeEpochEvent
    
    for (batch_idx, batch) in enumerate(learner |> data_bunch |> train)
        gradients = gradient(parameters) do 
            batch_step(learner, batch, batch_idx, true)
        end
        # callback here - BeforeUpdateEvent
        # TODO: multiple optimizers and parameter groups
        # TODO: Also have schedulers and update them
        update!(opt(learner), parameters, gradients)
        # callback here - CancelBatchEvent 
    end
    # callback here - AfterEpochEvent
    # callback here - CancelEpochEvent
end


function valid_epoch(learner::AbstractLearner, epoch_idx::Integer)
    testmode!(learner) # TODO: move to TrainEvalCallback ?
    # callback here - BeforeEpochEvent
    for (batch_idx, batch) in enumerate(learner |> data_bunch |> valid)
        batch_step(learner, batch, batch_idx, false)
    end
    # callback here - AfterEpochEvent

    # callback here - CancelEpochEvent
end

function fit!(learner::AbstractLearner, epochs::Integer)
    # TODO: the user can be able to specify multiple parameter
    # groups for differential learning rates
    handler = CallbackHandler(learner)
    handler(Phases.InitializationPhase(), Events.BeforeFitEvent())
    # callback here - BeforeFitEvent
    parameters = Flux.params(model(learner))
    for epoch_idx in 1:epochs
        train_epoch!(learner, epoch_idx)
        valid_epoch(learner, epoch_idx)
    end
    # callback here - AfterFitEvent
    # callback here - CancelFitEvent
end