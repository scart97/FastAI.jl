using ..Callback.Phases
using ..Callback.Events

struct DummyCallback <: AbstractCallback end

handle_callback!(::DummyCallback, ::AbstractLearner, ::InitializationPhase, ::BeforeFitEvent) = println("Initializing training")

handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforeEpochEvent) = println("Train before epoch")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforePredictEvent) = println("Train before predict")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforeLossEvent) = println("Train before loss")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::AfterLossEvent) = println("Train after loss")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforeUpdateEvent) = println("Train before update")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::CancelBatchEvent) = println("Train cancel batch")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::AfterEpochEvent) = println("Train after epoch")
handle_callback!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::CancelEpochEvent) = println("Train cancel epoch")

handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforeEpochEvent) = println("Validation before epoch")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforePredictEvent) = println("Validation before predict")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforeLossEvent) = println("Validation before loss")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::AfterLossEvent) = println("Validation after loss")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforeUpdateEvent) = println("Validation before update")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::CancelBatchEvent) = println("Validation cancel batch")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::AfterEpochEvent) = println("Validation after epoch")
handle_callback!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::CancelEpochEvent) = println("Validation cancel epoch")

handle_callback!(::DummyCallback, ::AbstractLearner, ::CleanupPhase, ::AfterFitEvent) = println("Cleanup after fit")
handle_callback!(::DummyCallback, ::AbstractLearner, ::CleanupPhase, ::CancelFitEvent) = println("Cleanup cancel fit")