using ..Callback.Phases
using ..Callback.Events

struct DummyCallback <: AbstractCallback end

handle!(::DummyCallback, ::AbstractLearner, ::InitializationPhase, ::BeforeFitEvent) = println("Initializing training")

handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforeEpochEvent) = println("Train before epoch")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforePredictEvent) = println("Train before predict")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforeLossEvent) = println("Train before loss")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::AfterLossEvent) = println("Train after loss")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::BeforeUpdateEvent) = println("Train before update")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::CancelBatchEvent) = println("Train cancel batch")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::AfterEpochEvent) = println("Train after epoch")
handle!(::DummyCallback, ::AbstractLearner, ::TrainPhase, ::CancelEpochEvent) = println("Train cancel epoch")

handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforeEpochEvent) = println("Validation before epoch")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforePredictEvent) = println("Validation before predict")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforeLossEvent) = println("Validation before loss")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::AfterLossEvent) = println("Validation after loss")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::BeforeUpdateEvent) = println("Validation before update")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::CancelBatchEvent) = println("Validation cancel batch")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::AfterEpochEvent) = println("Validation after epoch")
handle!(::DummyCallback, ::AbstractLearner, ::ValidationPhase, ::CancelEpochEvent) = println("Validation cancel epoch")

handle!(::DummyCallback, ::AbstractLearner, ::CleanupPhase, ::AfterFitEvent) = println("Cleanup after fit")
handle!(::DummyCallback, ::AbstractLearner, ::CleanupPhase, ::CancelFitEvent) = println("Cleanup cancel fit")