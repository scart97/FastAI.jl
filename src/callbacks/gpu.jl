using Flux: gpu

struct CudaCallback <: AbstractCallback end

function handle!(::CudaCallback, learner::AbstractLearner, ::InitializationPhase, ::BeforeFitEvent)
    mod = gpu(model(learner))
    model!(learner, mod)
end

# TODO: change to use CuIterator to free memory
function handle!(::CudaCallback, ::AbstractLearner, ::AbstractPhase, e::BeforePredictEvent)
    e.x = gpu.(e.x)
    e.y = gpu.(e.y)
end    

