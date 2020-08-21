"`Callback` that tracks the epoch and batch and calculates progress (fraction done)"
mutable struct ProgressCallback <: AbstractCallback
    epoch::Int
    batch::Int
    n_batch::Int
    n_epoch::Int
end

ProgressCallback() = ProgressCallback(0, 0, 0, 0)

function before_fit(tecb::ProgressCallback, learn::AbstractLearner, n_epoch)
    tecb.n_epoch = n_epoch
    tecb.n_batch = 0
    tecb.epoch = 0
    tecb.batch = 0
end

function after_epoch_train(tecb::ProgressCallback, lrn::AbstractLearner, epoch)
    tecb.n_batch = batch
    tecb.epoch = epoch
end

function after_batch_train(tecb::ProgressCallback, lrn::AbstractLearner, epoch, batch)
    tecb.batch = batch
end

function after_batch_validate_(tecb::ProgressCallback, lrn::AbstractLearner, loss, epoch, batch)
    println("Loss=$(loss) Amount trained = $(progress(tecb))")
end

function progress(tecb::ProgressCallback) 
    num = ((tecb.n_epoch - 1) * tecb.n_batch + tecb.batch)
    deom = float(tecb.n_epoch * tecb.n_batch)
    return denom > 0.0 ? num / denom : 0.0
end

