struct DummyCallback <: AbstractCallback end

function before_fit(cb::DummyCallback, lrn::AbstractLearner, n_epoch) println("Before Fit") end
function after_fit(cb::DummyCallback, lrn::AbstractLearner) println("After Fit") end
function after_cancel_fit(cb::DummyCallback, lrn::AbstractLearner) println("After Cancel Fit") end
function before_epoch(cb::DummyCallback, lrn::AbstractLearner, epoch) println("Before Epoch") end
function after_epoch(cb::DummyCallback, lrn::AbstractLearner, epoch) println("After Epoch") end
function after_cancel_epoch(cb::DummyCallback, lrn::AbstractLearner, epoch) println("After Cancel Epoch") end
function before_epoch_train(cb::DummyCallback, lrn::AbstractLearner, epoch) println("\nBefore Epoch Train") end
function after_epoch_train(cb::DummyCallback, lrn::AbstractLearner, epoch) println("\nAfter Epoch Train") end
function after_cancel_epoch_train(cb::DummyCallback, lrn::AbstractLearner, epoch) println("\nAfter Cancel Epoch Train") end
function before_epoch_validate(cb::DummyCallback, lrn::AbstractLearner, epoch) println("\nBefore Epoch Validate") end
function after_epoch_validate(cb::DummyCallback, lrn::AbstractLearner, epoch) println("\nAfter Epoch Validate") end
function after_cancel_epoch_validate(cb::DummyCallback, lrn::AbstractLearner, epoch) println("\nAfter Cancel Epoch Validate") end
function before_batch_train(cb::DummyCallback, lrn::AbstractLearner, epoch, batch) print("Before Batch Train,") end
function after_batch_train(cb::DummyCallback, lrn::AbstractLearner, epoch, batch) print("After Batch Train,")  end
function batch_train_loss(cb::DummyCallback, lrn::AbstractLearner, loss, epoch, batch) print("Batch Train Loss = $(loss)")  end
function after_cancel_batch_train(cb::DummyCallback, lrn::AbstractLearner, epoch, batch) print("After Cancel Batch Train")  end
function before_batch_validate(cb::DummyCallback, lrn::AbstractLearner, epoch, batch) print("Before Batch Validate,")  end
function after_batch_validate(cb::DummyCallback, lrn::AbstractLearner, epoch, batch) print("After Batch Validate") end
function batch_validate_loss(cb::DummyCallback, lrn::AbstractLearner, loss, epoch, batch) println("Batch Validate Loss = $(loss)") end
function after_cancel_batch_validate(cb::DummyCallback, lrn::AbstractLearner, epoch, batch) println("After Cancel Batch Validate") end
