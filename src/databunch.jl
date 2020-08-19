"""
    DataBunch
    DataBunch(train::Flux.Data.DataLoader, valid::Flux.Data.DataLoader, n_inp::Integer)

A `DataBunch` is a bunched `train` and `valid` (validation) dataloader.
"""
struct DataBunch
    train::Flux.Data.DataLoader
    valid::Flux.Data.DataLoader
    n_inp::Integer
end

"""
    estimate_number_inp(db::Flux.Data.DataLoader)

Estimate the number of inputs in the batch as all but the last element.
"""
estimate_number_inp(dl::Flux.Data.DataLoader) = length(one_batch(dl)) - 1

"""
    DataBunch(train::Flux.Data.DataLoader, valid::Flux.Data.DataLoader)

Creates a DataBunch from train and validation dataloaders.
"""
DataBunch(train::Flux.Data.DataLoader, valid::Flux.Data.DataLoader) = DataBunch(train, valid, estimate_number_inp(train))

"""
    n_inp(db::DataBunch)

Get the `db.n_inp` number of inputs in the batch.
"""
n_inp(db::DataBunch) = db.n_inp

"""
    train(db::DataBunch)

Get the `db.train` dataloader.
"""
train(db::DataBunch) = db.train
"""
    valid(db::DataBunch)

Get the `db.valid` (validation) dataloader.
"""
valid(db::DataBunch) = db.valid