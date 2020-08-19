using Flux: @functor,mse

"A simple DataBunch where `x` is random and `y = a*x + b` plus some noise."
function synth_dbunch(;a=2, b=3, bs=16, n_train=10, n_valid=2)

    function get_data(n)
        xy = [] #Array{Float32}(undef,bs*n,2)
        for i in 1:bs*n  
            x = rand()
            y = a*x .+ b + 0.1*rand()
            #xy[i,1] = x
            #xy[i,2] = y
            push!(xy,([x],[y]))
        end
        xy = hcat.(xy...)
        return Flux.Data.DataLoader(xy, batchsize=bs, shuffle=true) 
    end

    train_dl = get_data(n_train)
    valid_dl = get_data(n_valid)
    return DataBunch(train_dl, valid_dl)
end

function synth_learner(n_train=10, n_valid=2)
    data = synth_dbunch(n_train=n_train,n_valid=n_valid)
    return Learner(data, Dense(1,1), loss=mse, opt=Descent(0.001))
end

function one_batch(dl::Flux.Data.DataLoader)
    for d in dl
        return d
    end
end

function foo()
    learn = synth_learner()

    model = RegModel()
    @show model
    ps = params(model)
    @show ps

    xb,yb = learn |> data_bunch |> train |> one_batch
    @show xb[1]
    @show yb[1]

    gs = gradient(ps) do 
        Flux.mse(model(xb[1]),yb[1])
    end
    @show gs
    update!(Descent(0.1), ps, gs)
    @show model
end

function bah()
    learn = synth_learner()
    model = Dense(1,1)
    @show model.W
    p = params(model)
    @show p
    ps = Params(p)
    @show ps
    xb,yb = learn |> data_bunch |> train |> one_batch
    @show xb[1]
    @show yb[1]
    gs = gradient(ps) do
        Flux.mse(model([xb[1]]),yb[1])
    end
    @show gs
    update!(Descent(0.1), ps, gs)
    @show model.W
end

"Woooo"
function exercise()
    #println("Yow")
    learn = synth_learner()
    add_cb!(learn,Recorder(add_time=true, train_metrics=false, valid_metrics=true, alpha=0.98))
    
    xys = learn |> data_bunch |> train |> one_batch
    lf = loss(learn)
    mf = model(learn)

    function lff(xy)
        x = xy[1]
        p = mf(x)[1]
        y = xy[2]
        #@show ("Woo",x,p,y)
        return lf(p,y)
    end
    
    init_loss = sum(lff.(xys))
    fit!(learn,16)
    final_loss = sum(lff.(xys))
    
    println(init_loss)
    println(final_loss)
    @assert final_loss < init_loss
end
