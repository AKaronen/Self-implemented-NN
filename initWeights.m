function weights = initWeights(inputs, outputs)
    weights = (rand(inputs+1,outputs)-0.5) / 10; % small random initial weights
end