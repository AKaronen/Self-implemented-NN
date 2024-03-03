function [Loss, trainedWeights, acc, pred] = NN(traindata, trainclass, validation, layers, maxEpochs, activation_fun, rho, verbose)
% Implementation of a fully connected neural network.
% INPUT VARIABLES:
% TRAINDATA = MxN matrix where M is the number of samples and N is the
% number of features
% TRAINCLASS = Mx1 matrix of numeric class labels. Class labels are one-hot
% encoded for the training process.
% VALIDATION = Mx(N+1) matrix which contains the validation data in the 
% first N columns and the class labels in the last column.
% LAYERS = A vector which determines all sizes of layers
% MAXEPOCHS = Maximum number of training iterations
% ACTIVATION_FUN = A string from the set of implemented activation
% functions i.e. {'relu', 'sigmoid', 'tanh'}
% RHO = Learning rate parameter, if not specified default value is 1e-5.
% VERBOSE = Boolean value to opt out of verbose learning process (default True) 

    % Default values
    if ~exist('maxEpochs', 'var')
        maxEpochs = 50000;
    end
    
    if ~exist('verbose', 'var')
       verbose = true;
    end
    
    if ~exist('activation_fun', 'var')
        type = 'sigmoid';
    else
        type = activation_fun;
    end
    
    if ~exist('rho', 'var')
        rho  = 1e-5; % learning rate
    end
    
    % Initialize some useful parameters
    X = traindata;
    num_layers = size(layers,2);
    num_labels = layers(end);
    M = size(X, 1); % # samples
    N = size(X, 2); % # features
    input = [ones(M, 1) X]; % Extended input
    
    
    for i = 1:num_layers
        hidden{i} = zeros(M, layers(i)+1);
        activations{i} = zeros(M,layers(i)+1);
    end
    activations{1} = input;
    
    
    % Initialisation
    train_J = zeros(1,maxEpochs); % loss function value vector initialisation
    val_J = zeros(1,maxEpochs); % loss function value vector initialisation
    train_acc = zeros(1,maxEpochs);  % train accuracy vector initialisation
    val_acc = zeros(1,maxEpochs);  % validation accuracy vector initialisation
    
    
    Y = zeros(num_labels,M);
    for i = 1:M
        Y(trainclass(i), i) = 1; % One hot encoding of class labels
    end
    
    for i = 1:(num_layers-1) % initialize weights
        weights{i} = initWeights(layers(i), layers(i+1));
    end


    if ~isempty(validation)
            X_val = validation(:,1:end-1);
            valclass = validation(:,end);
            M_val = size(X_val, 1);

            Y_val = zeros(num_labels,M_val);
            for i = 1:M_val
                Y_val(valclass(i),i) = 1; % One hot encoding of class labels
            end

            for i = 1:num_layers
                val_hidden{i} = zeros(M_val, layers(i)+1);
                val_activations{i} = zeros(M_val, layers(i)+1);
            end

            val_activations{1} = [ones(M_val, 1) X_val]; 
    end
    
    if verbose
        figure;
        ax1 = subplot(2,1,1);
        ax2 = subplot(2,1,2);
    end
    t = 0;

    model_index = 0;
    last_ten = {1,50};
    while 1 % iterative training "forever"
        t = t+1;
        for i = 2:num_layers
            w = weights{i-1};
            x = activations{i-1};
            % Feed-forward operation
            hidden{i} = x*w;
            activations{i} = activation_function(hidden{i}, type);
    
            if i ~= num_layers
                bias = ones(size(hidden{i},1),1);
                activations{i} = [bias activations{i}]; % extended activations
            end
        end

        %Test with the testing/validation data
        if ~isempty(validation)
            for i = 2:num_layers
                w = weights{i-1};
                x = val_activations{i-1};
                % Feed-forward operation
                val_hidden{i} = x*w;
                val_activations{i} = activation_function(val_hidden{i}, type);

                if i ~= num_layers
                    bias = ones(size(val_hidden{i},1),1);
                    val_activations{i} = [bias val_activations{i}]; % extended activations
                end
            end
            [~, val_pred] = max(val_activations{end}, [], 2);
            val_acc(t) = mean(double(valclass == val_pred));
            val_J(t) = 0.5*sum((Y_val - val_activations{end}').^2, 'all');
        end

        
        [~, pred] = max(activations{end}, [], 2);
        train_acc(t) = mean(double(trainclass == pred')); % training accuracy
        train_J(t) = 0.5*sum((Y - activations{end}').^2, 'all'); % loss function evaluation (MSE)
    
        if (mod(t, 500) == 0) && verbose % Plot learning process for every 500 epochs
            index = 0:100:t;
            index(1) = 1;
            if ~isempty(validation)
                acc = [train_acc; val_acc];
                J = [train_J; val_J];
                plotLearning(J(:,index),index, acc(:,index), [ax1,ax2], maxEpochs, verbose)
            else
                acc = [train_acc];
                J = [train_J];
                plotLearning(J(:,index),index, acc(:,index), [ax1,ax2], maxEpochs, verbose)
            end
            
        end
    
        if train_J(t) < eps % the learning is good enough
            break;
        end
    
        if t == maxEpochs % too many epochs would be done
            break;
        end
    
        if t > 1 % this is not the first epoch
            if norm(train_J(t) - train_J(t-1)) < sqrt(eps) % the improvement is small enough
                break;
            end
        end
        
        
        % Back propagation
        for i = num_layers:-1:2
            if i == num_layers
                sigma{i} = activations{i} - Y';
            else
                %
                if strcmp(type, 'sigmoid')
                    grad = activation_function(hidden{i}, type).*(1-activation_function(hidden{i}, type));
                    sigma{i} = (sigma{i+1}*weights{i}(2:end, :)') .* grad; % sigmoid gradient = s(x)*(1-s(x))
                elseif strcmp(type, 'relu')
                    grad = zeros(size(hidden{i}));
                    grad(hidden{i}>=0) = 1; % relu gradient = (1 | x >= 0 and 0 | x<0)
                    sigma{i} = (sigma{i+1}*weights{i}(2:end, :)') .* grad;
                else
                    grad = 1-activation_function(hidden{i},type).^2; % tanh gradient = 1-tanh(x)^2
                    sigma{i} = (sigma{i+1}*weights{i}(2:end, :)') .* grad;  
                end
                
            end
            
        end
        % deltas
        for i = 1:(num_layers-1)
            delta{i} = sigma{i+1}'*activations{i};
        end
        
    
    
        for i = 1:(num_layers-1) 
    
            weight_grad{i} = delta{i};
            
            newWeights{i} = zeros(size(weights{i}));
    
            dOut = - rho*mean(weight_grad{i}(1,:),1)';
            dHid = - rho*weight_grad{i}(2:end,:)';
    
            newWeights{i}(:,1) = weights{i}(:,1) + dOut;
            newWeights{i}(:,2:end) = weights{i}(:,2:end) + dHid;
            
        end
        weights = newWeights;
        model_index = model_index + 1;
        if mod(t,51) == 0
            model_index = 1;
        end
        last_ten{model_index} = weights;

        if t>50
            if all(train_J(t-50:t)-val_J(t-50:t)<0)
                newWeights = last_ten{1};
                break;
            end
        end


    end

    trainedWeights = newWeights;
    Loss = [train_J;val_J];
    acc = [train_acc; val_acc];
    if ~isempty(validation)
        pred = val_pred;
    end
end


function y = activation_function(x, type)
    % activation functions
    if strcmp(type, 'sigmoid')
        y = 1./(1+exp(-x));
    elseif strcmp(type, 'relu')
        y = max(0,x);
    elseif strcmp(type, 'tanh')
        y = tanh(x);
    end
end

