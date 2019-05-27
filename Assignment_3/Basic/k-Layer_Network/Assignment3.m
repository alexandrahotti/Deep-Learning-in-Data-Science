function [m]= Assignment3()
clear;
addpath C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Datasets\cifar-10-batches-mat;

rng(400);


% Loading all availabe training data
[X1, Y1, y1] = LoadBatch('data_batch_1.mat'); %training data
[X2, Y2, y2] = LoadBatch('data_batch_2.mat'); %validation data
[X3, Y3, y3] = LoadBatch('data_batch_3.mat'); %training data
[X4, Y4, y4] = LoadBatch('data_batch_4.mat'); %training data
[X5, Y5, y5] = LoadBatch('data_batch_5.mat'); %training data

% Using 9000 data points for training
Xtr = [X1, X2, X3, X4, X5(:,1:5000)];
Ytr = [Y1, Y2, Y3, Y4, Y5(:,1:5000)];
ytr = [y1; y2; y3; y4; y5(1:5000)];

% and using the last 1000 for validation
Xva = X2(:,5001:10000);
Yva = Y2(:,5001:10000);
yva = y2(5001:10000);

[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data


%%%%%   Checking the analytical gradients against the numerical gradients%
%
% %preprocessing the data to give it a zero mean
no_dim = 3072;
[Xtr, Xva, Xte] = preProcessData(Xtr(1:no_dim,:), Xva(1:no_dim,:), Xte(1:no_dim,:));
[~, n] = size(Xtr);

m = [50,50];%[50, 30, 20, 20, 10, 10, 10, 10]; % 
no_layers = 3;
lambda = 0.005;
K = 10;
eps = 0.1;
h =1e-6;
d = no_dim;

[W_init, b_init] = initilizeWeightsAndBiasesXavier(no_layers, m, K , d);



% NetParameters = NetParams;
% NetParameters.W = W_init;
% NetParameters.b = b_init;
% 
% NetParameters.use_bn = false;
% 
% 
% 
% %[rel_err_gradb,rel_err_gradW] = computeRelativeErrorNumericalAnalyticalGradients(W_init, b_init, Xtr, Xva, Xte, Ytr, lambda, h, eps, m, n, no_dim, no_layers, NetParams );
% 
% %disp('DONE');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % % %%%%%% Excersice 3 - Figure 3 assignment 2 %%%%%%%%%%%%%%%
% 
% % Mini batch gradient descent with cyclical learning
% 
% GDparameters = GDparams;
% GDparameters.n_batch = 100;
% GDparameters.eta_min = 1e-5;
% GDparameters.eta_max = 1e-1;
% GDparameters.eta = 0;
% GDparameters.n_s = 2250;
% GDparameters.l = 0;
% GDparameters.update_step = 0;
% GDparameters.n_epochs = 20;
% no_points = 45000;
% 
% [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers);
% 
% 
% plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 9 layer NN - cyclical learning', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 9 layer NN - cyclical learning', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 9 layer NN - cyclical learning', 'update step', 'accuracy');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar);
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% % %%%%%% Excersice 3 - Figure 4 assignment 2 %%%%%%%%%%%%%%%
% 
% % Mini batch gradient descent with cyclical learning
% 
% GDparameters = GDparams;
% GDparameters.n_batch = 100;
% GDparameters.eta_min = 1e-5;
% GDparameters.eta_max = 1e-1;
% GDparameters.eta = 0;
% GDparameters.n_s = 800;
% GDparameters.l = 0;
% GDparameters.update_step = 0;
% GDparameters.n_epochs = 48;
% no_points = 800*2*3;
% 
% [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W, b, lambda);
% 
% 
% plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 2 layer NN - cyclical learning', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 2 layer NN - cyclical learning', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 2 layer NN - cyclical learning', 'update step', 'accuracy');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar);
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% EXCERCISE 2 - training a 3 neural network %

% Mini batch gradient descent with cyclical learning
% 
% GDparameters = GDparams;
% GDparameters.n_batch = 100;
% GDparameters.eta_min = 1e-5;
% GDparameters.eta_max = 1e-1;
% GDparameters.eta = 0;
% GDparameters.n_s = 5 * 45000 / GDparameters.n_batch;
% GDparameters.l = 0;
% GDparameters.update_step = 0;
% GDparameters.n_epochs = 9*2;
% no_points = 2*9*49000/100;
% 
% [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda);
% 
% 
% plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 3 layer NN - cyclical learning', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 3 layer NN - cyclical learning', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 3 layer NN - cyclical learning', 'update step', 'accuracy');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar);
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% % % EXCERCISE 2 - training a 9 layer neural network %
% 
% % Mini batch gradient descent with cyclical learning
% 
% GDparameters = GDparams;
% GDparameters.n_batch = 100;
% GDparameters.eta_min = 1e-5;
% GDparameters.eta_max = 1e-1;
% GDparameters.eta = 0;
% GDparameters.n_s = 5 * 45000 / GDparameters.n_batch;
% GDparameters.l = 0;
% GDparameters.update_step = 0;
% GDparameters.n_epochs = 20;
% no_points = 45000;
% 
% [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers);
% 
% 
% plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 9 layer NN - cyclical learning', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 9 layer NN - cyclical learning', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 9 layer NN - cyclical learning', 'update step', 'accuracy');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar);
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f=0;


%%%% Sensitivity to initilaization %%%%%%%%%%
GDparameters = GDparams;
GDparameters.n_batch = 100;
GDparameters.eta_min = 1e-5;
GDparameters.eta_max = 1e-1;
GDparameters.eta = 0;
GDparameters.n_s = 2250;
GDparameters.l = 0;
GDparameters.update_step = 0;
GDparameters.n_epochs = 20;
no_points = 45000;

[W_init, b_init] = initilizeWeightsAndBiases(no_layers, m, K , d);
[cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers);


plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 3 layer NN - cyclical learning', 'update step', 'cost');
plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 3 layer NN - 1e-3', 'update step', 'loss');
plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 9 layer NN - cyclical learning', 'update step', 'accuracy');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar);

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


function [grad_J_W, grad_J_b] = ComputeGradients(X, Y, W, b, lambda, no_layers)
% Backwardpass for a k layer network


[~, n] = size(X);
vec_ones = ones(n,1); % Nx1

% Vectors to store results for weights and biases
grad_J_W = cell(no_layers, 1); 
grad_J_b = cell(no_layers,1); 

%Forward pass
[P_batch, h_batch ] = EvaluateClassifier(X, W, b);

%Backward pass
G_batch = -(Y - P_batch);


for l = no_layers: -1 : 2
    
    %The gradiets of J wrt bias vector b_k and W_k
    grad_J_W{l} = (1/n) * G_batch * h_batch{l}' + 2 * lambda * W{l};
    grad_J_b{l} = (1/n) * G_batch * vec_ones;


    % Propagate G_batch to the previous layer
    G_batch = W{l}' * G_batch;
    G_batch = G_batch .* (h_batch{l} > 0);
    

end

%The gradiets of J wrt bias vector b_1 and W_1
grad_J_W{1} = (1/n) * G_batch * h_batch{1}' + 2 * lambda * W{1};
grad_J_b{1} = (1/n) * G_batch * vec_ones;
    
end


function [relative_error_gradb,relative_error_gradW] = computeRelativeErrorNumericalAnalyticalGradients(W,b, Xtr, Xva, Xte, Ytr, lambda, h, eps, m, no_points, no_dim, no_layers, NetParams)
% Computes the relative error between a numerically estimated gradient and
% an analytically computed gradient

relative_error_gradW = cell(no_layers,1); 
relative_error_gradb = cell(no_layers,1);

NetParams.W = W;
NetParams.b = b;

% compute the gradients
[grad_W, grad_b] = ComputeGradients(Xtr, Ytr, W, b, lambda, no_layers);

Grads = ComputeGradsNumSlow(Xtr, Ytr, NetParams, lambda, h);

% compute the relative errors

for i = 1:no_layers
    
relative_error_gradb{i} = abs(grad_b{i} - Grads.b{i}) ./ max(eps, abs(Grads.b{i}) + abs(grad_b{i}));
relative_error_gradW{i} = abs(grad_W{i} - Grads.W{i}) ./ max(eps, abs(Grads.W{i}) + abs(grad_W{i}));

end
end


function [W, b] = initilizeWeightsAndBiasesXavier(no_layers, m, K, d)
% b and W are initilized using Xavier Initilization
mu = 0;

W = cell(no_layers,1); 
b = cell(no_layers,1);

% For the connection between the input layer and the first hidden layer. To get dimensions: m_1 x d 
std = 1 / sqrt( d );
W{1} = std .* randn( m(1), d ) + mu;
b{1} = zeros(m(1),1);


% For connecting hidden layers. To get dimensions such as: m_2 x m_1
if no_layers > 2
    for l = 2 : no_layers-1

        std = 1 / sqrt( m(l) );
        W{l} = std .* randn(m(l), m(l-1)) + mu;
        b{l} = zeros(m(l), 1);

    end
end

% For last hidden layer to get the output dimensions: K x m_end
std = 1 / sqrt( m(end) );
W{end} = std .* randn(K, m(end)) + mu;
b{end} = zeros(K, 1);

end


function [W, b] = initilizeWeightsAndBiases(no_layers, m, K, d)
% b and W are initilized using a normal distribution 
% Sensitivity Initilization

mu = 0;

W = cell(no_layers,1); 
b = cell(no_layers,1);

sig1 = 1e-1;
sig2 = 1e-3;
sig3 = 1e-4;

% For the connection between the input layer and the first hidden layer. To get dimensions: m_1 x d 
std = sig2; 
W{1} = std .* randn( m(1), d ) + mu;
b{1} = zeros(m(1),1);


% For connecting hidden layers. To get dimensions such as: m_2 x m_1
if no_layers > 2
    for l = 2 : no_layers-1
        W{l} = std .* randn(m(l), m(l-1)) + mu;
        b{l} = zeros(m(l), 1);

    end
end

% For last hidden layer to get the output dimensions: K x m_end
    W{end} = std .* randn(K, m(end)) + mu;
    b{end} = zeros(K, 1);

end


function [J, loss] = ComputeCost(X, Y, W, b, lambda)
%Computes the loss and cost J with the amount of regularization given by
%lambda.

[~, n] = size(X);
regularization_term = 0;

% Calculate the regularization term
for i = 1:size(W,1)
   
regularization_term = regularization_term + lambda * sum( sum(W{i}.^2 ));

end

% Classify the given data using the parameters W and B.
[P, ~] = EvaluateClassifier(X, W, b); 

%Compute the loss
l_cross = -log(dot(Y,P));
loss = (sum(l_cross)/n);

%Compute the cost
J = loss + regularization_term;
end


function plotResults(training_vals, validation_vals, x_step, x_max, step_sz , legend_1, legend_2, title_txt, x_label, y_label)
% Plots validation and training values in the same plot 

figure();
x_range = 0:x_step:(size(training_vals,1)-1)*x_step;
disp(size(x_range));

disp(size(training_vals));
disp(size(validation_vals));

plot(x_range', training_vals, x_range', validation_vals);

legend(legend_1,legend_2);
title(title_txt)
xlabel(x_label)
ylabel(y_label)
hold on;

end


function [W, Xtr, Ytr, ytr, Xte, Yte, yte, Xva, Yva, yva] = reduceDimData(W, Xtr, Ytr, ytr, Xte, Yte, yte, Xva, Yva, yva, no_points, no_dim)
% Reduces the dimensionality of all data wrt d and n. Where d = data
% dimensions and n = no data points. For reference: X: dxn

W = {W{1}(:,1:no_dim),W{2}};

Xtr = Xtr(1:no_dim,1:no_points);
Ytr = Ytr(:,1:no_points);
ytr = ytr(1:no_points);

Xva = Xva(1:no_dim,1:no_points);
Yva = Yva(:,1:no_points);
yva = yva(1:no_points);

Xte = Xte(1:no_dim,1:no_points);
Yte = Yte(:,1:no_points);
yte = yte(1:no_points);

end


function [Xtr, Xva, Xte] = preProcessData(Xtr,Xva,Xte)
% Preprocesses the data by giving it a zero mean.

mean_X = mean(Xtr, 2); 
std_X = std(Xtr, 0, 2);

Xtr = Xtr - repmat(mean_X, [1,size(Xtr,2)]);
Xtr = Xtr ./ repmat(std_X, [1, size(Xtr, 2)]);

Xva = Xva - repmat(mean_X, [1,size(Xva,2)]);
Xva = Xva ./ repmat(std_X, [1, size(Xva, 2)]);

Xte = Xte - repmat(mean_X, [1,size(Xte,2)]);
Xte = Xte ./ repmat(std_X, [1, size(Xte, 2)]);

end


function [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, W, b] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparams, W, b, lambda, no_layers)
% This function performs gradient descent on minibatches using cyclical learning and run through
% the the training images the number of times given by: GDparams.n_epochs


% Arrays used to store costs, losses and accuracies
cost_training = zeros(GDparams.n_epochs+1, 1);
cost_validation = zeros(GDparams.n_epochs+1, 1);

loss_training = zeros(GDparams.n_epochs+1, 1);
loss_validation = zeros(GDparams.n_epochs+1, 1);

acc_training = zeros(GDparams.n_epochs+1, 1);
acc_validation = zeros(GDparams.n_epochs+1, 1);


for i = 1:GDparams.n_epochs
    
    % Compute costs and losses on the training data
    [cost_tr, loss_tr] = ComputeCost(Xtr, Ytr, W, b, lambda);
    
    cost_training(i) = cost_tr;
    loss_training(i) = loss_tr;
    
    % Compute costs and losses on the validation data
    [cost_val, loss_val]  = ComputeCost(Xva, Yva, W, b, lambda);
    
    cost_validation(i) = cost_val;
    loss_validation(i) = loss_val;
    
    % Compute accuracies on the validation & training data
    acc_training(i) = ComputeAccuracy(Xtr, ytr, W, b);
    acc_validation(i) = ComputeAccuracy(Xva, yva, W, b);
    
    % Update the parameters for a new epoch.
    [W, b, GDparams] = MiniBatchGDCyclicalLearning(Xtr, Ytr, GDparams, W, b, lambda,no_layers);
    
    
    if i == GDparams.n_epochs
        
        [cost_tr, loss_tr] = ComputeCost(Xtr, Ytr, W, b, lambda);
        cost_training(i+1) = cost_tr;
        loss_training(i+1) = loss_tr;

        [cost_val, loss_val]  = ComputeCost(Xva, Yva, W, b, lambda);
        cost_validation(i+1) = cost_val;
        loss_validation(i+1) = loss_val;

        acc_training(i+1) = ComputeAccuracy(Xtr, ytr, W, b);
        acc_validation(i+1) = ComputeAccuracy(Xva, yva, W, b);
    end
    
    % The data points are shuffled after every epoch.
    [ Xtr, Ytr, ytr ] = shuffleDataPoints(Xtr, Ytr, ytr);
end

end


function[ X_shuffled, Y_shuffled, y_shuffled ] = shuffleDataPoints(X, Y, y)
% Uniformly shuffles the data by shuffling columns of a matrix
% here we shuffle X which has the form dxN.

new_indicies_shuffled = randperm(size(X,2));

X_shuffled = X(:,new_indicies_shuffled);
Y_shuffled = Y(:,new_indicies_shuffled);
y_shuffled = y(new_indicies_shuffled);

end


function [Wstar, bstar,GDparams ] = MiniBatchGDCyclicalLearning(X, Y, GDparams, W, b, lambda, no_layers)
%Performs gradient descent on batches of data with a cyclical learning
%rate.

[~, N] = size(X);

    for j=1:N/GDparams.n_batch
        
        % Get the indicies of the data for the current minibatch.
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        
        
        % Compute the gradients of the parameters being learned.
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda, no_layers);
        
        %Increase the update step
        GDparams.update_step = GDparams.update_step + 1;
        
        % The cyclical learning rate is updated
        [ eta_t ] = updateCyclicalLearningRate(GDparams);
        GDparams.eta = eta_t;
        
       
        % We want to update the l parameter to keep track of the current
        % cycle.
        if mod( GDparams.update_step, GDparams.n_s*2 ) == 0
            GDparams.l = GDparams.l + 1;
        end
        
        % The weights and biases are updated.
         for i = 1:no_layers
            b{i} = b{i} - GDparams.eta * grad_b{i};
            W{i} = W{i} - GDparams.eta * grad_W{i};
         end
    end
    
    % The final b and W trained 
    bstar = b;
    Wstar = W;
end


function [eta_t] = updateCyclicalLearningRate(GDparams)
% Updates the cyclical learning rate at time t.

eta_t = double(0);

% Check if t is within a lower and upper range. I.e. If nt is increasing. 
lower1 = (2*GDparams.l*GDparams.n_s) <= GDparams.update_step;
upper1 = GDparams.update_step <= ((2*GDparams.l+1)*GDparams.n_s) ;

if lower1 && upper1
    eta_t = GDparams.eta_min + ((GDparams.update_step - 2*GDparams.l*GDparams.n_s)/( GDparams.n_s ))*(GDparams.eta_max - GDparams.eta_min);
end



% We check if t is within another lower and upper range. I.e. if nt is deacreasing.
upper2 = GDparams.update_step >= (2 * GDparams.l + 1)*GDparams.n_s ;
lower2 = GDparams.update_step <= 2*(GDparams.l + 1)*GDparams.n_s ;

if upper2 && lower2
    eta_t = GDparams.eta_max - ((GDparams.update_step - (2*GDparams.l+1)*GDparams.n_s)/GDparams.n_s)*(GDparams.eta_max - GDparams.eta_min);
end
end


function ComputeAndPlotRelativeError(X, Y, batch_sizes, W, b, lambda, eps)
% Computes and creates boxplots of the relative errors between the analytical and 
% the numerical gradients for b and W for the mini batch sizes given in the vector
% batch_sizes.

[K, ~] = size(b);
[K, d] = size(W);

h = 1e-5;
no_batches = length(batch_sizes);

gradients_rel_errors_b = zeros (K, no_batches);
gradients_rel_errors_W = zeros (K*d, no_batches);

for i = 1:no_batches
    
    [grad_W, grad_b] = ComputeGradients(Xtr, Ytr, Wtemp, b, lambda);
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtr, Ytr, Wtemp, b, lambda, h);
    
    relative_error_gradb1 = abs(grad_b{1} - ngrad_b{1}) ./ max(eps, abs(ngrad_b{1}) + abs(grad_b{1}));
    relative_error_gradW1 = abs(grad_W{1} - ngrad_W{1}) ./ max(eps, abs(ngrad_W{1}) + abs(grad_W{1}));

    relative_error_gradb2 = abs(grad_b{2} - ngrad_b{2}) ./ max(eps, abs(ngrad_b{2}) + abs(grad_b{2}));
    relative_error_gradW2 = abs(grad_W{2} - ngrad_W{2}) ./ max(eps, abs(ngrad_W{2}) + abs(grad_W{2}));
    


%     relative_error_gradW = reshape(relative_error_gradW, K*d,1);
%     
%     gradients_rel_errors_b(:, i) = relative_error_gradb;
%     gradients_rel_errors_W(:, i) = relative_error_gradW;
%     
%     
%     max_grad_b_relative_error = max(relative_error_gradb);
%     max_grad_W_relative_error = max(relative_error_gradW);
%     
%     min_grad_b_relative_error = min(relative_error_gradb);
%     min_grad_W_relative_error = min(relative_error_gradW);
% 
%     mean_grad_b_relative_error = mean(relative_error_gradb);
%     mean_grad_W_relative_error = mean(relative_error_gradW);
    
end

boxplot(gradients_rel_errors_W, batch_sizes);
title('Relative error between numerical and analytical gradient for W1')
xlabel('Batch size')
ylabel('relavtive error')

% 
% boxplot(gradients_rel_errors_b, batch_sizes);
% title('Relative error between numerical and analytical gradient for b1')
% xlabel('Batch size')
% ylabel('relavtive error')
% 
% 
% boxplot(gradients_rel_errors_W, batch_sizes);
% title('Relative error between numerical and analytical gradient for W1')
% xlabel('Batch size')
% ylabel('relavtive error')
% 
% boxplot(gradients_rel_errors_b, batch_sizes);
% title('Relative error between numerical and analytical gradient for b1')
% xlabel('Batch size')
% ylabel('relavtive error')

end


function[X, Y, y] = LoadBatch(filename)
% Loads a batch of data and creates one hot encoded labels corresponding to
% the labels of the data.

A = load(filename);

X = double(A.data');
%X = X / 255;

y = A.labels;
y = y + 1 ;
Y = oneHotEncoder(y);
end


function[Y] = oneHotEncoder(y)
%Creates a matrix of one hot encoded data from an array of labels.

N = length(y);
K = max(y) ;
Y = zeros(K, N);

for i = 1 : K
    dp_current_label = find( i == y);
    Y(i, dp_current_label) = 1; 
end
end


function acc = ComputeAccuracy(X, y, W, b)
%Evaluates the accuracy of trained Weights and bias parameters

% Compute the probabilities for the datapoints X. 
[P, ~ ] = EvaluateClassifier(X, W, b);

[total_no_classifications, ~] = size(y);

% Compute the total number of correct classifications.
[~, I] = max(P,[],1); 
[~, no_accurate_classifications] = size(find(I==y'));

% Compute the accuracy.
acc = no_accurate_classifications/total_no_classifications;
end



function [P, h] = EvaluateClassifier(X, W, b)
%Evaluates a k-layer network via a forwardpass.

[~, n] = size(X);
k = size(W, 1); 

% Cells used to store intermediary results.
s = cell(k,1);
h = cell(k,1);

h{1} = X; 

% Forwardpass through all layers except the last.
for l = 1:k-1
    s{l} = W{l} * h{l} + repmat( b{l}, [1, n] );
    h{l+1} = max( 0, s{l} ); 
end

% Forwardpass through last layers without activation.
s{k} = W{k}*h{k} + repmat( b{k}, [1, n] );

% Softmax is applied to the last layers to get probabilities.
P = softmax(s{k});
end


function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)
% Used to numerically compute gradients to compare to the analytical
% garidents.

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
            
            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda);
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end