function Assignment3()
clear;
addpath C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Datasets\cifar-10-batches-mat;

rng(13);


% Loading all availabe training data
[X1, Y1, y1] = LoadBatch('data_batch_1.mat'); %training data
[X2, Y2, y2] = LoadBatch('data_batch_2.mat'); %validation data
[X3, Y3, y3] = LoadBatch('data_batch_3.mat'); %training data
[X4, Y4, y4] = LoadBatch('data_batch_4.mat'); %training data
[X5, Y5, y5] = LoadBatch('data_batch_5.mat'); %training data

% Using 45000 data points for training
Xtr = [X1, X2, X3, X4, X5(:,1:5000)];
Ytr = [Y1, Y2, Y3, Y4, Y5(:,1:5000)];
ytr = [y1; y2; y3; y4; y5(1:5000)];

% Using 5000 data points for validation
Xva = X5(:,5001:10000);
Yva = Y5(:,5001:10000);
yva = y5(5001:10000);

[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data



%%%%% Data and hyperparameter initilization and preprocessing %%%%%
no_dim = 3072;
[Xtr, Xva, Xte] = preProcessData(Xtr(1:no_dim,:), Xva(1:no_dim,:), Xte(1:no_dim,:));
[~, n] = size(Xtr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   Checking the analytical gradients against the numerical gradients%
% 
% 
% NetParameters = NetParams;
% NetParameters.W = W_init;
% NetParameters.b = b_init;
% 
% NetParameters.use_bn = true;
% no_points = 10000;
% batch_size = 100;

% 
% alpha = 0;

% [relative_error_gradb,relative_error_gradW,relative_error_gradga,relative_error_gradbe] = computeRelativeErrorNumericalAnalyticalGradients(W_init, b_init, Xtr, Xva, Xte, Ytr, lambda, h, eps, m, no_points, no_dim, no_layers, NetParams, batch_size, mu, variance, alpha, betas_init, gammas_init, true);
% %[rel_err_gradb,rel_err_gradW] = computeRelativeErrorNumericalAnalyticalGradients(W_init, b_init, Xtr, Xva, Xte, Ytr, lambda, h, eps, m, n, no_dim, no_layers, NetParams );
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% %%%% A 3 layer Neural Network with Batch Normalization %%%%%%%%%%%%%%%%%%%%
% 
% 
% m =  [50, 50]%[50, 30, 20, 20, 10, 10, 10, 10];%[50,30,20,20,10,10,10,10]; % %[50, 50];%
% no_layers = 3;
% lambda = 0.005;
% K = 10;
% eps = 0.1;
% h =1e-6;
% d = no_dim;
% 
% [W_init, b_init] = initilizeWeightsAndBiases(no_layers, m, K , d);
% [mu, variance] = initilizeMuVarAVG(m, no_layers);
% [betas_init, gammas_init] = initilizeBetasAndGammas(no_layers, m);
% 
% 
% 
% 
% %%%%%%%% Parametersetting on page 8 in the assignment description %%%%%%
% GDparameters = GDparams;
% GDparameters.n_batch = 100;
% GDparameters.eta_min = 1e-5;
% GDparameters.l = 0;
% GDparameters.update_step = 0;
% GDparameters.eta_max = 1e-1;
% GDparameters.eta = 0;
% GDparameters.n_s = 5 * 45000/GDparameters.n_batch;
% GDparameters.n_epochs = 20;
% alpha = [0, 0.9];
% no_points = 45000;
% 
% [betas_init, gammas_init] = initilizeBetasAndGammas(no_layers, m);
% 
% [mu_av, var_av] = initilizeMuVarAVG(m, no_layers);
% 
% [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar, gammasstar, betastar, mu_av, var_av] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers,betas_init, gammas_init, alpha, mu_av, var_av);
% 
% plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 9 layer NN - cyclical learning & BN', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 9 layer NN - cyclical learning & BN', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 9 layer NN - cyclical learning & BN', 'update step', 'accuracy');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar, gammasstar, betastar, GDparameters.n_batch , mu_av, var_av);
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%% A 9-layer Neural Network with Batch Normalization %%%%%%%%%%%%%%%%%%%%


% m = [50, 30, 20, 20, 10, 10, 10, 10];%[50,30,20,20,10,10,10,10]; % %[50, 50];%
% no_layers = 9;
% lambda = 0.005;
% K = 10;
% eps = 0.1;
% h =1e-6;
% d = no_dim;
% 
% [W_init, b_init] = initilizeWeightsAndBiases(no_layers, m, K , d);
% [mu, variance] = initilizeMuVarAVG(m, no_layers);
% [betas_init, gammas_init] = initilizeBetasAndGammas(no_layers, m);
% 
% 
% 
% 
%%%%%% Parametersetting on page 8 in the assignment description %%%%%%
% GDparameters = GDparams;
% GDparameters.n_batch = 100;
% GDparameters.eta_min = 1e-5;
% GDparameters.l = 0;
% GDparameters.update_step = 0;
% GDparameters.eta_max = 1e-1;
% GDparameters.eta = 0;
% GDparameters.n_s = 5 * 45000/GDparameters.n_batch;
% GDparameters.n_epochs = 20;
% alpha = [0, 0.9];
% no_points = 45000;
% 
% [betas_init, gammas_init] = initilizeBetasAndGammas(no_layers, m);
% 
% [mu_av, var_av] = initilizeMuVarAVG(m, no_layers);
% 
% [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar, gammasstar, betastar, mu_av, var_av] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers,betas_init, gammas_init, alpha, mu_av, var_av);
% 
% plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 9 layer NN - cyclical learning & BN', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 9 layer NN - cyclical learning & BN', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 9 layer NN - cyclical learning & BN', 'update step', 'accuracy');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar, gammasstar, betastar, GDparameters.n_batch , mu_av, var_av);
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%% Coarse to Fine Grid Search to Find set lambda for a 3 layer network%%%
% %%%% A 3 layer Neural Network with Batch Normalization %%%%%%%%%%%%%%%%%%%%
% 
% 
m = [60, 50, 40];
no_layers = 4;
lambda = 0.005;
K = 10;
eps = 0.1;
h =1e-6;
d = no_dim;

%[W_init, b_init] = HEInitilizeWeightsAndBiases(no_layers, m, K , d);
[W_init, b_init] = initilizeWeightsAndBiases_sensitivity(no_layers, m, K , d);
[mu, variance] = initilizeMuVarAVG(m, no_layers);
[betas_init, gammas_init] = initilizeBetasAndGammas(no_layers, m);

% %%%%%%%% Parametersetting on page 8 in the assignment description %%%%%%
GDparameters = GDparams;
GDparameters.n_batch = 100;
GDparameters.eta_min = 1e-5;
GDparameters.l = 0;
GDparameters.update_step = 0;
GDparameters.eta_max = 1e-1;
GDparameters.eta = 0;
GDparameters.n_s = 5 * 45000/GDparameters.n_batch;
GDparameters.n_epochs = 20;
alpha = [0, 0.9];
no_points = 45000;



[cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar, gammasstar, betastar, mu_av, var_av] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers,betas_init, gammas_init, alpha, mu, variance);

plotResults(cost_training, cost_validation, 100, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 9 layer NN - cyclical learning & BN', 'update step', 'cost');
plotResults(loss_training, loss_validation, 100, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 9 layer NN - cyclical learning & BN', 'update step', 'loss');
plotResults(acc_training, acc_validation, 100, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 9 layer NN - cyclical learning & BN', 'update step', 'accuracy');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar, gammasstar, betastar, GDparameters.n_batch , mu_av, var_av);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% % %%%% Coarse Grid Search %%%%%%%%%
% % l_min = -4;
% % l_max = -1;
% % no_runs = 20;
% % GridSearch(Xte, yte, Xtr, Ytr, ytr, Xva, Yva, yva, no_runs, GDparameters, l_min, l_max, b_init, W_init,no_layers, betas_init, gammas_init, alpha, mu_av, var_av)
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % %%%% Fine Grid Search %%%%%%%%%
% % l_min = -2.884626952;
% % l_max = -2.23064548;
% % no_runs = 20;
% % GridSearch(Xte, yte, Xtr, Ytr, ytr, Xva, Yva, yva, no_runs, GDparameters, l_min, l_max, b_init, W_init,no_layers, betas_init, gammas_init, alpha, mu_av, var_av)
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%% Finer Grid Search %%%%%%%%%
% l_min = -2.612272498000000;%-2.562272498;%-2.692294055;
% l_max = -2.499724355000000;%-2.549724355;%-2.492294055;
% no_runs = 150;
% GridSearch(Xte, yte, Xtr, Ytr, ytr, Xva, Yva, yva, no_runs, GDparameters, l_min, l_max, b_init, W_init,no_layers, betas_init, gammas_init, alpha, mu, variance);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





end


function GridSearch(Xte, yte, Xtr, Ytr, ytr, Xva, Yva, yva, no_runs,GDparameters, l_min, l_max, b_init, W_init,no_layers, betas_init, gammas_init, alpha, mu_av, var_av)
% Trains a k layer netural network using mini batch gradient descent, a
% cyclical learning rate and batch normalization for a number of lambdas given by: no_runs.

result_tables = cell(no_runs,1);

% no runs controlls how many lambdas we want to train the network for
for i = 1:no_runs
    
    % Drawing a lambda value
    l = l_min + ( l_max - l_min ) * rand(1,1);
    lambda = 10^l;
    results_grid_search(1,i) = lambda;
    
    % Gradient Descent 

    [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar, gammasstar, betastar, mu_av, var_av] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda, no_layers,betas_init, gammas_init, alpha, mu_av, var_av);
   
    
    acc_test = ComputeAccuracy(Xte, yte, Wstar, bstar, gammasstar, betastar, GDparameters.n_batch , mu_av, var_av);
%     % Storing the lambda and its loss, cost and accuracy
%     lambda_vec = zeros(size(cost_training,1),1);     
%     lambda_vec(1) = lambda;
%    
%     T = table(lambda_vec, cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation);
%     writetable(T,strcat('tabledata', int2str(i), '.txt'),'Delimiter',';');
%     result_tables{i} = table_name;

    end
end


function [mu_av, var_av] = initilizeMuVarAVG(m, no_layers)
% The moving averages for mu and the variances used during the batch
% normalization ares initilized as zero.

mu_av = cell( no_layers-1, 1); 
var_av = cell( no_layers-1, 1);

for i = 1:no_layers-1
    mu_av{i} = zeros(m(i),1);
    var_av{i} = zeros(m(i),1);
end
end


function [betas, gammas] = initilizeBetasAndGammas(no_layers, m)
% The scales is initilized to one and the shift to zero.

% We do not normalize the weights at the last layer. We want the last layer
% to learn the statistics of the data. Therefore we initilize values for:
% no_layers-1  number of layers.

betas = cell( no_layers-1, 1); 
gammas = cell( no_layers-1, 1);

    for l = 1 : no_layers-1
        betas{l} = zeros( m(l), 1);
        gammas{l}= ones( m(l), 1);
    end
    
end

function [W, b] = HEInitilizeWeightsAndBiases(no_layers, m, K, d)
% b and W are initilized using a normal distribution 
% HE initilization.

mu = 0;

W = cell(no_layers,1); 
b = cell(no_layers,1);

% For the connection between the input layer and the first hidden layer. To get dimensions: m_1 x d 
std = sqrt( 2 /  d );
W{1} = std .* randn( m(1), d ) + mu;
b{1} = zeros(m(1),1);

% For connecting hidden layers. To get dimensions such as: m_2 x m_1
if no_layers > 2
    for l = 2 : no_layers-1
        std = sqrt( 2 / m(l-1) ); % obs bytte till m(l-1)
        W{l} = std .* randn(m(l), m(l-1)) + mu;
        b{l} = zeros(m(l), 1);

    end
end

% For last hidden layer to get the output dimensions: K x m_end
    std = sqrt( 2 / m(no_layers-1) );
    W{end} = std .* randn(K, m(end)) + mu;
    b{end} = zeros(K, 1);

end

function [W, b] = initilizeWeightsAndBiases_sensitivity(no_layers, m, K, d)
% b and W are initilized using a normal distribution 
% Sensitivity Initilization.

mu = 0;

W = cell(no_layers,1); 
b = cell(no_layers,1);

sig1 = 1e-1;
sig2 = 1e-3;
sig3 = 1e-4;
std = sig3;

% For the connection between the input layer and the first hidden layer. To get dimensions: m_1 x d 
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


function [grad_J_W, grad_J_b, grad_J_be, grad_J_ga,  mu_av, var_av] = ComputeGradients(X, Y, W, b, lambda, no_layers, betas, gammas,  mu_av, var_av, label_smoothing, varargin)
% Backwardpass for a k layer network with batch normalization.
        
[~, n] = size(X);
vec_ones = ones(n,1);

% %%%%%% (e) BONUS %%%%%%
%X = applyJitter(X);
% %%%%%%%%%%%%%%%%%%%%%%%



% Vectors to store results for weights and biases
grad_J_W = cell(no_layers, 1); 
grad_J_b = cell(no_layers,1); 

% For scale and shift parameters
grad_J_ga = cell(no_layers-1, 1); 
grad_J_be = cell(no_layers-1, 1); 


% The alpha value is set based on if it is the very first epoch or not.
if size(varargin,2) == 3
    batch = varargin{1};
    epoch = varargin{2};
    
    if batch == 1 && epoch == 1
        a = varargin{3}(1);
    else
        a = varargin{3}(2);
    end
else
    a = varargin{1};
end


[P_batch, h_batch, mus, variances, s_norm, s,  mu_av, var_av ] = EvaluateClassifier(X, W, b, gammas, betas, true, a, mu_av, var_av, false);

%Propagate The gradients through the loss and softmax operations


% BONUS: one sided label smoothing.
if label_smoothing
    smoothed_labels = sampleSmoothedLabels(Y);
    G_batch = -(smoothed_labels - P_batch); 

else
    G_batch = -(Y - P_batch); 
end

% For the k:th layer
% The gradients of J w.r.t bias vector bk and Wk
grad_J_W{no_layers} = (1/n) * G_batch * h_batch{no_layers}' + 2 * lambda * W{no_layers};
grad_J_b{no_layers} = (1/n) * G_batch * vec_ones;

% Propagate G_batch to the previous layer
G_batch = W{no_layers}' * G_batch;
G_batch = G_batch .* (h_batch{no_layers} > 0);


for l = no_layers-1: -1 : 1
    
    % Compute gradients for the scale and shift parameters for layer l
    grad_J_ga{l} = (1/n)*( G_batch .* s_norm{l}) * vec_ones;
    grad_J_be{l} = (1/n)* G_batch * vec_ones;
    
    %Propagate the gradients through the scale and shift
    G_batch = G_batch .* (gammas{l}* vec_ones');
    
    % Propagate G_batch through the batch normalization
    G_batch = BatchNormBackPass(G_batch, s{l}, mus{l}, variances{l}, vec_ones, n);
    
    %The gradiets of J wrt bias vector b_k and W_k
    grad_J_W{l} = (1/n) * G_batch * h_batch{l}' + 2 * lambda * W{l};
    grad_J_b{l} = (1/n) * G_batch * vec_ones;
    
    if l > 1
        % Propagate G_batch to the previous layer
        G_batch = W{l}' * G_batch;
        G_batch = G_batch .* (h_batch{l} > 0);
    end
end 
end


function [relative_error_gradb,relative_error_gradW,relative_error_gradga,relative_error_gradbe] = computeRelativeErrorNumericalAnalyticalGradients(W,b, Xtr, Xva, Xte, Ytr, lambda, h, eps, m, no_points, no_dim, no_layers, NetParams, batch_size, mu, variance, alpha, gammas, betas, norm)
% Computes the relative error between a numerically estimated gradient and
% an analytically computed gradient.

relative_error_gradW = cell(no_layers,1); 
relative_error_gradb = cell(no_layers,1);
relative_error_gradga = cell(no_layers,1); 
relative_error_gradbe = cell(no_layers,1);

NetParams.W = W;
NetParams.b = b;
NetParams.gammas = gammas;
NetParams.betas = betas;

% compute the gradients
[grad_W, grad_b, grad_betas, grad_gammas] = ComputeGradients(Xtr, Ytr, W, b, lambda, no_layers, betas, gammas, mu, variance, alpha);


alpha = 0;
Grads = ComputeGradsNumSlow(Xtr, Ytr, NetParams, lambda, h, batch_size, mu, variance, alpha, norm);

% compute the relative errors

for i = 1:no_layers
    
relative_error_gradb{i} = abs(grad_b{i} - Grads.b{i}) ./ max(eps, abs(Grads.b{i}) + abs(grad_b{i}));
relative_error_gradW{i} = abs(grad_W{i} - Grads.W{i}) ./ max(eps, abs(Grads.W{i}) + abs(grad_W{i}));

if i< no_layers
    relative_error_gradga{i} = abs(grad_gammas{i} - Grads.gammas{i}) ./ max(eps, abs(Grads.gammas{i}) + abs(grad_gammas{i}));
    relative_error_gradbe{i} = abs(grad_betas{i} - Grads.betas{i}) ./ max(eps, abs(Grads.betas{i}) + abs(grad_betas{i}));
end
end
end


function [J, loss] = ComputeCost(X, Y, W, b, lambda, gammas, betas, n_batch, mu, var, alpha, precomputed)
%Computes the loss and cost J with the amount of regularization given by
%lambda.

[d, n] = size(X);
regularization_term = 0;

% Calculate the regularization term
for i = 1:size(W,1)
    regularization_term = regularization_term + lambda * sum( sum(W{i}.^2 ));
end


loss = 0;
for j=1:n/n_batch
    
    % Get the minibatch.
    j_start = (j-1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start:j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
    
    
    % Calculate the probabilities for the current minibatch.
    [P, ~, ~, ~, ~, ~,  ~, ~] = EvaluateClassifier(Xbatch, W, b, gammas, betas, true, alpha,  mu, var, precomputed );
    

    %Compute the loss and the cost for the minibatch.
    l_cross = -log(dot(Ybatch,P));
    
    loss_batch = (sum(l_cross)/n);
    
    %Update the total loss.
    loss = loss + loss_batch;
end

% Regularize the loss to get the cost.
J = loss + regularization_term;
end

function [ X ] = applyJitter(X)
% BONUS task e - applies a random jitter to each element in the training
% data from a uniform distribution in the range of +/- x %
[d, n] = size(X);

jitter = -0.001 + (0.001 + 0.001)*rand(d,n);

jitter = X.*jitter;

X = X + jitter;
end


function smoothed_labels = sampleSmoothedLabels(Y)
% BONUS: Performs one sided label smoothing within a certain range.
% Kxn

[K, n] = size(Y);

% Uniformly sample a new label for each data point within a certain range.
uppper_range = 1;
lower_range = 0.95;

% Generate values from the uniform distribution on the interval [lower_range, uppper_range]


p = lower_range + (uppper_range-lower_range).*rand(1,n); % 1 x n
p_matrix = repmat(p, [K,1]); % K x n

smoothed_labels = Y .* p_matrix;

end

function G_batch = BatchNormBackPass( G_b, s_norm, mu, variance, vec_ones, n)
% Normalize a batch of data such that they are unit gaussian along each
% dimension d.

sigma_1 = ((variance + eps).^-0.5)';
sigma_2 = ((variance + eps).^-1.5)';

G1 = G_b .* (sigma_1' * vec_ones');
G2 = G_b .* (sigma_2' * vec_ones');

D = s_norm - mu * vec_ones';
c = ( G2 .* D) * vec_ones;

G_batch = G1 - (1/n)* (G1 * vec_ones) * vec_ones' - (1/n)* D .* (c * vec_ones');

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


function [W, Xtr, Ytr, ytr, Xte, Yte, yte, Xva, Yva, yva] = reduceDimData(W,Xtr,Ytr,ytr, Xte,Yte,yte, Xva,Yva,yva, no_points, no_dim )
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
% Preprocesses the data by giving it a zero mean

mean_X = mean(Xtr, 2); 
std_X = std(Xtr, 0, 2);

Xtr = Xtr - repmat(mean_X, [1,size(Xtr,2)]);
Xtr = Xtr ./ repmat(std_X, [1, size(Xtr, 2)]);

Xva = Xva - repmat(mean_X, [1,size(Xva,2)]);
Xva = Xva ./ repmat(std_X, [1, size(Xva, 2)]);

Xte = Xte - repmat(mean_X, [1,size(Xte,2)]);
Xte = Xte ./ repmat(std_X, [1, size(Xte, 2)]);

end


function [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, W, b, gammas, betas, mu_av, var_av] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparams, W, b, lambda, no_layers, betas, gammas, alpha, mu_av, var_av)
% This function performs gradient descent on minibatches using cyclical learning and run through
% the the training images the number of times given by: GDparams.n_epochs


% Arrays used to store costs, losses and accuracies
cost_training = zeros(GDparams.n_epochs+1, 1);
cost_validation = zeros(GDparams.n_epochs+1, 1);

loss_training = zeros(GDparams.n_epochs+1, 1);
loss_validation = zeros(GDparams.n_epochs+1, 1);

acc_training = zeros(GDparams.n_epochs+1, 1);
acc_validation = zeros(GDparams.n_epochs+1, 1);


Xtr_unaugmented = Xtr;

% We run through the epochs
for epoch = 1:GDparams.n_epochs

    
    % alpha equal to 1 means that we only use the precomputed averages as
    % we do not updates the mu and var averages during testing, i.e. when
    % we compute the losses and costs.
    alpha_cost = 1;
    [cost_tr, loss_tr] = ComputeCost(Xtr, Ytr, W, b, lambda, gammas, betas, GDparams.n_batch, mu_av, var_av, alpha_cost, true );
    
    cost_training(epoch) = cost_tr;
    loss_training(epoch) = loss_tr;
    
    [cost_val, loss_val]  = ComputeCost(Xva, Yva, W, b, lambda, gammas, betas, GDparams.n_batch, mu_av, var_av, alpha_cost, true );
    
    cost_validation(epoch) = cost_val;
    loss_validation(epoch) = loss_val;
    
    acc_training(epoch) = ComputeAccuracy(Xtr, ytr, W, b, gammas, betas, GDparams.n_batch, mu_av, var_av);
    acc_validation(epoch) = ComputeAccuracy(Xva, yva, W, b, gammas, betas, GDparams.n_batch, mu_av, var_av);
    
   
    [W, b, gammas, betas, GDparams, mu_av, var_av] = MiniBatchGDCyclicalLearning(Xtr, Ytr, GDparams, W, b, lambda, no_layers, betas, gammas, alpha, mu_av, var_av, epoch);
    
    if epoch == GDparams.n_epochs
        
        
        [cost_tr, loss_tr] = ComputeCost(Xtr, Ytr, W, b, lambda, gammas, betas, GDparams.n_batch, mu_av, var_av, alpha_cost, true );
        cost_training(epoch+1) = cost_tr;
        loss_training(epoch+1) = loss_tr;

        [cost_val, loss_val]  = ComputeCost(Xva, Yva, W, b, lambda, gammas, betas, GDparams.n_batch, mu_av, var_av, alpha_cost, true );
        cost_validation(epoch+1) = cost_val;
        loss_validation(epoch+1) = loss_val;

        acc_training(epoch+1) = ComputeAccuracy(Xtr, ytr, W, b, gammas, betas, GDparams.n_batch, mu_av, var_av);
        acc_validation(epoch+1) = ComputeAccuracy(Xva, yva, W, b, gammas, betas, GDparams.n_batch, mu_av, var_av);
    end
    
    
    % BONUS: Mirroring the datapoints
%     if mod(epoch,2) == 0
%         prob = 0.01;
%         Xtr = mirrorBatch( Xtr, prob);
% %     else
% %         Xtr = Xtr_unaugmented;
%          
%     end

    [ Xtr, Ytr, ytr ] = shuffleDataPoints(Xtr, Ytr, ytr);
    

    
end
end


function [Wstar, bstar, gammasstar, betastar, GDparams, mu_av, var_av ] = MiniBatchGDCyclicalLearning(X, Y, GDparams, W, b, lambda, no_layers, betas, gammas, alpha, mu_av, var_av, epoch)
%Performs gradient descent on minibatches of data

[~, N] = size(X);

    for batch=1:N/GDparams.n_batch
        
        % Get a minibatch of data
        batch_start = (batch-1) * GDparams.n_batch + 1;
        batch_end = batch * GDparams.n_batch;
        inds = batch_start:batch_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        
        % Compute the gradients for the parameters to be learned.
        label_smoothing = false;
        [grad_W, grad_b, grad_betas, grad_gammas,  mu_av, var_av] = ComputeGradients(Xbatch, Ybatch, W, b, lambda, no_layers, betas, gammas,  mu_av, var_av, label_smoothing, batch, epoch, alpha);
        
        % We increase the update step
        GDparams.update_step = GDparams.update_step + 1;
        
        % The cyclical learning rate is updated
        [ eta_t ] = updateCyclicalLearningRate(GDparams);
        GDparams.eta = eta_t;
        
       
        % We want to update the l parameter to keep track of the current cycle
        if mod( GDparams.update_step, GDparams.n_s*2 ) == 0
            GDparams.l = GDparams.l + 1;
        end
        
        % The weights and biases are updated
        for i = 1:no_layers
            b{i} = b{i} - GDparams.eta * grad_b{i};
            W{i} = W{i} - GDparams.eta * grad_W{i};
            
            if i < no_layers
                betas{i} = betas{i} - GDparams.eta * grad_betas{i};
                gammas{i} = gammas{i} - GDparams.eta * grad_gammas{i};
            end
        end
    end
    
    % The final b and W, gammas and betas trained 
    bstar = b;
    Wstar = W;
    gammasstar = gammas;
    betastar = betas;
end


function [eta_t ] = updateCyclicalLearningRate(GDparams)
% Updates the cyclical learning rate at time t

eta_t = double(0);

% We check if t is within a lower and upper range. I.e if n_t is increasing 
lower1 = (2*GDparams.l*GDparams.n_s) <= GDparams.update_step;
upper1 = GDparams.update_step <= ((2*GDparams.l+1)*GDparams.n_s) ;


if lower1 && upper1
    eta_t = GDparams.eta_min + ((GDparams.update_step - 2*GDparams.l*GDparams.n_s)/( GDparams.n_s ))*(GDparams.eta_max - GDparams.eta_min);
end


upper2 = GDparams.update_step >= (2 * GDparams.l + 1)*GDparams.n_s ;
lower2 = GDparams.update_step <= 2*(GDparams.l + 1)*GDparams.n_s ;

% We check if t is within a lower and upper range. I.e if n_t is deacreasing
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
    
    [grad_W, grad_b, grad_betas, grad_gammas] = ComputeGradients(Xtr, Ytr, Wtemp, b, lambda, no_layers, betas, gammas);
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
% Loades a batch of data and creates one-hot encoding corresponding to their labels. 

A = load(filename);

X = double(A.data');
%X = X / 255;


y = A.labels;
y = y + 1 ;
Y = oneHotEncoder(y);

end

function X_batch = mirrorBatch( X_batch, prob)


%X_batch = X_batch(1:7,1:8);

[d , n] = size(X_batch);

% Sampling indicies of images to be mirrored.
%prob = 1 - prob;
mask = ((rand(size(X_batch,2),1)) < prob)';

ims_to_be_mirrored = X_batch .* repmat(mask,[d,1]); 
indicies_mirrored_ims = find(ims_to_be_mirrored);


% Reshaping the images to be mirrored into a matrix format.
s_im = reshape(ims_to_be_mirrored, 32, 32, 3, n);

% This row can be used during debugging to make sure that the mirroring is
% correct.
%s_im = (s_im - min(s_im(:))) / (max(s_im(:)) - min(s_im(:)));

s_im = permute(s_im, [2, 1, 3, 4]);

% Mirroring the images.
s_im = flipdim(s_im, 2);

% Going back to the vector format.
im = reshape(s_im, 3072, n);


% These rows can be used during debugging to make sure that the mirroring is
% correct.
% Reshaping all of the images in the batch into a matrix format.
%s_im_all = reshape(X_batch, 32, 32, 3, n);
%s_im_all = (s_im_all - min(s_im_all(:))) / (max(s_im_all(:)) - min(s_im_all(:)));
%s_im_all = permute(s_im_all, [2, 1, 3, 4]);
% Going back to the vector format.
%s_im_all = reshape(s_im_all, 3072, n);


% Putting the mirrored images into the matrix with the regular images.
X_batch(indicies_mirrored_ims) = im(indicies_mirrored_ims);



%Debugging. To make sure that the images were mirrored correctly.
% s_im = reshape(s_im_all,32,32,3,8);
% figure();
% montage(s_im);
% hold on;
end


function[Y] = oneHotEncoder(y)
%creates matrix of one hot encoded data from an array of labels between 

N = length(y);
K = max(y) ;
Y = zeros(K, N);

for i = 1 : K
    dp_current_label = find( i == y);
    Y(i, dp_current_label) = 1; 
end

end


function acc = ComputeAccuracy(X, y, W, b ,gammas, betas, n_batch, mu_av, var_av)
%Evaluates the accuracy of trained Weights and bias parameters on
%minibatches using the precomputed moving averages of the variances and
%means.

n = size(X,2);

total_no_accurate_classifications = 0;
for j=1:n/n_batch
    
    % Get the indicies for a minibatch.
    j_start = (j-1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start:j_end;
    Xbatch = X(:, inds);
    ybatch = y(inds);
    
    % alpha = 1 to use the moving, pre computed, mu_av and var_av values 
    alpha = 1;
    [P, ~, ~, ~, ~, ~,~,~] = EvaluateClassifier(Xbatch, W, b, gammas, betas, true, alpha, mu_av, var_av, true);

    
    [~, I] = max(P,[],1); 
    [~, no_accurate_classifications] = size(find(I==ybatch'));
    
    total_no_accurate_classifications = total_no_accurate_classifications + no_accurate_classifications;
end

acc = total_no_accurate_classifications/n;

end


function [P, h, mu_vec, variance_vec, s_norm, s, mu_av, var_av ] = EvaluateClassifier_org(X, W, b, gammas, betas, normalize, alpha, mu_av, var_av, precomputed)
%Evaluates a k layer network via a forwardpass

[~, n] = size(X);
k = size(W, 1);

s = cell(k,1);
s_norm = cell(k-1, 1);
h = cell(k,1);

mu_vec = cell(k-1,1);
variance_vec = cell(k-1,1);


h{1} = X; 


for l = 1:k-1
    
    s{l} = W{l} * h{l} + repmat( b{l}, [1, n] );  % 50x1 
    
    % Batch Normlaization
    % Compute the mean and variances for the current layer.
    [ mu, variance ] = compute_mean_variance( s{l} );
    mu_vec{l} = mu ;
    variance_vec{l} = variance;
    
    % 1. The first minibatch of the first epoch alpha is set to 0 to get
    % mu_avg = mu and var_avg = var
    % 2. If we are using precomputed averages during testing alpha is set to 1. 
    % Thus, the precomputed values are not updated.
    mu_av{l} = alpha*mu_av{l} + (1-alpha)*mu_vec{l};
    var_av{l} = alpha*var_av{l} + (1-alpha)*variance_vec{l};
    
    if precomputed
        s_norm{l} = BatchNormalize( s{l} , mu_av{l}, var_av{l}, eps );
    else
        s_norm{l} = BatchNormalize( s{l} , mu, variance, eps );
    end
    
    s_shifted_scaled = repmat( gammas{l}, [1, size(s_norm{l},2)] ) .* s_norm{l} + repmat( betas{l}, [1, size(s_norm{l},2)]);
    
    h{l+1} = max( 0, s_shifted_scaled ); 
        
end

% For the kth layer
s{k} = W{k}*h{k} + repmat( b{k}, [1, n] );

P = softmax(s{k});
end

function [P, h, mu_vec, variance_vec, s_norm, s, mu_av, var_av ] = EvaluateClassifier(X, W, b, gammas, betas, normalize, alpha, mu_av, var_av, precomputed)
%Evaluates a k layer network via a forwardpass

[~, n] = size(X);
k = size(W, 1);

s = cell(k,1);
s_norm = cell(k-1, 1);
h = cell(k,1);

mu_vec = cell(k-1,1);
variance_vec = cell(k-1,1);


h{1} = X; 


for l = 1:k-1
    
    s{l} = W{l} * h{l} + repmat( b{l}, [1, n] );  % 50x1 
    
    % Batch Normlaization
    % Compute the mean and variances for the current layer.
    [ mu, variance ] = compute_mean_variance( s{l} );
    mu_vec{l} = mu ;
    variance_vec{l} = variance;
    
    % 1. The first minibatch of the first epoch alpha is set to 0 to get
    % mu_avg = mu and var_avg = var
    % 2. If we are using precomputed averages during testing alpha is set to 1. 
    % Thus, the precomputed values are not updated.
    mu_av{l} = alpha*mu_av{l} + (1-alpha)*mu_vec{l};
    var_av{l} = alpha*var_av{l} + (1-alpha)*variance_vec{l};
    
    if precomputed
        s_norm{l} = BatchNormalize( s{l} , mu_av{l}, var_av{l}, eps );
    else
        s_norm{l} = BatchNormalize( s{l} , mu, variance, eps );
    end
    
    s_shifted_scaled = repmat( gammas{l}, [1, size(s_norm{l},2)] ) .* s_norm{l} + repmat( betas{l}, [1, size(s_norm{l},2)]);
    
    h{l+1} = max( 0, s_shifted_scaled ); 
        
end

% For the kth layer
s{k} = W{k}*h{k} + repmat( b{k}, [1, n] );

P = softmax(s{k});
end


function [P, h, mu_vec, variance_vec, s_norm, s, mu_av, var_av ] = EvaluateClassifier_DropOut(X, W, b, gammas, betas, normalize, alpha, mu_av, var_av, precomputed)
%Evaluates a k layer network via a forwardpass
% Uses both Batch Normalization and inverted droput and is used during the
% training phase of the network.

prob = 0.01;
p = 1 - prob;
[~, n] = size(X);
k = size(W, 1);

s = cell(k,1);
s_norm = cell(k-1, 1);
h = cell(k,1);

mu_vec = cell(k-1,1);
variance_vec = cell(k-1,1);


h{1} = X; 


for l = 1:k-1
    
    s{l} = W{l} * h{l} + repmat( b{l}, [1, n] );  % 50x1 
    
    % Batch Normlaization
    % Compute the mean and variances for the current layer.
    [ mu, variance ] = compute_mean_variance( s{l} );
    mu_vec{l} = mu ;
    variance_vec{l} = variance;
    
    % 1. The first minibatch of the first epoch alpha is set to 0 to get
    % mu_avg = mu and var_avg = var
    % 2. If we are using precomputed averages during testing alpha is set to 1. 
    % Thus, the precomputed values are not updated.
    mu_av{l} = alpha*mu_av{l} + (1-alpha)*mu_vec{l};
    var_av{l} = alpha*var_av{l} + (1-alpha)*variance_vec{l};
    
    if precomputed
        s_norm{l} = BatchNormalize( s{l} , mu_av{l}, var_av{l}, eps );
    else
        s_norm{l} = BatchNormalize( s{l} , mu, variance, eps );
    end
    
    s_shifted_scaled = repmat( gammas{l}, [1, size(s_norm{l},2)] ) .* s_norm{l} + repmat( betas{l}, [1, size(s_norm{l},2)]);
    
    h{l+1} = max( 0, s_shifted_scaled );
    
    % Inverted dropout
    inds = size(h{l+1});
    u = (rand(inds) < p)/p;
    h{l+1} = h{l+1}.* u;
  
        
end

% For the kth layer
s{k} = W{k}*h{k} + repmat( b{k}, [1, n] );

P = softmax(s{k});
end


function  [ mu, variance ] = compute_mean_variance( s )
% Computes the mean and variances of a layer.

n = size(s, 2);
mu = mean(s , 2);
variance = var(s, 0, 2);
variance = variance * (n-1) / n;
end


function s_norm = BatchNormalize( s, mu, var, epsilon )
% Normalize the data to give it a gaussian distribution.

s_norm = ( (diag( var + epsilon ) )^ (-0.5))*(s - repmat(mu, 1, size(s, 2)));

end


function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h, batch_size, mu, variance, alpha, norm)
% Used to numerically compute gradients to compare to the analytical
% garidents.

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);

if norm
    
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
         
        [c1, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false);    
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        

        [c2, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
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
     
        [c1, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
     
        [c2, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
        
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if norm
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
           
            [c1, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            
             [c2, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
            
            
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
           
            [c1, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
         
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            
            [c2, ~]  = ComputeCost(X, Y, NetTry.W, NetTry.b, lambda, NetTry.gammas, NetTry.betas, batch_size, mu, variance, alpha, false); 
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end


function[ X_shuffled, Y_shuffled, y_shuffled ] = shuffleDataPoints(X, Y, y)
% Randomly shuffles the data by shuffling columns of a matrix.

new_indicies_shuffled = randperm(size(X,2));

X_shuffled = X(:,new_indicies_shuffled);
Y_shuffled = Y(:,new_indicies_shuffled);
y_shuffled = y(new_indicies_shuffled);

end