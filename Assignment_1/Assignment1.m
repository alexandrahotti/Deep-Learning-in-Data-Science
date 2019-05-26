
function [ngrad_b, ngrad_W]= Assignment1()
clear;
addpath C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Datasets\cifar-10-batches-mat;

rng(400);

% Loading data

[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat'); %training data
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat'); %validation data
[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data

[d, N] = size(Xtr);
K = max(ytr); 
[b_init, W_init] = initilizeWeightsAndBiases(K, d);


%%%%Comparison of numerically and analytically computed mini batch gradients for b and W %%
% h = 1e-6;
% lambda = 0;
% batch_sizes = [1, 40, 100];
% eps=0;

%ComputeAndPlotRelativeError(Xtr, Ytr, batch_sizes, W, b, lambda, eps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%% Mini batch gradient descent %%%%
% GDparameters = GDparams;
% lambda = 0;
% GDparameters.n_batch = 100;
% GDparameters.eta = 0.01; % 0.001
% GDparameters.n_epochs = 40; % 70
% 
% [cost_training, cost_validation, loss_training, loss_validation, Wstar, bstar] = GradientDescent(Xtr, Ytr, Xva, Yva, GDparameters, W_init, b_init, lambda);
% 
%%% Plotting the training costs and losses computed after each epoch
% plotResults(cost_training, cost_validation, GDparameters.n_epochs , 'training cost', 'validation cost', 'Decaying learning rate (d) - Mini batch graddient descent costs', 'epoch', 'cost');
% plotResults(loss_training, loss_validation, GDparameters.n_epochs , 'training loss', 'validation loss', 'Decaying learning rate (d) - Mini batch graddient descent losses', 'epoch', 'loss');
%
% %%% Accuracy of Mini batch gradient descent %%%%
% acc = ComputeAccuracy(Xte, yte, Wstar, bstar);
% 
%%% Plot weights
% plotWeights(Wstar);

end

function plotResults(training_vals, validation_vals,n_epochs , legend_1, legend_2, title_txt, x_label, y_label)
figure();
x_range = 0:1:n_epochs-1;

plot(x_range', training_vals, x_range', validation_vals);

legend(legend_1,legend_2);
title(title_txt)
xlabel(x_label)
ylabel(y_label)
hold on;

end


function [cost_training, cost_validation, loss_training, loss_validation, W, b] = GradientDescent(Xtr, Ytr, Xva, Yva, GDparams, W, b, lambda)
% This function performs gradient descent on minibatches and run through
% the the training images the number of times given by: GDparams.n_epochs 

% Vectors used to store computed costs and losses
cost_training = zeros(GDparams.n_epochs, 1);
cost_validation = zeros(GDparams.n_epochs, 1);

loss_training = zeros(GDparams.n_epochs, 1);
loss_validation = zeros(GDparams.n_epochs, 1);



for i = 1:GDparams.n_epochs

    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, GDparams.n_batch);
    
    % Compute costs and losses on training and validation data set 
    [cost_tr, loss_tr] = ComputeCost(Xtr, Ytr, W, b, lambda);
    cost_training(i) = cost_tr;
    loss_training(i) = loss_tr;
    
    [cost_val, loss_val]  = ComputeCost(Xva, Yva, W, b, lambda);
    cost_validation(i) = cost_val;
    loss_validation(i) = loss_val;
end
end


function ComputeAndPlotRelativeError(X, Y, batch_sizes, W, b, lambda, eps)
% Computes and creates boxplots of the relative errors between the analytical and 
% the numerical gradients for b and W for the mini batch sizes given in the vector
% batch_sizes.

[K, ~] = size(b);
[K, d] = size(W);

h = 1e-6;
no_batches = length(batch_sizes);

gradients_rel_errors_b = zeros (K, no_batches);
gradients_rel_errors_W = zeros (K*d, no_batches);

for i = 1:no_batches
    
    [grad_W, grad_b] = ComputeGradients(X(:, 1:batch_sizes(i)), Y(:, 1:batch_sizes(i)), W, b, lambda);
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:, 1:batch_sizes(i)), Y(:, 1:batch_sizes(i)), W, b, lambda, h);
    

    relative_error_gradb = abs(grad_b - ngrad_b) ./ max(eps, abs(ngrad_b) + abs(grad_b));
    relative_error_gradW = abs(grad_W - ngrad_W) ./ max(eps, abs(ngrad_W) + abs(grad_W));
    relative_error_gradW = reshape(relative_error_gradW, K*d,1);
    
    gradients_rel_errors_b(:, i) = relative_error_gradb;
    gradients_rel_errors_W(:, i) = relative_error_gradW;
    
    
    max_grad_b_relative_error = max(relative_error_gradb);
    max_grad_W_relative_error = max(relative_error_gradW);
    
    min_grad_b_relative_error = min(relative_error_gradb);
    min_grad_W_relative_error = min(relative_error_gradW);

    mean_grad_b_relative_error = mean(relative_error_gradb);
    mean_grad_W_relative_error = mean(relative_error_gradW);
    
end

boxplot(gradients_rel_errors_W, batch_sizes);
title('Relative error between numerical and analytical gradient for W')
xlabel('Batch size')
ylabel('relavtive error')

boxplot(gradients_rel_errors_b, batch_sizes);
title('Relative error between numerical and analytical gradient for b')
xlabel('Batch size')
ylabel('relavtive error')

end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
% Computes the numerical gradient

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    [c1, ~]  = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    [c1, ~]  = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    [c1, ~]  = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

end

function[X, Y, y] = LoadBatch(filename)
% Loads data and normalizes it
% Also creates one hot encoded labels for the data
A = load(filename);

X = double(A.data');
X = X / 255;

y = A.labels;
y = y + 1 ;
Y = oneHotEncoder(y);
end



function[b, W] = initilizeWeightsAndBiases(K, d)
% Initilization of b and W using a normal distribution 

mu = 0;
std = 0.01;
W = std.*randn(K, d) + mu;
b = std.*randn(K, 1) + mu;

end

function[Y] = oneHotEncoder(y)
%creates matrix of one hot encoded data from an array of labels

N = length(y);
K = max(y) ;
Y = zeros(K, N);

for i = 1 : K
    dp_current_label = find( i == y);
    Y(i, dp_current_label) = 1; 
end

end

function P = EvaluateClassifier(X, W, b)
% Evaluates a classifer by implementing equation 1, 2 and 3 on page 1 in the
% assignment description.

% Equation 1: s = Wx+b
% Equation 2: p = softmax(s)
% Equation 3: softmax(s) = exp(s) / ((l^T)exp(s))

% X: d×n.
% W and b: the parameters of the network.
% P: K×n. probability for each possible label for each image


[d, n] = size(X); 
[K, ~] = size(W);

b_matrix = repmat( b, [1, n] );
s = W*X + b_matrix;

% Softmax
exp_row_sum = sum(exp(s));
P = exp(s)./ repmat(exp_row_sum, K, 1);
end

function [J, loss]  = ComputeCost(X, Y, W, b, lambda)
% Computes the cross entropy cost on a batch of data

%X:  d×n.
%Y: K×n. The one-hot ground truth labels
%J: sum of the loss of the network’s predictions

[d, n] = size(X);
regularization_term = sum( sum(W.^2) )*lambda;

P = EvaluateClassifier(X, W, b); 

l_cross = -log(dot(Y,P));
loss = sum(l_cross)/n;

J = loss + regularization_term;
end


function acc = ComputeAccuracy(X, y, W, b)
% Computes the accuracy by doing one forward pass and comparing these
% results to the gorund truth

% X: d×n.
% y: labels of length n
% acc: the accuracy.

% forward pass
P = EvaluateClassifier(X, W, b); % Kxn

[M, I] = max(P,[],1); 
[total_no_classifications, ~] = size(y);
[~, no_accurate_classifications] = size(find(I==y'));

acc = no_accurate_classifications/total_no_classifications;
end


function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
% computed the gradients of w and b using the efficient matrix computations
% described during the lecture

% X: d×n.
% Y: K×n - the one-hot ground truth labels
% P: the probability for each label. K×n
% W: grad W is the gradient matrix of the cost J relative to W and has size K×d.

[~, n] = size(X);
vec_ones = ones(n,1);

%Forward pass
P_batch = EvaluateClassifier(X, W, b);

%Backward pass
G_batch = -(Y - P_batch);

dL_dW =  (1/n)*G_batch*X';
grad_W = dL_dW + 2*lambda*W;

grad_b =  (1/n)*G_batch*vec_ones;
end

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, n_batch)
%Performs mini batchs gradient descent 
% X: dxN
[~, N] = size(X);

    for j=1:N/n_batch
        
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
       
        b = b - GDparams.eta * grad_b;
        W = W - GDparams.eta * grad_W;
    end
    
    bstar = b;
    Wstar = W;
end

function plotWeights( W )
figure();

for i=1:10
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

montage(s_im);
hold on;
end


