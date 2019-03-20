
function [ngrad_b, ngrad_W]= Assignment1()
clear;
addpath Datasets/cifar-10-batches-mat/;

rng(38);

[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat'); %training data
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat'); %validation data
[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data

[d, N] = size(Xtr);
K = max(ytr); 

[b_init, W_init] = initilizeWeightsAndBiases(K, d);

%%%%Comparison of numerically and analytically computed mini batch gradients for b and W %%
h = 1e-6;
lambda = 0;
batch_sizes = [1, 40, 100];
eps=0;
%ComputeAndPlotRelativeError(Xtr, Ytr, batch_sizes, W, b, lambda, eps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Mini batch gradient descent %%%%
n_batch = 100;
GDparameters = GDparams;
GDparameters.n_batch = 100;
GDparameters.eta = 0.01;
GDparameters.n_epochs = 40;

[cost_training, cost_validation, Wstar, bstar] = GradientDescent(Xtr, Ytr, Xva, Yva, GDparameters, W_init, b_init, lambda);
%plotValidationTrainingCost(cost_training, cost_validation, GDparameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Accuracy of Mini batch gradient descent %%%%
acc = ComputeAccuracy(Xte, yte, Wstar, bstar);

%%%% Plot weights %%%%
plotWeights(Wstar);
end

function plotValidationTrainingCost(cost_training, cost_validation, GDparams)
%Plots the 
n_epochs_range = [1:1:GDparams.n_epochs];
disp(size(n_epochs_range));
disp(size(cost_training));
disp(size(cost_validation));

plot(n_epochs_range, cost_training, n_epochs_range, cost_validation);
legend('training cost','validation cost');
title('Cost/loss computed on the training and validation data sets')
xlabel('epoch')
ylabel('loss')
end

function [cost_training, cost_validation, W, b] = GradientDescent(Xtr, Ytr, Xva, Yva, GDparams, W, b, lambda)
% This function performs gradient descent on minibatches and run through
% the the training images the number of times given by: GDparams.n_epochs 
disp(GDparams.n_epochs);

cost_training = zeros(GDparams.n_epochs, 1);
cost_validation = zeros(GDparams.n_epochs, 1);

for i = 1:GDparams.n_epochs
    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, GDparams.n_batch);
    cost_training(i) = ComputeCost(Xtr, Ytr, W, b, lambda);
    cost_validation(i) = ComputeCost(Xva, Yva, W, b, lambda);
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

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

end

function[X, Y, y] = LoadBatch(filename)
%X: dxN
%Y: is K×N
%y: is a vector of length N containing the one hot encoding labels for each image. 

A = load(filename);

X = double(A.data');
X = X / 255;

y = A.labels;
y = y + 1 ;
Y = oneHotEncoder(y);
end



function[b, W] = initilizeWeightsAndBiases(K, d)
% b and W are initilized using a normal distribution 

mu = 0;
std = 0.01;
W = std.*randn(K, d) + mu;
b = std.*randn(K, 1) + mu;

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

exp_row_sum = sum(exp(s));
P = exp(s)./ repmat(exp_row_sum, K, 1);
end

function J = ComputeCost(X, Y, W, b, lambda)
%X:  d×n.
%Y: K×n. The one-hot ground truth labels
%J: sum of the loss of the network’s predictions

[d, n] = size(X);
regularization_term = sum( sum(W.^2) )*lambda;
P = EvaluateClassifier(X, W, b); 
l_cross = -log(dot(Y,P));
J = (sum(l_cross)/n) + regularization_term;
end


function acc = ComputeAccuracy(X, y, W, b)
% X: d×n.
% y: labels of length n
% acc: the accuracy.

P = EvaluateClassifier(X, W, b); % Kxn

[M, I] = max(P,[],1); 
[total_no_classifications, ~] = size(y);
[~, no_accurate_classifications] = size(find(I==y'));

acc = no_accurate_classifications/total_no_classifications;
end


function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
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
%Performs gradient descent on batches of data
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

for i=1:10
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

montage(s_im, 'Size', [1,3]);

end


