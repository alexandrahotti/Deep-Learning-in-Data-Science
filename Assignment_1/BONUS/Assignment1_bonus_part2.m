
function [ngrad_b, ngrad_W]= Assignment1_BONUS_2()
% performs gradient descent using the SVM loss described during the
% lectures

clear;
addpath C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Datasets\cifar-10-batches-mat;

rng(38);

% Load data
[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat'); %training data
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat'); %validation data
[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data

[d, N] = size(Xtr);
K = max(ytr); 

%%%%Comparison of numerically and analytically computed mini batch gradients for b and W %%
% h = 1e-6;
% batch_sizes = [1, 40, 100];
% eps=0.1;

%ComputeAndPlotRelativeError(Xtr, Ytr, batch_sizes, W, b, lambda, eps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%% Mini batch gradient descent %%%%
GDparameters = GDparams; % Creating a GDparams object as suggested in the asisgnment description
lambda = 0;
GDparameters.n_batch = 100;
GDparameters.eta = 0.1; 
GDparameters.n_epochs = 40;

% Initilizing the weights and biases via the bias trick where the bias is
% included in the weight matrix. Source: http://cs231n.github.io/linear-classify/
[weights_and_bias] = initilizeWeightsAndBiases_trick(K, d);

% When we do the bias trick we need to insert an extra dimension equal to
% one in our data

vec_ones_tr = ones(1,N);
vec_ones_va = ones(1, size(Xva,2));
vec_ones_te = ones(1, size(Xte,2));

Xtr = [Xtr; vec_ones_tr];
Xva = [Xva; vec_ones_va];
Xte = [Xte; vec_ones_te];

% Mini Batch Gradient Descent

[cost_training, cost_validation, W_bstar] = GradientDescent(Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, weights_and_bias, lambda);
%plotValidationTrainingCost(cost_training, cost_validation, GDparameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Accuracy of Mini batch gradient descent %%%%
acc = ComputeAccuracy(Xte, yte, W_bstar);

% %%%% Plot weights %%%%
% plotWeights(Wstar);
end


function plotValidationTrainingCost(cost_training, cost_validation, GDparams)
%Plots the costs of the validation and training data sets 

n_epochs_range = [1:1:GDparams.n_epochs];

plot(n_epochs_range, cost_training, n_epochs_range, cost_validation);
legend('training cost','validation cost');
title('Cost/loss computed on the training and validation data sets')
xlabel('epoch')
ylabel('loss')
end

function [cost_training, cost_validation, weights_and_bias] = GradientDescent(Xtr, Ytr, ytr, Xva, Yva, yva, GDparams, weights_and_bias, lambda)
% This function performs gradient descent on minibatches and run through
% the the training images the number of times given by: GDparams.n_epochs 

% init vectors to store computed costs after each epoch
cost_training = zeros(GDparams.n_epochs, 1);
cost_validation = zeros(GDparams.n_epochs, 1);

[d,n] = size(Xtr);

% going through all epochs
for i = 1:GDparams.n_epochs
   weights_and_bias = MiniBatchGD(Xtr, Ytr, ytr, GDparams, weights_and_bias, lambda, GDparams.n_batch);
   
   % Computing the svm cost
   [~, cost_training(i)] = ComputeCostSVM(Xtr, Ytr, weights_and_bias, lambda);
   [~, cost_validation(i)] = ComputeCostSVM(Xva, Yva, weights_and_bias, lambda);
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
    [~, c1] = ComputeCostSVM(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    [~, c2] = ComputeCostSVM(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    [~, c1] = ComputeCostSVM(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    [~, c2] = ComputeCostSVM(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

end

function[X, Y, y] = LoadBatch(filename)
% loads  and normalizes a batch of data
% Also cerates a one hot encoded representation of the data labels

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


function[weights_and_bias] = initilizeWeightsAndBiases_trick(K, d)
% b and W are initilized using a normal distribution
% b is added to the weight matrix since we are using the bias trick.
% Source: http://cs231n.github.io/linear-classify/


mu = 0;
std = 0.01;

W = std.*randn(K, d) + mu;
b = std.*randn(K, 1) + mu;

weights_and_bias = [W, b];
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

function P = EvaluateClassifier(X, W)
% Evaluates a classifer by implementing equation 1, 2 and 3 on page 1 in the
% assignment description. However since we are doing the bias trick we do
% not need to add the bias.

% Equation 1: s = Wx
% Equation 2: p = softmax(s)
% Equation 3: softmax(s) = exp(s) / ((l^T)exp(s))


[d, n] = size(X); 
[K, ~] = size(W);

s = W*X ;

exp_row_sum = sum(exp(s));
P = exp(s)./ repmat(exp_row_sum, K, 1);
end


function [marg, J] = ComputeCostSVM(X, Y, W_b, lambda)
% Computes the svm cost as described in the lecture

[d,N] = size(X);
[K,N] = size(Y);

s = W_b*X;
sj = repmat(s(logical(Y))',K,1);

Y_excluding_j = ~Y;
marg = max( 0, s-sj+1 ).* Y_excluding_j;

regularization_term = sum( sum(W_b.^2) )*lambda;
J = (1/N)*sum( sum(marg,2) ) + regularization_term;
end



function acc = ComputeAccuracy(X, y, W_b)
% X: d×n.
% y: labels of length n
% acc: the accuracy.

P  = EvaluateClassifier(X, W_b); % Kxn

[M, I] = max(P,[],1); 
[total_no_classifications, ~] = size(y);
[~, no_accurate_classifications] = size(find(I==y'));

acc = no_accurate_classifications/total_no_classifications;
end


function grad_W = ComputeGradients(X, Y, y, W_b, marg, lambda)
% Computes the gradient of W. Since we are using the bias trick we do not
% need to explicitly compute the gradient of b. Instead since it is
% included in W we get it by computing grad W.

[d, n] = size(X);
[K, n] = size(Y);
vec_ones = ones(n,1);

filter = zeros(K,n);
filter( marg >0 );

for i = 1 : K
    dps_over_marg = find( marg(i,:) >0);
    filter(i, dps_over_marg) = 1; 
    
end
tot_over_marg = sum(filter,1); 

filter(logical(Y))= - sum(filter,1); 

dL_dW =  (1/n) * filter * X';
grad_W = dL_dW + 2*lambda*W_b;

end

function Wstar = MiniBatchGD(X, Y, y, GDparams, W_b, lambda, n_batch)
%Performs mini batch gradient descent on batches of data
% X: dxN

[~, N] = size(X);

    for j=1:N/n_batch
        
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        ybatch = y(j_start:j_end);
        
        [marg, ~] = ComputeCostSVM(Xbatch, Ybatch, W_b, lambda);
        grad_W = ComputeGradients(Xbatch, Ybatch,ybatch, W_b, marg, lambda);
       
        W_b = W_b - GDparams.eta * grad_W;
    end
    
    Wstar = W_b;
end

function plotWeights( W )

for i=1:10
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

montage(s_im, 'Size', [3,4]);

end
