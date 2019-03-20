
function [ngrad_b, ngrad_W]= Assignment1()
clear;
addpath Datasets/cifar-10-batches-mat/;

rng(38);

[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat'); %training data
%[Xv, Yv, yv] = LoadBatch('data_batch_2.mat'); %validation data
%[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data

[d, N] = size(Xtr);
K = max(ytr); 
[b, W] = initilizeWeightsAndBiases(K, d);

h = 1e-6;
lambda = 1;
batch_sizes = [1, 40, 100];
eps=0;


[jjnj] = ComputeAndPlotRelativeError(Xtr, Ytr, batch_sizes, W, b, lambda, eps);

end


function [K] = ComputeAndPlotRelativeError(X, Y, batch_sizes, W, b, lambda, eps)
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
%X: contains the image pixel data, has size dxN, is of type double or
%single and has entries between 0 and 1. N is the number of images
%(10000) and d the dimensionality of each image (3072=32×32×3).

%Y: is K×N (K= # of labels = 10) and contains the one-hot representation
%of the label for each image.

%y: is a vector of length N containing the label for each image. A note
%of caution. CIFAR-10 encodes the labels as integers between 0-9 but
%Matlab indexes matrices and vectors starting at 1. Therefore it may be
%easier to encode the labels between 1-10.

A = load(filename);

X = double(A.data');
X = X / 255;

y = A.labels;
y = y + 1 ;
Y = oneHotEncoder(y);
end



function[b, W] = initilizeWeightsAndBiases(K, d)

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

% X: each column of X corresponds to an image and it has size d×n.
% W and b: are the parameters of the network.
% P: each column of P contains the probability for each label for the image
% in the corresponding column of X. P has size K×n.


[d, n] = size(X); 
[K, ~] = size(W);

b_matrix = repmat( b, [1, n] );
s = W*X + b_matrix;

exp_row_sum = sum(exp(s));
P = exp(s)./ repmat(exp_row_sum, K, 1);
end

function J = ComputeCost(X, Y, W, b, lambda)
%X: each column of X corresponds to an image and X has size d×n.
%Y : each column of Y (K×n) is the one-hot ground truth label for the corresponding 
%column of X or Y is the (1×n) vector of ground truth labels.
%J: is a scalar corresponding to the sum of the loss of the network’s
%predictions for the images in X relative to the ground truth labels and
%the regularization term on W.

[d, n] = size(X);

regularization_term = sum( sum(W.^2) )*lambda;

% P: each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.
P = EvaluateClassifier(X, W, b); %Kxn

l_cross = -log(dot(Y,P));
J = (sum(l_cross)/n) + regularization_term;
end


function acc = ComputeAccuracy(X, y, W, b)
% X: each column of X corresponds to an image and X has size d×n.
% y: is the vector of ground truth labels of length n.
% acc: is a scalar value containing the accuracy.

P = EvaluateClassifier(X, W, b); % Kxn


[M, I] = max(P,[],1); 

[total_no_classifications, ~] = size(y);

[~, no_accurate_classifications] = size(find(I==y'));

acc = no_accurate_classifications/total_no_classifications;
end


function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
%Assuming that X,Y = one batch

% X: each column of X corresponds to an image and it has size d×n.
% Y: each column of Y (K×n) is the one-hot ground truth label for the corresponding column of X.
% P: each column of P contains the probability for each label for the image
% in the corresponding column of X. P has size K×n.
% W: grad W is the gradient matrix of the cost J relative to W and has size K×d.

[~, n] = size(X);

vec_ones = ones(n,1);

%Forward pass
P_batch = EvaluateClassifier(X, W, b);


%Backward pass
G_batch = -(Y - P_batch);

dL_dW =  (1/n)*(G_batch*X');
grad_W = dL_dW + 2*lambda*W;

grad_b =  (1/n)*(G_batch*vec_ones);
end

