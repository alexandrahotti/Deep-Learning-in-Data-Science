
function[P] = Assignment1()
addpath Datasets/cifar-10-batches-mat/;

[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat'); %training data
%[Xv, Yv, yv] = LoadBatch('data_batch_2.mat'); %validation data
%[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data

[N, d] = size(Xtr);
K = max(ytr)+1; 
[b, W] = initilizeWeightsAndBiases(K, d);

P = EvaluateClassifier(transpose(Xtr(1:100,:)), W, b);

end


function[X, Y, y] = LoadBatch(filename)
%X: contains the image pixel data, has size N×d, is of type double or
%single and has entries between 0 and 1. N is the number of images
%(10000) and d the dimensionality of each image (3072=32×32×3).

%Y: is K×N (K= # of labels = 10) and contains the one-hot representation
%of the label for each image.

%y: is a vector of length N containing the label for each image. A note
%of caution. CIFAR-10 encodes the labels as integers between 0-9 but
%Matlab indexes matrices and vectors starting at 1. Therefore it may be
%easier to encode the labels between 1-10.

A = load(filename);
X = double(A.data);
X = X / 255;

y = A.labels;
Y = oneHotEncoder(y);

end

function[b, W] = initilizeWeightsAndBiases(K, d)
% b = zeros(K);
% W = zeros(K, d);

mu = 0;
std = 0.01;
W = mu.*randn(K, d) + std;
b = mu.*randn(K, 1) + std;

end

function[Y] = oneHotEncoder(y)
%creates matrix of one hot encoded data from an array of labels between 

N = length(y);
K = max(y) + 1; % since matlab uses 1 as the smallest indicies +1 is added
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

% disp('W');
% disp(size(W));
% 
% disp('X');
% disp(size(X));
% 
% disp('b');
% disp(size(b));


[d, n] = size(X); 
p_result = zeros(d, n);

s = bsxfun(@plus, W*X, b);
%s = W*X + b;
%P = exp(s)/ sum(exp(s));

exp_row_sum = sum(exp(s), 2);
P = bsxfun(@rdivide, exp(s), exp_row_sum);
end

