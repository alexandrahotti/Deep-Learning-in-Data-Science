

function[Xtr, Ytr, ytr] = Assignment1()
addpath Datasets/cifar-10-batches-mat/;


%[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat'); %training data
%[Xv, Yv, yv] = LoadBatch('data_batch_2.mat'); %validation data
%[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data


[d, N] = size(Xtr);
K = max(ytr); 




end

function[X, Y, y] = LoadBatch(filename)
%X: contains the image pixel data, has size d×N, is of type double or
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
    Y(i,dp_current_label) = 1; 
end

end

