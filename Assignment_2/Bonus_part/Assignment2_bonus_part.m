function [ngrad_b, ngrad_W]= Assignment2()
clear;
addpath C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Datasets\cifar-10-batches-mat;

rng(400);

% Loading all availabe training data
[X1, Y1, y1] = LoadBatch('data_batch_1.mat'); %training data
[X2, Y2, y2] = LoadBatch('data_batch_2.mat'); %validation data
[X3, Y3, y3] = LoadBatch('data_batch_3.mat'); %training data
[X4, Y4, y4] = LoadBatch('data_batch_4.mat'); %training data
[X5, Y5, y5] = LoadBatch('data_batch_5.mat'); %training data

Xtr = [X1, X2, X3, X4, X5(:,1:9000)];
Ytr = [Y1, Y2, Y3, Y4, Y5(:,1:9000)];
ytr = [y1; y2; y3; y4; y5(1:9000)];

Xva = X5(:,9001:10000);
Yva = Y5(:,9001:10000);
yva = y5(9001:10000);

[Xte, Yte, yte] = LoadBatch('test_batch.mat'); %testing data

%preprocessing the data by giving it zero mean
[Xtr, Xva, Xte] = preProcessData(Xtr, Xva, Xte);
[d, n] = size(Xtr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% Best performing lambda %%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_points = 49000;

%%%%%% BONUS (b) %%%%%%%%%
lambda = 0.000648717 * 1;
m = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Mini batch gradient descent with cyclical learning

GDparameters = GDparams;
GDparameters.n_batch = 100;
GDparameters.eta_min = 1e-5;
GDparameters.eta_max = 1e-1;
GDparameters.eta = 0;
GDparameters.eta_vec = zeros(1005,1);
GDparameters.n_s = 980;
GDparameters.l = 0;
GDparameters.update_step = 0;
GDparameters.n_epochs = 12;

%Get the dimensions of the data
[d, ~] = size(Xtr);
K = max(ytr); 

% initilize the weights and biases
[b_init, W_init] = initilizeWeightsAndBiases(K, d, m);


[cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda);


% plotResults(cost_training, cost_validation, 490, no_points , GDparameters.n_batch, 'training cost', 'validation cost', 'Costs - 2 layer NN - cyclical learning', 'update step', 'cost');
% plotResults(loss_training, loss_validation, 490, no_points , GDparameters.n_batch, 'training loss', 'validation loss', 'Loss - 2 layer NN - cyclical learning', 'update step', 'loss');
% plotResults(acc_training, acc_validation, 490, no_points , GDparameters.n_batch, 'training accuracy', 'validation accuracy', 'Accuracy - 2 layer NN - cyclical learning', 'update step', 'accuracy');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST ACCURACY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc_test_fig3 = ComputeAccuracy(Xte, yte, Wstar, bstar);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


function GridSearch(Xtr, Ytr, ytr, Xva, Yva, yva, no_runs, no_cycles,GDparameters, l_min, l_max, b_init, W_init, file_name);
% Trains a 2 layer netural network using mini batch gradient descent and a
% cyclical learning rate for a number of lambdas given by: no_runs.

result_tables = cell(no_runs,1);

% no runs controlls how many lambdas we want to train the network for
for i = 1:no_runs
    
    % Drawing a lambda value
    l = l_min + ( l_max - l_min ) * rand(1,1);
    lambda = 10^l;
    results_grid_search(1,i) = lambda;
    
    % Gradeitn Descent 
    [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, Wstar, bstar] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparameters, W_init, b_init, lambda)
    
   
    % Storing the lambda and its loss, cost and accuracy
    lambda_vec = zeros(size(cost_training,1),1);     
    lambda_vec(1) = lambda;
   
    T = table(lambda_vec, cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation);
    writetable(T,strcat('tabledata', int2str(i), '.txt'),'Delimiter',';');
    result_tables{i} = table_name;
    
end


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
% dimensions and n = no data points.
% For reference: X: dxn

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


function [rel_err_gradb1, rel_err_gradb2, rel_err_gradW1, rel_err_gradW2] = computeRelativeErrorNumericalAnalyticalGradients(Xtr, Xva, Xte, Ytr, lambda, h, eps, m, no_points, no_dim )
% Computes the relative error between a numerically estimated gradient and
% an analytically computed gradient


% change the dimensions of the data 
%W = {W{1}(:,1:no_points),W{2}};
Xtr = Xtr(1:no_dim,1:no_points);
Ytr = Ytr(:,1:no_points);
Xva = Xtr(1:no_dim,1:no_points);
Yva = Ytr(:,1:no_points);
Xte = Xtr(1:no_dim,1:no_points);
Yte = Ytr(:,1:no_points);

%preprocessing the data
[Xtr, ~, ~] = preProcessData(Xtr, Xva, Xte);

%Get the dimensions of teh data
[d, ~] = size(Xtr);
K = 10; 

% initilize the weights and biases
[b, W] = initilizeWeightsAndBiases(K, d, m);


% compute the gradients
[grad_W, grad_b] = ComputeGradients(Xtr, Ytr, W, b, lambda);
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtr, Ytr, W, b, lambda, h);

% compute the relative errors
relative_error_gradb1 = abs(grad_b{1} - ngrad_b{1}) ./ max(eps, abs(ngrad_b{1}) + abs(grad_b{1}));
relative_error_gradW1 = abs(grad_W{1} - ngrad_W{1}) ./ max(eps, abs(ngrad_W{1}) + abs(grad_W{1}));
[w1_row, w1_col] = size(relative_error_gradW1);

relative_error_gradb2 = abs(grad_b{2} - ngrad_b{2}) ./ max(eps, abs(ngrad_b{2}) + abs(grad_b{2}));
relative_error_gradW2 = abs(grad_W{2} - ngrad_W{2}) ./ max(eps, abs(ngrad_W{2}) + abs(grad_W{2}));
[w2_row, w2_col] = size(relative_error_gradW2);

% create boxplots for each relative error
figure(1);
boxplot(relative_error_gradb1);
title('Relative error - numerical & analytical gradient - b1')
ylabel('relavtive error')
hold on;

figure(2);
boxplot(relative_error_gradb2);
title('Relative error - numerical & analytical gradient - b2')
ylabel('relavtive error')
hold on;

figure(3);
boxplot(reshape(relative_error_gradW1,w1_row*w1_col,1));
title('Relative error - numerical & analytical gradient - W1')
ylabel('relavtive error')
hold on;

figure(4);
boxplot(reshape(relative_error_gradW2,w2_row*w2_col,1));
title('Relative error - numerical & analytical gradient - W2')
ylabel('relavtive error')
hold on;
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


function[b, W] = initilizeWeightsAndBiases(K, d, m)
% b and W are initilized using a normal distribution 

mu = 0;
std_1 = 1/sqrt(d);
std_2 = 1/sqrt(m);

W_1 = std_1.*randn(m,d) + mu;
W_2 = std_2.*randn(K,m) + mu;

b_1 = zeros(m,1);
b_2 = zeros(K,1);

W = {W_1,W_2};
b = {b_1,b_2};

end


function [cost_training, cost_validation, loss_training, loss_validation, acc_training, acc_validation, W, b] = GradientDescentCyclicalLearning (Xtr, Ytr, ytr, Xva, Yva, yva, GDparams, W, b, lambda)
% This function performs gradient descent on minibatches using cyclical learning and run through
% the the training images the number of times given by: GDparams.n_epochs


% Arrays used to store costs, losses and accuracies
cost_training = zeros(GDparams.n_epochs+1, 1);
cost_validation = zeros(GDparams.n_epochs+1, 1);

loss_training = zeros(GDparams.n_epochs+1, 1);
loss_validation = zeros(GDparams.n_epochs+1, 1);

acc_training = zeros(GDparams.n_epochs+1, 1);
acc_validation = zeros(GDparams.n_epochs+1, 1);


% We run through the epochs
for i = 1:GDparams.n_epochs
    
    [cost_tr, loss_tr] = ComputeCost(Xtr, Ytr, W, b, lambda);
    
    cost_training(i) = cost_tr;
    loss_training(i) = loss_tr;
    
    [cost_val, loss_val]  = ComputeCost(Xva, Yva, W, b, lambda);
    
    cost_validation(i) = cost_val;
    loss_validation(i) = loss_val;
    
    acc_training(i) = ComputeAccuracy(Xtr, ytr, W, b);
    acc_validation(i) = ComputeAccuracy(Xva, yva, W, b);
    
   
    [W, b, GDparams] = MiniBatchGDCyclicalLearning(Xtr, Ytr, GDparams, W, b, lambda);
    
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
end



end

function [Wstar, bstar,GDparams ] = MiniBatchGDCyclicalLearning(X, Y, GDparams, W, b, lambda)
%Performs gradient descent on batches of data
% X: dxN

[~, N] = size(X);

    for j=1:N/GDparams.n_batch
        
        j_start = (j-1)*GDparams.n_batch + 1;
        j_end = j*GDparams.n_batch;
        inds = j_start:j_end;
        
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
        
        % We increase the update step
        GDparams.update_step = GDparams.update_step + 1;
        
        % The cyclical learning rate is updated
        [ eta_t ] = updateCyclicalLearningRate(GDparams);
        GDparams.eta = eta_t;
        GDparams.eta_vec(GDparams.update_step) = eta_t;
       
        % We want to update the l parameter to keep track of the current
        % cycle
        if mod( GDparams.update_step, GDparams.n_s*2 ) == 0
            GDparams.l = GDparams.l + 1;
        end
        
        % The weights and biases are updated
        b{1} = b{1} - GDparams.eta * grad_b{1};
        W{1} = W{1} - GDparams.eta * grad_W{1};
        
        b{2} = b{2} - GDparams.eta * grad_b{2};
        W{2} = W{2} - GDparams.eta * grad_W{2};
    end
    
    % The final b and W trained 
    bstar = b;
    Wstar = W;
end

function [eta_t ] = updateCyclicalLearningRate(GDparams)
% Updates the cyclical learning rate at time t

eta_t = double(0);

% We check if t is within a lower and upper range. If nt is increasing 
lower1 = (2*GDparams.l*GDparams.n_s) <= GDparams.update_step;
upper1 = GDparams.update_step <= ((2*GDparams.l+1)*GDparams.n_s) ;


if lower1 && upper1
    eta_t = GDparams.eta_min + ((GDparams.update_step - 2*GDparams.l*GDparams.n_s)/( GDparams.n_s ))*(GDparams.eta_max - GDparams.eta_min);
end


upper2 = GDparams.update_step >= (2 * GDparams.l + 1)*GDparams.n_s ;
lower2 = GDparams.update_step <= 2*(GDparams.l + 1)*GDparams.n_s ;

% We check if t is within a lower and upper range. If nt is deacreasing
if upper2 && lower2
    
    eta_t = GDparams.eta_max - ((GDparams.update_step - (2*GDparams.l+1)*GDparams.n_s)/GDparams.n_s)*(GDparams.eta_max - GDparams.eta_min);
end

end


function [cost_training, cost_validation, W, b] = GradientDescent(Xtr, Ytr, Xva, Yva, GDparams, W, b, lambda)
% This function performs gradient descent on minibatches using vanilla update and run through
% the the training images the number of times given by: GDparams.n_epochs 


% Vectors to store results
cost_training = zeros(GDparams.n_epochs, 1);
cost_validation = zeros(GDparams.n_epochs, 1);

loss_training = zeros(GDparams.n_epochs, 1);
loss_validation = zeros(GDparams.n_epochs, 1);

% we go through all the epochs
for i = 1:GDparams.n_epochs
    
    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, GDparams.n_batch);
    
    % compute results
    [cost_training(i), loss_training(i)] = ComputeCost(Xtr, Ytr, W, b, lambda);
    [cost_validation(i), loss_validation(i)]  = ComputeCost(Xva, Yva, W, b, lambda);

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
%X: dxN
%Y: is K×N
%y: is a vector of length N containing the one hot encoding labels for each image. 

A = load(filename);

X = double(A.data');
%X = X / 255;

y = A.labels;
y = y + 1 ;
Y = oneHotEncoder(y);
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

function [J, loss] = ComputeCost(X, Y, W, b, lambda)
%X:  d×n.
%Y: K×n. The one-hot ground truth labels
%J: sum of the loss of the network’s predictions

[d, n] = size(X);
regularization_term = 0;

% Calculate the regularization term
for i = 1:2
regularization_term = regularization_term + lambda * sum( sum(W{i}.^2 ));
end

P = EvaluateClassifier(X, W, b); 

%Compute the loss and the cost
l_cross = -log(dot(Y,P));
loss = (sum(l_cross)/n);
J = loss + regularization_term;
end


function acc = ComputeAccuracy(X, y, W, b)
%Evaluates the accuracy of trained Weights and bias parameters

% X: d×n.
% y: labels of length n
% acc: the accuracy.

P = EvaluateClassifier(X, W, b); % Kxn

[M, I] = max(P,[],1); 
[total_no_classifications, ~] = size(y);
[~, no_accurate_classifications] = size(find(I==y'));

acc = no_accurate_classifications/total_no_classifications;

end

function [ X ] = applyJitter(X)
% BONUS task e - applies a random jitter to each element in the training
% data from a uniform distribution in the range of +/- 2%
[d, n] = size(X);

jitter = -0.0025 + (0.0025+0.0025)*rand(d,n);

jitter = X.*jitter;

X = X + jitter;
end


function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
% Backwardpass: computes the gradients of the loss wrt W1, W2, b1 and b2
% The steps follow slide 30 in lecture 4

%%%%%% (e) BONUS %%%%%%
%X = applyJitter(X);
%%%%%%%%%%%%%%%%%%%%%%%

[~, n] = size(X);
vec_ones = ones(n,1); % Nx1

%Forward pass
[P_batch, h_batch] = EvaluateClassifier_fp(X, W, b); %  KxN  ,   mxN

%Backward pass
G_batch = -(Y - P_batch); % KxN


dL_dW2 = (1/n) * G_batch * h_batch' + 2 * lambda * W{2}; %Kxm
dL_db2 = (1/n) * G_batch * vec_ones; %Kx1

%Propagate the gradient back through the second layer 
G_batch = W{2}' * G_batch; %mxN
G_batch = G_batch .* (h_batch > 0); %mxN

dL_dW1 = (1/n) * G_batch * X' + 2 * lambda * W{1}; %mxd
dL_db1 = (1/n) * G_batch * vec_ones; %mx1

grad_W = {dL_dW1, dL_dW2};
grad_b = {dL_db1, dL_db2};
end

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, n_batch)
%Performs gradient descent on batches of data
% X: dxN

[~, N] = size(X);

    for j=1:N/n_batch
        
        
        j_end = j*n_batch;
        inds = j_start:j_end;
        
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
       
        b{1} = b{1} - GDparams.eta * grad_b{1};
        W{1} = W{1} - GDparams.eta * grad_W{1};
        
        b{2} = b{2} - GDparams.eta * grad_b{2};
        W{2} = W{2} - GDparams.eta * grad_W{2};
    end
    
    bstar = b;
    Wstar = W;
end


function [P, h] = EvaluateClassifier_fp(X, W, b)
% Evaluates a classifer via the forwardpass

%%%% BONUS (d) drop out according to lecture 5 slide 82 %%
p = 0.3; % probability of dropping an activation

[d, n] = size(X); 

% slide 29 lecture 4
b_1_matrix = repmat( b{1}, [1, n] );
b_2_matrix = repmat( b{2}, [1, n] );


s_j = W{1} * X + b_1_matrix;

h = max(0, s_j);

%%% BONUS: Randomly dropping activations
u1 = rand(size(h)) < p;
h = h.*u1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s =  W{2}*h + b_2_matrix;

% According to the assignment description p.1, eq. (5)
[K, ~] = size(s);
exp_row_sum = sum(exp(s));
P = exp(s)./ repmat(exp_row_sum, K, 1);
end

function [P, h] = EvaluateClassifier(X, W, b)
% Evaluates a classifer via the forwardpass

[d, n] = size(X); 

% slide 29 lecture 4
b_1_matrix = repmat( b{1}, [1, n] );
b_2_matrix = repmat( b{2}, [1, n] );


s_j = W{1} * X + b_1_matrix;

h = max(0, s_j);


s =  W{2}*h + b_2_matrix;

% According to the assignment description p.1, eq. (5)
[K, ~] = size(s);
exp_row_sum = sum(exp(s));
P = exp(s)./ repmat(exp_row_sum, K, 1);
end




function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
% computes numerical gradients for W and b

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        [c1, ~] = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        [c1, ~] = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end
