 function Assignment4()
clear;

% set seed
rng(4001);

% Get data
data_path = 'C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Assignment_4\data/goblet_book.txt';
book_data = loadData(data_path); 

% Unique characters in the book.
book_chars = unique(book_data);

% Maps between indicies and characters.
[char_to_ind, ind_to_char, K] = createMaps( book_chars );

eta = 0.1; % Learning rate during training
m = 100; % Dims of the hidden states
seq_length = 25; % Input sequence length
sig = 0.01;  

[b, c, U, W, V] = InitilizeRNNWeights(m, K, sig);

% Initilization of Bias vectors
RNN.b = b; RNN.c = c;
RNN.U = U; RNN.W = W; RNN.V = V;

RNN_M.b = zeros(size(RNN.b));RNN_M.c = zeros(size(RNN.c));
RNN_M.W = zeros(size(RNN.W)); RNN_M.U = zeros(size(RNN.U)); RNN_M.V = zeros(size(RNN.V));

% Get one hot encoding of the data.
[X, Y] = getData(tweet_all_text, seq_length, char_to_ind, K);

iteration = 1; numEpochs = 15;
loss_vec = []; curr_loss = 0;
hprev = []; min_loss = 444;
min_rnn_list =[]; min_losses_list =[];
min_RNN = RNN;

losses = trainRNN( iteration, numEpochs, curr_loss, loss_vec, hprev, min_loss, min_rnn_list, min_losses_list, min_RNN ,RNN, X, Y,seq_length, eta, RNN_M, ind_to_char, book_data,char_to_ind ); );

% Plot the computed losses.
plotLosses(losses, 'update step', 'smooth loss', 'Smooth loss for 2 epochs of training' )

 end
 
 
 function loss_vec = trainRNN( iteration, numEpochs, curr_loss, loss_vec, hprev, min_loss, min_rnn_list, min_losses_list, min_RNN ,RNN, X, Y,seq_length, eta, RNN_M, ind_to_char, book_data,char_to_ind )
 % Trains a RNN with several minibatches for several  epochs.
 
 for epoch = 1 : numEpochs
  
    [RNN, curr_loss, iteration, RNN_M, min_RNN, new_min_loss] = MiniBatchGD(RNN, X', Y', seq_length, K, m, eta, iteration, RNN_M, ind_to_char, curr_loss(end), loss_min, book_data, char_to_ind, min_RNN, epoch);
    loss_vec = [loss_vec, curr_loss];

    
    if new_min_loss < loss_min
        min_losses_list = [min_losses_list, new_min_loss ];
        min_rnn_list = [min_rnn_list, min_RNN ];
        
    end
    
    loss_min = new_min_loss;
    
 end
 end

 
 function plotLosses(losses, x_label, y_label, plot_title )
% Plots a sequence of losses.

number_of_losses= 1 : length(losses);
plot(number_of_losses , losses)
title(plot_title)
xlabel(x_label)
ylabel(y_label)

end


function clipped_grad = clipGradients(grads, ind)
% Clip the gradients inorder to prevent exploding gradient .

clipped_grad = max( min(grads.(ind) ,5), -5);

end


function [RNN, loss_vec, iteration, M, min_RNN, min_loss] = MiniBatchGD(RNN, X, Y, n, K, m, eta, iteration, RNN_M, ind_to_char, smoothed_loss, min_loss, book_data, char_to_ind, min_RNN, epoch)
% Perform minibatch gradient descent on mini batches of data.


textlen = 200; loss_vec = []; e = 1;


while endTextNotReached(e, n, X) 
    
    % Get the current minibatch given by the sequence length.
    X_batch = X(:, e : e + n - 1);
    Y_batch = Y(:, e : e + n - 1);
    
    % Update hprev.
    hprev = updateHprev( m, e );
    
    % Forward pass
    [curr_loss, H, P, A, O] = ForwardPass(RNN, X_batch, Y_batch, hprev, n);
    
    % Backward pass
    Gradient_vec = ComputeGradients(X_batch, Y_batch, n, P, H, A, RNN.V, RNN.W);
    [RNN_M, RNN] = AdaGrad(Gradient_vec, RNN_M, eta, RNN);
    
    
    if iteration == 1 && e == 1
        smoothed_loss = curr_loss;
    end
    
    smoothed_loss = 0.001 * curr_loss + .999 * smoothed_loss ;
    if smoothed_loss < min_loss
        
    % A new RNN model which achived the current smallest loss was detected.
         min_RNN = RNN;
         min_loss = smoothed_loss;
        
        % Synthesize text.
        getSynthesizedText(1000, X_batch(:, 1)', K, RNN, hprev, ind_to_char, iteration, epoch, smoothed_loss, char_to_ind );
        
    end
    
    % Generate  text every 10000 update step.
    if mod(iteration, 10000) || (iteration == 1 && e == 1 ) == 0
        getSynthesizedText(textlen, X_batch(:, 1)', K, RNN, hprev, ind_to_char, iteration, epoch, smoothed_loss, char_to_ind );
    end
    
    % Store the calculated loss in the loss vector.
    loss_vec = [loss_vec, smoothed_loss];
    
    % Update where we are in the text and the update step.
    e = e + n; iteration = iteration + 1;
end

end


function not_reached = endTextNotReached(e, n, X)
% Checks if the end of the text is reached.

 if e <= length(X) - n - 1 
    not_reached = false;
 else
     not_reached = true;
 end

end


function hprev = updateHprev( m, e )
% Checks if hprev should be updated. It should be zero at the beginning of
% each epoch.

  % Beginning of an epoch
  if e == 1
        hprev = zeros(m, 1);
   
  else 
   % A regular update in the middle of the text.
        hprev = h(:, end);
    end

end


function getSynthesizedText(tweetChars, X, K, RNN, hprev, ind_to_char, iteration, epoch, loss_smooth, char_to_ind )
% Synthesizes a tweet with a specific lenght from the current trained
% parameters of the network.

        
    textInds = GenerateSequence(RNN, hprev, X(:, 1)', tweetChars);
    genText = [];

    for char = 1 : tweetChars
        genText = [genText ind_to_char(textInds(char))];
    end


    disp(['epoch = ' num2str(epoch) 'iter = ' num2str(iteration) ', smoothed loss = ' num2str(loss_smooth)]);
    disp(genText);

end


function Y = GenerateSequence(RNN, h0, x0, n)
% Generates a sequence of a specified length.

K = size(RNN.V,1); 

% Used to store one-hot encoded sampled characters.
Y = zeros(1, n);

h_t_1 = h0; 
x_next = x0;

for l = 1:n
    a_t = RNN.W * h_t_1 + RNN.U * x_next + RNN.b;
    h_t = tanh( a_t );
    o_t = RNN.V * h_t + RNN.c;
    
    % probabilities for each possible character.
    p_t = softmax( o_t );
    
    % Sample a label and convert into a character.
    label = sampleChar(p_t);
    x_next = oneHotEncoding( label , K)';
    
    Y(l) = label;
    h_t_1 = h_t;
end

end


function [RNN_M, RNN] = AdaGrad(Gradient_vec, RNN_M, learning_rate, RNN)
% Performs the backward pass using AdaGrad.

for f = fieldnames(RNN)'
    
	% Clips the gradients.
     curr_grad = f{1};
     Gradient_vec.(curr_grad) = clipGradients(Gradient_vec, curr_grad);

    RNN_M.(curr_grad) = RNN_M.(curr_grad) + Gradient_vec.(curr_grad).^2;
    RNN.(curr_grad) = RNN.(curr_grad) - learning_rate * (Gradient_vec.(curr_grad) ./( RNN_M.(curr_grad) + eps ).^0.5);
end
end


function [X, Y] = getData(book_data, seq_length, char_to_ind, K)

X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

% Convert X_chars and Y_chars into one-hot encoding of their characters. 

X = zeros(K,seq_length);
Y = zeros(K, seq_length);

for i = 1:seq_length
    Y(:,i) = oneHotEncoding(char_to_ind(Y_chars(i)),K);
    X(:,i) = oneHotEncoding(char_to_ind(X_chars(i)),K);
    
end
end


function gradients = ComputeGradients(X, Y, tau, P, h, a, V, W)
% Computes the gradients of the parameters of the network to be learned.

% initialize parameters
m = size(W, 1);
dlda = zeros(tau, m);
dldh = zeros(tau, m);


g = -(Y - P)';                                         
gradients.c = (sum(g))';                                           
gradients.V = g'*h(:, 2 : end)';                            

dldh(tau, :) = g(tau, :) * V;                             
dlda(tau, :) = dldh(tau, :) * diag( 1 - ( tanh( a(:, tau) ) ).^2 );      

% Iteratively compute the graidents for the loss wrt h and a.
for t = tau - 1 : -1 : 1
    dldh(t, :) = g(t, :) * V + dlda(t + 1, :) * W;
    dlda(t, :) = dldh(t, :) * diag( 1 - ( tanh( a(:, t) ) ).^2 );
end

gradients.W = dlda' * h(:, 1 : end - 1)';
gradients.U = dlda' * X';
gradients.b = ( sum(dlda) )';

end


function onehotencoded_labels = oneHotEncoding(label, K)
% creates one hot encoded labels.

Identity_matrix = eye(K);
onehotencoded_labels = Identity_matrix(label, :);
end


function ii = sampleChar(p)
% Samples a charcter based on the probability p.

cp = cumsum(p);
a = rand;
ixs = find(cp-a > 0 );
ii = ixs(1);
end


function [loss, h_t, prob_t, a_t, o_t] = ForwardPass(RNN, x, y, h_prev, tau)
% Performs the forwardpass in the RNN. 

[K, m] = size(RNN.V);

% Vectors to store partial results
h_t = zeros(m, tau); a_t = zeros(m, tau);
prob_t = zeros(K, tau); o_t = zeros(K, tau);

% Initilize parameters.
h(:, 1) = h_prev; 
loss = 0;

for t = 1 : tau
    a_t(:, t) = RNN.W * h_t(:, 1) + RNN.U  x(:, t) + RNN.b;
    h_t(:, t) = tanh( a_t(:, t) );
    o_t(:, t) = RNN.V * h_t(:, t) + RNN.c;
    prob_t(:, t) = exp( o_t(:, t) )/sum(exp( o_t(:, t) ));
    loss = loss - log( y(:, t)' * prob_t(:, t));
end

h_t = [h_prev, h_t];

end


function [b, c, U, W, V] = InitilizeRNNWeights(m, K, sig)

% bias vectors initilized to zero.
b = zeros(m,1);
c = zeros(K,1);

% Sample weight initilization from a normal distribution with
% standardeviation given by sig.
U = randn(m, K)*sig;
W = randn(m, m)*sig;
V = randn(K, m)*sig;

end


function [char_to_ind, ind_to_char, K] = createMaps( book_chars )
% Creates mapping between indices and characters.

K = size(book_chars , 2);

map_char_ind = containers.Map(num2cell(tweet_chars), 1 : length(num2cell(tweet_chars)));
map_ind_char = containers.Map(1 : length(num2cell(tweet_chars)), num2cell(tweet_chars));

char_to_ind = [containers.Map('KeyType','char','ValueType','int32'); map_char_ind];
ind_to_char = [containers.Map('KeyType','int32','ValueType','char'); map_ind_char];

end


function [relative_error_gradb, relative_error_gradc, relative_error_gradW, relative_error_gradU,relative_error_gradV] = computeRelativeErrorsGradients(Gradients, numerical_Gradients )


relative_error_gradb = abs(Gradients.b - numerical_Gradients.b) ./ max(eps, abs(Gradients.b) + abs(numerical_Gradients.b));
relative_error_gradc = abs(Gradients.c - numerical_Gradients.c) ./ max(eps, abs(Gradients.c) + abs(numerical_Gradients.c));

relative_error_gradW = abs(Gradients.W - numerical_Gradients.W) ./ max(eps, abs(Gradients.W) + abs(numerical_Gradients.W));
relative_error_gradU = abs(Gradients.U - numerical_Gradients.U) ./ max(eps, abs(Gradients.U) + abs(numerical_Gradients.U));
relative_error_gradV = abs(Gradients.V - numerical_Gradients.V) ./ max(eps, abs(Gradients.V) + abs(numerical_Gradients.V));

figure();
boxplot(relative_error_gradW(:));
title('Relative error between numerical and analytical gradient for W')
xlabel('Relative Error - Gradient W')
ylabel('relavtive error')

figure();
boxplot(relative_error_gradU(:));
title('Relative error between numerical and analytical gradient for U')
xlabel('Relative Error - Gradient U')
ylabel('relavtive error')

figure();
boxplot(relative_error_gradV(:));
title('Relative error between numerical and analytical gradient for V')
xlabel('Relative Error - Gradient V')
ylabel('relavtive error')

figure();
boxplot(relative_error_gradb(:));
title('Relative error between numerical and analytical gradient for b')
xlabel('Relative Error - Gradient b')
ylabel('relavtive error')

figure();
boxplot(relative_error_gradc(:));
title('Relative error between numerical and analytical gradient for c')
xlabel('Relative Error - Gradient c')
ylabel('relavtive error')

end


function data = loadData(file_name)
% Given a txt filename, the content of this file is loaded into a vector.

file_id = fopen(file_name,'r');
data = fscanf(file_id,'%c');
fclose(file_id);

end


function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end
end


function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end

end


function loss = ComputeLoss(X_batch, Y_batch, RNN, h)
% Computes the loss between the ground truth labels and computed
% probabilities for training data in X.

n = size(X_batch, 2);

[~, ~, P, ~, ~] = ForwardPass(RNN, X_batch, Y_batch, h, n);

loss_cross = diag( -log(Y_batch' * P) );

loss = sum( loss_cross );
end
