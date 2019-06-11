function Assignment4_BONUS()
clear;

% set seed
rng(400);

% Get data
data_path = 'C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Assignment_4\BONUS\data\condensed\condensed_';
start_year = 2015;
end_year = 2018;

tweet_all_text = readTwitterData(start_year, end_year, data_path );


% Unique characters from tweets.
tweet_chars = unique(tweet_all_text);

% Maps between indicies and characters.
[char_to_ind, ind_to_char, K] = createMaps( tweet_chars );

eta = 0.1;% Learning rate during training
m = 100; % Dims of the hidden states
seq_length = 17; % Input sequence length
sig = 0.01;  

[b, c, U, W, V] = InitilizeRNNWeights(m, K, sig);

% Initilization of Bias vectors
RNN.b = b; RNN.c = c;
RNN.U = U; RNN.W = W; RNN.V = V;

RNN_M.b = zeros(size(RNN.b));RNN_M.c = zeros(size(RNN.c));
RNN_M.W = zeros(size(RNN.W)); RNN_M.U = zeros(size(RNN.U)); RNN_M.V = zeros(size(RNN.V));

% Get one hot encoding of the data.
[X, Y] = getData(tweet_all_text, seq_length, char_to_ind, K);



iter = 1; n_epochs = 15; smallest_loss = []; current_loss = 0;
hprev = []; RNN_min = RNN;
h_min = hprev;



% Train the RNN for n_epochs.
losses = trainingRNN(n_epochs,RNN, X, Y, K, m, eta, iter, RNN_M, ind_to_char, current_loss, tweet_all_text, char_to_ind, RNN_min, seq_length, h_min);


% Plot the smooth losses computed during training.
plotLosses(losses, x_label, y_label, plot_title );

end


function losses = trainingRNN(n_epochs,RNN, X, Y, K, m, eta, iter, RNN_M, ind_to_char, current_loss, tweet_all_text, char_to_ind, RNN_min, seq_length, h_min)
% Trains a RNN with several minibatches for several  epochs.

smallest_loss = 400;

for epoch = 1 : n_epochs
    
 
[RNN, current_loss, iter, RNN_M, RNN_min, current_min_loss, h_min] = MiniBatchGD(RNN, X', Y', seq_length, K, m, eta, iter, RNN_M, ind_to_char, current_loss(end), smallest_loss, tweet_all_text, char_to_ind, RNN_min, seq_length, h_min, epoch);
losses = [losses, current_loss];
  
         
%%%%%%%%%%%%% SAVE RESULTS %%%%%%%%%%%%%
save_path = 'C:\Users\Alexa\Desktop\KTH\årskurs_4\DeepLearning\Assignments\github\Deep-Learning-in-Data-Science\Assignment_4\BONUS\results\seq_lenghts\17_long\';
save_current_RNN(epoch, min_RNN, h_min, smallest_loss, save_path );


end    
    smallest_loss = current_min_loss; 
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


function save_current_RNN(epoch, min_RNN, min_h, min_loss, save_path )
% Saves a RNN struct.

min_loss_rounded = int32(round(min_loss));
file_name = [save_path, 'RNN_epoch_', num2str(epoch),'_loss_', num2str(min_loss_rounded),'.mat'];
file_name_h = [ save_path ,'\h_epoch_', num2str(epoch),'_loss_', num2str(min_loss_rounded),'.mat'];
save(file_name, 'min_RNN');
save(file_name_h, 'min_h');

end

 
function [X, Y] = getData(book_data, seq_length, char_to_ind, K)
% Converts textual data into one hot encoded format. 

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


function tweet_all_text = readTwitterData(start_year, end_year, data_path )
% Reads the twitter data in the form of jsonobjects between a year range.

tweet_all_text = [];

for year = start_year : end_year
    tweet_fname = [ data_path num2str(year) '.json'];
  
    tweets = jsondecode(fileread(tweet_fname));
    for tweet = 1 : length(tweets)
        try
          
        tweet_text = [tweets{tweet, 1}.full_text '§'];
        
        catch
            
            try
                tweet_text = [tweets{tweet, 1}.text '§'];
            catch
                tweet_text = [tweets(tweet).text '§'];
            end
            
        end
        
        tweet_all_text = [tweet_all_text, tweet_text];
    end 
end

end


function [RNN, all_loss, iteration, RNN_M, min_RNN, min_loss, min_h] = MiniBatchGD(RNN, X, Y, n, K, m, eta, iteration, RNN_M, ind_to_char, all_loss, min_loss, tweet_all_text, char_to_ind, min_RNN, seq_length, min_h, epoch)


smooth_losses = []; e = 1; tweet_text_left = true; end_of_tweet = false;
decay_rate = 0.9;


while tweet_text_left 

    % Get a batch of data
    X_batch = X(:, e : e + n - 1);
    Y_batch = Y(:, e + 1 : e + n);
    
    % Check if we ha reached the end of the current tweet.
    end_of_tweet_char = oneHotEncoding(char_to_ind('§'), K);
    [end_reached, index] = ismember(end_of_tweet_char, Xe', 'rows');
    
    if end_reached
        % Then we know that we passed an end of tweet character.

        % Update n such that it is updated to the number of steps right
        % before the end of tweet character.
        n = index;
        
        % Extract the batch again so that it stops at the end of tweet
        % character.
        X_batch = X(:, e : e + n - 1);
        Y_batch = Y(:, e : e + n - 1);
    end
    
    [hprev ]= updateHprev(m, e )
    
    
    if iteration == 1 || mod(iter, 10000) == 0
        getSynthesizedTweet(tweetChars, X, K, RNN, hprev, ind_to_char, iteration, epoch, all_loss, char_to_ind );
    end
    
    % Forward pass
    [curr_loss, H, P, A, o] = ForwardPass(RNN, X_batch, Y_batch, hprev, n);
    
    % Backward pass
    Gradient_vec = ComputeGradients(X_batch, Y_batch, n, P, H, A, RNN.V, RNN.W);
    [RNN_M, RNN] = AdaGrad(Gradient_vec, RNN_M, eta);
    
    
    if iter == 1 && e == 1
        all_loss = curr_loss;
    end
    
    % Update the total smooth loss.
    all_loss = 0.999 * all_loss + 0.001 * curr_loss;
    
    % If the newest computed loss is smaller than anny previosuly computed
    % loss we store this value and also synthesize text.
    if all_loss < min_loss
        
        getSynthesizedTweet(140, X_batch, K, RNN,hprev, ind_to_char, iteration, epoch, all_loss, char_to_ind );
        
        min_loss = all_loss; min_RNN = RNN; min_h = hprev;
        
    end
    
    % Store all the smooth losses for plotting.
    smooth_losses = [smooth_losses, all_loss];

    % Update were we are in the text.
    iteration = iteration + 1;
    e = e + n;
    
    % Check all the tweet text is read. I.e. if the epoch is done.
    if e > length(X) - n - 1
        tweet_text_left = false;
    end
    
    % Check if the end of a tweet is reached. We need to know this next 
    % loop to reset hprev to 0.
    if end_reached
        end_of_tweet = true;
        n = seq_length;
    end
    
    % Used to decay the learning rate.
    update_step_decay = 55000;
    if mod(iter, 55000 )
        eta = eta * decay_rate;
    end

end

end


function clipped_grad = clipGradients(grads, ind)
% Clip the gradients inorder to prevent exploding gradient .

clipped_grad = max( min(grads.(ind) ,5), -5);

end


function [RNN_M, RNN] = AdaGrad(Gradient_vec, RNN_M, learning_rate)
% Performs the backward pass using AdaGrad.

for f = fieldnames(RNN)'
    
	% Clips the gradients.
     curr_grad = f{1};
     Gradient_vec.(curr_grad) = clipGradients(Gradient_vec, curr_grad);

    RNN_M.(curr_grad) = RNN_M.(curr_grad) + Gradient_vec.(curr_grad).^2;
    RNN.(curr_grad) = RNN.(curr_grad) - learning_rate * (Gradient_vec.(curr_grad) ./( RNN_M.(curr_grad) + eps ).^0.5);
end
end


function getSynthesizedTweet(tweetChars, X, K, RNN, hprev, ind_to_char, iteration, epoch, loss_smooth, char_to_ind )
% Synthesizes a tweet with a specific lenght from the current trained
% parameters of the network.

        
    tweetInds = GenerateSequence(RNN, hprev, X(:, 1)', tweetChars, ind_to_char, char_to_ind);
    genTweet = [];

    for char = 1 : tweetChars
        genTweet = [genTweet ind_to_char(tweetInds(char))];
    end


    disp(['epoch = ' num2str(epoch) 'iter = ' num2str(iteration) ', smooth_loss = ' num2str(loss_smooth)]);
    disp(genTweet);

end


function Y = GenerateSequence(RNN, h0, x0, tweetChars, ind_to_char, char_to_ind)
% h0: hiddenstate at time 0. x0: dummy input to RNN.
% n: length of seugence to be genearted.

K = size(RNN.V,1); 

% Used to store one-hot encoded sampled characters.
Y = zeros(1, n);

h_t_1 = h0; 
x_next = x0;


end_of_tweet_char = oneHotEncoding(char_to_ind('§'), K);
% [end_reached, index] = ismember(end_of_tweet_char, Xe', 'rows');

generate_length_left = tweetChars;
curr_tweet_length = 0;

l = 0;

while curr_tweet_length < 140

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
    
    % Check if end of tweet char generated.
    curr_char = ind_to_char(label);
    [end_char_gen, ~] = ismember(end_of_tweet_char, curr_char, 'rows');
    
    % If an end of tweet char is generated. Then the user can still get a
    % new tweet with the remaining chars requested.
    if end_char_gen
        % We need to keep track of how many chars are generated so that the
        % tweet does not become longer than 140 chars.
        curr_tweet_length = 0;
    else
        % If a normal char is generated. Then the user has one less char
        % left. 
        generate_length_left = generate_length_left - 1;
        curr_tweet_length = curr_tweet_length + 1;
        
    end
end

end


function [hprev, end_of_tweet ] = updateHprev( m, e, end_of_tweet )
% Checks if hprev should be updated. It should be zero at the beginning of
% each epoch and at the beginning of each tweet.

  % Beginning of an epoch
  if e == 1
        hprev = zeros(m, 1);
   
   % Beginning of an tweet.
    elseif end_of_tweet
        hprev = zeros(m, 1);
        end_of_tweet = false;
  else 
   % A regular update in the middle of a tweet.
        hprev = h(:, end);
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


function onehotencoded_labels = oneHotEncoding(label, K)
% creates one hot encoded labels.

Identity_matrix = eye(K);
onehotencoded_labels = Identity_matrix(label, :);
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
% Computesrelative errors between gradients of the network and numerically
% computed gradients and creates boxplots.

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


function [loss, h, p, a, o] = ForwardPass(RNN, x, y, h_prev, tau)
% Performs the forwardpass in the RNN. 

[K, m] = size(RNN.V);

% Vectors to store partial results
h = zeros(m, tau); a_t = zeros(m, tau);
prob_t = zeros(K, tau); o_t = zeros(K, tau);

% Initilize parameters.
h(:, 1) = h_prev; 
loss = 0;

for t = 1 : tau
    a_t(:, t) = RNN.W * h(:, 1) + RNN.U  x(:, t) + RNN.b;
    h(:, t) = tanh( a_t(:, t) );
    o_t(:, t) = RNN.V * h(:, t) + RNN.c;
    prob_t(:, t) = exp( o_t(:, t) )/sum(exp( o_t(:, t) ));
    loss = loss - log( y(:, t)' * prob_t(:, t));
end

h = [h_prev, h];

end


function loss = ComputeLoss(X_batch, Y_batch, RNN, h)
% Computes the loss between the ground truth labels and computed
% probabilities for training data in X.

n = size(X_batch, 2);

[~, ~, P, ~, ~] = ForwardPass(RNN, X_batch, Y_batch, h, n)

loss_cross = diag( -log(Y_batch' * P) );

loss = sum( loss_cross );
end
