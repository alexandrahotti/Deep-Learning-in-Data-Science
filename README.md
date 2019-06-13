# Deep Learning in Data Science
This repository contains assignment solutions for the course DD2424 Deep Learning in Data Science.


## Assignment 1 - A one layer network
In assignment 1 a one layer network with multiple outputs is trained to 
classify images from the CIFAR-10 dataset. The network is trained using mini-batch gradient descent applied to a cost function
that computes the cross-entropy loss with an L2 regularization term on the weight matrix.

In the first part of the bonus assignment the performance of this one layer network is improved. In the next part of the bonus assignment a one layer network is trained using a SVM multi-class loss.

## Assignment 2 - A two layer network
In assignment 2 a two layer network with multiple outputs is trained to classify images from the CIFAR-10 dataset. 
This network is trained using a cyclical learning rate as this approach eliminates much of the trial-and-error associated 
with finnding a good learning rate and some of the costly hyper- parameter optimization over multiple parameters associated 
with training with momentum. The main idea of cyclical learning rates is that during training the learning rate is periodically 
changed in a systematic fashion from a small value to a large one and then from this large value back to the small value. And 
this process is then repeated again and again until training is stopped. 

## Assignment 3 - A k layer network
In assignment 3 the code from assignment 2 is generalized such that it can handle any number of layers. Also the network incorporates batch normalization.

## Assignment 4 - A RNN used to synthesize text
In assignment 4 a RNN is trained to synthesize English text character
by character. First  a vanilla RNN is trained using the text from the book 
The Goblet of Fire by J.K. Rowling. AdaGrad is used for optimization.

In the Bonus Part of the assignment text was synthesized using text from Donald Trumps twitter account.
