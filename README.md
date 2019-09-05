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

In the Bonus Part of the assignment text was synthesized using text from Donald Trump's twitter account.

### Excerpt of results

#### Synthesized text - Harry Potter and the Goblet of Fire

 your voice ears, yesuse said, bus beteragring?" said **Bagman** you's soppome a?"
There Cuntupily leoc kitch the Midingicidestly."
"Sod numored to **Cedric** weires.  Nide feal **Hagrid**.
"Nor  liftered old secled, ans root.  I yining abots mound. ... Tress fired one to chingested else a here tay gated to Debbasley, lunget and lot the **Durmst**ended. . . ." Paint ac Sert. . . straid she spomsthed the rofecodyy **Hermione** had for student from all onto mound Death; they wopping.
"What up they interumped at rail it.  Yeh.  They're was hand **Harry** have toward telfing appide awrothould **Siri**age she had nothor ickiet, a fit to schoundoning ay.
"Mapperly tust sixts.  He seieven she back in ontigind **Hogwarts**," said **Ron**."
"Dad dundridesble onebly.
	"stath that the deed in he fromt giving, was **Krum's** chom, the closed of that the not anything have, **Harry** all odd they shought excapped, so fliensint.
  
  #### Synthesized text - Donald Trump’s twitter account
  atad and **borders** has uncetich wondess, **good** revemennerny!§Joxecoblougit Vator) of **Democrat**on in InIC **Korea** latiot **bad** TODRE!
https://t.co/M3

im you! **Love** stthis such Mem Your offict."SCATK!
**#Trump2016**
#Menjesl you 2016 
**poor, and of through America take has riched**§A whogalyghed 

clawing the kneais on doy trun't hanyive I werroong hinn goime Strie. **#MakeAmericaGreat**

ill bichin! 
#O.NP on a STroucested, even touation who with unyorm. I **fear sv sartary **Hillary!** httpshaspucal **President U.S. @realDonald**Trciegh
