# Reinforcement Learning Task 
## Tic-tac-toe Bot

In this task it was proposed to solve a Tic-tac-toe problem using reinforcement learning algorithms. 
The PPO algorithm was used as basis to solve the problem.

### The Problem
The tic-tac-toe simulation was made using Python 3. A game class was created, this class is capable to detect wrong movements, win conditions and its rewards. 

### Inputs and Outputs
The input to the critic and actor neural networks are the occupation grid of the x markers and the o markers, a domain of size 9*2 = 18


The output of the neural network are the probabilities of each possible action, which are to occupy any of the 9 squares of the game, only the action with highest probability is taken.
Note that any action is possible, even actions that are considered wrong movements. Those actions are punished with a reward of -5, encouraging the network to take only possible actions.

### Training

The training is done using the PPO algorithm, using a batch size of 2048 samples, and a minibatch size of 1024 samples. The network topography consists of 3 linear layers of size 1024 using the relu activation function, except the last layer of the critic and actor networks. The last layer of the critic network does not have a activation function, and the last layer of the actor network has a softmax activation function.

The training process consists of determining if the policy, or the random operator, should have the first go, using a random function. Then the policy and random operator takes turns on the board, until the game is won or a wrong movement is taken. If the random operator wins, the policy gets the double the opposite reward of the random operator win condition reward.

The training is relatively fast until the network learns how to play by the rules, after this point, it slows down a lot, as the random operator does not play with the objective to win. Only eventually the random operator will show some resemblance to a human playing the game. With enough time the neural network should learn all the random combinations that lead to success and failure, increasing its returns and win rate. 

To get a better training response, an adversarial network was implemented, using an older copy of the agent. As the adversarial network goal is to win the game, the training should be faster. 

### GUI

A simple GUI was developed using tkinter. It works exactly as the training process, replacing the random operator to a human operator.

### Performance


Currently the network is showing a win rate of 95.3% over the random operator. 


The returns of training are shown below. The trend is still upward, should be improving slowly over a lot of training episodes still. 

![Returns](returns.png?raw=true "Returns")

### How to use

The trained network are available in ./policy.

The training runs are stored in ./runs using tensorboard. Install tensorboard and run "tensorboard --logdir ./runs" to visualize

To train the network, just run train.py

To play with the network, just run gui.py

To see the demo, change the demo flag inside gui.py to "demo = True" and just click on any square of the gui, the random operator should work by itself, while the policy takes alternate turns.

To play via text with the network, as requested by the GithubGist, import the functions from game.py and use a python interactive console.
