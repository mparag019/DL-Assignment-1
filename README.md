# CS6910: DL Assignment 1


This repository contains Python code for implementing various gradient descent optimization algorithms for training neural networks using back propagation algorithm for a classification task. The primary goal of this project is to explore and compare different optimization techniques and their impact on model training performance.

##Wandb report link: 
https://api.wandb.ai/links/cs23m047/8bnbwbn1

## Features

- Implementation of gradient descent, stochastic gradient descent (SGD), momentum, Nesterov accelerated gradient (NAG), RMSprop, Adam, and NAdam optimization algorithms.
- Support for different activation functions including sigmoid, ReLU, tanh, and identity.
- Flexible configuration options including the choice of loss function (mean squared error or cross-entropy), learning rate, batch size, number of epochs, and network architecture.
- Utilization of the Fashion-MNIST dataset for training and evaluation.
- Integration with Weights and Biases (wandb) for experiment tracking and visualization.

## Usage Instructions
Run the `train.py` script with appropriate command-line arguments to train and evaluate the neural network using different optimization algorithms and configurations.
Example
```bash
python train.py --wandb_entity myname --wandb_project myprojectname --dataset fashion_mnist --epochs 10 --batch_size 64 --loss cross_entropy --optimizer adam --learning_rate 0.001 --momentum 0.9 --beta 0.9 --beta1 0.9 --beta2 0.999 --epsilon 1e-10 --weight_decay 0.0005 --weight_init Xavier --num_layers 5 --hidden_size 64 --activation ReLU
```
This command runs the train.py script with the specified parameters for training and evaluating the neural network. Adjust the arguments as needed for your specific experiment. After running this you will get the plots for validation accuracy, validation loss, training accuracy, and training loss on wandb

## Results
Achieved maximum accuracy of 87.83% for Fashion MNIST dataset and 96.95% for MNIST dataset for the following configuration

- Model Configuration:
  - Number of Hidden Layers: 5
  - Number of Hidden Neurons: 64
  - Weight Decay: 0.0005
  - Activation Function: tanh
  - Weight Initialization: Xavier
  - Optimizer: ADAM
  - Learning Rate: 0.001
  - Batch Size: 64
  - Loss Type: Cross Entropy




