# CS6910: DL Assignment 1


This repository contains Python code for implementing various gradient descent optimization algorithms for training neural networks using back propagation algorithm for a classification task. The primary goal of this project is to explore and compare different optimization techniques and their impact on model training performance.

## Wandb Report link: 
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

### Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.999 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 1e-10 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.0005 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 5 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 64 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

<br>

This command runs the train.py script with the specified parameters for training and evaluating the neural network. Adjust the arguments as needed for your specific experiment. After running this you will get the plots for validation accuracy, validation loss, training accuracy, and training loss on wandb

## Results
Maximum accuracy of 87.83% for Fashion MNIST dataset and 96.95% for MNIST dataset is achieved for the following configuration

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




