import argparse
import numpy as np
import random
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.datasets import mnist
import sys
import wandb
wandb.login()



class_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

k = 10

def showImage():
  label = []
  images = []

  for i in range(len(x_train)):
    if len(label) == 10:
      break
    if y_train[i] not in label:
      images.append(x_train[i])
      label.append(y_train[i])
  for i in range(len(images)):
    plt.imshow(images[i], cmap = 'gray')
    plt.title(class_labels[label[i]])
    plt.axis("off")
    plt.show()


def sigmoid(x):
  return 1/(1 + np.exp(-x))

def relu(x):
  if x < 0:
    return 0
  else:
    return x

def tanh(x):
  return np.tanh(x)

def identity(x):
    return x

def softmax(x):
  max_float_value = sys.float_info.max
  sum = 0.0
  for i in range(len(x)):
    if (sum + np.exp(x[i]) <= max_float_value):
      sum += np.exp(x[i])
    else:
      sum = max_float_value
  y = []
  for i in range(len(x)):
    y_i = np.exp(x[i])/sum
    y.append(y_i)
  return y

def initialize_param(input_size, n, neurons):
    W = []
    b = []
    for i in range(0, n+1):
        if i == 0:
            W_i = np.random.rand(neurons,input_size) * 0.01
            W.append(W_i)
        elif i == n:
            W_i = np.random.rand(10,neurons) * 0.01
            W.append(W_i)
        else :
            W_i = np.random.rand(neurons,neurons) * 0.01
            W.append(W_i)

        if i == n:
            b_i = np.random.rand(10,1) * 0.01
            b.append(b_i)
        else :
            b_i = np.random.rand(neurons,1) * 0.01
            b.append(b_i)
    return W, b

def xavier_initialization(input_size, n, neurons):
  W = []
  b = []
  for i in range(0, n+1):
    if i == 0:
        variance = 6.0 / (input_size + neurons)
        std_dev = np.sqrt(variance)
        W_i = np.random.normal(0, std_dev, size=(neurons, input_size))
        W.append(W_i)
    elif i == n:
        variance = 6.0 / (10 + neurons)
        std_dev = np.sqrt(variance)
        W_i = np.random.normal(0, std_dev, size=(10, neurons))
        W.append(W_i)
    else :
        variance = 6.0 / (neurons + neurons)
        std_dev = np.sqrt(variance)
        W_i = np.random.normal(0, std_dev, size=(neurons, neurons))
        W.append(W_i)

    if i == n:
        variance = 2.0 / (10 + 1)
        std_dev = np.sqrt(variance)
        b_i = np.random.normal(0, std_dev, size=(10, 1))
        # b_i = np.zeros((10,1))
        b.append(b_i)
    else :
        variance = 2.0 / (neurons + 1)
        std_dev = np.sqrt(variance)
        b_i = np.random.normal(0, std_dev, size=(neurons, 1))
        # b_i = np.zeros((neurons,1))
        b.append(b_i)
  return W, b

def forward_propogation(func, x, W, b, n, neurons):
  a = []
  h = []
  for k in range(n+1):
    if k == 0:
      a_k = np.add(np.dot(W[k], x), b[k])
      a.append(a_k)
    else:
      a_k = np.add(np.dot(W[k], h[k-1]), b[k])
      a.append(a_k)
    if k == n: break
    h_k = []
    a[k] = np.clip(a[k], -709.78, 709.78)

    for j in range(neurons):
      if func == "sigmoid":
        h_kj = sigmoid(a[k][j][0])
      elif func == "ReLU":
        h_kj = relu(a[k][j][0])
      elif func == "tanh":
        h_kj = tanh(a[k][j][0])
      elif func == "identity":
        h_kj = identity(a[k][j][0])
      h_k.append(h_kj)
    h_k = np.array(h_k).reshape(neurons,1)
    h.append(h_k)
  a[n] = np.clip(a[n], -500, 500)
  y_pred = np.array(softmax(a[n]))
  return a, h, y_pred

def diff_g(func, a):
  diff_a = []
  if func == "sigmoid":
    for i in a:
      i = np.clip(i, -709.78, 709.78)
      diff_a.append(sigmoid(i[0]) * (1 - sigmoid(i[0])))
  elif func == "tanh":
    for i in a:
      diff_a.append(1 - tanh(i[0]) * tanh(i[0]))
  elif func == "ReLU":
    for i in a:
      if i[0] > 0: diff_a.append(1)
      else :diff_a.append(0)
  elif func == "identity":
    for i in a:
      diff_a.append(1)
  diff_a = np.array(diff_a).reshape(len(diff_a),1)
  return diff_a

def back_propogation(func, a, h, y_pred, y, x, n, W,loss_type):
  e_y = np.zeros(10).reshape(10,1)
  e_y[y] = 1

  grad_W = []
  grad_b = []
  grad_h = []
  grad_a = []
  if loss_type == "cross_entropy":
    grad_a_n = np.subtract(y_pred, e_y)
  else :
    grad_a_n = 2 * (y_pred - e_y) * (y_pred * (1 - y_pred))
  grad_a.append(grad_a_n)
  for k in range(n, -1, -1):
    if k == 0:
      grad_W_k = np.dot(grad_a[len(grad_a) - 1], x.T)
    else:
      grad_W_k = np.dot(grad_a[len(grad_a) - 1], h[k-1].T)

    grad_b_k = grad_a[len(grad_a) - 1]
    grad_W.append(grad_W_k)
    grad_b.append(grad_b_k)

    if k == 0: break

    grad_hprev = np.dot(W[k].T, grad_a[len(grad_a) - 1])

    g = diff_g(func, a[k-1])
    grad_aprev = grad_hprev * g

    grad_h.append(grad_hprev)
    grad_a.append(grad_aprev)

  return grad_W, grad_b


def gradient_descent(func, x, y, W, b, n, neurons,loss_type):
  a, h, y_pred = forward_propogation(func, x, W, b, n, neurons)
  grad_W, grad_b = back_propogation(func, a, h, y_pred, y, x, n, W,loss_type)
  return grad_W, grad_b, y_pred


def mean_squared_error(y_pred, y):
  loss = 0
  for i in range(len(y_pred)):
    if (i == y):
      loss += (y_pred[i][0] - 1)**2
    else:
      loss += (y_pred[i][0])**2
  return loss


def cross_entropy_loss(y_pred, y):
  return -np.log(np.clip(y_pred[y][0], 1e-10, 1))


def training_accuracy(x_train, activation_func, W, b, n, y_train, neurons):
  count = 0
  for i in range(int(0.9 * len(x_train))):
    a, h, y_pred = forward_propogation(activation_func, x_train[i].flatten().reshape(784,1), W, b, n, neurons)
    y_p = np.argmax(y_pred)
    if (y_train[i] == y_p):
      count+=1
  wandb.log({"Training Acc: " : count/54000})

def validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum):
  count = 0
  loss = 0
  for i in range(54000, len(x_train)):
    a, h, y_pred = forward_propogation(activation_func, x_train[i].flatten().reshape(784,1), W, b, n, neurons)
    y_p = np.argmax(y_pred)
    loss += cross_entropy_loss(y_pred, y_train[i])
    if (y_train[i] == y_p):
      count+=1
  wandb.log({"Validation loss: " : (loss + sum)/6000})
  wandb.log({"Validation Acc: ": count/6000})


def sgd(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train):
  batch = batch_size
  for j in range(epochs):
    loss = 0
    for i in range(int(0.9 * len(x_train))):

      if batch == batch_size:
        dw = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bias) for bias in b]

      grad_W, grad_b, y_pred = gradient_descent(activation_func, x_train[i].flatten().reshape(784,1), y_train[i], W, b, n, neurons,loss_type)

      for k in range(len(dw)):
        dw[k] = np.add(dw[k], grad_W[len(dw) - k - 1])
      for k in range(len(db)):
        db[k] = np.add(db[k], grad_b[len(db) - k - 1])
      batch-=1

      if batch == 0:
        for k in range(len(dw)):
          W[k] = np.subtract(W[k],(eta * dw[k]/batch_size)) - eta * alpha * W[k]
        for i in range(len(db)):
          b[k] = np.subtract(b[k],(eta * db[k]/batch_size))
        batch = batch_size

      if (loss_type == "cross_entropy"):
        loss += cross_entropy_loss(y_pred, y_train[i])
      else :
        loss += mean_squared_error(y_pred, y_train[i])

    sum_loss = 0
    for i in range(len(W)):
      sum_loss += np.sum(np.square(W[i]))
    sum_loss = sum_loss * alpha / 2

    wandb.log({"Epoch : ": j+1})
    wandb.log({"Training loss: " : (loss + sum_loss) /54000})

    training_accuracy(x_train, activation_func, W, b, n, y_train, neurons)

    validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum_loss)
  return W, b


def momentum(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, beta):
  batch = batch_size

  prev_vw = [np.zeros_like(w) for w in W]
  prev_vb = [np.zeros_like(bias) for bias in b]

  for j in range(epochs):
    loss = 0
    for i in range(int(0.9 * len(x_train))):

      if batch == batch_size:
        dw = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bias) for bias in b]
        vw = [np.zeros_like(w) for w in W]
        vb = [np.zeros_like(bias) for bias in b]
      grad_W, grad_b, y_pred = gradient_descent(activation_func, x_train[i].flatten().reshape(784,1), y_train[i], W, b, n, neurons,loss_type)

      for k in range(len(dw)):
        dw[k] = np.add(dw[k], grad_W[len(dw) - k - 1])
      for k in range(len(db)):
        db[k] = np.add(db[k], grad_b[len(db) - k - 1])
      batch-=1

      if batch == 0:
        for k in range(len(dw)):
          vw[k] = np.add(vw[k],np.add(beta * prev_vw[k], eta* dw[k]/batch_size))
          W[k] = np.subtract(W[k],vw[k]) - eta * alpha * W[k]
        for k in range(len(db)):
          vb[k] = np.add(vb[k],np.add(beta * prev_vb[k], eta* db[k]/batch_size))
          b[k] = np.subtract(b[k],vb[k])
        prev_vw = vw
        prev_vb = vb
        batch = batch_size
      if (loss_type == "cross_entropy"):
        loss += cross_entropy_loss(y_pred, y_train[i])
      else :
        loss += mean_squared_error(y_pred, y_train[i])

    sum_loss = 0
    for i in range(len(W)):
      sum_loss += np.sum(np.square(W[i]))
    sum_loss = sum_loss * alpha / 2

    wandb.log({"Epoch : ": j+1})
    wandb.log({"Training loss: " : (loss + sum_loss) /54000})

    training_accuracy(x_train, activation_func, W, b, n, y_train, neurons)

    validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum_loss)
  return W,b 

def nestrov(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, beta):
  batch = batch_size

  prev_vw = [np.zeros_like(w) for w in W]
  prev_vb = [np.zeros_like(bias) for bias in b]
  vw = [np.zeros_like(w) for w in W]
  vb = [np.zeros_like(bias) for bias in b]

  for j in range(epochs):
    loss = 0
    for i in range(int(0.9 * len(x_train))):
      if batch == batch_size:
        dw = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bias) for bias in b]
        W_new = [np.zeros_like(w) for w in W]
        b_new = [np.zeros_like(bias) for bias in b]
        for k in range(len(dw)):
          vw[k] = beta * prev_vw[k]
        for k in range(len(db)):
          vb[k] = beta * prev_vb[k]
        for k in range(len(dw)):
          W_new[k] = np.subtract(W[k],vw[k])
        for k in range(len(db)):
          b_new[k] = np.subtract(b[k],vb[k])

      grad_W, grad_b, y_pred = gradient_descent(activation_func, x_train[i].flatten().reshape(784,1), y_train[i], W_new, b_new, n, neurons,loss_type)

      for k in range(len(dw)):
        dw[k] = np.add(dw[k], grad_W[len(dw) - k - 1])
      for k in range(len(db)):
        db[k] = np.add(db[k], grad_b[len(db) - k - 1])
      batch-=1

      if batch == 0:
        for k in range(len(dw)):
          vw[k] = np.add(vw[k],eta* dw[k]/batch_size)
          W[k] = np.subtract(W[k],vw[k]) - eta * alpha * W[k]
        for k in range(len(db)):
          vb[k] = np.add(vb[k],eta*db[k]/batch_size)
          b[k] = np.subtract(b[k],vb[k])
        prev_vw = vw
        prev_vb = vb
        batch = batch_size
      if (loss_type == "cross_entropy"):
        loss += cross_entropy_loss(y_pred, y_train[i])
      else :
        loss += mean_squared_error(y_pred, y_train[i])

    sum_loss = 0
    for i in range(len(W)):
      sum_loss += np.sum(np.square(W[i]))
    sum_loss = sum_loss * alpha / 2

    wandb.log({"Epoch : ": j+1})
    wandb.log({"Training loss: " : (loss + sum_loss) /54000})

    training_accuracy(x_train, activation_func, W, b, n, y_train, neurons)

    validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum_loss)
  return W,b


def rmsprop(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, eps, beta):
  batch = batch_size

  vw = [np.zeros_like(w) for w in W]
  vb = [np.zeros_like(bias) for bias in b]

  for j in range(epochs):
    loss = 0
    for i in range(int(0.9 * len(x_train))):
      if batch == batch_size:
        dw = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bias) for bias in b]

      grad_W, grad_b, y_pred = gradient_descent(activation_func, x_train[i].flatten().reshape(784,1), y_train[i], W, b, n, neurons,loss_type)

      for k in range(len(dw)):
        dw[k] = np.add(dw[k], grad_W[len(dw) - k - 1])
      for k in range(len(db)):
        db[k] = np.add(db[k], grad_b[len(db) - k - 1])
      batch-=1

      if batch == 0:
        for k in range(len(dw)):
          vw[k] = np.add(beta * vw[k],(1 - beta)* (dw[k]/batch_size)**2)
          W[k] = np.subtract(W[k],eta * (dw[k]/batch_size) / (np.sqrt(vw[k]) + eps)) - eta * alpha * W[k]
        for k in range(len(db)):
          vb[k] = np.add(beta * vb[k],(1 - beta)*(db[k]/batch_size)**2)
          b[k] = np.subtract(b[k],eta * (db[k]/batch_size) / (np.sqrt(vb[k]) + eps))
        batch = batch_size
      if (loss_type == "cross_entropy"):
        loss += cross_entropy_loss(y_pred, y_train[i])
      else :
        loss += mean_squared_error(y_pred, y_train[i])

    sum_loss = 0
    for i in range(len(W)):
      sum_loss += np.sum(np.square(W[i]))
    sum_loss = sum_loss * alpha / 2

    wandb.log({"Epoch : ": j+1})
    wandb.log({"Training loss: " : (loss + sum_loss) /54000})

    training_accuracy(x_train, activation_func, W, b, n, y_train, neurons)

    validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum_loss)
  return W, b

def adam(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, eps, beta1, beta2):
  batch = batch_size


  mw = [np.zeros_like(w) for w in W]
  mb = [np.zeros_like(bias) for bias in b]
  mw_hat = [np.zeros_like(w) for w in W]
  mb_hat = [np.zeros_like(bias) for bias in b]
  vw = [np.zeros_like(w) for w in W]
  vb = [np.zeros_like(bias) for bias in b]
  vw_hat = [np.zeros_like(w) for w in W]
  vb_hat = [np.zeros_like(bias) for bias in b]

  t = 0
  for j in range(epochs):
    loss = 0
    for i in range(int(0.9 * len(x_train))):
      if batch == batch_size:
        dw = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bias) for bias in b]

      grad_W, grad_b, y_pred = gradient_descent(activation_func, x_train[i].flatten().reshape(784,1), y_train[i], W, b, n, neurons,loss_type)

      for k in range(len(dw)):
        dw[k] = np.add(dw[k], grad_W[len(dw) - k - 1])
      for k in range(len(db)):
        db[k] = np.add(db[k], grad_b[len(db) - k - 1])
      batch-=1

      if batch == 0:
        for k in range(len(dw)):
          mw[k] = np.add(beta1 * mw[k], (1 - beta1) * (dw[k]/batch_size))
          vw[k] = np.add(beta2 * vw[k], (1 - beta2) * (dw[k]/batch_size)**2)
        for k in range(len(db)):
          mb[k] = np.add(beta1 * mb[k], (1 - beta1) * (db[k]/batch_size))
          vb[k] = np.add(beta2 * vb[k], (1 - beta2) * (db[k]/batch_size)**2)

        for k in range(len(dw)):
          mw_hat[k] = mw[k]/(1 - np.power(beta1, t+1))
          vw_hat[k] = vw[k]/(1 - np.power(beta2, t+1))
        for k in range(len(db)):
          mb_hat[k] = mb[k]/(1 - np.power(beta1, t+1))
          vb_hat[k] = vb[k]/(1 - np.power(beta2, t+1))

        for k in range(len(dw)):
          W[k] = np.subtract(W[k],eta * mw_hat[k] / (np.sqrt(vw_hat[k]) + eps)) - eta * alpha * W[k]
        for k in range(len(db)):
          b[k] = np.subtract(b[k],eta * mb_hat[k] / (np.sqrt(vb_hat[k]) + eps))
        batch = batch_size
        t+=1
      if (loss_type == "cross_entropy"):
        loss += cross_entropy_loss(y_pred, y_train[i])
      else :
        loss += mean_squared_error(y_pred, y_train[i])

    sum_loss = 0
    for i in range(len(W)):
      sum_loss += np.sum(np.square(W[i]))
    sum_loss = sum_loss * alpha / 2

    wandb.log({"Epoch : ": j+1})
    wandb.log({"Training loss: " : (loss + sum_loss) /54000})

    training_accuracy(x_train, activation_func, W, b, n, y_train, neurons)

    validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum_loss)
  return W, b


def nadam(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, eps, beta1, beta2):

  batch = batch_size

  mw = [np.zeros_like(w) for w in W]
  mb = [np.zeros_like(bias) for bias in b]
  mw_hat = [np.zeros_like(w) for w in W]
  mb_hat = [np.zeros_like(bias) for bias in b]
  vw = [np.zeros_like(w) for w in W]
  vb = [np.zeros_like(bias) for bias in b]
  vw_hat = [np.zeros_like(w) for w in W]
  vb_hat = [np.zeros_like(bias) for bias in b]

  t = 1
  for j in range(epochs):
    loss = 0
    for i in range(int(0.9 * len(x_train))):
      if batch == batch_size:
        dw = [np.zeros_like(w) for w in W]
        db = [np.zeros_like(bias) for bias in b]

      grad_W, grad_b, y_pred = gradient_descent(activation_func, x_train[i].flatten().reshape(784,1), y_train[i], W, b, n, neurons,loss_type)

      for k in range(len(dw)):
        dw[k] = np.add(dw[k], grad_W[len(dw) - k - 1])
      for k in range(len(db)):
        db[k] = np.add(db[k], grad_b[len(db) - k - 1])
      batch-=1

      if batch == 0:
        for k in range(len(dw)):
          mw[k] = np.add(beta1 * mw[k], (1 - beta1) * (dw[k]/batch_size))
          vw[k] = np.add(beta2 * vw[k], (1 - beta2) * (dw[k]/batch_size)**2)
        for k in range(len(db)):
          mb[k] = np.add(beta1 * mb[k], (1 - beta1) * (db[k]/batch_size))
          vb[k] = np.add(beta2 * vb[k], (1 - beta2) * (db[k]/batch_size)**2)

        for k in range(len(dw)):
          mw_hat[k] = mw[k]/(1 - np.power(beta1, j+1))
          vw_hat[k] = vw[k]/(1 - np.power(beta2, j+1))
        for k in range(len(db)):
          mb_hat[k] = mb[k]/(1 - np.power(beta1, j+1))
          vb_hat[k] = vb[k]/(1 - np.power(beta2, j+1))

        for k in range(len(dw)):
          W[k] = np.subtract(W[k],(eta / (np.sqrt(vw_hat[k]) + eps)) * np.add((beta1 * mw_hat[k]), (1 - beta1) * (dw[k] / batch_size) / (1-beta1**(t + 1)))) - eta * alpha * W[k]
        for k in range(len(db)):
          b[k] = np.subtract(b[k],(eta / (np.sqrt(vb_hat[k]) + eps)) * np.add((beta1 * mb_hat[k]), (1 - beta1) * (db[k] / batch_size) / (1-beta1**(t + 1))) )
        batch = batch_size
        t+=1
      if (loss_type == "cross_entropy"):
        loss += cross_entropy_loss(y_pred, y_train[i])
      else :
        loss += mean_squared_error(y_pred, y_train[i])

    sum_loss = 0
    for i in range(len(W)):
      sum_loss += np.sum(np.square(W[i]))
    sum_loss = sum_loss * alpha / 2

    wandb.log({"Epoch : ": j+1})
    wandb.log({"Training loss: " : (loss + sum_loss) /54000})

    training_accuracy(x_train, activation_func, W, b, n, y_train, neurons)

    validation_loss_and_accuracy(x_train, activation_func, W, b, n, y_train, neurons, sum_loss)
  return W, b

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument('-wp', '--wandb_project', default='myprojectname')
    parser.add_argument('-we', '--wandb_entity', default='myname')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=["mnist", "fashion_mnist"])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument('-o', '--optimizer', default='adam', choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-beta', '--beta', type=float, default=0.9)
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9)
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-10)
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005)
    parser.add_argument('-w_i', '--weight_init', default='Xavier', choices=["random", "Xavier"])
    parser.add_argument('-nhl', '--num_layers', type=int, default=5)
    parser.add_argument('-sz', '--hidden_size', type=int, default=64)
    parser.add_argument('-a', '--activation', default='tanh', choices=["identity", "sigmoid", "tanh", "ReLU"])

    # Parse arguments
    args = parser.parse_args()

    # Set arguments to variables
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    loss_type = args.loss
    optimizer = args.optimizer
    eta = args.learning_rate
    momentum = args.momentum
    beta = args.beta
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.epsilon
    alpha = args.weight_decay
    init = args.weight_init
    n = args.num_layers
    neurons = args.hidden_size
    activation_func = args.activation

    wandb.init(project=wandb_project, entity=wandb_entity)

    run_name="hl_"+ str(n) + "_bs_" + str(batch_size) + "_ac_" + str(activation_func)
    wandb.run.name=run_name

    if dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train/255
        x_test = x_test/255
    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train/255
        x_test = x_test/255


    if init == "Xavier":
        W, b= xavier_initialization(len(x_train[0].flatten()), n, neurons)
    elif init == "random":
        W, b= initialize_param(len(x_train[0].flatten()), n, neurons)

    if (optimizer == "sgd"):
        W,b = sgd(n, neurons, epochs, alpha, eta, 1, activation_func, W, b, loss_type, x_train, y_train)
    elif (optimizer  == "momentum"):
        W,b = momentum(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b, loss_type, x_train, y_train, momentum)
    elif(optimizer  == "nesterov"):
        W,b = nestrov(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, momentum)
    elif (optimizer  == "rmsprop"):
        W,b = rmsprop(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, epsilon, beta)
    elif (optimizer  == "adam"):
        W,b = adam(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, epsilon, beta1, beta2)
    elif (optimizer  == "nadam"):
        W,b = nadam(n, neurons, epochs, alpha, eta, batch_size, activation_func, W, b,loss_type, x_train, y_train, epsilon, beta1, beta2)

    wandb.finish()
if __name__ == "__main__":
    main()
