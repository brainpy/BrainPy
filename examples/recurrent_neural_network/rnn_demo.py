# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # Recurrent Neural Network Demo

# %% [markdown]
# Thanks to: https://github.com/gabrielloye/RNN-walkthrough

# %% [markdown]
# In this implementation, we'll be building a model that can complete your sentence based on a few characters or a word used as input.
#
# ![Example](img/Slide4.jpg)
#
# To keep this short and simple, we won't be using any large or external datasets. Instead, we'll just be defining a few sentences to see how the model learns from these sentences. The process that this implementation will take is as follows:
#
# ![Overview](img/Slide5.jpg)

# %%
import sys

import jax.lax

sys.path.append('../../')

import brainpy as bp
import numpy as np

# %% [markdown]
# First, we'll define the sentences that we want our model to output when fed with the first word or the first few characters.
#
# Then we'll create a dictionary out of all the characters that we have in the sentences and map them to an integer. This will allow us to convert our input characters to their respective integers (*char2int*) and vice versa (*int2char*).

# %%
text = ['hey how are you good i am fine',
        'good i am fine have a nice day',
        'The cell then uses gates to regulate the information to be kept or discarded at '
        'each time step before passing on the long-term and short-term information to the '
        'next cell'.lower(),
        'This repo holds the code for the implementation in my FloydHub article on RNNs'.lower(),
        'The secret sauce to the LSTM lies in its gating mechanism within each LSTM cell'.lower(),
        'the input at a time-step and the hidden state from the previous time step is '
        'passed through a tanh activation function'.lower()]
text = ['hey how are you',
        'good i am fine',
        'have a nice day']

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# %%
print(char2int)

# %% [markdown]
# Next, we'll be padding our input sentences to ensure that all the sentences are of the sample length. While RNNs are typically able to take in variably sized inputs, we will usually want to feed training data in batches to speed up the training process. In order to used batches to train on our data, we'll need to ensure that each sequence within the input data are of equal size.
#
# Therefore, in most cases, padding can be done by filling up sequences that are too short with **0** values and trimming sequences that are too long. In our case, we'll be finding the length of the longest sequence and padding the rest of the sentences with blank spaces to match that length.

# %%
maxlen = len(max(text, key=len))
print("The longest string has {} characters".format(maxlen))

# %%
# Padding

# A simple loop that loops through the list of sentences and adds 
# a ' ' whitespace until the length of the sentence matches the
# length of the longest sentence
for i in range(len(text)):
  while len(text[i]) < maxlen:
    text[i] += ' '

# %% [markdown]
# As we're going to predict the next character in the sequence at each time step, we'll have to divide each sentence into
#
# - Input data
#     - The last input character should be excluded as it does not need to be fed into the model
# - Target/Ground Truth Label
#     - One time-step ahead of the Input data as this will be the "correct answer" for the model at each time step corresponding to the input data

# %%
# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
  # Remove last character for input sequence
  input_seq.append(text[i][:-1])

  # Remove firsts character for target sequence
  target_seq.append(text[i][1:])
  print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

# %% [markdown]
# Now we can convert our input and target sequences to sequences of integers instead of characters by mapping them using the dictionaries we created above. This will allow us to one-hot-encode our input sequence subsequently.

# %%
for i in range(len(text)):
  input_seq[i] = [char2int[character] for character in input_seq[i]]
  target_seq[i] = [char2int[character] for character in target_seq[i]]

# %% [markdown]
# Before encoding our input sequence into one-hot vectors, we'll define 3 key variables:
#
# - *dict_size*: The number of unique characters that we have in our text
#     - This will determine the one-hot vector size as each character will have an assigned index in that vector
# - *seq_len*: The length of the sequences that we're feeding into the model
#     - As we standardised the length of all our sentences to be equal to the longest sentences, this value will be the max length - 1 as we removed the last character input as well
# - *batch_size*: The number of sentences that we defined and are going to feed into the model as a batch

# %%
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
  # Creating a multi-dimensional array of zeros with the desired output shape
  features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

  # Replacing the 0 at the relevant character index with a 1 to represent that character
  for i in range(batch_size):
    for u in range(seq_len):
      features[i, u, sequence[i][u]] = 1
  return features


# %% [markdown]
# We also defined a helper function that creates arrays of zeros for each character
# and replaces the corresponding character index with a **1**.

# %%
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

# %% [markdown]
# Since we're done with all the data pre-processing, we can now move the data from
# numpy arrays to PyTorch's very own data structure - **Torch Tensors**

# %%
input_seq = bp.math.array(input_seq)
target_seq = bp.math.array(target_seq)


# %% [markdown]
# Now we've reached the fun part of this project! We'll be defining the model using the
# Torch library, and this is where you can add or remove layers, be it fully connected
# layers, convolutational layers, vanilla RNN layers, LSTM layers, and many more! In
# this post, we'll be using the basic nn.rnn to demonstrate a simple example of how RNNs
# can be used.
#
# Before we start building the model, let's use a build in feature in PyTorch to check
# the device we're running on (CPU or GPU). This implementation will not require GPU as
# the training is really simple. However, as you progress on to large datasets and models
# with millions of trainable parameters, using the GPU will be very important to speed up
# your training.


# %% [markdown]
# To start building our own neural network model, we can define a class that inherits
# PyTorch’s base class (nn.module) for all neural network modules. After doing so, we
# can start defining some variables and also the layers for our model under the constructor.
# For this model, we’ll only be using 1 layer of RNN followed by a fully connected layer.
# The fully connected layer will be in-charge of converting the RNN output to our desired
# output shape.
#
# We’ll also have to define the forward pass function under forward() as a class method.
# The order the forward function is sequentially executed, therefore we’ll have to pass
# the inputs and the zero-initialized hidden state through the RNN layer first, before
# passing the RNN outputs to the fully-connected layer. Note that we are using the layers
# that we defined in the constructor.
#
# The last method that we have to define is the method that we called earlier to initialize
# the hidden state - init_hidden(). This basically creates a tensor of zeros in the shape
# of our hidden states.

# %%
class Model(bp.dnn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(Model, self).__init__()

    # Defining some parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    # Defining the layers
    # RNN Layer
    self.w_ir = bp.math.TrainVar(bp.math.random.random((input_size, hidden_dim)))
    self.w_rr = bp.math.TrainVar(bp.math.random.random((hidden_dim, hidden_dim)))
    self.b_rr = bp.math.TrainVar(bp.math.zeros((hidden_dim,)))
    # Fully connected layer
    self.w_ro = bp.math.TrainVar(bp.math.random.random((hidden_dim, output_size)))
    self.b_ro = bp.math.TrainVar(bp.math.zeros((output_size,)))

  def __call__(self, x):
    def scan_fun(hidden, x):
      hidden = bp.dnn.relu(x @ self.w_ir + hidden @ self.w_rr + self.b_rr)
      return hidden, hidden

    @jax.partial(jax.vmap, in_axes=(None, 0))
    def readout(params, hidden):
      return hidden @ params['w_ro'] + params['b_ro']

    init_hidden = bp.math.zeros((x.shape[0], self.hidden_dim))
    _, hist_hidden = jax.lax.scan(scan_fun, init_hidden, x.transpose(1, 0, 2))
    outputs = readout(dict(w_ro=self.w_ro, b_ro=self.b_ro), hist_hidden)
    return outputs.transpose(1, 0, 2)


# %% [markdown]
# After defining the model above, we'll have to instantiate the model with the
# relevant parameters and define our hyperparamters as well. The hyperparameters
# we're defining below are:
#
# - *n_epochs*: Number of Epochs --> This refers to the number of times our model will go through the entire training dataset
# - *lr*: Learning Rate --> This affects the rate at which our model updates the weights in the cells each time backpropogation is done
#     - A smaller learning rate means that the model changes the values of the weight with a smaller magnitude
#     - A larger learning rate means that the weights are updated to a larger extent for each time step
#
# Similar to other neural networks, we have to define the optimizer and loss function
# as well. We’ll be using CrossEntropyLoss as the final output is basically a classification task.

# %%
# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

# Define hyperparameters
n_epochs = 100
lr = 0.01

# Define Loss, Optimizer
optimizer = bp.dnn.Adam(lr=lr, train_vars=model.train_vars())


# %%
@bp.math.function(nodes=(model, optimizer))
def loss(x, y):
  outputs = model(x)
  loss = bp.dnn.cross_entropy_loss(outputs, y)
  return loss


vg = bp.math.value_and_grad(loss)


@bp.math.jit
@bp.math.function(nodes=(model, optimizer))
def train(x, y):
  loss, grads = vg(x, y)
  optimizer(grads)
  return loss


# %% [markdown]
# Now we can begin our training! As we only have a few sentences, this training
# process is very fast. However, as we progress, larger datasets and deeper
# models mean that the input data is much larger and the number of parameters
# within the model that we have to compute is much more.

# %%
# Training Run
for epoch in range(1, n_epochs + 1):
  loss = train(input_seq, target_seq)

  if epoch % 10 == 0:
    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss))


# %% [markdown]
# Let’s test our model now and see what kind of output we will get.
# Before that, let’s define some helper function to convert our model
# output back to text.

# %%
def predict(model, character):
  # One-hot encoding our input to fit into the model
  character = np.array([[char2int[c] for c in character]])
  character = one_hot_encode(character, dict_size, character.shape[1], 1)
  character = bp.math.array(character)

  out = model(character)

  prob = bp.dnn.softmax(out[-1], axis=0)
  # Taking the class with the highest probability score from the output
  char_ind = bp.math.max(prob, axis=0)[1]
  return int2char[int(char_ind)]


# %%
def sample(model, out_len, start='hey'):
  start = start.lower()
  # First off, run through the starting characters
  chars = [ch for ch in start]
  size = out_len - len(chars)
  # Now pass in the previous characters and get a new one
  for ii in range(size):
    char = predict(model, chars)
    chars.append(char)

  return ''.join(chars)


# %%
print(sample(model, 15, 'good'))

# %% [markdown]
# As we can see, the model is able to come up with the sentence ‘good i am fine ‘
# if we feed it with the words ‘good’, achieving what we intended for it to do!
