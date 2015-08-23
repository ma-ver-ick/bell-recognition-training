__author__ = 'msei'

import math
import numpy as np

import theano
import theano.tensor as T

import time

from scipy.io.wavfile import read, write
from pylab import plot, show, subplot, specgram

import lasagne

import traindata_mix


FILE = '150821_1700/test_mlp_002-epoch-31.npz'


def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    window_size = 256

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, window_size * 2),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid_1 = lasagne.layers.DenseLayer(
            l_in, num_units=window_size * 2,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

    l_hid_2 = lasagne.layers.DenseLayer(
            l_hid_1, num_units=window_size * 2,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid_2, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')

# restore network
network = build_mlp(input_var)
network_parameters = np.load(FILE)
lasagne.layers.set_all_param_values(network, network_parameters['arr_0'])

# prepare prediction
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

complete_x = list()
complete_y = list()

for position, fft, c in traindata_mix.test_data_iterator(traindata_mix.RING_01_TEST_DATA):
    try:
        complete_x.append(predict_fn([[[fft]]]) * 10000)
    except:
        print position, fft
        complete_x.append(0)
    complete_x.append(0)

rate, data = read(traindata_mix.RING_01_TEST_DATA + ".wav")

plot(range(0, len(data)), data)
plot(range(0, len(complete_x)), complete_x)

show()
