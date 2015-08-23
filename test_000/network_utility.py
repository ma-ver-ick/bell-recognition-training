__author__ = 'msei'
import numpy as np

import theano
import theano.tensor as T

import time

import lasagne
import pickle

FLOAT = 'float'
INPUT_VAR = 'i_'
NEURON_VAR = 'n_'

FILE = '../test_002/150821_2200_512-256-256-2/test_mlp_002-epoch-79.npz'


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
            l_in, num_units=window_size,
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())

    l_hid_2 = lasagne.layers.DenseLayer(
            l_hid_1, num_units=window_size,
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid_2, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

network = build_mlp()
network_parameters = np.load(FILE)
lasagne.layers.set_all_param_values(network, network_parameters['arr_0'])

func = list()


def add_to_func(func, new_line):
    func.append(new_line)

last_neuron_count = -1
last_layer_input_layer = True
layer_count = 0

for layer in lasagne.layers.get_all_layers(network):
    print layer
    if isinstance(layer, lasagne.layers.InputLayer):
        print "\t", layer.shape

        arguments = ""
        last_neuron_count = layer.shape[len(layer.shape) - 1]
        for i in range(0, last_neuron_count):
            arguments += "%s %s%i, " % (FLOAT, INPUT_VAR, i)

        arguments = arguments[:-2]

        add_to_func(func, "float evaluate(" + arguments + ") {")

    if isinstance(layer, lasagne.layers.DenseLayer):

        add_to_func(func, "\t// layer: " + str(layer_count) + ", name: " + str(layer.name))

        print "- Weights:"
        for i in layer.W.get_value():
            print "\t\t", i

        print "- Bias"
        for i in layer.b.get_value():
            print "\t\t", i

        # Generate function
        for neuron in range(0, layer.num_units):
            code = "\t%s %s%i_%i =" % (FLOAT, NEURON_VAR, layer_count, neuron)

            close_bracket = False
            if layer.nonlinearity == lasagne.nonlinearities.sigmoid:
                code += " sigmoid("
                close_bracket = True
            if layer.nonlinearity == lasagne.nonlinearities.very_leaky_rectify:
                code += " very_leaky_rectify("
                close_bracket = True

            for l_n in range(0, last_neuron_count):

                if not last_layer_input_layer:
                    var_name = NEURON_VAR + str(layer_count - 1) + "_" + str(l_n)
                else:
                    var_name = INPUT_VAR + str(l_n)

                code += " (%f) * %s +" % (layer.W.get_value()[l_n][neuron], var_name)

            code += " (%f)" % (layer.b.get_value()[neuron])

            if close_bracket:
                code += ");"
            else:
                code += ";"

            add_to_func(func, code)

        # Prepare for next layer
        if last_layer_input_layer:
            last_layer_input_layer = False
        layer_count += 1
        last_neuron_count = layer.num_units

    print "-"


add_to_func(func, "")


# generate return value - this assumes a softmax output layer (!)
for neuron in range(0, last_neuron_count):
    var_name = "%s%i_%i" % (NEURON_VAR, layer_count-1, neuron)
    if neuron == 0:
        code = "\tif("
    else:
        code = "\telse if("
    code += var_name + " > "

    if last_neuron_count > 2:
        max_stmt = "max("
        for others in range(0, last_neuron_count):
            if others == neuron:
                continue
            max_stmt += "%s%i_%i, " % (NEURON_VAR, layer_count-1, others)
        code += max_stmt[:-2] + ")"
    else:
        for others in range(0, last_neuron_count):
            if others == neuron:
                continue
            code += "%s%i_%i" % (NEURON_VAR, layer_count-1, others)

    code += ") { return %i; }" % neuron

    add_to_func(func, code)

add_to_func(func, "")
add_to_func(func, "\treturn -1;")
add_to_func(func, "}")

with open('temp.txt', 'w') as f:
    f.writelines(func)

