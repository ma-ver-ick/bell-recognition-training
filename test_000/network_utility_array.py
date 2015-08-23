__author__ = 'msei'
import numpy as np

import theano
import theano.tensor as T

import time

import lasagne
import pickle

DATA_TYPE = 'double'
DATA_TYPE_P = '%.32f'
INPUT_VAR = 'input'
NEURON_VAR = 'n_'
NEURON_WEIGHT_VAR = 'nw_'
NEURON_BIAS_VAR = 'nb_'

FILE = '../test_002/150821_2200_512-256-256-2/test_mlp_002-epoch-79.npz'
FILE = 'model.npz'
FILE = '../test_002/150822_2200_512_256-256-2-my-acti/test_mlp_002-epoch-98.npz'


def my_activation(x):
    return x / (1. + abs(x))


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
            nonlinearity=my_activation,
            W=lasagne.init.GlorotUniform())

    l_hid_2 = lasagne.layers.DenseLayer(
            l_hid_1, num_units=window_size,
            nonlinearity=my_activation,
            W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid_2, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


network = build_mlp()
network_parameters = np.load(FILE)
lasagne.layers.set_all_param_values(network, network_parameters['arr_0'])

func = list()
pre = list()


def add_to(func, new_line):
    func.append(new_line)


last_neuron_count = -1
last_layer_input_layer = True
layer_count = 0

for layer in lasagne.layers.get_all_layers(network):
    print layer
    if isinstance(layer, lasagne.layers.InputLayer):
        print "\t", layer.shape

        last_neuron_count = layer.shape[len(layer.shape) - 1]
        add_to(func, DATA_TYPE + " evaluate(" + DATA_TYPE + " " + INPUT_VAR + "[]) {")

    if isinstance(layer, lasagne.layers.DenseLayer):

        add_to(func, "\t// layer: " + str(layer_count) + ", name: " + str(layer.name))

        print "- Weights:"
        # for i in layer.W.get_value():
        #     print "\t\t", i

        print "- Bias"
        # for i in layer.b.get_value():
        #     print "\t\t", i

        # Generate function
        # for(int i = 0; i < layer.num_units; i++) {
        #     for(int ii = 0; ii < last_neuron_count; ii++) {
        #         n_0[i] = input_0 * X
        #     }
        #     n_0[i] = sigmoid(n_0[i] + bias)
        # }

        var_name = NEURON_VAR + str(layer_count)
        if not last_layer_input_layer:
            last_var_name = NEURON_VAR + str(layer_count - 1)
        else:
            last_var_name = INPUT_VAR

        var_weight_name = NEURON_WEIGHT_VAR + str(layer_count)

        af = ""
        if layer.nonlinearity == lasagne.nonlinearities.sigmoid:
            af = " sigmoid"
        if layer.nonlinearity == lasagne.nonlinearities.very_leaky_rectify:
            af = " very_leaky_rectify"

        add_to(func, "\t%s %s[%i];" % (DATA_TYPE, var_name, layer.num_units))
        add_to(func, "\tfor(int i = 0; i < " + str(layer.num_units) + "; i++) {")
        add_to(func, "\t\tfor(int ii = 0; ii < " + str(last_neuron_count) + "; ii++) {")
        add_to(func, "\t\t\t" + ("%s[i] += %s[ii] * %s[i][ii];" % (var_name, last_var_name, var_weight_name)))
        add_to(func, "\t\t}")
        add_to(func, "\t\t%s%i[i] = %s(%s%i[i] + %s[i]);" % (NEURON_VAR, layer_count, af, NEURON_VAR, layer_count, NEURON_BIAS_VAR + str(layer_count)))
        add_to(func, "\t}")

        code = "%s %s[][%i] = {" % (DATA_TYPE, var_weight_name, last_neuron_count)
        for neuron in range(0, layer.num_units):
            code += "{"
            for l_n in range(0, last_neuron_count):
                code += (DATA_TYPE_P + ", ") % layer.W.get_value()[l_n][neuron]
            code = code[:-2] + "}, "

        add_to(pre, code[:-2] + "};")

        code = "%s %s[] = {" % (DATA_TYPE, NEURON_BIAS_VAR + str(layer_count))
        for neuron in range(0, layer.num_units):
            code += (DATA_TYPE_P + ", ") % layer.b.get_value()[neuron]
        add_to(pre, code[:-2] + "};")

        # Prepare for next layer
        if last_layer_input_layer:
            last_layer_input_layer = False
        layer_count += 1
        last_neuron_count = layer.num_units

    print "-"


add_to(func, "")


# generate return value - this assumes a softmax output layer (!)
for neuron in range(0, last_neuron_count):
    last_var_name = "%s%i[%i]" % (NEURON_VAR, layer_count-1, neuron)
    if neuron == 0:
        code = "\tif("
    else:
        code = "\telse if("
    code += last_var_name + " > "

    if last_neuron_count > 2:
        max_stmt = "max("
        for others in range(0, last_neuron_count):
            if others == neuron:
                continue
            max_stmt += "%s%i[%i], " % (NEURON_VAR, layer_count-1, others)
        code += max_stmt[:-2] + ")"
    else:
        for others in range(0, last_neuron_count):
            if others == neuron:
                continue
            code += "%s%i[%i]" % (NEURON_VAR, layer_count-1, others)

    code += ") { return %i; }" % neuron

    add_to(func, code)

add_to(func, "")
add_to(func, "\treturn -1;")
add_to(func, "}")

with open('temp.txt', 'w') as f:
    for line in pre:
        f.write(line + "\n")
    for line in func:
        f.write(line + "\n")

