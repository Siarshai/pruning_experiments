import tensorflow as tf
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.core import Dense


def get_kernel_and_bias(layer_name, repacked_weights, debug=False):
    kernel, bias = None, None
    if "dense" in layer_name:
        kernel = repacked_weights[layer_name + "/kernel"]
        bias = repacked_weights[layer_name + "/bias"]
        if debug:
            print("restore_network: dense layer: ", layer_name)
    elif "conv" in layer_name:
        kernel = repacked_weights[layer_name + "/kernel"]
        if layer_name + "/biases" in repacked_weights.keys():
            bias = repacked_weights[layer_name + "/biases"]
        if debug:
            print("restore_network: convolutional layer: ", layer_name)
    else:
        print("No weights for layer: " + layer_name)
    return kernel, bias


def create_conv_from_weights(input_tensor, layer_name, repacked_weights, debug):
    kernel, bias = get_kernel_and_bias(layer_name, repacked_weights, debug)
    conv_filters_num = kernel.shape[-1]
    conv_layer = Conv2D(
        filters=conv_filters_num,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=bias is not None,
        activation=None,
        kernel_initializer=tf.constant_initializer(kernel),
        bias_initializer=tf.constant_initializer(bias) if bias else None,
        name=layer_name + "_repacked")
    x = conv_layer.apply(input_tensor)
    x = tf.nn.relu(x, name=layer_name + "_relu")
    return x


def create_dense_from_weights(input_tensor, layer_name, repacked_weights, debug):
    kernel, bias = get_kernel_and_bias(layer_name, repacked_weights, debug)
    units_num = kernel.shape[-1]
    dense = Dense(units=units_num,
                  activation=tf.nn.relu,
                  kernel_initializer=tf.constant_initializer(kernel),
                  bias_initializer=tf.constant_initializer(bias),
                  name=layer_name + "_repacked")
    return dense.apply(input_tensor)


# Must match create_network(...)
def restore_network_mnist(network_input, layers_order, repacked_weights, debug=False):
    x = create_conv_from_weights(network_input, layers_order[0], repacked_weights, debug)
    x = create_conv_from_weights(x, layers_order[1], repacked_weights, debug)
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name=layers_order[1] + "_maxpool2d")
    x = create_conv_from_weights(x, layers_order[2], repacked_weights, debug)
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name=layers_order[1] + "_maxpool2d")
    x = tf.layers.flatten(x)
    x = create_dense_from_weights(x, layers_order[3], repacked_weights, debug)
    x = create_dense_from_weights(x, layers_order[4], repacked_weights, debug)
    return x


def restore_network_cifar_10(network_input, layers_order, repacked_weights, debug=False):
    x = create_conv_from_weights(network_input, layers_order[0], repacked_weights, debug)
    x = create_conv_from_weights(x, layers_order[1], repacked_weights, debug)
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name=layers_order[1] + "_maxpool2d")
    x = create_conv_from_weights(x, layers_order[2], repacked_weights, debug)
    x = create_conv_from_weights(x, layers_order[3], repacked_weights, debug)
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name=layers_order[1] + "_maxpool2d")
    x = tf.layers.flatten(x)
    x = create_dense_from_weights(x, layers_order[4], repacked_weights, debug)
    x = create_dense_from_weights(x, layers_order[5], repacked_weights, debug)
    return x

def get_restore_network_function(dataset_label):
    return {
        "mnist": restore_network_mnist,
        "cifar_10": restore_network_cifar_10
    }[dataset_label]
