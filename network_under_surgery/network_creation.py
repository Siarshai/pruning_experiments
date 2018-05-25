import tensorflow as tf
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.core import Dense

from bonesaw.masked_layers import MaskedConv2D, MaskedDense


def create_network_mnist(network_input, classes_num):
    conv1 = MaskedConv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv1")
    x = conv1.apply(network_input)
    x = tf.nn.relu(x, name="conv1_relu")

    conv2 = MaskedConv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv2")
    x = conv2.apply(x)
    x = tf.nn.relu(x, name="conv2_relu")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name="conv2_maxpool2d")
    x = tf.layers.dropout(x, 0.4, name="conv2_dropout")

    conv3 = MaskedConv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv3")
    x = conv3.apply(x)
    x = tf.nn.relu(x, name="conv3_relu")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name="conv4_maxpool2d")
    x = tf.layers.dropout(x, 0.4, name="conv4_dropout")

    x = tf.layers.flatten(x)

    dense1 = MaskedDense(units=32,
                         activation=tf.nn.relu,
                         name="dense1")
    x = dense1.apply(x)

    dense2 = Dense(units=classes_num,
                   name="dense2_logits")
    x = dense2.apply(x)

    stripable_layers = {
        "conv1": conv1, "conv2": conv2, "conv3": conv3
    }
    return x, stripable_layers


def create_network_cifar_10(network_input, classes_num):
    conv1 = MaskedConv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv1")
    x = conv1.apply(network_input)
    x = tf.nn.relu(x, name="conv1_relu")
    x = tf.layers.dropout(x, 0.4, name="conv1_dropout")

    conv2 = MaskedConv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv2")
    x = conv2.apply(x)
    x = tf.nn.relu(x, name="conv2_relu")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name="conv2_maxpool2d")
    x = tf.layers.dropout(x, 0.4, name="conv2_dropout")

    conv3 = MaskedConv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv3")
    x = conv3.apply(x)
    x = tf.nn.relu(x, name="conv3_relu")
    x = tf.layers.dropout(x, 0.4, name="conv3_dropout")

    conv4 = MaskedConv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv4")
    x = conv4.apply(x)
    x = tf.nn.relu(x, name="conv4_relu")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name="conv4_maxpool2d")
    x = tf.layers.dropout(x, 0.5, name="conv4_dropout")

    x = tf.layers.flatten(x)

    dense1 = MaskedDense(units=32,
                         activation=tf.nn.relu,
                         name="dense1")
    x = dense1.apply(x)
    x = tf.layers.dropout(x, 0.5, name="conv4_dropout")

    dense2 = Dense(units=classes_num,
                   name="dense2_logits")
    x = dense2.apply(x)

    stripable_layers = {
        "conv1": conv1, "conv2": conv2, "conv3": conv3, "conv4": conv4, "dense1": dense1
    }
    return x, stripable_layers


def create_network_cifar_100(network_input, classes_num):
    conv1 = MaskedConv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv1")
    x = conv1.apply(network_input)
    x = tf.nn.relu(x, name="conv1_relu")
    x = tf.layers.dropout(x, 0.4, name="conv1_dropout")

    conv2 = MaskedConv2D(
        filters=48,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv2")
    x = conv2.apply(x)
    x = tf.nn.relu(x, name="conv2_relu")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name="conv2_maxpool2d")
    x = tf.layers.dropout(x, 0.4, name="conv2_dropout")

    conv3 = MaskedConv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv3")
    x = conv3.apply(x)
    x = tf.nn.relu(x, name="conv3_relu")
    x = tf.layers.dropout(x, 0.4, name="conv3_dropout")

    conv4 = MaskedConv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv4")
    x = conv4.apply(x)
    x = tf.nn.relu(x, name="conv4_relu")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2,
        padding="valid",
        name="conv4_maxpool2d")
    x = tf.layers.dropout(x, 0.5, name="conv4_dropout")

    x = tf.layers.flatten(x)

    dense1 = MaskedDense(units=128,
                         activation=tf.nn.relu,
                         name="dense1")
    x = dense1.apply(x)
    x = tf.layers.dropout(x, 0.5, name="conv4_dropout")

    dense2 = Dense(units=classes_num,
                   name="dense2_logits")
    x = dense2.apply(x)

    stripable_layers = {
        "conv1": conv1, "conv2": conv2, "conv3": conv3, "conv4": conv4, "dense1": dense1
    }
    return x, stripable_layers


def get_create_network_function(dataset_label):
    return {
        "mnist": create_network_mnist,
        "cifar_10": create_network_cifar_10,
        "cifar_100": create_network_cifar_100
    }[dataset_label]


def get_layers_names_for_dataset(dataset_label):
    return {
        "mnist": ["conv1", "conv2", "conv3", "dense1", "dense2_logits"],
        "cifar_10": ["conv1", "conv2", "conv3", "conv4", "dense1", "dense2_logits"],
        "cifar_100": ["conv1", "conv2", "conv3", "conv4", "dense1", "dense2_logits"]
    }[dataset_label]
