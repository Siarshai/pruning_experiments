import tensorflow as tf
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.core import Dense

from bonesaw.masked_layers import MaskedConv2D


def create_network(network_input, classes_num):
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

    dense1 = Dense(units=256,
                   activation=tf.nn.relu,
                   name="dense1")
    x = dense1.apply(x)

    dense2 = Dense(units=classes_num,
                   name="dense2_logits")
    x = dense2.apply(x)
    return x
