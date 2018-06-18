import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

from bonesaw.masked_layers import MaskingLayer


def create_network_mnist(network_input, classes_num, is_training):
    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv1")(network_input)
    x = tf.nn.relu(x, name="conv1_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=24,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv2")(x)
    x = tf.nn.relu(x, name="conv2_relu")
    x = MaskingLayer()(x)

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv2_maxpool2d"
    )(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv4")(x)
    x = tf.nn.relu(x, name="conv4_relu")
    x = MaskingLayer()(x)

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv4_maxpool2d"
    )(x)

    x = Flatten()(x)
    x = Dropout(0.5, name="conv4_dropout")(x, training=is_training)

    x = Dense(units=48, activation=tf.nn.relu, name="dense1")(x)
    x = MaskingLayer()(x)

    x = Dense(units=classes_num, name="dense2_logits")(x)

    return x


def create_network_cifar_10(network_input, classes_num, is_training):
    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv1")(network_input)
    x = tf.nn.relu(x, name="conv1_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=24,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv2")(x)
    x = tf.nn.relu(x, name="conv2_relu")
    x = MaskingLayer()(x)

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv2_maxpool2d"
    )(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv3")(x)
    x = tf.nn.relu(x, name="conv3_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv4")(x)
    x = tf.nn.relu(x, name="conv4_relu")
    x = MaskingLayer()(x)

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv4_maxpool2d"
    )(x)

    x = Flatten()(x)
    x = Dropout(0.5, name="conv4_dropout")(x, training=is_training)

    x = Dense(units=48, activation=tf.nn.relu, name="dense1")(x)
    x = MaskingLayer()(x)

    x = Dense(units=classes_num, name="dense2_logits")(x)

    return x


def create_network_cifar_100(network_input, classes_num, is_training):
    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv1")(network_input)
    x = tf.nn.relu(x, name="conv1_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv2")(x)
    x = tf.nn.relu(x, name="conv2_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=48,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv3")(x)
    x = tf.nn.relu(x, name="conv3_relu")
    x = MaskingLayer()(x)

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv3_maxpool2d"
    )(x)

    x = Dropout(0.5, name="conv3_dropout")(x, training=is_training)

    x = Conv2D(
        filters=48,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv4")(x)
    x = tf.nn.relu(x, name="conv4_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=48,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv5")(x)
    x = tf.nn.relu(x, name="conv5_relu")
    x = MaskingLayer()(x)

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv5_maxpool2d"
    )(x)
    x = Dropout(0.5, name="conv5_dropout")(x, training=is_training)

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv6")(x)
    x = tf.nn.relu(x, name="conv6_relu")
    x = MaskingLayer()(x)

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=True,
        activation=None,
        name="conv7")(x)
    x = tf.nn.relu(x, name="conv7_relu")

    x = MaxPooling2D(
        pool_size=[2, 2],
        padding="valid",
        name="conv7_maxpool2d"
    )(x)

    x = Flatten()(x)
    x = Dropout(0.5, name="conv7_dropout")(x, training=is_training)

    x = Dense(units=48, activation=tf.nn.relu, name="dense1")(x)
    x = MaskingLayer()(x)
    # x = Dropout(0.5, name="dense1_dropout")(x, training=is_training)

    x = Dense(units=48, activation=tf.nn.relu, name="dense2")(x)
    x = MaskingLayer()(x)

    x = Dense(units=classes_num, name="dense3_logits")(x)

    return x


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
        "cifar_100": ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7",
                      "dense1", "dense2", "dense3_logits"]
    }[dataset_label]


def get_masking_correspondencies_for_dataset(dataset_label):
    return {
        "cifar_10": {"conv1": "masking_layer_1",
                     "conv2": "masking_layer_2",
                     "conv3": "masking_layer_3",
                     "conv4": "masking_layer_4",
                     "dense1": "masking_layer_5"},
        "cifar_100": {"conv1": "masking_layer_1",
                     "conv2": "masking_layer_2",
                     "conv3": "masking_layer_3",
                     "conv4": "masking_layer_4",
                     "conv5": "masking_layer_5",
                     "conv6": "masking_layer_6",
                     "conv7": "",
                     "dense1": "masking_layer_7",
                     "dense2": "masking_layer_8"},
    }[dataset_label]

