import tensorflow as tf
from tensorflow.python.layers import base, utils
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.ops import nn_ops, init_ops

MASK_NAME = "mask"
MASKED_OUTPUT_NAME = "masked_output"
MASKS_COLLECTION = 'masks'
WEIGHT_NAME = 'kernel'
WEIGHTS_COLLECTION = 'kernels'
BIAS_NAME = 'bias'
BIASES_COLLECTION = 'biases'


class MaskedConv2D(Conv2D):
    def __init__(self, **kwargs):
        super(MaskedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.mask = self.add_variable(
            name=MASK_NAME,
            shape=(self.filters,),
            initializer=init_ops.ones_initializer(),
            trainable=False,
            dtype=self.dtype)
        self.kernel = self.add_variable(name=WEIGHT_NAME,
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)

        tf.add_to_collection(MASKS_COLLECTION, self.mask)
        tf.add_to_collection(WEIGHTS_COLLECTION, self.kernel)

        if self.use_bias:
            self.bias = self.add_variable(name=BIAS_NAME,
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            tf.add_to_collection(BIASES_COLLECTION, self.bias)
        else:
            self.bias = None

        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        assert self.rank == 2

        outputs = tf.nn.convolution(
            input=inputs,
            filter=self.kernel,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, self.rank + 2))

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        outputs = tf.multiply(self.mask, outputs, MASKED_OUTPUT_NAME)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
