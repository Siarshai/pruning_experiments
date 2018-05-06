import tensorflow as tf
from tensorflow.python.layers import base, utils
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.ops import nn_ops, init_ops

MASK_COLLECTION = 'masks'
THRESHOLD_COLLECTION = 'thresholds'
MASKED_WEIGHT_COLLECTION = 'masked_weights'
WEIGHT_COLLECTION = 'kernel'
# The 'weights' part of the name is needed for the quantization library
# to recognize that the kernel should be quantized.
MASKED_WEIGHT_NAME = 'weights/masked_weight'


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
            name='mask',
            shape=kernel_shape,
            initializer=init_ops.ones_initializer(),
            trainable=False,
            dtype=self.dtype)
        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        self.masked_kernel = tf.multiply(self.mask, self.kernel, MASKED_WEIGHT_NAME)

        tf.add_to_collection(MASK_COLLECTION, self.mask)
        tf.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_kernel)
        tf.add_to_collection(WEIGHT_COLLECTION, self.kernel)

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None

        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        assert self.rank == 2

        outputs = tf.nn.convolution(
            input=inputs,
            filter=self.masked_kernel,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, self.rank + 2))

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
