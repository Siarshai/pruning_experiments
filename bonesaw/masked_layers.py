import tensorflow as tf
from tensorflow.python.layers import base, utils
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.ops import nn_ops, init_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn


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
        self.mask_plh = tf.placeholder(tf.float32, (self.filters,), 'mask_plh')
        self.mask_assign_op = tf.assign(self.mask, self.mask_plh)

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


class MaskedDense(base.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MaskedDense, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})

        self.kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)

        self.mask = self.add_variable(
            name='mask',
            shape=(self.units,),
            initializer=init_ops.ones_initializer(),
            trainable=False,
            dtype=self.dtype)

        ops.add_to_collection(MASKS_COLLECTION, self.mask)
        ops.add_to_collection(WEIGHTS_COLLECTION, self.kernel)

        if self.use_bias:
            self.bias = self.add_variable(
                'bias',
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        if len(output_shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel,
                                             [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)

        outputs = tf.multiply(self.mask, outputs, MASKED_OUTPUT_NAME)

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)
