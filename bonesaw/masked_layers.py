import tensorflow as tf
from keras.engine.topology import Layer
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops


MASK_NAME = "trainable_mask"
CAPTURE_MASK_NAME = "capture_mask"
MASKED_OUTPUT_NAME = "masked_output"

MASKS_COLLECTION = 'masks'
MASKS_ASSIGN_COLLECTION = 'masks_assign_ops'
MASKS_PLH_COLLECTION = 'masks_placeholders_ops'

CAPTURE_MASKS_COLLECTION = 'capture_masks'
CAPTURE_MASKS_ASSIGN_COLLECTION = 'capture_masks_assign_ops'
CAPTURE_MASKS_PLH_COLLECTION = 'capture_masks_placeholders_ops'

WEIGHT_NAME = 'kernel'
WEIGHTS_COLLECTION = 'kernels'
BIAS_NAME = 'bias'
BIASES_COLLECTION = 'biases'
MASKABLE_TRAINABLES = 'maskable_trainables'

LO_VARIABLES_COLLECTION = "l0_variables"
LO_ALPHA_SET_OP_COLLECTION = "l0_set_op_collection"
LO_ALPHA_PLH_COLLECTION = "l0_set_op_collection"
LO_VARIABLES_BETA_COLLECTION = "l0_variables_beta"

LO_LOSSES_COLLECTION = "l0_losses"
LO_TRAIN_TOGGLE_ON = "l0_train_toggle_on_ops"
LO_TRAIN_TOGGLE_OFF = "l0_train_toggle_off_ops"


class L0MaskableMixin:
    def init_l0_masks(self, units):
        log_epsilon = 0.0001
        beta_epsilon = 0.01
        ksi = 1.1
        gamma = -0.1
        self.l0_alpha = tf.Variable([2.75] * units, name="hard_concrete_alpha", dtype=tf.float32, trainable=False)
        self.l0_beta = tf.Variable([0.05] * units, name="hard_concrete_beta", dtype=tf.float32, trainable=False)

        self.l0_alpha_plh = tf.placeholder(tf.float32, (units,), 'hard_concrete_alpha_plh')
        self.l0_beta_plh = tf.placeholder(tf.float32, (units,), 'hard_concrete_plh')

        self.l0_alpha_set_op = tf.assign(self.l0_alpha, self.l0_alpha_plh, name="l0_alpha_set_op")
        self.l0_beta_set_op = tf.assign(self.l0_beta, self.l0_beta_plh, name="l0_beta_set_op")

        tf.add_to_collection(LO_VARIABLES_COLLECTION, self.l0_alpha)
        tf.add_to_collection(LO_ALPHA_SET_OP_COLLECTION, self.l0_alpha_set_op)
        tf.add_to_collection(LO_ALPHA_PLH_COLLECTION, self.l0_alpha_plh)
        tf.add_to_collection(LO_VARIABLES_BETA_COLLECTION, self.l0_beta)

        self.abs_l0_beta = tf.abs(self.l0_beta)
        self.l0_u = tf.random_uniform(shape=(units,), minval=0, maxval=1, name="hard_concrete_u_input",
                                      dtype=tf.float32)
        self.l0_s = tf.nn.sigmoid((tf.log(self.l0_u + log_epsilon) - tf.log(1.0 - self.l0_u + log_epsilon) + self.l0_alpha) / (
                self.abs_l0_beta + beta_epsilon))

        self.l0_s_stretched = self.l0_s * (ksi - gamma) + gamma
        self.l0_z = tf.minimum(1.0, tf.maximum(0.0, self.l0_s_stretched))

        self.L0_complexity = tf.nn.sigmoid(self.l0_alpha/2.0)

        tf.add_to_collection(LO_LOSSES_COLLECTION, self.L0_complexity)

        self.l0_train_toggle = tf.Variable(False, trainable=False, name="l0_train_toggle")
        self.l0_train_toggle_on = tf.assign(self.l0_train_toggle, True)
        tf.add_to_collection(LO_TRAIN_TOGGLE_ON, self.l0_train_toggle_on)
        self.l0_train_toggle_off = tf.assign(self.l0_train_toggle, False)
        tf.add_to_collection(LO_TRAIN_TOGGLE_OFF, self.l0_train_toggle_off)


class MaskingLayer(Layer, L0MaskableMixin):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(trainable=False, **kwargs)

    def build(self, input_shape):
        units = input_shape[-1]  # number of neurons or filters in previous layer

        self.init_l0_masks(units)
        self.trainable_mask = tf.Variable(
            [1.0] * units, name=MASK_NAME, dtype=tf.float32, trainable=False)
        self.capture_mask = tf.Variable(
            [1.0] * units, name=CAPTURE_MASK_NAME, dtype=tf.float32, trainable=False)

        self.trainable_mask_plh = tf.placeholder(tf.float32, (units,), 'mask_plh')
        self.capture_mask_plh = tf.placeholder(tf.float32, (units,), 'capture_mask_plh')

        self.trainable_mask_assign_op = tf.assign(self.trainable_mask, self.trainable_mask_plh)
        self.capture_mask_assign_op = tf.assign(self.capture_mask, self.capture_mask_plh)

        ops.add_to_collection(MASKS_COLLECTION, self.trainable_mask)
        ops.add_to_collection(MASKS_ASSIGN_COLLECTION, self.trainable_mask_assign_op)
        ops.add_to_collection(MASKS_PLH_COLLECTION, self.trainable_mask_plh)
        ops.add_to_collection(CAPTURE_MASKS_COLLECTION, self.capture_mask)
        ops.add_to_collection(CAPTURE_MASKS_ASSIGN_COLLECTION, self.trainable_mask_assign_op)
        ops.add_to_collection(CAPTURE_MASKS_PLH_COLLECTION, self.trainable_mask_plh)

        super(MaskingLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = inputs
        outputs = tf.cond(self.l0_train_toggle, lambda: tf.multiply(self.l0_z, outputs),
                          lambda: tf.multiply(self.trainable_mask, outputs))
        outputs = tf.multiply(self.capture_mask, outputs, MASKED_OUTPUT_NAME)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
