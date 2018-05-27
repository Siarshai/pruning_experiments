import datetime
from collections import namedtuple
from bonesaw.masked_layers import MASKS_COLLECTION, MASKABLE_TRAINABLES, LO_LOSSES_COLLECTION, LO_VARIABLES_COLLECTION

import tensorflow as tf

from bonesaw.network_restoration import get_restore_network_function
from network_under_surgery.model_creation import get_create_network_function


def create_network_under_surgery(sess, dataset, FLAGS, repacked_weights=None, layers_order=None):
    network_input = tf.placeholder(tf.float32, [None] + list(dataset.image_shape), 'main_input')
    network_target = tf.placeholder(tf.int32, [None, dataset.classes_num], 'main_target')

    is_training = tf.Variable(initial_value=True, trainable=False, name="is_training")
    is_training_plh = tf.placeholder(tf.bool, [], 'is_training_plh')
    is_training_assign_op = tf.assign(is_training, is_training_plh)

    begin_ts = datetime.datetime.now()
    if repacked_weights is not None and layers_order is not None:
        restore_network_fn = get_restore_network_function(dataset.dataset_label)
        network_logits = restore_network_fn(network_input, layers_order, repacked_weights, debug=False)
        stripable_layers = None
    else:
        create_network_fn = get_create_network_function(dataset.dataset_label)
        network_logits, stripable_layers = create_network_fn(network_input, dataset.classes_num, is_training)

    print("Network created ({}), preparing ops".format(datetime.datetime.now() - begin_ts))
    network = create_training_ops(network_input, network_logits, network_target, is_training_assign_op, is_training_plh, FLAGS)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    return network, saver, train_writer, stripable_layers


def create_training_ops(network_input, network_logits, network_target, is_training_assign_op, is_training_plh, FLAGS):
    global_step = tf.train.get_or_create_global_step()
    mask_update_op = None
    direct_summaries, mean_summaries, mean_summaries_plh = {}, {}, {}

    classes_op = tf.argmax(input=network_logits, axis=1, output_type=tf.int32)
    probabilities_op = tf.nn.softmax(network_logits)
    accuracy_op = tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(input=network_target, axis=1, output_type=tf.int32), classes_op)))

    regularizer_loss = 0
    for weight in tf.get_collection(MASKABLE_TRAINABLES):
        regularizer_loss += FLAGS.l2 * tf.nn.l2_loss(weight)
        regularizer_loss += FLAGS.l1 * tf.reduce_sum(tf.abs(weight))

    loss = regularizer_loss + tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=network_target, logits=network_logits))

    lasso_masks_lambda = tf.Variable(FLAGS.masks_lasso_lambda_step, trainable=False, name="lasso_masks_lambda")
    all_masks = tf.concat(tf.get_collection(MASKS_COLLECTION), axis=0)
    number_of_channels_loss = lasso_masks_lambda * tf.reduce_mean(tf.abs(all_masks))
    lasso_masks_loss = loss + number_of_channels_loss

    l0_masks_lambda = tf.Variable(FLAGS.masks_l0_lambda_step, trainable=False, name="l0_masks_lambda")
    l0_complexities = tf.get_collection(LO_LOSSES_COLLECTION)
    l0_loss = l0_masks_lambda*tf.reduce_sum(l0_complexities)
    l0_masks_loss = loss + l0_loss

    direct_summaries['accuracy'] = tf.summary.scalar('accuracy', accuracy_op)
    direct_summaries['loss'] = tf.summary.scalar('loss', loss)

    mean_accuracy_op = tf.placeholder(tf.float32, shape=(), name='mean_accuracy_plh')
    mean_loss_op = tf.placeholder(tf.float32, shape=(), name='mean_loss_plh')
    mean_summaries_plh['mean_accuracy'] = mean_accuracy_op
    mean_summaries_plh['mean_loss'] = mean_loss_op
    mean_summaries['mean_train_accuracy'] = tf.summary.scalar('mean_train_accuracy', mean_accuracy_op)
    mean_summaries['mean_train_loss'] = tf.summary.scalar('mean_train_loss', mean_loss_op)
    mean_summaries['mean_val_accuracy'] = tf.summary.scalar('mean_val_accuracy', mean_accuracy_op)
    mean_summaries['mean_val_loss'] = tf.summary.scalar('mean_val_loss', mean_loss_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            var_list=tf.get_collection(MASKABLE_TRAINABLES))

        lasso_masks_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
        lasso_update_masks_op = lasso_masks_optimizer.minimize(
            loss=lasso_masks_loss,
            global_step=tf.train.get_global_step(),
            var_list=tf.get_collection(MASKS_COLLECTION))
        update_lasso_masks_lambda_op = tf.assign(lasso_masks_lambda, lasso_masks_lambda + FLAGS.masks_lasso_lambda_step)
        lasso_update_masks_op = tf.group(lasso_update_masks_op, update_lasso_masks_lambda_op)

        l0_masks_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
        l0_update_masks_op = l0_masks_optimizer.minimize(
            loss=l0_masks_loss,
            global_step=tf.train.get_global_step(),
            var_list=tf.get_collection(LO_VARIABLES_COLLECTION))
        update_l0_masks_lambda_op = tf.assign(l0_masks_lambda, l0_masks_lambda + FLAGS.masks_lasso_lambda_step)
        l0_update_masks_op = tf.group(l0_update_masks_op, update_l0_masks_lambda_op)

    Network = namedtuple("Network", "input_plh, target_plh, network_logits, train_op, "
                                    "is_training_assign_op, is_training_plh, "
                                    "classes_op, probabilities_op, accuracy_op, "
                                    "loss, mask_update_op, global_step, "
                                    "direct_summaries, mean_summaries, mean_summaries_plh, "
                                    "lasso_update_masks_op, l0_update_masks_op")
    return Network(
        input_plh=network_input,
        target_plh=network_target,
        network_logits=network_logits,
        is_training_assign_op=is_training_assign_op,
        is_training_plh=is_training_plh,
        train_op=train_op,
        classes_op=classes_op,
        probabilities_op=probabilities_op,
        accuracy_op=accuracy_op,
        loss=loss,
        mask_update_op=mask_update_op,
        global_step=global_step,
        direct_summaries=direct_summaries,
        mean_summaries=mean_summaries,
        mean_summaries_plh=mean_summaries_plh,
        lasso_update_masks_op=lasso_update_masks_op,
        l0_update_masks_op=l0_update_masks_op
    )
