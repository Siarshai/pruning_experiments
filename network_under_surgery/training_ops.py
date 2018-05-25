import datetime
from collections import namedtuple
from copy import deepcopy
from math import inf
from random import shuffle
from bonesaw.masked_layers import MASKS_COLLECTION, MASKABLE_TRAINABLES

import os
import tensorflow as tf
import numpy as np


def create_training_ops(network_input, network_logits, network_target, FLAGS):

    global_step = tf.train.get_or_create_global_step()
    mask_update_op = None
    direct_summaries, mean_summaries, mean_summaries_plh = {}, {}, {}

    classes_op = tf.argmax(input=network_logits, axis=1, output_type=tf.int32)
    probabilities_op = tf.nn.softmax(network_logits)
    accuracy_op = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(input=network_target, axis=1, output_type=tf.int32), classes_op)))

    regularizer_loss = 0
    for weight in tf.get_collection(MASKABLE_TRAINABLES):
        regularizer_loss += FLAGS.l2*tf.nn.l2_loss(weight)
        regularizer_loss += FLAGS.l1*tf.reduce_sum(tf.abs(weight))

    loss = regularizer_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=network_target, logits=network_logits))

    masks_loss = None
    masks_lasso_lambda = None
    if FLAGS.task == "train_lasso":
        masks_lasso_lambda = tf.Variable(FLAGS.masks_lasso_lambda_step, trainable=False, name="masks_lasso_lambda")
        all_masks = tf.concat(tf.get_collection(MASKS_COLLECTION), axis=0)
        number_of_channels_loss = masks_lasso_lambda*tf.reduce_mean(tf.abs(all_masks))
        masks_loss = loss + number_of_channels_loss

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
        update_masks_op = None
        if FLAGS.task == "train_lasso":
            masks_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
            update_masks_op = masks_optimizer.minimize(
                loss=masks_loss,
                global_step=tf.train.get_global_step(),
                var_list=tf.get_collection(MASKS_COLLECTION))
            update_masks_lambda_op = tf.assign(masks_lasso_lambda, masks_lasso_lambda + FLAGS.masks_lasso_lambda_step)
            update_masks_op = tf.group(update_masks_op, update_masks_lambda_op)

    Network = namedtuple("Network", "input_plh, target_plh, train_op, "
                                    "classes_op, probabilities_op, accuracy_op, "
                                    "loss, mask_update_op, global_step, "
                                    "direct_summaries, mean_summaries, mean_summaries_plh, "
                                    "update_masks_op")
    return Network(
        input_plh=network_input,
        target_plh=network_target,
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
        update_masks_op=update_masks_op
    )


def simple_train(sess, saver, train_writer, network, dataset, FLAGS):
    for epoch in range(FLAGS.epochs):
        print("Training epoch {}".format(epoch))
        begin_ts = datetime.datetime.now()
        train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        print("... took {}".format(datetime.datetime.now() - begin_ts))
        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))


def write_mean_summary(sess, step, network, train_writer, mean_accuracy, mean_loss, tag):
    summary_loss, summary_accuracy = sess.run(
        [network.mean_summaries["mean_" + tag + "_loss"], network.mean_summaries["mean_" + tag + "_accuracy"]],
        feed_dict={
            network.mean_summaries_plh["mean_loss"]: mean_loss,
            network.mean_summaries_plh["mean_accuracy"]: mean_accuracy
        })
    train_writer.add_summary(summary_loss, step)
    train_writer.add_summary(summary_accuracy, step)
    print("{} loss: {}".format(tag, mean_loss))
    print("{} accuracy: {}".format(tag, mean_accuracy))


def train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="train_op"):

    assert len(dataset.train_labels) == len(dataset.train_images)

    all_idxes = np.random.randint(0, dataset.train_images_num-1, dataset.train_images_num)

    mean_loss, mean_accuracy, summary_batch_num, batch_num = 0.0, 0.0, 0, 0
    max_batches = dataset.train_images_num // FLAGS.batch_size
    summary_step_freq_divisor = 6
    summary_step_freq_batches = max_batches//summary_step_freq_divisor

    if train_op_name == "update_masks_op":
        train_op = network.update_masks_op
    else:
        train_op = network.train_op

    while batch_num < max_batches:
        i = batch_num * FLAGS.batch_size
        batch_idxes = all_idxes[i:i + FLAGS.batch_size]

        _, loss, accuracy = sess.run(
            [train_op, network.loss, network.accuracy_op],
            feed_dict={
                network.input_plh: dataset.train_images[batch_idxes],
                network.target_plh: dataset.train_labels[batch_idxes]
            })

        mean_loss += loss
        mean_accuracy += accuracy
        batch_num += 1
        summary_batch_num += 1

        if summary_batch_num >= summary_step_freq_batches:
            mean_loss /= summary_batch_num
            mean_accuracy /= summary_batch_num
            step = epoch*summary_step_freq_divisor + batch_num//summary_step_freq_batches
            write_mean_summary(sess, step, network, train_writer, mean_accuracy, mean_loss, tag="train")
            mean_loss, mean_accuracy, summary_batch_num = 0.0, 0.0, 0


def val_epoch(sess, train_writer, network, dataset, epoch, FLAGS):

    all_idxes = list(range(dataset.test_images_num))
    shuffle(all_idxes)
    mean_loss, mean_accuracy, batch_num = 0.0, 0.0, 0
    max_batches = dataset.test_images_num // FLAGS.batch_size

    while batch_num < max_batches:
        i = batch_num * FLAGS.batch_size
        batch_idxes = all_idxes[i:i + FLAGS.batch_size]

        loss, accuracy = sess.run(
            [network.loss, network.accuracy_op],
            feed_dict={
                network.input_plh: dataset.test_images[batch_idxes],
                network.target_plh: dataset.test_labels[batch_idxes]
            })

        mean_loss += loss
        mean_accuracy += accuracy
        batch_num += 1

    mean_loss /= batch_num
    mean_accuracy /= batch_num
    write_mean_summary(sess, epoch, network, train_writer, mean_accuracy, mean_loss, tag="val")


def pretend_train_epoch(sess, network, dataset, batches_to_feed, FLAGS):

    assert len(dataset.train_labels) == len(dataset.train_images)

    all_idxes = list(range(dataset.train_images_num))
    shuffle(all_idxes)
    mean_loss, mean_accuracy, summary_batch_num, batch_num = 0.0, 0.0, 0, 0
    max_batches = dataset.train_images_num // FLAGS.batch_size

    while batch_num < min(max_batches, batches_to_feed):
        i = batch_num * FLAGS.batch_size
        batch_idxes = all_idxes[i:i + FLAGS.batch_size]

        loss, = sess.run(
            [network.loss],
            feed_dict={
                network.input_plh: dataset.train_images[batch_idxes],
                network.target_plh: dataset.train_labels[batch_idxes]
            })

        mean_loss += loss
        batch_num += 1

    mean_loss /= batch_num
    return mean_loss


def train_bruteforce(sess, saver, train_writer, network, dataset, stripable_layers, FLAGS):
    epoch = 0
    for _ in range(FLAGS.epochs):
        print("Epoch {}".format(epoch))
        train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        epoch += 1

    for filter_prune_idx in range(FLAGS.filters_to_prune):
        best_loss = inf
        best_mask = None
        best_layer_name = None
        for layer_name, layer in stripable_layers.items():
            old_mask = layer.mask.eval()
            for channel_idx, m in enumerate(old_mask):
                if m:
                    new_mask = deepcopy(old_mask)
                    new_mask[channel_idx] = 0.0
                    sess.run([layer.mask_assign_op], feed_dict={
                        layer.mask_plh: new_mask
                    })
                    loss_after_masking = pretend_train_epoch(sess, network, dataset, 64, FLAGS)
                    if loss_after_masking < best_loss:
                        best_loss = loss_after_masking
                        best_mask = new_mask
                        best_layer_name = layer_name
                        print("New best mask with loss {} for layer {}".format(best_loss, best_layer_name))
                    sess.run([layer.mask_assign_op], feed_dict={
                        layer.mask_plh: old_mask
                    })

        print("Applying mask to layer {}, {} channels left in channel, {} channels left to prune".format(
            best_layer_name, int(np.sum(best_mask)), FLAGS.filters_to_prune-filter_prune_idx))
        sess.run([stripable_layers[best_layer_name].mask_assign_op], feed_dict={
            stripable_layers[best_layer_name].mask_plh: best_mask
        })

        for _ in range(FLAGS.epochs_finetune):
            print("Fine-tune epoch {}".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))


def train_lasso(sess, saver, train_writer, network, dataset, stripable_layers, FLAGS):
    epoch = 0
    for _ in range(FLAGS.epochs):
        print("Epoch {}".format(epoch))
        train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        epoch += 1

    for cycle in range(FLAGS.masks_lasso_cycles):
        print("Lasso cycle {}".format(cycle))

        for _ in range(FLAGS.masks_lasso_epochs):
            print("Epoch {} (lasso mask train)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="update_masks_op")
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

        # capture
        channels_left = 0
        for layer_name, layer in stripable_layers.items():
            mask_val = layer.mask.eval()
            capture_mask = np.zeros(mask_val.shape)
            for channel_idx, m in enumerate(mask_val):
                if m < FLAGS.masks_lasso_capture_range:
                    capture_mask[channel_idx] = 0.0
                    mask_val[channel_idx] = 0.0
                else:
                    capture_mask[channel_idx] = 1.0
                    channels_left += 1
            sess.run([layer.mask_assign_op], feed_dict={
                layer.mask_plh: mask_val
            })
            sess.run([layer.capture_mask_assign_op], feed_dict={
                layer.capture_mask_plh: capture_mask
            })

        for _ in range(FLAGS.masks_lasso_epochs_finetune):
            print("Epoch {} (finetune)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

        print("Channels left {}".format(channels_left))

    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))
