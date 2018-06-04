import datetime
import os
from copy import deepcopy
from math import inf
from random import shuffle

import tensorflow as tf
import numpy as np

from bonesaw.masked_layers import LO_VARIABLES_COLLECTION, LO_TRAIN_TOGGLE_OFF
from bonesaw.masked_layers import LO_TRAIN_TOGGLE_ON


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


def train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="train_op", actually_train=True, batches_to_feed=None):
    assert len(dataset.train_labels) == len(dataset.train_images)

    all_idxes = np.random.randint(0, dataset.train_images_num - 1, dataset.train_images_num)

    mean_loss, mean_accuracy, summary_batch_num, batch_num = 0.0, 0.0, 0, 0
    max_batches = dataset.train_images_num // FLAGS.batch_size
    summary_step_freq_divisor = 6
    summary_step_freq_batches = max_batches // summary_step_freq_divisor

    if train_op_name == "lasso_update_masks_op":
        train_op = network.lasso_update_masks_op
        loss = network.loss
        update_masks_lambda_op = None
    elif train_op_name == "l0_update_masks_op":
        train_op = network.l0_update_masks_op
        loss = network.lasso_masks_loss
        update_masks_lambda_op = network.lasso_update_masks_op
    else:
        train_op = network.train_op
        loss = network.l0_masks_loss
        update_masks_lambda_op = network.update_l0_masks_lambda_op

    if batches_to_feed is None:
        batches_to_feed = max_batches

    while batch_num < min(batches_to_feed, max_batches):
        i = batch_num * FLAGS.batch_size
        batch_idxes = all_idxes[i:i + FLAGS.batch_size]

        if actually_train:
            _, loss, accuracy = sess.run(
                [train_op, network.loss, network.accuracy_op],
                feed_dict={
                    network.input_plh: dataset.train_images[batch_idxes],
                    network.target_plh: dataset.train_labels[batch_idxes]
                })
        else:
            loss, accuracy = sess.run(
                [loss, network.accuracy_op],
                feed_dict={
                    network.input_plh: dataset.train_images[batch_idxes],
                    network.target_plh: dataset.train_labels[batch_idxes]
                })

        mean_loss += loss
        mean_accuracy += accuracy
        batch_num += 1
        summary_batch_num += 1

        if actually_train and summary_batch_num >= summary_step_freq_batches:
            mean_loss /= summary_batch_num
            mean_accuracy /= summary_batch_num
            step = epoch * summary_step_freq_divisor + batch_num // summary_step_freq_batches
            write_mean_summary(sess, step, network, train_writer, mean_accuracy, mean_loss, tag="train")
            mean_loss, mean_accuracy, summary_batch_num = 0.0, 0.0, 0

    if update_masks_lambda_op is not None:
        print("Updating masks lambda")
        sess.run(update_masks_lambda_op)

    return mean_loss/batch_num


def val_epoch(sess, train_writer, network, dataset, epoch, FLAGS):
    sess.run([network.is_training_assign_op], feed_dict={network.is_training_plh: False})

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
    sess.run([network.is_training_assign_op], feed_dict={network.is_training_plh: True})


def network_pretrain(sess, saver, train_writer, network, dataset, FLAGS):
    epoch = 0
    for _ in range(FLAGS.epochs):
        print("Training epoch {}".format(epoch))
        begin_ts = datetime.datetime.now()
        train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        print("... took {}".format(datetime.datetime.now() - begin_ts))
        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        epoch += 1
    print("Pretraining done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_pretrained_' + dataset.dataset_label))
    return epoch


def train_mask_bruteforce(sess, saver, train_writer, network, dataset, stripable_layers, last_epoch, FLAGS):
    epoch = last_epoch
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
                    loss_after_masking = train_epoch(sess, train_writer, network, dataset, epoch, FLAGS,
                                                     actually_train=False, batches_to_feed=64)
                    if loss_after_masking < best_loss:
                        best_loss = loss_after_masking
                        best_mask = new_mask
                        best_layer_name = layer_name
                        print("New best mask with loss {} for layer {}".format(best_loss, best_layer_name))
                    sess.run([layer.mask_assign_op], feed_dict={
                        layer.mask_plh: old_mask
                    })

        print("Applying mask to layer {}, {} channels left in channel, {} channels left to prune".format(
            best_layer_name, int(np.sum(best_mask)), FLAGS.filters_to_prune - filter_prune_idx))
        sess.run([stripable_layers[best_layer_name].mask_assign_op], feed_dict={
            stripable_layers[best_layer_name].mask_plh: best_mask
        })

        for _ in range(FLAGS.epochs_finetune):
            print("Fine-tune epoch {}".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

    print("Bruteforce mask training done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))


def train_mask_lasso(sess, saver, train_writer, network, dataset, stripable_layers, last_epoch, FLAGS):
    epoch = last_epoch
    for cycle in range(FLAGS.masks_lasso_cycles):
        print("Lasso cycle {}".format(cycle))

        for _ in range(FLAGS.masks_lasso_epochs):
            print("Epoch {} (lasso mask train)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="lasso_update_masks_op")
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

    print("Lasso mask training done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))


def train_mask_l0(sess, saver, train_writer, network, dataset, stripable_layers, last_epoch, FLAGS):
    epoch = last_epoch

    for toggle in tf.get_collection(LO_TRAIN_TOGGLE_ON):
        sess.run(toggle)

    for cycle in range(FLAGS.masks_l0_cycles):
        print("L0 cycle {}".format(cycle))
        print("Alpha")
        print(stripable_layers["conv1"].l0_alpha.eval())
        print("Beta")
        print(stripable_layers["conv1"].l0_beta.eval())

        sess.run([network.is_training_assign_op], feed_dict={network.is_training_plh: False})
        for _ in range(FLAGS.masks_l0_epochs):
            print("Epoch {} (L0 mask train)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="l0_update_masks_op")
            epoch += 1

        sess.run([network.is_training_assign_op], feed_dict={network.is_training_plh: True})
        for _ in range(FLAGS.masks_l0_epochs_finetune):
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)

    for name, layer in stripable_layers.items():
        alpha = layer.l0_alpha.eval()
        for i in range(len(alpha)):
            if alpha[i] > 0.0:
                alpha[i] = 1.0
            else:
                alpha[i] = 0.0
        sess.run(layer.mask_assign_op, feed_dict={
            layer.mask_plh: alpha
        })

    for toggle in tf.get_collection(LO_TRAIN_TOGGLE_OFF):
        sess.run(toggle)

    sess.run([network.is_training_assign_op], feed_dict={network.is_training_plh: True})
    for _ in range(FLAGS.masks_l0_epochs_finetune):
        train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        epoch += 1

    print("L0 mask training done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))
