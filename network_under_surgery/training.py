import datetime
import os
from random import shuffle

import tensorflow as tf
import numpy as np

from bonesaw.masked_layers import LO_TRAIN_TOGGLE_OFF, LO_VARIABLES_COLLECTION, MASKS_COLLECTION, \
    MASKS_ASSIGN_COLLECTION, MASKS_PLH_COLLECTION, CAPTURE_MASKS_PLH_COLLECTION, CAPTURE_MASKS_ASSIGN_COLLECTION
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
        loss = network.lasso_masks_loss
        update_masks_lambda_op = network.update_lasso_masks_lambda_op
    elif train_op_name == "l0_update_masks_op":
        train_op = network.l0_update_masks_op
        loss = network.l0_masks_loss
        update_masks_lambda_op = network.update_l0_masks_lambda_op
    else:
        train_op = network.train_op
        loss = network.loss
        update_masks_lambda_op = None

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
    sess.run([network.is_training_assign_op], feed_dict={
        network.is_training_plh: False
    })

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
    sess.run([network.is_training_assign_op], feed_dict={
        network.is_training_plh: True
    })


def network_pretrain(sess, saver, train_writer, network, dataset, FLAGS):
    epoch = 0
    sess.run([network.is_training_assign_op], feed_dict={
        network.is_training_plh: True
    })
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


def train_with_random_drop(sess, saver, train_writer, network, dataset, last_epoch, FLAGS):
    epoch = last_epoch

    masks = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_COLLECTION)}
    masks_plhs = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_PLH_COLLECTION)}
    masks_assign_ops = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_ASSIGN_COLLECTION)}

    for cycle in range(FLAGS.randomdrop_cycles):
        print("Random drop cycle {}".format(cycle))

        for layer_name, mask_var in masks.items():
            mask_plh = masks_plhs[layer_name]
            mask_assign_op = masks_assign_ops[layer_name]
            mask = mask_var.eval()
            last_channel_idx = FLAGS.randomdrop_percent*cycle*len(mask)//FLAGS.randomdrop_cycles
            for channel_idx, m in enumerate(mask):
                mask[channel_idx] = 0.0 if channel_idx <= last_channel_idx else 1.0
            sess.run([mask_assign_op], feed_dict={
                mask_plh: mask
            })

        print("Finetuning...".format(cycle))
        for _ in range(FLAGS.randomdrop_finetune_epochs):
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

    print("Lasso mask training done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))


def train_mask_lasso(sess, saver, train_writer, network, dataset, last_epoch, FLAGS):
    epoch = last_epoch

    masks = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_COLLECTION)}
    masks_plhs = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_PLH_COLLECTION)}
    masks_assign_ops = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_ASSIGN_COLLECTION)}
    capture_masks_plhs = {var.name.split("/")[0]: var for var in tf.get_collection(CAPTURE_MASKS_PLH_COLLECTION)}
    capture_masks_assign_ops = {var.name.split("/")[0]: var for var in
                                tf.get_collection(CAPTURE_MASKS_ASSIGN_COLLECTION)}

    for cycle in range(FLAGS.masks_lasso_cycles):
        print("Lasso cycle {}".format(cycle))

        for _ in range(FLAGS.masks_lasso_epochs):
            print("Epoch {} (lasso mask train)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="lasso_update_masks_op")
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

        channels_left = 0
        for layer_name, mask_var in masks.items():
            mask_plh = masks_plhs[layer_name]
            mask_assign_op = masks_assign_ops[layer_name]
            capture_mask_plh = capture_masks_plhs[layer_name]
            capture_mask_assign_op = capture_masks_assign_ops[layer_name]
            mask = mask_var.eval()
            capture_mask = np.zeros(mask.shape)
            for channel_idx, m in enumerate(mask):
                if m < FLAGS.masks_lasso_capture_range:
                    capture_mask[channel_idx] = 0.0
                    mask[channel_idx] = 0.0
                else:
                    capture_mask[channel_idx] = 1.0
                    channels_left += 1
            sess.run([mask_assign_op], feed_dict={
                mask_plh: mask
            })
            sess.run([capture_mask_assign_op], feed_dict={
                capture_mask_plh: capture_mask
            })

        for _ in range(FLAGS.masks_lasso_epochs_finetune):
            print("Epoch {} (finetune)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

        print("Channels left {}".format(channels_left))

    print("Lasso mask training done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))


def train_mask_l0(sess, saver, train_writer, network, dataset, last_epoch, FLAGS):
    epoch = last_epoch

    for toggle in tf.get_collection(LO_TRAIN_TOGGLE_ON):
        sess.run(toggle)

    alphas = {var.name.split("/")[0]: var for var in tf.get_collection(LO_VARIABLES_COLLECTION)}
    masks_plhs = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_PLH_COLLECTION)}
    masks_assign_ops = {var.name.split("/")[0]: var for var in tf.get_collection(MASKS_ASSIGN_COLLECTION)}

    for cycle in range(FLAGS.masks_l0_cycles):
        print("L0 cycle {}".format(cycle))
        print("Alpha")
        print(alphas["masking_layer_1"].eval())
        sess.run([network.is_training_assign_op], feed_dict={
            network.is_training_plh: False
        })
        for _ in range(FLAGS.masks_l0_epochs):
            print("Epoch {} (L0 mask train)".format(epoch))
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS, train_op_name="l0_update_masks_op")
            epoch += 1

        sess.run([network.is_training_assign_op], feed_dict={
            network.is_training_plh: True
        })
        for _ in range(FLAGS.masks_l0_epochs_finetune):
            train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
            epoch += 1

        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)

    for layer_name, alpha_var in alphas.items():
        mask_assign_op = masks_assign_ops[layer_name]
        mask_plh = masks_plhs[layer_name]
        alpha = alpha_var.eval()
        for i in range(len(alpha)):
            if alpha[i] > 0.0:
                alpha[i] = 1.0
            else:
                alpha[i] = 0.0
        sess.run(mask_assign_op, feed_dict={
            mask_plh: alpha
        })

    for toggle in tf.get_collection(LO_TRAIN_TOGGLE_OFF):
        sess.run(toggle)

    sess.run([network.is_training_assign_op], feed_dict={
        network.is_training_plh: True
    })
    print("Finetuning...")
    for _ in range(FLAGS.masks_l0_epochs_finetune):
        train_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        val_epoch(sess, train_writer, network, dataset, epoch, FLAGS)
        epoch += 1

    print("L0 mask training done, saving model")
    saver.save(sess, os.path.join(FLAGS.output_dir, 'model_' + dataset.dataset_label))
