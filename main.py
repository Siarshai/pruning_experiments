import glob
import os
import shutil

import tensorflow as tf

from bonesaw.weights_stripping import repack_graph
from network_under_surgery.model_creation import get_layers_names_for_dataset
from network_under_surgery.training import network_pretrain, train_mask_lasso, train_mask_l0, train_with_random_drop
from network_under_surgery.data_reading import load_dataset_to_memory
from network_under_surgery.training_ops_creation import create_network_under_surgery
from result_show import show_results_against_compression

Flags = tf.app.flags
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('log_dir', None, 'Summary directory for tensorboard log')
Flags.DEFINE_string('source_model_name', None, 'Model name to search in output_dir, will train from scratch if None')
Flags.DEFINE_string('pruned_model_name', "pruned_network", 'Name for saved pruned network')

Flags.DEFINE_float('learning_rate', 0.0015, 'The learning rate for the network')
Flags.DEFINE_float('beta1', 0.9, 'beta1 of Adam optimizer')
Flags.DEFINE_float('l2', 0.00025, 'l2 regularizer')
Flags.DEFINE_float('l1', 0.00005, 'l2 regularizer')
Flags.DEFINE_integer('batch_size', 32, 'Batch size of the input batch')
Flags.DEFINE_float('decay', 1e-6, 'Gamma of decaying')
Flags.DEFINE_integer('epochs', 30, 'The max epoch for the training')
Flags.DEFINE_integer('filters_to_prune', 96, 'Number of filters to drop with bruteforce algorithm')
Flags.DEFINE_integer('epochs_finetune', 1, 'Fine-tune epochs after filter drop')

Flags.DEFINE_float('randomdrop_percent', 0.5, '---')
Flags.DEFINE_integer('randomdrop_cycles', 30, '---')
Flags.DEFINE_integer('randomdrop_finetune_epochs', 3, '---')

Flags.DEFINE_float('masks_lasso_lambda_step', 0.0002, '---')
Flags.DEFINE_integer('masks_lasso_cycles', 20, '---')
Flags.DEFINE_integer('masks_lasso_epochs', 1, '---')
Flags.DEFINE_integer('masks_lasso_epochs_finetune', 3, 'Fine-tune epochs after filter drop with lasso train')
Flags.DEFINE_float('masks_lasso_capture_range', 0.075, '---')

Flags.DEFINE_float('masks_l0_lambda_step', 1.0, '---')
Flags.DEFINE_float('masks_l0_lambda_max', 10.0, '---')
Flags.DEFINE_integer('masks_l0_cycles', 30, '---')
Flags.DEFINE_integer('masks_l0_epochs', 1, '---')
Flags.DEFINE_integer('masks_l0_epochs_finetune', 1, '---')
Flags.DEFINE_float('masks_l0_learning_rate', 0.0035, 'The learning rate for the network')
Flags.DEFINE_integer('masks_l0_epochs_final_finetune', 20, '---')

Flags.DEFINE_string('task', "mask_l0", 'What we gonna do')
Flags.DEFINE_string('dataset', "cifar_10", 'What to feed to network')

FLAGS = Flags.FLAGS

# Preparing directory, checking passed arguments
if FLAGS.output_dir is None:
    FLAGS.output_dir = "output_dir"
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

dataset = load_dataset_to_memory(FLAGS.dataset)

if FLAGS.log_dir is None:
    if not os.path.exists(os.path.join(FLAGS.output_dir, "log")):
        os.mkdir(os.path.join(FLAGS.output_dir, "log"))
    run_idx = 0
    while os.path.exists(os.path.join(FLAGS.output_dir, "log", str(run_idx))):
        run_idx += 1
    os.mkdir(os.path.join(FLAGS.output_dir, "log", str(run_idx)))
    FLAGS.log_dir = os.path.join(FLAGS.output_dir, "log", str(run_idx))

print("Loaded data from {}:\n\t{} train examples\n\t{} test examples\n\t{} classes\n\tinput shape: {}\n".format(
    dataset.dataset_label, dataset.train_images_num, dataset.test_images_num, dataset.classes_num, dataset.image_shape))


def relocate_trained_model(model_folder, prefix, FLAGS):
    try:
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        shutil.copy2(os.path.join(FLAGS.output_dir, "checkpoint"), os.path.join(model_folder, "checkpoint"))
        for filename in glob.glob(os.path.join(FLAGS.output_dir, prefix)):
            shutil.copy2(filename, os.path.join(model_folder, os.path.split(filename)[-1]))
    except Exception as e:
        print("Could not relocate trained model: ", str(e))


if FLAGS.task in ["only_pretrain", "train_bruteforce", "train_lasso",
                  "train_l0", "mask_bruteforce", "mask_lasso", "mask_l0"]:

    with tf.Session() as sess:
        model_folder = dataset.dataset_label + "_model_pretrained_bak"
        last_epoch = 0
        network, saver, train_writer = create_network_under_surgery(sess, dataset, FLAGS)
        if "mask" not in FLAGS.task:
            print("Begin training")
            last_epoch = network_pretrain(sess, saver, train_writer, network, dataset, FLAGS)
            relocate_trained_model(model_folder, "model_pretrained_*", FLAGS)
        else:
            print("Loading pretrained weights")
            ckpt = tf.train.get_checkpoint_state(model_folder)
            saver.restore(sess, ckpt.model_checkpoint_path)

        need_to_save = False
        if "lasso" in FLAGS.task:
            print("Continue training with lasso")
            train_mask_lasso(sess, saver, train_writer, network, dataset, last_epoch, FLAGS)
            need_to_save = True
        elif "l0" in FLAGS.task:
            print("Continue training with l0")
            train_mask_l0(sess, saver, train_writer, network, dataset, last_epoch, FLAGS)
            need_to_save = True
        elif "randomdrop" in FLAGS.task:
            print("Continue training with l0")
            train_with_random_drop(sess, saver, train_writer, network, dataset, last_epoch, FLAGS)
            need_to_save = True

        if need_to_save:
            model_folder = dataset.dataset_label + "_model_masked_bak"
            print("Training is over, moving model to separate folder")
            try:
                relocate_trained_model(model_folder, "model_*", FLAGS)
            except Exception as e:
                print("Could not relocate trained model: {}", str(e))

elif FLAGS.task in ["eval", "eval_repack", "eval_repack_randomdrop"]:
    model_folder = dataset.dataset_label + "_model_masked_bak"
    repacked_weights_list, compressions = None, []
    with tf.Session() as sess:
        network, saver, train_writer, stripable_layers = create_network_under_surgery(sess, dataset, FLAGS)
        ckpt = tf.train.get_checkpoint_state(model_folder)
        saver.restore(sess, ckpt.model_checkpoint_path)

        if FLAGS.task in ["eval", "eval_repack"]:
            loss, accuracy = sess.run([network.loss, network.accuracy_op], feed_dict={
                network.input_plh: dataset.test_images,
                network.target_plh: dataset.test_labels
            })
            print("Val loss after loading: {}".format(loss))
            print("Val accuracy after loading: {}".format(accuracy))

        if FLAGS.task != "eval_repack_randomdrop":
            random_drop_order = [0.0]
            random_drop_tries = 1
        else:
            random_drop_order = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
            random_drop_tries = 3

        if FLAGS.task == "eval":
            exit(0)

        repacked_weights_list = []
        for random_drop_p in random_drop_order:
            for random_drop_try in range(random_drop_tries):
                print("Repacking with {} random drop".format(random_drop_p))
                evaluated_trainable_variables, compression = repack_graph(
                    sess.graph, get_layers_names_for_dataset(dataset.dataset_label), random_drop=random_drop_p, debug=False)
                repacked_weights_list.append(evaluated_trainable_variables)
                compressions.append(compression)

    if FLAGS.task in ["eval_repack", "eval_repack_randomdrop"]:
        assert repacked_weights_list
        losses, accuracies = [], []
        for i, repacked_weights in enumerate(repacked_weights_list):
            print("{}/{}".format(i+1, len(repacked_weights_list)))
            with tf.Session() as sess:
                print("Restoring network with stripped weights...")
                layers_order = get_layers_names_for_dataset(dataset.dataset_label)
                network, saver, train_writer, _ = create_network_under_surgery(sess, dataset, FLAGS, repacked_weights, layers_order)
                print("Running...")
                loss, accuracy = sess.run([network.loss, network.accuracy_op], feed_dict={
                    network.input_plh: dataset.test_images,
                    network.target_plh: dataset.test_labels
                })
                print("Val loss after repacking: {}".format(loss))
                print("Val accuracy after repacking: {}".format(accuracy))
                losses.append(loss)
                accuracies.append(accuracy)
        if FLAGS.task == "eval_repack_randomdrop":
            print(compressions)
            print(accuracies)
            print(losses)
            show_results_against_compression(compressions, accuracies, losses)
        else:
            print("compression: {}".format(compressions))
            print("accuracy: {}".format(accuracies))
            print("loss: {}".format(losses))


else:
    raise ValueError("Unknown task: " + FLAGS.task)

print("Done")
