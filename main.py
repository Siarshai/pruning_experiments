import glob
import os
import shutil

import tensorflow as tf

from bonesaw.network_restoration import restore_network
from bonesaw.weights_stripping import repack_graph
from network_under_surgery.network_creation import create_network
from network_under_surgery.training_ops import create_training_ops, simple_train
from network_under_surgery.data_reading import load_cifar_10_to_memory, load_mnist_to_memory

Flags = tf.app.flags
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('log_dir', None, 'Summary directory for tensorboard log')
Flags.DEFINE_string('source_model_name', None, 'Model name to search in output_dir, will train from scratch if None')
Flags.DEFINE_string('pruned_model_name', "pruned_network", 'Name for saved pruned network')

Flags.DEFINE_float('learning_rate', 0.001, 'The learning rate for the network')
Flags.DEFINE_float('beta1', 0.975, 'beta1 of Adam optimizer')
Flags.DEFINE_integer('batch_size', 32, 'Batch size of the input batch')
Flags.DEFINE_float('decay', 1e-6, 'Gamma of decaying')
Flags.DEFINE_integer('epochs', 30, 'The max epoch for the training')

Flags.DEFINE_string('task', "eval_repack", 'What we gonna do')
Flags.DEFINE_string('dataset', "mnist", 'What to feed to network')

FLAGS = Flags.FLAGS

# Preparing directory, checking passed arguments
if FLAGS.output_dir is None:
    FLAGS.output_dir = "output_dir"
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

if FLAGS.log_dir is None:
    if not os.path.exists(os.path.join(FLAGS.output_dir, "log")):
        os.mkdir(os.path.join(FLAGS.output_dir, "log"))
    run_idx = 0
    while os.path.exists(os.path.join(FLAGS.output_dir, "log", str(run_idx))):
        run_idx += 1
    os.mkdir(os.path.join(FLAGS.output_dir, "log", str(run_idx)))
    FLAGS.log_dir = os.path.join(FLAGS.output_dir, "log", str(run_idx))


# Reading data
if FLAGS.dataset == "mnist":
    dataset = load_mnist_to_memory(True)
elif FLAGS.dataset == "cifar_10":
    dataset = load_cifar_10_to_memory(True)
else:
    raise ValueError("Unknown dataset: " + FLAGS.dataset)

print("Loaded data from {}:\n\t{} train examples\n\t{} test examples\n\t{} classes\n\tinput shape: {}\n".format(
    dataset.dataset_label, dataset.train_images_num, dataset.test_images_num, dataset.classes_num, dataset.image_shape))


def create_network_under_surgery(sess, repacked_weights=None, layers_order=None):
    network_input = tf.placeholder(tf.float32, [None] + list(dataset.image_shape), 'main_input')
    network_target = tf.placeholder(tf.int32, [None, dataset.classes_num], 'main_target')
    if repacked_weights is not None and layers_order is not None:
        network_logits = restore_network(network_input, layers_order, repacked_weights, debug=True)
    else:
        network_logits = create_network(network_input, dataset.classes_num)
    network = create_training_ops(network_input, network_logits, network_target, FLAGS)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    return network_input, network_target, network_logits, network, saver, train_writer


model_folder = dataset.dataset_label + "_model_bak"

if FLAGS.task == "train":
    with tf.Session() as sess:
        network_input, network_target, network_logits, network, saver, train_writer = create_network_under_surgery(sess)
        print("Begin training")
        simple_train(sess, saver, train_writer, network, dataset, FLAGS)
        print("Training is over, moving model to separate folder")
        try:
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            shutil.copy2(os.path.join(FLAGS.output_dir, "checkpoint"), os.path.join(model_folder, "checkpoint"))
            for filename in glob.glob(os.path.join(FLAGS.output_dir, "model_*")):
                shutil.copy2(filename, os.path.join(model_folder, os.path.split(filename)[-1]))
        except Exception as e:
            print("Could not relocate trained model: {}", str(e))
elif FLAGS.task in ["eval", "repack", "eval_repack"]:
    repacked_weights = None
    with tf.Session() as sess:
        network_input, network_target, network_logits, network, saver, train_writer = create_network_under_surgery(sess)

        ckpt = tf.train.get_checkpoint_state(model_folder)
        saver.restore(sess, ckpt.model_checkpoint_path)

        network_input = sess.graph.get_tensor_by_name("main_input:0")
        network_target = sess.graph.get_tensor_by_name("main_target:0")
        network_logits = sess.graph.get_tensor_by_name("dense2_logits/BiasAdd:0")

        network = create_training_ops(network_input, network_logits, network_target, FLAGS)

        if FLAGS.task in ["eval", "eval_repack"]:
            loss, accuracy = sess.run([network.loss, network.accuracy_op], feed_dict={
                network.input_plh: dataset.test_images,
                network.target_plh: dataset.test_labels
            })
            print("Val loss after loading: {}".format(loss))
            print("Val accuracy after loading: {}".format(accuracy))

        if FLAGS.task in ["repack", "eval_repack"]:
            print("Repacking...")
            repacked_weights = repack_graph(sess.graph, ["conv1", "conv2", "conv3"], debug=True)

    if FLAGS.task in ["repack", "eval_repack"]:
        assert repacked_weights
        with tf.Session() as sess:
            print("Restoring network with stripped weights...")
            network_input, network_target, network_logits, network, saver, train_writer = \
                create_network_under_surgery(
                    sess, repacked_weights,
                    ["conv1", "conv2", "conv3", "dense1", "dense2_logits"])

            loss, accuracy = sess.run([network.loss, network.accuracy_op], feed_dict={
                network.input_plh: dataset.test_images,
                network.target_plh: dataset.test_labels
            })
            print("Val loss after repacking: {}".format(loss))
            print("Val accuracy after repacking: {}".format(accuracy))
else:
    raise ValueError("Unknown task: " + FLAGS.task)

print("Done")
