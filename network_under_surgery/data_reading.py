import struct

import collections
import pickle
import os
import numpy as np
import keras

Dataset = collections.namedtuple("dataset_tensors",
                                 'classes_num, image_shape, train_images_num, test_images_num, '
                                 'train_images, train_labels, test_images, test_labels, dataset_label')


def reshape_and_normalize(image):
    normalized_image = image.astype(dtype=np.float32)
    if len(normalized_image.shape) == 2:
        normalized_image = np.reshape(normalized_image, (normalized_image.shape[0], normalized_image.shape[1], 1))
    elif len(normalized_image.shape) == 3:
        pass
    elif len(normalized_image.shape) == 1:
        normalized_image = np.reshape(normalized_image, (32, 32, 3)) # TODO: unhardcode
    else:
        raise RuntimeError("Illegal input to reshape_and_normalize: shape: {}", normalized_image.shape)
    normalized_image = 2.0 * (normalized_image / 255.0) - 1.0
    return normalized_image


def load_mnist_to_memory(ohe=False):
    MNIST_PREFIX = "D:\MLData\MNIST"

    with open(os.path.join(MNIST_PREFIX, "train-labels-idx1-ubyte", "train-labels.idx1-ubyte"), 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        train_labels = np.fromfile(flbl, dtype=np.int8)
    with open(os.path.join(MNIST_PREFIX, "train-images-idx3-ubyte", "train-images.idx3-ubyte"), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        train_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_labels), rows, cols)
        train_images = np.asarray([reshape_and_normalize(img) for img in train_images])

    with open(os.path.join(MNIST_PREFIX, "t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"), 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        test_labels = np.fromfile(flbl, dtype=np.int8)
    with open(os.path.join(MNIST_PREFIX, "t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        test_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_labels), rows, cols)
        test_images = np.asarray([reshape_and_normalize(img) for img in test_images])

    classes_num = len(set(train_labels))
    if ohe:
        train_labels = keras.utils.to_categorical(train_labels, classes_num)
        test_labels = keras.utils.to_categorical(test_labels, classes_num)

    assert len(train_images) == len(train_labels)
    assert len(test_images) == len(test_labels)

    dataset = Dataset(
        classes_num=classes_num,
        image_shape=train_images[0].shape,
        train_images_num=len(train_images),
        test_images_num=len(test_images),
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        dataset_label="mnist"
    )
    return dataset


def load_cifar_10_to_memory(ohe=False):
    CIFAR_10_PREFIX = "D:\MLData\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py"
    CIFAR_10_TRAIN_FILES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    CIFAR_10_TEST_FILES = ["test_batch"]

    train_images, train_labels = [], []
    for filename in CIFAR_10_TRAIN_FILES:
        with open(os.path.join(CIFAR_10_PREFIX, filename), 'rb') as fo:
            filedata = pickle.load(fo, encoding='bytes')
            for image, label in zip(filedata[b'data'], filedata[b'labels']):
                train_images.append(reshape_and_normalize(image))
                train_labels.append(label)

    test_images, test_labels = [], []
    for filename in CIFAR_10_TEST_FILES:
        with open(os.path.join(CIFAR_10_PREFIX, filename), 'rb') as fo:
            filedata = pickle.load(fo, encoding='bytes')
            for image, label in zip(filedata[b'data'], filedata[b'labels']):
                test_images.append(reshape_and_normalize(image))
                test_labels.append(label)

    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)
    classes_num = len(set(train_labels))
    if ohe:
        train_labels = keras.utils.to_categorical(train_labels, classes_num)
        test_labels = keras.utils.to_categorical(test_labels, classes_num)
    else:
        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)

    assert len(train_images) == len(train_labels)
    assert len(test_images) == len(test_labels)

    dataset = Dataset(
        classes_num=classes_num,
        image_shape=train_images[0].shape,
        train_images_num=len(train_images),
        test_images_num=len(test_images),
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        dataset_label="cifar_10"
    )

    return dataset

def load_dataset_to_memory(dataset_label):
    return {
        "mnist": load_mnist_to_memory,
        "cifar_10": load_cifar_10_to_memory
    }[dataset_label](True)