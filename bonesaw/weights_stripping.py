import random
from collections import OrderedDict
from itertools import cycle

import numpy as np

from bonesaw.masked_layers import WEIGHT_NAME, BIAS_NAME, MASK_NAME, MASKS_COLLECTION


def eval_weights_from_graph(graph, collection_name, debug=False):
    trainable_variables_list = graph.get_collection(collection_name)
    if debug:
        print("get_weights_from_graph: fetched trainable variables")
    evaluated_weights = OrderedDict()
    for var in trainable_variables_list:
        if debug:
            print(" {}, shape {}".format(var.name, var.shape))
        evaluated_weights[var.name.split(":")[0]] = var.eval()
    return evaluated_weights


def _get_bias_from_weight_path(evaluated_weights, weight_name):
    if not weight_name.endswith(WEIGHT_NAME):
        raise ValueError("Unexpected name: {}", weight_name)
    origin_name = weight_name.split("/")[0]
    bias_name = origin_name + "/" + BIAS_NAME
    for key, val in evaluated_weights.items():
        if key == bias_name:
            return key, val
    return None, None


def compute_number_of_parameters(evaluated_weights):
    total = 0
    for key, val in evaluated_weights.items():
        total += np.cumprod(val.shape)[-1]
    return total


def _strip_empty_weights_conv_with_biases(weights, next_weights, bias, mask):

    weights_T = np.transpose(weights, axes=(3, 2, 1, 0))
    next_weights_T = np.transpose(next_weights, axes=(2, 0, 1, 3))

    repacked_weights_T = []
    next_repacked_weights_T = []
    repacked_biases = []

    for j in range(len(mask)):
        if mask[j] != 0.0:
            repacked_weights_T.append(weights_T[j])
            next_repacked_weights_T.append(next_weights_T[j])
            repacked_biases.append(bias[j])

    repacked_weights = np.transpose(repacked_weights_T, axes=(3, 2, 1, 0))
    next_repacked_weights = np.transpose(next_repacked_weights_T, axes=(1, 2, 0, 3))
    repacked_biases = np.asarray(repacked_biases)

    return repacked_weights, next_repacked_weights, repacked_biases


def _strip_empty_weights_conv(masked_weights, next_masked_weights, mask):

    masked_weights_T = np.transpose(masked_weights, axes=(3, 2, 1, 0))
    next_masked_weights_T = np.transpose(next_masked_weights, axes=(2, 0, 1, 3))

    repacked_weights_T = []
    next_repacked_weights_T = []

    for j in range(len(mask)):
        if mask[j] != 0.0:
            repacked_weights_T.append(masked_weights_T[j])
            next_repacked_weights_T.append(next_masked_weights_T[j])

    repacked_weights = np.transpose(repacked_weights_T, axes=(3, 2, 1, 0))
    next_repacked_weights = np.transpose(next_repacked_weights_T, axes=(1, 2, 0, 3))

    return repacked_weights, next_repacked_weights


def _strip_empty_weights_dense_with_biases(masked_weights, next_masked_weights, bias, mask):

    masked_weights_T = np.transpose(masked_weights, axes=(1, 0))

    repacked_weights_T = []
    next_repacked_weights = []
    repacked_biases = []

    for j in range(len(mask)):
        if mask[j] != 0.0:
            repacked_weights_T.append(masked_weights_T[j])
            next_repacked_weights.append(next_masked_weights[j])
            repacked_biases.append(bias[j])

    repacked_weights = np.transpose(repacked_weights_T, axes=(1, 0))
    next_repacked_weights = np.asarray(next_repacked_weights)
    repacked_biases = np.asarray(repacked_biases)

    return repacked_weights, next_repacked_weights, repacked_biases


def _strip_empty_weights_dense(masked_weights, next_masked_weights, mask):

    masked_weights_T = np.transpose(masked_weights, axes=(1, 0))

    repacked_weights_T = []
    next_repacked_weights = []

    for j in range(len(mask)):
        if mask[j] != 0.0:
            repacked_weights_T.append(masked_weights_T[j])
            next_repacked_weights.append(next_masked_weights[j])

    repacked_weights = np.transpose(repacked_weights_T, axes=(1, 0))
    next_repacked_weights = np.asarray(next_repacked_weights)

    return repacked_weights, next_repacked_weights


def _strip_empty_weights_conv_to_dense_with_biases(weights, next_weights, bias, mask):

    weights_T = np.transpose(weights, axes=(3, 2, 1, 0))

    repacked_weights_T = []
    next_repacked_weights = []
    repacked_biases = []

    for j in range(len(mask)):
        if mask[j] != 0.0:
            repacked_weights_T.append(weights_T[j])
            repacked_biases.append(bias[j])

    for row, m in zip(next_weights, cycle(mask)):
        if m:
            next_repacked_weights.append(row)

    repacked_weights = np.transpose(repacked_weights_T, axes=(3, 2, 1, 0))
    next_repacked_weights = np.asarray(next_repacked_weights)
    repacked_biases = np.asarray(repacked_biases)

    return repacked_weights, next_repacked_weights, repacked_biases


def _strip_empty_weights_conv_to_dense(weights, next_weights, mask):

    weights_T = np.transpose(weights, axes=(3, 2, 1, 0))

    repacked_weights_T = []
    next_repacked_weights = []

    for j in range(len(mask)):
        if mask[j] != 0.0:
            repacked_weights_T.append(weights_T[j])

    for row, m in zip(next_weights, cycle(mask)):
        if m:
            next_repacked_weights.append(row)

    repacked_weights = np.transpose(repacked_weights_T, axes=(3, 2, 1, 0))
    next_repacked_weights = np.asarray(next_repacked_weights)

    return repacked_weights, next_repacked_weights


def strip_all_empty_weights(trainable_variables, masks, layer_order, debug=False):
    for i in range(len(layer_order)-1):

        layer_name = layer_order[i]
        next_layer_name = layer_order[i+1]

        mask_name = layer_name + "/" + MASK_NAME
        weight_name = layer_name + "/" + WEIGHT_NAME
        next_weight_name = next_layer_name + "/" + WEIGHT_NAME

        if debug:
            print("strip_all_empty_weights: picked layers")
            print(" ", weight_name)
            print(" ", next_weight_name)

        weights = trainable_variables[weight_name]
        next_weights = trainable_variables[next_weight_name]

        bias_name, bias = _get_bias_from_weight_path(trainable_variables, weight_name)
        mask = masks[mask_name]
        if len(weights.shape) == 4 and len(next_weights.shape) == 4:
            # conv -> conv
            if bias is None:
                repacked_weights, next_repacked_weights = \
                    _strip_empty_weights_conv(weights, next_weights, mask)
            else:
                repacked_weights, next_repacked_weights, repacked_bias = \
                    _strip_empty_weights_conv_with_biases(weights, next_weights, bias, mask)
                trainable_variables[bias_name] = repacked_bias
        elif len(weights.shape) == 2 and len(next_weights.shape) == 2:
            # dense -> dense
            if bias is None:
                repacked_weights, next_repacked_weights = \
                    _strip_empty_weights_dense(weights, next_weights, mask)
            else:
                repacked_weights, next_repacked_weights, repacked_bias = \
                    _strip_empty_weights_dense_with_biases(weights, next_weights, bias, mask)
                trainable_variables[bias_name] = repacked_bias
        elif len(weights.shape) == 4 and len(next_weights.shape) == 2:
            # conv -> dense
            if bias is None:
                repacked_weights, next_repacked_weights = \
                    _strip_empty_weights_conv_to_dense(weights, next_weights, mask)
            else:
                repacked_weights, next_repacked_weights, repacked_bias = \
                    _strip_empty_weights_conv_to_dense_with_biases(weights, next_weights, bias, mask)
                trainable_variables[bias_name] = repacked_bias
        else:
            raise ValueError("Unsupported weights shapes: {} -> {}".format(
                weights.shape, next_weights.shape))

        trainable_variables[weight_name] = repacked_weights
        trainable_variables[next_weight_name] = next_repacked_weights

    return trainable_variables


def repack_graph(graph, layer_order, random_drop=0.0, debug=False):
    evaluated_trainable_variables = eval_weights_from_graph(graph, collection_name="trainable_variables", debug=debug)
    masks = eval_weights_from_graph(graph, collection_name=MASKS_COLLECTION, debug=debug)

    if random_drop:
        for key in masks.keys():
            elements_num = len(masks[key])
            elements_to_drop = int(random_drop*elements_num)
            if elements_to_drop > 0:
                idxes = list(range(elements_num))
                random.shuffle(idxes)
                masks[key][idxes[:elements_to_drop]] = 0.0

    initial_parameters_num = compute_number_of_parameters(evaluated_trainable_variables)

    if initial_parameters_num == 0:
        raise ValueError("No weights to repack")

    evaluated_trainable_variables = strip_all_empty_weights(
        evaluated_trainable_variables, masks, layer_order, debug)

    repacked_parameters_num = compute_number_of_parameters(evaluated_trainable_variables)

    if debug:
        print("initial_parameters_num: ", initial_parameters_num)
        print("repacked_parameters_num: ", repacked_parameters_num)

    print("Finished repacking, compression: ",
          100*(1.0 - float(repacked_parameters_num)/initial_parameters_num), " percent")

    return evaluated_trainable_variables
