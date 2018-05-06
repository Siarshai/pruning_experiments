from collections import OrderedDict

import numpy as np

from bonesaw.masked_layers import MASKED_WEIGHT_NAME


def eval_weights_from_graph(graph, collection_name, debug=False):
    trainable_variables_list = graph.get_collection(collection_name)
    if debug:
        print("get_weights_from_graph: fetched trainable variables")
    evaluated_weights = OrderedDict()
    for var in trainable_variables_list:
        print(" {}, shape {}".format(var.name, var.shape))
        evaluated_weights[var.name.split(":")[0]] = var.eval()
    return evaluated_weights


def _get_bias_from_masked_weight_path(evaluated_weights, weight_name):
    if not weight_name.endswith("weights/masked_weight"):
        raise AttributeError("Expected .../weights/masked_weight variable")
    origin_name = weight_name[:-len("/weights/masked_weight")]
    bias_name = origin_name + "/bias"
    for key, val in evaluated_weights.items():
        if key == bias_name:
            return key, val
    return None, None


def compute_number_of_parameters(evaluated_weights):
    total = 0
    for key, val in evaluated_weights.items():
        total += np.cumprod(val.shape)[-1]
    return total


def _strip_empty_weights_with_biases(masked_weights, next_masked_weights, biases, debug=False):

    if debug:
        print("strip_empty_weights_with_biases: got weights tensors")
        print(" masked_weights ", masked_weights.shape)
        print(" next_masked_weights ", next_masked_weights.shape)
        print(" biases ", biases.shape)

    masked_weights_T = np.transpose(masked_weights, axes=(3, 2, 1, 0))
    next_masked_weights_T = np.transpose(next_masked_weights, axes=(2, 0, 1, 3))

    repacked_weights_T = []
    next_repacked_weights_T = []
    repacked_biases = []

    for j in range(len(masked_weights_T)):
        if np.max(masked_weights_T[j]) != 0.0 or np.min(masked_weights_T[j]) != 0.0 or biases[j] != 0.0:
            repacked_weights_T.append(masked_weights_T[j])
            next_repacked_weights_T.append(next_masked_weights_T[j])
            repacked_biases.append(biases[j])

    repacked_biases = np.asarray(repacked_biases)
    repacked_weights = np.transpose(repacked_weights_T, axes=(3, 2, 1, 0))
    next_repacked_weights = np.transpose(next_repacked_weights_T, axes=(1, 2, 0, 3))

    if debug:
        print("strip_empty_weights_with_biases: after transformation")
        print(" masked_weights ", repacked_weights.shape)
        print(" next_masked_weights ", next_repacked_weights.shape)
        print(" biases ", repacked_biases.shape)

    return repacked_weights, next_repacked_weights, repacked_biases


def strip_empty_weights(masked_weights, next_masked_weights, debug=False):

    if debug:
        print("strip_empty_weights: got weights tensors")
        print(" masked_weights ", masked_weights.shape)
        print(" next_masked_weights ", next_masked_weights.shape)

    masked_weights_T = np.transpose(masked_weights, axes=(3, 2, 1, 0))
    next_masked_weights_T = np.transpose(next_masked_weights, axes=(2, 0, 1, 3))

    repacked_weights_T = []
    next_repacked_weights_T = []

    for j in range(len(masked_weights_T)):
        if np.max(masked_weights_T[j]) != 0.0 or np.min(masked_weights_T[j]) != 0.0:
            repacked_weights_T.append(masked_weights_T[j])
            next_repacked_weights_T.append(next_masked_weights_T[j])

    repacked_weights = np.transpose(repacked_weights_T, axes=(3, 2, 1, 0))
    next_repacked_weights = np.transpose(next_repacked_weights_T, axes=(1, 2, 0, 3))

    if debug:
        print("strip_empty_dimensions: after transformation")
        print(" masked_weights ", repacked_weights.shape)
        print(" next_masked_weights ", next_repacked_weights.shape)

    return repacked_weights, next_repacked_weights


def strip_all_empty_weights(evaluated_masked_weights, evaluated_trainable_variables, layer_order, debug=False):
    all_repacked_weights = {k: v for k, v in evaluated_masked_weights.items()}
    all_repacked_biases = {}
    for i in range(len(layer_order)):
        if i+1 == len(layer_order):
            continue

        layer_name = layer_order[i]
        next_layer_name = layer_order[i+1]

        weight_name = layer_name + "/" + MASKED_WEIGHT_NAME
        next_weight_name = next_layer_name + "/" + MASKED_WEIGHT_NAME

        if debug:
            print("strip_all_empty_weights: picked layers")
            print(" ", weight_name)
            print(" ", next_weight_name)

        masked_weights = all_repacked_weights[weight_name]
        next_masked_weights = all_repacked_weights[next_weight_name]

        bias_name, bias = _get_bias_from_masked_weight_path(evaluated_trainable_variables, weight_name)
        if bias is None:
            repacked_weights, next_repacked_weights = \
                strip_empty_weights(masked_weights, next_masked_weights, debug)
        else:
            repacked_weights, next_repacked_weights, repacked_bias = \
                _strip_empty_weights_with_biases(masked_weights, next_masked_weights, bias, debug)
            all_repacked_biases[bias_name] = repacked_bias

        all_repacked_weights[weight_name] = repacked_weights
        all_repacked_weights[next_weight_name] = next_repacked_weights

    return all_repacked_weights, all_repacked_biases


def repack_graph(graph, layer_order, debug=False):
    evaluated_masked_weights = eval_weights_from_graph(graph, collection_name="masked_weights")
    evaluated_trainable_variables = eval_weights_from_graph(graph, collection_name="trainable_variables")

    # ignoring biases
    initial_parameters_num = compute_number_of_parameters(evaluated_trainable_variables)

    if initial_parameters_num == 0:
        raise ValueError("No weights to repack")

    all_repacked_weights, all_repacked_biases = strip_all_empty_weights(
        evaluated_masked_weights, evaluated_trainable_variables, layer_order, debug)

    for k, v in all_repacked_biases.items():
        all_repacked_weights[k] = v

    for key in list(all_repacked_weights.keys()):
        if key.endswith("weights/masked_weight"):
            weight_name = key.split("/")[0] + "/kernel"
            all_repacked_weights[weight_name] = all_repacked_weights[key]
            del all_repacked_weights[key]

    for k, v in evaluated_trainable_variables.items():
        if k not in all_repacked_weights:
            all_repacked_weights[k] = v

    repacked_parameters_num = compute_number_of_parameters(all_repacked_weights)

    if debug:
        print("initial_parameters_num: ", initial_parameters_num)
        print("repacked_parameters_num: ", repacked_parameters_num)

    print("Finished repacking, compression: ",
          100*(1.0 - float(repacked_parameters_num)/initial_parameters_num), " percent")

    return all_repacked_weights
