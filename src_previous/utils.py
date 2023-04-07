import re
import torch
import numpy as np
import random
import os

from torchtext.datasets import WikiText2
from torchtext.datasets import WikiText103
from torchtext.datasets import PennTreebank

from data import build_cognitive_dataset, get_splits, build_dataset_lang

import logging
import sys

from scipy import sparse
from scipy.spatial.distance import cdist

_logger = logging.getLogger("data")
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName("DEBUG"))

#FIXME turn tasks into objects with .name .dim . etc


def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if not cuda:
            raise ValueError("WARNING: You have a CUDA device, so you should probably run with --cuda")


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def mask2d(N_x, N_y, cutoff, periodic):
    """Define anatomical mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Create a 2d mask of distances between neurons
    A neuron is connected to all neurons within a certain distance
    Args:
        :param N_x: number of neurons in x direction
        :param N_y: number of neurons in y direction
        :param cutoff: cutoff distance
        :param periodic: if True, use periodic boundary conditions
    Returns:
        :2d mask

    # cutoff = d in paper
    # apply anatomical mask on h2h weights
    # model.rnn.h2h.weight.data = model.rnn.h2h.weight.data*(mask2d)
    """

    # create 2d sheet of coordinates
    x1 = np.linspace(-N_x // 2, N_x // 2 - 1, N_x)
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.linspace(-N_y // 2, N_y // 2 - 1, N_y)
    x2 = np.expand_dims(x2, axis=1)
    x_coordinates = np.expand_dims(np.repeat(x1, N_y, axis=0).reshape(N_x, N_y).transpose().flatten(), axis=1)
    y_coordinates = np.expand_dims(np.repeat(x2, N_x, axis=0).reshape(N_x, N_y).flatten(), axis=1)

    # calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates, x_coordinates)
    distances_y = cdist(y_coordinates, y_coordinates)

    if periodic:
        distances_y = np.minimum(N_y - distances_y, distances_y)
        distances_x = np.minimum(N_x - distances_x, distances_x)

    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N_y, N_x, N_y, N_x)
    dist = dist.reshape(N_x * N_y, N_x * N_y)
    assert dist.shape == (N_x * N_y, N_x * N_y) #added by me
    # mask connections based on distance
    dist[dist < cutoff] = 1
    dist[dist > cutoff - 1] = 0
    return dist


def sparsemask2d(N_x, N_y, sparsity):
    """
    Define Erdos-Renyi sparse mask for h2h weights in CTRNN
    Source: https://github.com/mikailkhona/Multi-Task-learning/blob/main/Multitask_with_mask.ipynb
    Args:
        :param N_x: number of neurons in x direction
        :param N_y: number of neurons in y direction
        :param sparsity: sparsity of the mask
    Returns:
        :2d mask

    # sparsity for d=2 is 0.03228759765625
    # sparsity for d=3 is 0.0836
    # sparsity for d=4 is 0.1423
    """
    elements = np.random.uniform(0, 1, (N_x, N_y))
    mask = (elements < sparsity).astype(int)
    return mask


def find_matches(d, item):
    """ Find matches for item in dictionary keys
    Used in function: get_dataset_names
    Args:
        d: dictionary
        item: item to be matched against keys

    Returns:
        dataset value for matched regex
    """
    for k in d:
        if re.match(k, item):
            return d[k]


def get_dataset_names(training_tasks):
    """
    Args:
        training_tasks: list of training tasks, specified via flags.

    Returns:
        - list of names of the dataset assignment tuples, e.g.: [('wikitext', 'lang')]
        - list of names of the datasets, e.g. ["lang"]
        - tuple of:
            * dictionary from task to dataset assignment
            * dictionary from dataset assignment to task
    """

    task2dataset = {
        "wikitext(103)?|penntreebank|[a-z]{2}_wiki|pennchar(_perturbed)?": "lang",
        "yang19.*|contrib.*|khonaChandra22.*": "cog"
    }

    dataset2task = {
        "lang": "wikitext(103)?|penntreebank|[a-z]{2}_wiki|pennchar(_perturbed)?",
        "cog": "yang19.*|contrib.*|khonaChandra22.*"
    }

    dataset_assignments = [(x, find_matches(task2dataset, x)) for x in training_tasks]
    dataset_assignments = [x for x in dataset_assignments if x[1] is not None]
    dataset_names = list(set([x[1] for x in dataset_assignments]))

    return dataset_assignments, dataset_names, (task2dataset, dataset2task)


def build_training_tasks(training_tasks, args, seq_len=100):
    """
    Function that defines the task environments, gets dataset splits for langauge tasks, builds gym env for cog tasks
    Args:
        training_tasks: list of training tasks, specified via flags.
        args

    Returns:
        >> Output is dependent on which datasets {cog|lang|cog and lang} we're running on:
        - TASK2DIM: Dictionary from task (specified as f"{dataset}_{task}_mode={i}") to network input size & output size
        - TASK2MODE: Dictionary from task (specified as f"{dataset}_{task}_mode={i}") to mode i
        - MODE2TASK: Dictionary from mode i to task (specified as f"{dataset}_{task}_mode={i}")
        - dataset_assignments: e.g., [('wikitext', 'lang')]
        - lang_data_dict: Dictionary from task (specified as f"{dataset}_{task}_mode={i}") to dataset splits, vocab & vocab dimensions
        - cog_dataset_dict: Dictionary from task (specified as f"{dataset}_{task}_mode={i}") to cog dataset, environment
    """

    dataset_assignments, dataset_names, _ = get_dataset_names(training_tasks)
    print(f"dataset_assignments: {dataset_assignments}")

    TASK2DIM = {}  # dictionary used to build model
    MODE2TASK = {}  # dictionary used to identify current encoder/decoder
    TASK2MODE = {}

    TASK2DATASET = {
        "wikitext": WikiText2,
        "wikitext103": WikiText103,
        "penntreebank": PennTreebank
    }


    lang_data_dict = {}  # dictionary to map to correct vocab, training data, val data, test data
    cog_dataset_dict = {}

    for i, (task, dataset) in enumerate(dataset_assignments):
        print(f"*************\nBUILDING DATASET FOR TASK: {task}\n*************\n")
        if dataset == "lang":
            if not re.match("[a-z]{2}_wiki|pennchar", task):
                vocab, vocab_size, train_data, val_data, test_data = get_splits(TASK2DATASET[task], args.batch_size)
            else:
                BASE_DIR = os.path.abspath(os.path.join(__file__, '../..'))
                pathname = os.path.join(BASE_DIR, f"data/{task}/")
                vocab, vocab_size, train_data, val_data, test_data = build_dataset_lang(pathname, args)

            TASK2DIM[f"lang_{task}_mode={i}"] = {"input_size": vocab_size, "output_size": vocab_size}
            MODE2TASK[i] = f"lang_{task}_mode={i}"
            TASK2MODE[f"lang_{task}_mode={i}"] = i

            lang_data_dict[f"lang_{task}_mode={i}"] = {
                "vocab": vocab,
                "vocab_size": vocab_size,
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data}

        else:
            curr_training_tasks = [x for x in training_tasks if x.startswith(task)]
            dataset_cog, env, ob_size, act_size = build_cognitive_dataset(curr_training_tasks, args.batch_size, seq_len)
            if ob_size == 1: #TODO ADDED for verbal WM tasks
                ob_size = env.vocab_size
            TASK2DIM[f"cog_{task}_mode={i}"] = {"input_size": ob_size, "output_size": act_size}
            MODE2TASK[i] = f"cog_{task}_mode={i}"
            TASK2MODE[f"cog_{task}_mode={i}"] = i

            cog_dataset_dict[f"cog_{task}_mode={i}"] = {
                "dataset_cog": dataset_cog,
                "env": env,
                "ob_size": ob_size,
                "act_size": act_size}

    print("*" * 30)
    print(f"TASK2DIM: {TASK2DIM}")
    print(f"TASK2MODE: {TASK2MODE}")
    print(f"MODE2TASK: {MODE2TASK}")

    print("*" * 30)
    print("*Language data*")
    for key, value in lang_data_dict.items():
        print(
            f"{key} | vocab_size : {value['vocab_size']} | train_data.shape : {value['train_data'].shape} |"
            f" val_data.shape : {value['val_data'].shape} | test_data.shape : {value['test_data'].shape}")
            # train_data shape[1] = batch_size, other_data shape[1] = eval_batch_size

    print("*Cognitive function data*")
    for key, value in cog_dataset_dict.items():
        print(key)
    if all(x in dataset_names for x in ["lang", "cog"]):
        print("Running on language and cognitive tasks")
        return TASK2DIM, TASK2MODE, MODE2TASK, dataset_assignments, lang_data_dict, cog_dataset_dict
    elif dataset_names == ["lang"]:
        print("Running ONLY on language tasks")
        return TASK2DIM, TASK2MODE, MODE2TASK, dataset_assignments, lang_data_dict, None
    elif dataset_names == ["cog"]:
        print("Running ONLY on cognitive tasks")
        return TASK2DIM, TASK2MODE, MODE2TASK, dataset_assignments, None, cog_dataset_dict
    else:
        raise NotImplementedError("Unknown pair of tasks")
    print("*" * 30)