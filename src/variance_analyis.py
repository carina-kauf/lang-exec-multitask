import logging
import torch
import numpy as np

from utils_general import set_seed, repackage_hidden
from dataloader_lang_tasks import get_batch

import gym
import neurogym as ngym

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from neurogym.wrappers.block import MultiEnvs
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
import seaborn as sns
import pickle

import re

_logger = logging.getLogger(__name__)


def get_activity(args, model, cog_multi_env, task, tasks_env_dict, TRAINING_TASK_SPECS, device=None, num_trial=1000):
    """Get activity of equal-length trials
    Returns:
        activity: (num_trial, num_time_step, num_neuron)
        trial_list: list of trial index
    """
    activity_list, trial_list = list(), list()

    if tasks_env_dict[task]["dataset"] == "lang":
        if not args.CTRNN:
            hidden = model.init_hidden(10)  # eval batch size
        val_data = TRAINING_TASK_SPECS[task]["val_data"]
        num_trial = int(num_trial / 10)  # TODO check: added since each batch has 10 trials
    else:
        if not args.CTRNN:
            hidden = model.init_hidden(1)

    for i in range(num_trial):
        if tasks_env_dict[task]["dataset"] == "lang":
            inputs, targets = get_batch(val_data, i, args.bptt)
            if args.CTRNN:
                action_pred, activity = model(x=inputs, task=task)
            else:
                action_pred, activity, hidden = model(input=inputs, hidden=hidden, task=task)
        else:
            cog_multi_env.set_i(tasks_env_dict[task]["index"])
            cog_multi_env.new_trial()
            ob = cog_multi_env.ob
            ob = ob[:, np.newaxis, :]  # Add batch axis
            inputs = torch.from_numpy(ob).type(torch.float).to(device)
            if args.CTRNN:
                # can't index with task here, because the modules in the model are defined with the overall task set identifier
                action_pred, activity = model(x=inputs, task=tasks_env_dict[task]["task_set"])
            else:
                action_pred, activity = model(input=inputs, hidden=hidden, task=tasks_env_dict[task]["task_set"])

        activity = activity.detach().cpu().numpy()
        trial_list.append(cog_multi_env.trial)
        activity_list.append(activity)

    activity = np.concatenate(activity_list, axis=1)
    return activity, trial_list


def get_normalized_task_variance(args, TRAINING_TASK_SPECS, model, device=None):
    """Get normalized total variation of activity
    Returns:
        normalized_task_variance (float): normalized total variation of activity of RNN hidden units
            - shape: (n_units,), e.g., (300,)
            - unit variance normalized by max variance across tasks
        task_variance_dict (dict): dict of unnormalized unit task variance (key: task name, value: variance)
            - shape: (n_task, n_units), e.g., (10, 300)
        activity_dict (dict): dict of activity (key: task name, value: activity) [raw unit activity]
            - shape: (n_task, n_trial, n_time_step, n_units), e.g., (10, 1000, 100, 300)
    """
    cog_tasks = []
    for task in args.tasks:
        if TRAINING_TASK_SPECS[task]["dataset"] == "cog":
            all_subtasks = TRAINING_TASK_SPECS[task]["full_task_list"]
            task_set = [task] * len(all_subtasks)
            zipped = zip(all_subtasks, task_set)
            cog_tasks.extend(zipped)
    only_tasks = [subtask for (subtask, task_set) in cog_tasks]

    cog_multi_env = None
    tasks_env_dict = {}
    # individual env specifications for all collection tasks
    if len(cog_tasks) > 0:
        timing = {'fixation': ('constant', 500)}
        kwargs = {'dt': args.dt, 'timing': timing}
        envs = [gym.make(task, **kwargs) for task in only_tasks]
        cog_multi_env = MultiEnvs(envs, env_input=True) #todo when only training on one task, env_input is not necessary
        for i, (subtask, task_set) in enumerate(cog_tasks):
            tasks_env_dict[subtask] = {}
            tasks_env_dict[subtask]["task_set"] = task_set
            tasks_env_dict[subtask]["index"] = i
            tasks_env_dict[subtask]["dataset"] = "cog"
    for task in TRAINING_TASK_SPECS.keys():
        if TRAINING_TASK_SPECS[task]["dataset"] == "lang":
            tasks_env_dict[task] = {}
            tasks_env_dict[task]["task_set"] = task
            tasks_env_dict[task]["env"] = task
            tasks_env_dict[task]["dataset"] = "lang"

    task_variance_list = list()
    activity_dict = {}  # recording activity
    task_variance_dict = {}
    print(f"Getting normalized task variance for {len(tasks_env_dict.keys())} tasks...")
    for i, task in enumerate(tasks_env_dict.keys()):
        print(f"{i+1} | {task}")
        activity, trial_list = get_activity(args=args, model=model, cog_multi_env=cog_multi_env, task=task, tasks_env_dict=tasks_env_dict,
                                            TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, device=device, num_trial=500)
        activity_dict[i] = activity
        # Compute task variance across trials
        task_variance = np.var(activity, axis=1).mean(axis=0)
        task_variance_list.append(task_variance)
        task_variance_dict[task] = task_variance
    task_variance = np.array(task_variance_list)  # (n_task, n_units)
    thres = 1e-6
    # Filters out units that have a task_variance across all tasks with a sum less than or equal to the threshold thres.
    # The resulting task_variance array has the same number of rows as the original array, but a reduced number of columns.
    task_variance = task_variance[:, task_variance.sum(axis=0) > thres]
    # normalize task variance by max variance across tasks
    norm_task_variance = task_variance / np.max(task_variance, axis=0)

    return task_variance_dict, norm_task_variance, activity_dict


def figure_settings():
    figsize = (12, 8)
    rect = [0.25, 0.2, 0.6, 0.7]
    rect_color = [0.25, 0.15, 0.6, 0.05]
    rect_cb = [0.87, 0.2, 0.03, 0.7]
    fs = 12
    labelpad = 13
    return figsize, rect, rect_color, rect_cb, fs, labelpad


def cluster_plot(norm_task_variance, tasks):
    """Plot clustering of hidden units based on task preference
    Args:
        norm_task_variance (np.ndarray): normalized task variance of hidden units
            - shape: (n_task, n_units), e.g., (10, 300)
        tasks (list): list of task names
    """
    X = norm_task_variance.T
    silhouette_scores = list()
    n_clusters = np.arange(2, 20)
    for n in n_clusters:
        cluster_model = AgglomerativeClustering(n_clusters=n)
        labels = cluster_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')

    # Use the number of clusters that maximizes the silhouette score for clustering
    n_cluster = n_clusters[np.argmax(silhouette_scores)]
    cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
    labels = cluster_model.fit_predict(X)

    # Sort clusters by its task preference (important for consistency across nets)
    label_prefs = [np.argmax(norm_task_variance[:, labels == l].sum(axis=1)) for l in set(labels)]

    ind_label_sort = np.argsort(label_prefs)
    label_prefs = np.array(label_prefs)[ind_label_sort]
    # Relabel
    labels2 = np.zeros_like(labels)
    for i, ind in enumerate(ind_label_sort):
        labels2[labels == ind] = i
    labels = labels2

    # Sort neurons by labels
    ind_sort = np.argsort(labels)
    labels = labels[ind_sort]
    sorted_norm_task_variance = norm_task_variance[:, ind_sort]

    print("Plot Normalized Variance")
    # Plot Normalized Variance
    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    tick_names = [
        task.split(".")[1].split("-v0")[0] if any(x in task for x in ['yang19', 'khonaChandra22', 'contrib']) else task
        for task in tasks]
    tick_names = [re.sub("DelayMatchSample", "DMS-", x) for x in tick_names]

    vmin, vmax = 0, 1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    im = ax.imshow(sorted_norm_task_variance, cmap='magma',
                   aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

    plt.yticks(range(len(tick_names)), tick_names, rotation=0, va='center', fontsize=fs)
    plt.xticks([])
    plt.title('Units', fontsize=fs, y=0.97)
    plt.xlabel('Clusters', fontsize=fs, labelpad=labelpad)
    ax.tick_params('both', length=0)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[vmin, vmax])
    cb.outline.set_linewidth(0.5)
    clabel = 'Normalized Task Variance'

    cb.set_label(clabel, fontsize=fs, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=fs)

    print("Plot color bars indicating clustering")
    # Plot color bars indicating clustering
    cmap = matplotlib.cm.get_cmap('tab10')
    ax = fig.add_axes(rect_color)
    for il, l in enumerate(np.unique(labels)):
        color = cmap(il % 10)
        ind_l = np.where(labels == l)[0][[0, -1]]+np.array([0, 1])
        ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                color=color)
        ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=fs,
                ha='center', va='top', color=color)
    ax.set_xlim([0, len(labels)])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    fig.show()
    return sorted_norm_task_variance


def plot_task_similarity(norm_task_variance, tasks):
    """Plot task similarity matrix
    Args:
        norm_task_variance (np.ndarray): normalized task variance of hidden units
            - shape: (n_task, n_units), e.g., (10, 300)
        tasks (list): list of task names
    """
    similarity = cosine_similarity(norm_task_variance)  # TODO: add other metric

    # fname = f'files/seed={args.seed}_task_similarity.pkl'
    # with open(fname, 'wb') as fout:
    #     pickle.dump(similarity, fout)

    print(np.shape(norm_task_variance), np.shape(similarity))

    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)

    tick_names = [
        task.split(".")[1].split("-v0")[0] if any(x in task for x in ['yang19', 'khonaChandra22', 'contrib']) else task
        for task in tasks]
    tick_names = [re.sub("DelayMatchSample", "DMS-", x) for x in tick_names]
    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks(range(len(tick_names)), tick_names,
               rotation=90, va='top', fontsize=fs)

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=fs, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=fs)

    # plt.savefig(f'XXX_task_similarity.png') #TODO: save
    plt.show()


def plot_feature_similarity(sorted_norm_task_variance):
    """Plot feature similarity matrix
    Args:
        norm_task_variance (np.ndarray): normalized task variance of hidden units
            - shape: (n_task, n_units), e.g., (10, 300)
    """
    print(f"Shape of sorted_norm_task_variance: {np.shape(sorted_norm_task_variance)}")
    X = sorted_norm_task_variance.T
    similarity = cosine_similarity(X)  # TODO: add other metric
    print(np.shape(X), np.shape(similarity))
    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=fs, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    # plt.savefig(f'XXX_feature_similarity.png') #TODO: save
    plt.show()


def main(args, TRAINING_TASK_SPECS, model, device):
    task_variance_dict, norm_task_variance, activity_dict = \
        get_normalized_task_variance(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, model=model, device=device)
    tasks = list(task_variance_dict.keys())
    sorted_norm_task_variance = cluster_plot(norm_task_variance, tasks)
    plot_task_similarity(norm_task_variance, tasks)
    plot_feature_similarity(sorted_norm_task_variance)
