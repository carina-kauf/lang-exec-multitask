import logging
import torch
import numpy as np

from dataloader_cog_tasks import build_cognitive_dataset
from dataloader_lang_tasks import get_batch

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
import re

import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

_logger = logging.getLogger(__name__)


def get_activity(args, model, task, tasks_env_dict, TRAINING_TASK_SPECS, device=None, num_trial=1000):
    """Get activity of equal-length trials
    Returns:
        activity: (num_trial, num_time_step, num_neuron)
        trial_list: list of trial index
    """
    activity_list, trial_list = list(), list()
    eval_env = tasks_env_dict[task]["env"] # multitask env for collections, single task env for contrib/lang

    if tasks_env_dict[task]["dataset"] == "lang":
        if not args.CTRNN:
            hidden = model.init_hidden(args.eval_batch_size)
        val_data = TRAINING_TASK_SPECS[task]["val_data"]
        num_trial = int(num_trial / args.eval_batch_size)  # TODO check: added since each batch has args.eval_batch_size trials
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
            if 'contrib' not in task:
                eval_env.set_i(tasks_env_dict[task]["index"])
            eval_env.new_trial()
            ob = eval_env.ob
            if 'contrib' not in task:
                ob = ob[:, np.newaxis, :]  # Add batch axis
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
            else:
                ob = ob[:, np.newaxis]
                inputs = torch.from_numpy(ob).type(torch.long).to(device)

            if args.CTRNN:
                # can't index with task here, because the modules in the model are defined with the overall task set identifier
                action_pred, activity = model(x=inputs, task=tasks_env_dict[task]["task_set"])
            else:
                action_pred, activity = model(input=inputs, hidden=hidden, task=tasks_env_dict[task]["task_set"])

        activity = activity.detach().cpu().numpy()
        activity_list.append(activity)
        if tasks_env_dict[task]["dataset"] == "cog":
            trial_list.append(eval_env.trial)

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
    tasks_env_dict = {}
    cog_tasks = [elm for elm in TRAINING_TASK_SPECS.keys() if TRAINING_TASK_SPECS[elm]["dataset"] == "cog"]
    lang_tasks = [elm for elm in TRAINING_TASK_SPECS.keys() if TRAINING_TASK_SPECS[elm]["dataset"] == "lang"]

    for task_identifier in TRAINING_TASK_SPECS.keys():
        if task_identifier in cog_tasks:
            cog_multi_env, all_tasks, collections = build_cognitive_dataset(args, task_identifier, return_multienv=True)
            if task_identifier in collections:
                for i, subtask in enumerate(all_tasks):
                    tasks_env_dict[subtask] = {}
                    tasks_env_dict[subtask]["task_set"] = task_identifier
                    tasks_env_dict[subtask]["index"] = i
                    tasks_env_dict[subtask]["dataset"] = "cog"
                    tasks_env_dict[subtask]["env"] = cog_multi_env
            else:
                tasks_env_dict[task_identifier] = {}
                tasks_env_dict[task_identifier]["task_set"] = task_identifier
                tasks_env_dict[task_identifier]["dataset"] = "cog"
                tasks_env_dict[task_identifier]["env"] = cog_multi_env

        elif task_identifier in lang_tasks:
            tasks_env_dict[task_identifier] = {}
            tasks_env_dict[task_identifier]["task_set"] = task_identifier
            tasks_env_dict[task_identifier]["env"] = task_identifier
            tasks_env_dict[task_identifier]["dataset"] = "lang"

        else:
            raise ValueError(f"Task {task_identifier} not found in TRAINING_TASK_SPECS")

    task_variance_list = list()
    activity_dict = {}  # recording activity
    task_variance_dict = {}
    print(f"Getting normalized task variance for {len(tasks_env_dict.keys())} tasks...")
    for i, task in enumerate(tasks_env_dict.keys()):
        print(f"{i+1} | {task}")
        activity, trial_list = get_activity(args=args, model=model, task=task,
                                            tasks_env_dict=tasks_env_dict,
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


def cluster_plot(args, norm_task_variance, tasks, epoch, save_dir):
    """Plot clustering of hidden units based on task preference
    Args:
        norm_task_variance (np.ndarray): normalized task variance of hidden units
            - shape: (n_task, n_units), e.g., (10, 300)
        tasks (list): list of task names
    """
    X = norm_task_variance.T
    silhouette_scores = list()
    n_clusters = np.arange(2, args.max_cluster_nr + 1)

    for n in n_clusters:
        cluster_model = AgglomerativeClustering(n_clusters=n)
        labels = cluster_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for clustering')
    plt.close()


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
    plt.title('Units', fontsize=fs, y=1)
    plt.xlabel('\n\nClusters', fontsize=fs, labelpad=labelpad)
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
        ax.plot(ind_l, [0, 0], linewidth=4, solid_capstyle='butt',
                color=color)
        ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=fs,
                ha='center', va='top', color=color)
    ax.set_xlim([0, len(labels)])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    plt.savefig(f"{save_dir}/epoch={epoch}_cluster_plot.png", bbox_inches="tight", dpi=180)
    plt.show()
    plt.close()
    return sorted_norm_task_variance, silhouette_scores


def plot_silhouette_heatmap(args, silhouette_scores_per_epoch, epoch, save_dir):
    """Plots a heatmap of the silhouette score per number of predefined clusters per epoch

    Args:
        args: Arguments
        silhouette_scores_per_epoch: List of lists, shape [nr_epochs,nr_predefined_cluster]

    Returns:
        Plots heatmap
    """
    MAX_NUMBER = args.max_cluster_nr
    # specifying column names
    nr_epochs = np.arange(len(silhouette_scores_per_epoch))
    nr_clusters = np.arange(2, MAX_NUMBER + 1)

    # Create the pandas DataFrame
    plot_df = pd.DataFrame(silhouette_scores_per_epoch)

    plot_df.columns = nr_clusters
    plot_df.index = nr_epochs
    plot_df = plot_df.T #since we want epochs to be on the x-axis

    # ax = sns.heatmap(df)
    plt.figure()
    ax = sns.heatmap(plot_df, annot=True, cmap="coolwarm") #viridis
    ax.set(xlabel="Epoch number", ylabel="Nr. of clusters", title=f"Silhouette scores | Epoch {nr_epochs[-1]}")
    ax.collections[0].colorbar.set_label("Silhouette score")
    plt.savefig(f"{save_dir}/epoch={epoch}_silhouette_heatmap.png", bbox_inches="tight", dpi=180)
    plt.show()
    plt.close()


def plot_task_similarity(norm_task_variance, tasks, epoch, save_dir):
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
    plt.title('Task Similarity', fontsize=fs*1.5)

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=fs, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.savefig(f"{save_dir}/epoch={epoch}_task_similarity.png", bbox_inches="tight", dpi=180)
    plt.show()
    plt.close()


def plot_feature_similarity(sorted_norm_task_variance, epoch, save_dir):
    """Plot feature similarity matrix
    Args:
        norm_task_variance (np.ndarray): normalized task variance of hidden units
            - shape: (n_task, n_units), e.g., (10, 300)
    """
    X = sorted_norm_task_variance.T
    similarity = cosine_similarity(X)  # TODO: add other metric
    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    # ax.axis('off')
    # add axis labels
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Units', fontsize=fs, labelpad=labelpad)
    plt.ylabel('Units', fontsize=fs, labelpad=labelpad)
    plt.title('Network unit similarity', fontsize=fs * 1.5)

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=fs, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.savefig(f"{save_dir}/epoch={epoch}_feature_similarity.png", bbox_inches="tight", dpi=180)
    plt.show()
    plt.close()


def sorted_task_variance(task_variance_dict, env_id1, env_id2, f2_ax=None):
    """Task-variance analysis of units, line plot, units ordered by env_id1 task variance"""
    ind_sort = np.argsort(task_variance_dict[env_id1])
    plot_ids = [env_id1, env_id2]
    for id in plot_ids:
        if f2_ax is None:
            plt.figure()
            plt.plot(task_variance_dict[id][ind_sort], label=id)
            plt.legend()
            plt.title(f"Task variance of RNN units | Ordered by {env_id1} task variance")
            plt.show()
            plt.close()
        else:
            f2_ax.plot(task_variance_dict[id][ind_sort], label=id)
            f2_ax.legend()
            f2_ax.set_title(f"RNN units ordered by {env_id1} task variance")


def scatter_task_variance(task_variance_dict, env_id1, env_id2, f2_ax=None):
    """Task-variance analysis, scatter plot"""
    if f2_ax is None:
        plt.figure()
        plt.scatter(task_variance_dict[env_id1], task_variance_dict[env_id2])
        plt.xlabel(f"{env_id1}")
        plt.ylabel(f"{env_id2}")
        plt.title(f"{env_id1} vs. {env_id2}")
        plt.show()
        plt.close()
    else:
        f2_ax.scatter(task_variance_dict[env_id1], task_variance_dict[env_id2])
        f2_ax.xaxis.set_label_text(f"{env_id1}")
        f2_ax.yaxis.set_label_text(f"{env_id2}")
        f2_ax.set_title(f"Task variance of RNN units | {env_id1} vs. {env_id2}")


def frac_variance(task_variance_dict, env_id1, env_id2, f2_ax=None):
    """Fractional variance analysis, histogram plot"""
    print(f"{env_id1} | {env_id2}")
    diff_variance = (task_variance_dict[env_id1] - task_variance_dict[env_id2])
    sum_variance = task_variance_dict[env_id1] + task_variance_dict[env_id2]
    frac_variance = diff_variance / sum_variance

    if f2_ax is None:
        plt.figure()
        plt.hist(frac_variance, bins=100)
        plt.xlabel(f'{env_id2} < -- > {env_id1}')
        plt.xlim([-1, 1])
        plt.title(f"Frac variance of RNN units | {env_id1} vs. {env_id2}")
        plt.show()
        plt.close()
    else:
        f2_ax.hist(frac_variance, bins=100)
        f2_ax.xaxis.set_label_text(f'{env_id2} < -- > {env_id1}')
        f2_ax.set_xlim([-1, 1])
        f2_ax.set_title(f"Frac variance of RNN units | {env_id1} vs. {env_id2}")


def gridplot(task_variance_dict, env_id1, env_ids, epoch, save_dir):
    """Merging plot types"""

    fig2 = plt.figure(constrained_layout=True, figsize=(15, 3 * len(env_ids)))
    spec2 = GridSpec(ncols=3, nrows=len(env_ids), figure=fig2)
    f2_ax = []

    for i in range(len(env_ids)):
        # plotting
        j = 0
        f2_ax.append(fig2.add_subplot(spec2[i, j]))
        sorted_task_variance(task_variance_dict, env_id1, env_ids[i], f2_ax[-1])
        #
        j = 1
        f2_ax.append(fig2.add_subplot(spec2[i, j]))
        scatter_task_variance(task_variance_dict, env_id1, env_ids[i], f2_ax[-1])
        #
        j = 2
        f2_ax.append(fig2.add_subplot(spec2[i, j]))
        frac_variance(task_variance_dict, env_id1, env_ids[i], f2_ax[-1])
        print("*" * 30)
    fig2.suptitle(f"Unit variance analysis | Epoch {epoch}")
    plt.savefig(f'{save_dir}/epoch={epoch}_gridplot.png', bbox_inches='tight', dpi=280)
    plt.show()
    plt.close()


def main(args, TRAINING_TASK_SPECS, model, device, silhouette_scores_per_epoch, epoch, save_dir):

    model.eval()
    task_variance_dict, norm_task_variance, activity_dict = \
        get_normalized_task_variance(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, model=model, device=device)
    tasks = list(task_variance_dict.keys())
    sorted_norm_task_variance, silhouette_scores = cluster_plot(args=args, norm_task_variance=norm_task_variance, tasks=tasks,
                                                                epoch=epoch, save_dir=save_dir)
    silhouette_scores_per_epoch.append(silhouette_scores)
    plot_task_similarity(norm_task_variance, tasks, epoch=epoch, save_dir=save_dir)
    plot_feature_similarity(sorted_norm_task_variance, epoch=epoch, save_dir=save_dir)
    plot_silhouette_heatmap(args=args, silhouette_scores_per_epoch=silhouette_scores_per_epoch,
                            epoch=epoch, save_dir=save_dir)

    language_tasks = [elm for elm in TRAINING_TASK_SPECS if TRAINING_TASK_SPECS[elm]['dataset'] == 'lang']
    if len(language_tasks) > 0:
        env_id1 = language_tasks[0]
    else:
        env_id1 = tasks[0]
    if len(args.tasks) > 1:
        env_ids = language_tasks + [[elm for elm in tasks if elm not in language_tasks][0]] + \
                 [[elm for elm in tasks if elm not in language_tasks][-1]]  #add 2 non-language tasks
        gridplot(task_variance_dict=task_variance_dict, env_id1=env_id1, env_ids=env_ids, epoch=epoch, save_dir=save_dir)

    return silhouette_scores_per_epoch
