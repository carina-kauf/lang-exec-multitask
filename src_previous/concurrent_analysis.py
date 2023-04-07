import logging
import torch
import numpy as np

from utils import set_seed, get_dataset_names, repackage_hidden
from data import get_batch

import gym
from neurogym.wrappers.block import MultiEnvs
import neurogym as ngym

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
import seaborn as sns
import pickle

import re

from torch.utils.tensorboard import SummaryWriter

#from recurrent_unit_analysis import *

_logger = logging.getLogger(__name__)


def get_performance(net, curr_task_env, cog_mode, CTRNN=False, num_trial=1000, device='cpu', yang=True):
    """
    Get per-task performance of model on cognitive tasks
    source: https://github.com/neurogym/ngym_usage/blob/master/yang19/models.py#L108
    """
    if not CTRNN:
        hidden = net.init_hidden(1)
    perf = 0

    current_task_name = curr_task_env.spec.id
    #print(current_task_name)

    net.eval()
    with torch.no_grad(): #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/38
        for i in range(num_trial):
            curr_task_env.new_trial()
            ob, gt = curr_task_env.ob, curr_task_env.gt
            if yang:
                ob = ob[:, np.newaxis, :]
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
            else:
                ob = ob[:, np.newaxis]
                inputs = torch.from_numpy(ob).type(torch.long).to(device)

            if CTRNN:
                action_pred, rnn_activity = net(inputs, cog_mode)
            else:
                action_pred, hidden, rnn_activity = net(inputs, hidden, cog_mode)

            action_pred = action_pred.detach().cpu().numpy()
            action_pred = np.argmax(action_pred, axis=-1)

            #Note: for Yang19, action_pred is a list of lists, thus the double index. Get list elm of last list element
            # [[ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [11], [11], [ 0], [ 0]]
            perf += gt[-1] == action_pred[-1]

    perf /= num_trial
    return perf


def accuracy_cog(key, dataset, model, cog_mode, device, CTRNN=False, yang=True):
    """
    Get average accuracy on cognitive tasks
    """
    # Reset environment
    env = dataset.env
    env.reset(no_step=True)

    # Initialize variables for logging
    activity_dict = {}  # recording activity
    trial_infos = {}  # recording trial information

    num_trial = 1000
    if not CTRNN:
        hidden = model.init_hidden(1) #TODO check correct? > added since we're evaluating on one trial each per num_trials

    assert not model.training #assert that model is in eval mode

    with torch.no_grad():
        for i in range(num_trial):
            # Neurogym boilerplate
            # Sample a new trial
            trial_info = env.new_trial()
            # Observation and ground-truth of this trial
            ob, gt = env.ob, env.gt
            # Convert to numpy, add batch dimension to input
            if yang:
                inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            else:
                inputs = torch.from_numpy(ob[:, np.newaxis]).type(torch.long).to(device)

            # Run the model for one trial
            # inputs (SeqLen, Batch, InputSize)
            # action_pred (SeqLen, Batch, OutputSize)
            if CTRNN:
                action_pred, rnn_activity = model(inputs, cog_mode)
            else:
                action_pred, hidden, rnn_activity = model(inputs, hidden, cog_mode)
            # print(action_pred.shape, rnn_activity.shape)

            # Compute performance
            # First convert back to numpy
            action_pred = action_pred.cpu().detach().numpy()
            # Read out final choice at last time step
            choice = np.argmax(action_pred[-1, :]) #np.argmax returns index of highest value
            # Compare to ground truth
            correct = choice == gt[-1]

            # Record activity, trial information, choice, correctness
            rnn_activity = rnn_activity[:, 0, :].cpu().detach().numpy()
            activity_dict[i] = rnn_activity
            trial_infos[i] = trial_info  # trial_info is a dictionary
            trial_infos[i].update({'correct': correct})
            if env.spec:
                trial_infos[i].update({'spec.id': env.spec.id})

    # Print information for sample trials
    print(f"*****{key}******")
    for i in range(5):
        print('Trial', i, trial_infos[i], flush=True)

    average_acc = np.mean([val['correct'] for val in trial_infos.values()])
    print(f'Average performance after {num_trial} trials: {average_acc}\n', flush=True)
    return average_acc


def get_activity(model, env, current_task, TASK2MODE, CTRNN=False, lang_data_dict=None, device=None, bptt=None, num_trial=1000):
    """Get activity of equal-length trials"""

    trial_list = list()
    activity_list = list()

    if not CTRNN:
        if "lang" in current_task:
            hidden = model.init_hidden(10) #eval batch size
        else:
            hidden = model.init_hidden(1)

    with torch.no_grad():
        if "lang" in current_task:
            current_num_trial = int(num_trial/10) #TODO check: added since each batch has 10 trials
        else:
            current_num_trial = num_trial
        for i in range(current_num_trial):
            if "lang" in current_task:
                inputs, targets = get_batch(lang_data_dict[current_task]["test_data"], i, bptt) #FTODO check val_data or test_data?
                mode = TASK2MODE[current_task]

                if CTRNN:
                    action_pred, activity = model(inputs, mode=mode)
                else:
                    action_pred, hidden, activity = model(inputs, hidden, mode=mode)
                activity = activity.cpu().detach().numpy()

                trial_list.append(current_task)
                activity_list.append(activity)

            else:
                for cog_collection in ["yang19Dim32", "yang19", "khonaChandra22"]:
                    if (cog_collection in current_task) and (i == 0):
                        current_task = [x for x in list(TASK2MODE.keys()) if x.startswith(f"cog_{cog_collection}")][0]
                assert current_task in TASK2MODE.keys()
                cog_mode = TASK2MODE[current_task]
                env.new_trial()
                ob = env.ob
                # print(f'Trial observation shape {ob.shape}')
                if (env.spec is None) or (not env.spec.id.startswith("contrib")):
                    ob = ob[:, np.newaxis, :]  # Add batch axis
                    inputs = torch.from_numpy(ob).type(torch.float).to(device)
                else:
                    ob = ob[:, np.newaxis]
                    inputs = torch.from_numpy(ob).type(torch.long).to(device)

                if CTRNN:
                    action_pred, activity = model(inputs, mode=cog_mode)
                else:
                    action_pred, hidden, activity = model(inputs, hidden, mode=cog_mode)
                activity = activity.cpu().detach().numpy()
                trial_list.append(env.trial)
                activity_list.append(activity)

            if i == 0:
                _logger.debug(f'Shape of activity for task {current_task}: {activity.shape}')

    activity = np.concatenate(activity_list, axis=1)
    return activity, trial_list


def get_activity_lang_vs_cog(model, cog_dataset_dict, current_task, TASK2MODE, CTRNN=False, lang_data_dict=None, device=None, bptt=None, num_trial=200): #CK changed from 1000
    """Get activity of equal-length trials"""

    trial_list = list()
    activity_list = list()

    if not CTRNN:
        if "lang" in current_task:
            hidden = model.init_hidden(10) #TODO validation batch size, get automatically
        else:
            hidden = model.init_hidden(1)

    with torch.no_grad():
        if "lang" in current_task:
            current_num_trial = int(num_trial/10) #TODO check: added since each batch has 10 trials
        else:
            current_num_trial = num_trial
        for i in range(current_num_trial):
            if "lang" in current_task:
                inputs, targets = get_batch(lang_data_dict[current_task]["test_data"], i, bptt) #TODO CHECK val_data or test_data here?
                mode = TASK2MODE[current_task]

                if CTRNN:
                    action_pred, activity = model(inputs, mode=mode)
                else:
                    action_pred, hidden, activity = model(inputs, hidden, mode=mode)
                activity = activity.cpu().detach().numpy()

                trial_list.append(current_task)
                activity_list.append(activity)

            else:
                cog_mode = TASK2MODE[current_task]
                inputs, targets = cog_dataset_dict[current_task]["dataset_cog"]()
                if "contrib" in current_task:
                    inputs = torch.from_numpy(inputs).type(torch.long).to(device)
                else:
                    inputs = torch.from_numpy(inputs).type(torch.float).to(device)

                if CTRNN:
                    action_pred, activity = model(inputs, mode=cog_mode)
                else:
                    action_pred, hidden, activity = model(inputs, hidden, mode=cog_mode)
                    hidden = repackage_hidden(hidden)
                activity = activity.cpu().detach().numpy()
                activity_list.append(activity)

            if i == 0:
                _logger.debug(f'Shape of activity for task {current_task}: {activity.shape}')

    activity = np.concatenate(activity_list, axis=1)
    return activity


def get_normalized_tv(args, tasks, multi_envs_dict, model, TASK2MODE, save_dir, CTRNN=False, lang_data_dict=None, bptt=None, device=None):
    activity_dict = {}  # recording activity
    task_variance_list = []
    task_variance_dict = {}
    _logger.info(f'**** Getting activity ****')
    j = 0
    for task in tasks:
        if re.match('cog_yang19(Dim32)?_mode|cog_khonaChandra22_mode', task):
            # get tasks from collection
            if 'Dim32' in task:
                curr_tasks = ngym.get_collection('yang19Dim32')
            elif 'yang19' in task:
                curr_tasks = ngym.get_collection('yang19')
            elif 'khonaChandra22' in task:
                curr_tasks = ngym.get_collection('khonaChandra22')
            else:
                raise NotImplementedError

            for i in range(len(curr_tasks)):
                current_task = curr_tasks[i]
                print(current_task)
                env = multi_envs_dict[task]["multi_env"]
                env.set_i(i)

                activity, trial_list = get_activity(model, env, current_task, TASK2MODE, CTRNN=CTRNN,
                                                    lang_data_dict=lang_data_dict, bptt=bptt, device=device,
                                                    num_trial=500)
                activity_dict[j] = activity
                # Compute task variance
                task_variance = np.var(activity, axis=1).mean(axis=0)
                task_variance_list.append(task_variance)
                print(f"{j} | {current_task} | {task_variance.shape}")
                assert task_variance.shape == (args.hidden_size,)
                task_variance_dict[current_task] = task_variance
                j += 1

        else:
            current_task = task
            try:
                env = multi_envs_dict[task]["multi_env"]
            except:
                env = None #lang, we don't really care about the cognitive dataset here, we just need it as an argument
            activity, trial_list = get_activity(model, env, current_task, TASK2MODE, CTRNN=CTRNN, lang_data_dict=lang_data_dict, bptt=bptt, device=device, num_trial=500)
            activity_dict[j] = activity
            # Compute task variance
            task_variance = np.var(activity, axis=1).mean(axis=0)
            task_variance_list.append(task_variance)
            print(f"{j} | {current_task} | {task_variance.shape}")
            task_variance_dict[current_task] = task_variance
            j += 1
    task_variance = np.array(task_variance_list)  # (n_task, n_units)
    thres = 1e-6
    task_variance = task_variance[:, task_variance.sum(axis=0)>thres]
    norm_task_variance = task_variance / np.max(task_variance, axis=0)

    fname = f'{save_dir}/normalizedTV.pkl'
    with open(fname, 'wb') as fout:
        pickle.dump(norm_task_variance, fout)

    return task_variance_dict, norm_task_variance, activity_dict


def figure_settings():
    figsize = (3.5,2.5)
    rect = [0.25, 0.2, 0.6, 0.7]
    rect_color = [0.25, 0.15, 0.6, 0.05]
    rect_cb = [0.87, 0.2, 0.03, 0.7]
    fs = 6
    labelpad = 13
    return figsize, rect, rect_color, rect_cb, fs, labelpad


def cluster_plot(args, norm_task_variance, task_variance_dict, epoch, save_dir, last_cluster_nr=None):
    """
    Agglomerative clustering analysis of units based on normalized task variances
    """
    tasks = task_variance_dict.keys()
    X = norm_task_variance.T
    silhouette_scores = list()
    MAX_NUMBER = 30

    if args.continuous_cluster:
        if last_cluster_nr is None:
            n_clusters = np.arange(2, MAX_NUMBER)
        else:
            n_clusters = np.arange(last_cluster_nr, MAX_NUMBER)
    else:
        n_clusters = np.arange(2, MAX_NUMBER)

    wss_values = []
    for n in n_clusters:
        cluster_model = AgglomerativeClustering(n_clusters=n)
        labels = cluster_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        #elbow
    #     agglom = AgglomerativeClustering(n_clusters=k)
    #     agglom.fit(X)
    #     wss_values.append(agglom.inertia_)
    #
    # # Plot the WSS versus the number of clusters
    # plt.plot(n_clusters, wss_values, 'o-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    # plt.title('Elbow Method for Optimal k')
    # plt.savefig(f'{save_dir}/elbow_method.png')
    # plt.show()

    n_cluster = n_clusters[np.argmax(silhouette_scores)]

    if (args.continuous_cluster) and (last_cluster_nr is not None):
        silhouette_scores = list(np.zeros(last_cluster_nr-2)) + silhouette_scores #fill up so that heatmap plot works later

    last_cluster_nr = n_cluster

    # Write number of clusters to file
    filename = f'{save_dir}/epoch={epoch}_nr_clusters.txt'

    if os.path.exists(filename):
        with open(filename, 'a') as file:
            file.write(f'{args.seed}\t{n_cluster}\n')
    else:
        with open(filename, 'w') as file:
            file.write('seed\tnr_clusters\n')
            file.write(f'{args.seed}\t{n_cluster}\n')

    cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
    labels = cluster_model.fit_predict(X)

    # Sort clusters by its task preference (important for consistency across models)
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
    norm_task_variance = norm_task_variance[:, ind_sort]

    print("Plot Normalized Variance")
    # Plot Normalized Variance
    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    tick_names = [task.split(".")[1].split("-v0")[0] if any(x in task for x in ['yang19','khonaChandra22', 'contrib']) else task for task in tasks]
    tick_names = [re.sub("lang_", "", x) for x in tick_names]
    tick_names = [re.sub("_mode=\d+", "", x) for x in tick_names]
    tick_names = [re.sub("DelayMatchSample", "DMS-", x) for x in tick_names]


    vmin, vmax = 0, 1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    im = ax.imshow(norm_task_variance, cmap='magma',
                   aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks([])
    #TODO if only yang or contrib tasks, add training iterations instead
    plt.title(f'Units | Epoch {epoch}', fontsize=7, y=0.97)
    plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
    ax.tick_params('both', length=0)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[vmin, vmax])
    cb.outline.set_linewidth(0.5)
    clabel = 'Normalized Task Variance'

    cb.set_label(clabel, fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)

    print("Plot color bars indicating clustering")
    # Plot color bars indicating clustering
    cmap = matplotlib.cm.get_cmap('tab10')
    ax = fig.add_axes(rect_color)
    for il, l in enumerate(np.unique(labels)):
        color = cmap(il % 10)
        ind_l = np.where(labels == l)[0][[0, -1]] + np.array([0, 1])
        ax.plot(ind_l, [0, 0], linewidth=4, solid_capstyle='butt',
                color=color)
        ax.text(np.mean(ind_l), -0.5, str(il + 1), fontsize=6,
                ha='center', va='top', color=color)
    ax.set_xlim([0, len(labels)])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    plt.savefig(os.path.join(f'{save_dir}', f'seed={args.seed}_epoch={epoch}_clusterplot.png'), bbox_inches='tight', dpi=280)
    fig.show()

    return silhouette_scores, last_cluster_nr


def get_normalized_tv_lang_vs_cog(args, cog_dataset_dict, model, TASK2MODE, save_dir, CTRNN=False, lang_data_dict=None, bptt=None, device=None):
    #TODO CK: average over different lang tasks
    assert len(TASK2MODE) >= 2 #both lang and cog!
    task_variance_list = list()
    activity_dict = {}  # recording activity

    task_variance_dict = {}
    _logger.info(f'**** Getting activity ****')
    for i, name in enumerate(TASK2MODE.keys()):
        _logger.info(f"Getting activity for: {name}")
        activity = get_activity_lang_vs_cog(model, cog_dataset_dict, name, TASK2MODE, CTRNN=CTRNN, lang_data_dict=lang_data_dict, bptt=bptt, device=device, num_trial=500)
        activity_dict[i] = activity
        # Compute task variance
        task_variance = np.var(activity, axis=1).mean(axis=0)
        task_variance_list.append(task_variance)
        print(f"{i} | {name} | {task_variance.shape}")

        if any(x in name for x in ["lang", "contrib."]):
            key_name = name.split("_")[1]
            if len(key_name) == 2: #TODO hacky for language
                key_name += "_wiki"
            task_variance_dict[key_name] = task_variance
        else:
            task_variance_dict['cog'] = task_variance
    task_variance = np.array(task_variance_list)  # (n_task, n_units)
    thres = 1e-6
    task_variance = task_variance[:, task_variance.sum(axis=0)>thres]
    norm_task_variance = task_variance / np.max(task_variance, axis=0)

    fname = f'{save_dir}/normalizedTV_langvscog.pkl'
    with open(fname, 'wb') as fout:
        pickle.dump(norm_task_variance, fout)

    return task_variance_dict, norm_task_variance, activity_dict


def sorted_task_variance(task_variance_dict, env_id1, env_id2, f2_ax=None):
    """Task-variance analysis of units, line plot, units ordered by env_id1 task variance"""
    ind_sort = np.argsort(task_variance_dict[env_id1])
    plot_ids = [env_id1, env_id2]
    for id in plot_ids:
        if f2_ax is None:
            plt.plot(task_variance_dict[id][ind_sort], label=id)
            plt.legend()
            plt.title(f"Task variance of RNN units | Ordered by {env_id1} task variance")
            plt.show()
            # fname = os.path.join(fig_dir, f"{model_name}_sorted_task_variance.png")
            # plt.savefig(fname)
        else:
            f2_ax.plot(task_variance_dict[id][ind_sort], label=id)
            f2_ax.legend()
            f2_ax.set_title(f"RNN units ordered by {env_id1} task variance")
            # f2_ax.show()
            # fname = os.path.join(fig_dir, f"{model_name}_sorted_task_variance.png")
            # f2_ax.savefig(fname)


def scatter_task_variance(task_variance_dict, env_id1, env_id2, f2_ax=None):
    """Task-variance analysis, scatter plot"""
    if f2_ax is None:
        plt.figure()
        plt.scatter(task_variance_dict[env_id1], task_variance_dict[env_id2])
        plt.xlabel(f"{env_id1}")
        plt.ylabel(f"{env_id2}")
        plt.title(f"{env_id1} vs. {env_id2}")
        # fname = os.path.join(fig_dir, f"scatter_task_variance.png")
        # plt.savefig(fname)
        plt.show()
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
        plt.ylim([0, 10])
        plt.title(f"Frac variance of RNN units | {env_id1} vs. {env_id2}")
        # fname = os.path.join(fig_dir, f"{model_name}_frac_variance.png")
        # plt.savefig(fname)
        plt.show()
    else:
        f2_ax.hist(frac_variance, bins=100)
        f2_ax.xaxis.set_label_text(f'{env_id2} < -- > {env_id1}')
        f2_ax.set_xlim([-1, 1])
        f2_ax.set_title(f"Frac variance of RNN units | {env_id1} vs. {env_id2}")


def gridplot(task_variance_dict, env_id1, env_ids, epoch, save_dir, args):
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
    plt.savefig(os.path.join(f'{save_dir}', f'seed={args.seed}_epoch={epoch}_gridplot.png'), bbox_inches='tight', dpi=280)
    fig2.show()


def plot_task_similarity(args, norm_task_variance, task_variance_dict, epoch, save_dir):
    similarity = cosine_similarity(norm_task_variance)  # TODO: check

    # fname = f'files/seed={args.seed}_task_similarity.pkl'
    # with open(fname, 'wb') as fout:
    #     pickle.dump(similarity, fout)
    tasks = task_variance_dict.keys()

    print(np.shape(norm_task_variance), np.shape(similarity))

    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)

    tick_names = [task.split(".")[1].split("-v0")[0] if any(x in task for x in ['yang19', 'khonaChandra22', 'contrib']) else task for task
                  in tasks]
    tick_names = [re.sub("lang_", "", x) for x in tick_names]
    tick_names = [re.sub("_mode=\d+", "", x) for x in tick_names]
    tick_names = [re.sub("DelayMatchSample", "DMS-", x) for x in tick_names]

    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks(range(len(tick_names)), tick_names,
               rotation=90, va='top', fontsize=fs)

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)

    plt.savefig(f'{save_dir}/seed={args.seed}_epoch={epoch}_task_similarity.png', bbox_inches='tight', dpi=280)
    plt.show()


def main(args, model_save_dir, model, cog_dataset_dict, epoch, cog_accuracies, acc_dict, silhouette_scores_per_epoch,
         TASK2DIM, TASK2MODE, lang_data_dict, last_cluster_nr):
    # writer_name = re.sub("/", "+", model_save_dir)
    # writer_name = "+".join(writer_name.split("+")[2:])
    # writer = SummaryWriter(f"runs/{writer_name}")

    set_seed(args.seed, args.cuda)
    device = torch.device("cuda" if args.cuda else "cpu")
    _logger.debug(f"Running on this device: {device}")

    dataset_assignments, dataset_names, _ = get_dataset_names(args.tasks)

    print("################################", flush=True)
    print(f"##### ANALYZE MODEL EPOCH {epoch} ####", flush=True)
    print("################################", flush=True)

    model.eval()
    for key in TASK2MODE.keys():
        if "cog" in key:
            if any(x in key for x in ["yang", "khonaChandra"]):
                yang = True
            else:
                yang = False
            cog_mode = TASK2MODE[key]
            assert not model.training
            accuracy_cog(key, cog_dataset_dict[key]["dataset_cog"], model, cog_mode, device, args.CTRNN, yang=yang)

    multi_envs_dict = {}

    #prepare unit analysis
    all_cog_tasks, all_cog_modes, all_keys = [], [], []
    novel_dataset = []
    for x in TASK2DIM.keys(): #FIXME > convert tasks into objects
        collection_list = ["yang19", "yang19Dim32", "khonaChandra22"]
        if any(re.match(f"cog_{l}_mode", x) for l in collection_list):
            l_name = x.split("_")[1]  # TODO generalize!
            key = x
            mode = TASK2MODE[x]
            tasks = ngym.get_collection(l_name)
            to_append = tasks
        elif re.match(f"cog_yang19|cog_khonaChandra22", x): #if we're only training on one task from the yang19 collection
            key = x
            mode = TASK2MODE[x]
            tasks = [x.split("_")[1]]  # TODO generalize!
            to_append = key
        else:
            continue

        timing = {'fixation': ('constant', 500)} #FIXME: CK I had 300 here, but yang uses 500. Does it matter?
        kwargs = {'dt': 100, 'timing': timing}
        envs = [gym.make(task, **kwargs) for task in tasks]
        if len(tasks) == 1 and re.match("yang19\.|khonaChandra22\.", tasks[0]): #if we're only training on one task from the yang19 collection
            multi_env = MultiEnvs(envs, env_input=False) #don't add additional rule input!
        else:
            multi_env = MultiEnvs(envs, env_input=True)
        multi_envs_dict[key] = {
            "multi_env": multi_env,
            "yang": True
        }
        if re.match("cog_yang19(Dim32)?_mode|cog_khonaChandra22_mode", x):
            all_cog_tasks.extend(to_append)
            all_cog_modes.extend([mode for i in range(len(to_append))])
            all_keys.extend([key for i in range(len(to_append))])
            novel_dataset.extend([i for i in range(len(to_append))])
        else:
            all_cog_tasks.append(to_append)
            all_cog_modes.append(mode)
            all_keys.append(key)
            novel_dataset.append(0)

    for x in TASK2DIM.keys():
        if x.startswith("cog_contrib."):
            mode = TASK2MODE[x]
            tasks = [x.split("_")[1]]
            timing = {'fixation': ('constant', 500)} #FIXME: CK I had 300 here, but yang uses 500. Does it matter?
            kwargs = {'dt': 100, 'timing': timing}
            envs = [gym.make(task, **kwargs) for task in tasks]
            multi_env = MultiEnvs(envs, env_input=False) #don't add additional rule input!
            multi_envs_dict[x] = {
                "multi_env": multi_env,
                "yang": False
            }
            all_cog_tasks.append(x)
            all_cog_modes.append(mode)
            all_keys.append(x)
            novel_dataset.append(0)


    print("Get performance on individual cognitive tasks")
    # Check performance on other verbal WM tasks!
    for i in range(len(all_cog_tasks)):
        curr_task = all_keys[i]
        multi_env = multi_envs_dict[curr_task]["multi_env"]
        cog_mode = all_cog_modes[i]
        multi_env.set_i(novel_dataset[i]) #set current env, has to be novel_dataset[i] and not i
        #because otherwise it'll go out of bounds.
        curr_task_name = multi_env.spec.id
        #print(f"Current task is: {curr_task_name}")
        perf = get_performance(model, multi_env, cog_mode, CTRNN=args.CTRNN, num_trial=200, device=device, yang=multi_envs_dict[curr_task]["yang"])
        print('Average performance for cog task {:s} is: {:0.2f} '.format(all_cog_tasks[i], perf), flush=True)
        cog_accuracies[i].append(perf)
        acc_dict[curr_task_name] = cog_accuracies[i]
        # writer.add_scalars(f"Cognitive task performance:", {f"{multi_env.spec.id}": perf}, epoch)

    df = pd.DataFrame.from_dict(acc_dict)
    print(df)
    if epoch > 0:
        figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
        fig = plt.figure(figsize=figsize)

        ax = sns.lineplot(data=df, marker="o")
        ax.set(ylim=(0, 1), title=f"Accuracy on cog tasks | Epoch {epoch}")
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.savefig(f'{model_save_dir}/seed={args.seed}_epoch={epoch}_accuracy_lineplot.png', bbox_inches='tight', dpi=280)
        plt.show()

    if epoch == args.epochs:
        save_df = df.copy()
        save_df["average_perf"] = save_df.mean(axis=1)
        save_df["epoch"] = list(range(args.epochs + 1))
        save_df.to_csv(f'{model_save_dir}/seed={args.seed}_epoch={epoch}_accuracy_df.csv')

    avg_df = pd.DataFrame()
    avg_df["average_perf"] = df.mean(axis=1)
    if epoch > 0:
        figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
        fig = plt.figure(figsize=figsize)

        ax = sns.lineplot(data=avg_df, marker="o")
        ax.set(ylim=(0, 1), xlabel="Epoch", ylabel="Accuracy", title=f"Average accuracy on cog tasks | Epoch {epoch}")
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.savefig(f'{model_save_dir}/seed={args.seed}_epoch={epoch}_avg_accuracy_lineplot.png', bbox_inches='tight', dpi=280)
        plt.show()


    tasks = list(TASK2DIM.keys())
    if any("lang" in x for x in tasks):
        lang_data_dict = lang_data_dict

    task_variance_dict, norm_task_variance, activity_dict = get_normalized_tv(args, tasks, multi_envs_dict, model,
                                                                                  TASK2MODE, model_save_dir, CTRNN=args.CTRNN,
                                                                                  lang_data_dict=lang_data_dict,
                                                                                  bptt=args.bptt, device=device)
    silhouette_scores, last_cluster_nr = cluster_plot(args, norm_task_variance, task_variance_dict, epoch,
                                                      save_dir=model_save_dir, last_cluster_nr=last_cluster_nr)
    silhouette_scores_per_epoch.append(silhouette_scores)
    plot_task_similarity(args, norm_task_variance, task_variance_dict, epoch, save_dir=model_save_dir)

    # if any("yang19" in x for x in tasks):
    #     ua = UnitAnalysis(task_variance_dict, model, TASK2MODE)
    #     ua.plot_rec_connections()


    if len(dataset_names) > 1:
        task_variance_dict_lang_vs_cog, norm_task_variance, activity_dict = get_normalized_tv_lang_vs_cog(args, cog_dataset_dict, model, TASK2MODE, model_save_dir, CTRNN=args.CTRNN,
                                                                                  lang_data_dict=lang_data_dict, bptt=args.bptt, device=device)
        env_ids = [x.split("_")[1] for x in TASK2MODE.keys() if "yang" not in x]
        env_ids = [x+"_wiki" if len(x) == 2 else x for x in env_ids] #TODO hacky > for language
        if any("yang" in x for x in TASK2MODE.keys()):
            env_ids += ["cog"]

        if "wikitext" in env_ids:
            env_id1 = "wikitext"
        elif "penntreebank" in env_ids:
            env_id1 = "penntreebank"
        elif "pennchar" in env_ids:
            env_id1 = "pennchar"
        elif any(re.match("[a-z]{2}_wiki", task) for task in env_ids):
            env_id1 = [task for task in env_ids if re.match("[a-z]{2}_wiki", task)][0]
        else:
            raise NotImplementedError("Still implement other orders, but wikitext/PTB are the most important right now.")
        print(f"******* Getting gridplot of variance *******")
        gridplot(task_variance_dict_lang_vs_cog, env_id1, env_ids, epoch, model_save_dir, args)

    return last_cluster_nr


if __name__ == "__main__":
    last_cluster_nr = main(args, model_save_dir, model, cog_dataset_dict, epoch, cog_accuracies, acc_dict, silhouette_scores_per_epoch,
         TASK2DIM, TASK2MODE, lang_data_dict, last_cluster_nr)

