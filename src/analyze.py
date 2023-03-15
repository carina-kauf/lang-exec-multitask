import logging
import torch
import numpy as np

from utils import build_training_tasks, set_seed, get_dataset_names
from data import get_batch
from model import Multitask_RNNModel, Yang19_RNNNet

import gym
import neurogym as ngym
from neurogym.wrappers.block import MultiEnvs

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pickle

_logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


def get_performance(net, env, cog_mode, CTRNN=False, num_trial=1000, device='cpu'):
    """
    Get per-task performance of model on cognitive tasks
    """
    if not CTRNN:
        hidden = net.init_hidden(1)
    perf = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        if CTRNN:
            action_pred, rnn_activity = net(inputs)
        else:
            action_pred, hidden, rnn_activity = net(inputs, hidden, cog_mode)

        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)

        #Note: for Yang19, action_pred is a list of lists, thus the double index. Get list elm of last list element
        # [[ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [11], [11], [ 0], [ 0]]
        perf += gt[-1] == action_pred[-1]

    perf /= num_trial
    return perf


def accuracy_cog(dataset, model, cog_mode, device, CTRNN=False):
    """
    Get average accuracy on cognitive tasks
    """
    # Reset environment
    env = dataset.env
    env.reset(no_step=True)

    # Initialize variables for logging
    activity_dict = {}  # recording activity
    trial_infos = {}  # recording trial information

    num_trial = 200
    if not CTRNN:
        hidden = model.init_hidden(1) #TODO check correct?

    for i in range(num_trial):
        # Neurogym boilerplate
        # Sample a new trial
        trial_info = env.new_trial()
        # Observation and ground-truth of this trial
        ob, gt = env.ob, env.gt
        # Convert to numpy, add batch dimension to input
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)

        # Run the model for one trial
        # inputs (SeqLen, Batch, InputSize)
        # action_pred (SeqLen, Batch, OutputSize)
        if CTRNN:
            action_pred, rnn_activity = model(inputs)
        else:
            action_pred, hidden, rnn_activity = model(inputs, hidden, cog_mode)

        # print(action_pred.shape, rnn_activity.shape)

        # Compute performance
        # First convert back to numpy
        action_pred = action_pred.cpu().detach().numpy()
        # Read out final choice at last time step
        choice = np.argmax(action_pred[-1, :])  #TODO check correct?
        # Compare to ground truth
        correct = choice == gt[-1]

        # Record activity, trial information, choice, correctness
        rnn_activity = rnn_activity[:, 0, :].cpu().detach().numpy()
        activity_dict[i] = rnn_activity
        trial_infos[i] = trial_info  # trial_info is a dictionary
        trial_infos[i].update({'correct': correct})

    # Print information for sample trials
    for i in range(5):
        print('Trial ', i, trial_infos[i], flush=True)

    average_acc = np.mean([val['correct'] for val in trial_infos.values()])
    print(f'Average performance {average_acc}', flush=True)
    return average_acc


def get_activity(model, env, current_task, TASK2MODE, CTRNN=False, lang_data_dict=None, device=None, bptt=None, num_trial=1000):
    """Get activity of equal-length trials"""

    trial_list = list()
    activity_list = list()

    if not CTRNN:
        if "lang" in current_task:
            hidden = model.init_hidden(10) #TODO check correct?
        else:
            hidden = model.init_hidden(1)

    for i in range(num_trial):
        if "lang" in current_task:
            inputs, targets = get_batch(lang_data_dict[current_task]["val_data"], i, bptt)
            mode = TASK2MODE[current_task]

            if CTRNN:
                action_pred, activity = model(inputs)
            else:
                action_pred, hidden, activity = model(inputs, hidden, mode=mode)
            activity = activity.cpu().detach().numpy()

            if i == 0:
                _logger.info(f'Shape of activity for task {current_task}: {activity.shape}')

            trial_list.append(current_task)
            activity_list.append(activity)

        else:
            for key in TASK2MODE.keys():
                if "cog_" in key:
                    cog_mode = TASK2MODE[key]
            mode = cog_mode
            env.new_trial()
            ob = env.ob
            ob = ob[:, np.newaxis, :]  # Add batch axis
            inputs = torch.from_numpy(ob).type(torch.float).to(device)

            if CTRNN:
                action_pred, activity = model(inputs)
            else:
                action_pred, hidden, activity = model(inputs, hidden, mode=mode)
            activity = activity.cpu().detach().numpy()
            trial_list.append(env.trial)
            activity_list.append(activity)

            if i == 0:
                _logger.info(f'Shape of activity for task {current_task}: {activity.shape}')

    activity = np.concatenate(activity_list, axis=1)
    return activity, trial_list


def get_normalized_tv(args, tasks, env, model, TASK2MODE, CTRNN=False, lang_data_dict=None, bptt=None, device=None):
    task_variance_list = list()
    activity_dict = {}  # recording activity

    task_variance_dict = {}
    _logger.info(f'**** Getting activity ****')
    for i in range(len(tasks)):
        if tasks[i].startswith("yang19"):
            print(tasks[i])
            env.set_i(i)
        activity, trial_list = get_activity(model, env, tasks[i], TASK2MODE, CTRNN=CTRNN, lang_data_dict=lang_data_dict, bptt=bptt, device=device, num_trial=500)
        activity_dict[i] = activity
        # Compute task variance
        task_variance = np.var(activity, axis=1).mean(axis=0)
        task_variance_list.append(task_variance)
        print(f"{i} | {tasks[i]} | {task_variance.shape}")
        task_variance_dict[tasks[i]] = task_variance
    task_variance = np.array(task_variance_list)  # (n_task, n_units)
    thres = 1e-6
    task_variance = task_variance[:, task_variance.sum(axis=0)>thres]
    norm_task_variance = task_variance / np.max(task_variance, axis=0)

    if args.CTRNN:
        fname = f'files_CTRNN/seed={args.seed}_normalizedTV.pkl'
    else:
        fname = f'files_{args.model}/seed={args.seed}_normalizedTV.pkl'
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


def cluster_plot(args, norm_task_variance, tasks):
    """
    Agglomerative clustering analysis of units based on normalized task variances
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
    if args.CTRNN:
        plt.savefig(os.path.join(f'figures_CTRNN', f'seed={args.seed}_silhouette_score.png'), bbox_inches='tight', dpi=280)
    else:
        plt.savefig(os.path.join(f'figures_{args.model}', f'seed={args.seed}_silhouette_score.png'), bbox_inches='tight', dpi=280)
    plt.show()

    n_cluster = n_clusters[np.argmax(silhouette_scores)]

    # Write number of clusters to file
    if args.CTRNN:
        filename = f'files_CTRNN/nr_clusters.txt'
    else:
        filename = f'files_{args.model}/nr_clusters.txt'

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
    tick_names = [task[len('yang19.'):-len('-v0')] if 'yang19' in task else task.split("_")[1] for task in tasks]

    vmin, vmax = 0, 1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    im = ax.imshow(norm_task_variance, cmap='magma',
                   aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks([])
    plt.title('Units', fontsize=7, y=0.97)
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
    if args.CTRNN:
        plt.savefig(os.path.join(f'figures_CTRNN', f'seed={args.seed}_clusterplot.png'), bbox_inches='tight', dpi=280)
    else:
        plt.savefig(os.path.join(f'figures_{args.model}', f'seed={args.seed}_clusterplot.png'), bbox_inches='tight', dpi=280)
    #plt.savefig(os.path.join(model_save_dir, "clusterplot.png"), bbox_inches='tight', dpi=280)
    fig.show()


def sorted_task_variance(task_variance_dict, env_id1, env_ids, f2_ax=None):
    """Task-variance analysis of units, line plot, units ordered by env_id1 task variance"""
    ind_sort = np.argsort(task_variance_dict[env_id1])
    plot_ids = [env_id1, env_ids[0]]
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


def scatter_task_variance(task_variance_dict, env_id1, env_ids, f2_ax=None):
    """Task-variance analysis, scatter plot"""
    for env_id2 in env_ids:
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


def frac_variance(task_variance_dict, env_id1, env_ids, f2_ax=None):
    """Fractional variance analysis, histogram plot"""
    for env_id2 in env_ids:
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


def gridplot(task_variance_dict, env_ids, env_id1):
    """Merging plot types"""

    fig2 = plt.figure(constrained_layout=True, figsize=(15, 3 * len(env_ids)))
    spec2 = GridSpec(ncols=3, nrows=len(env_ids), figure=fig2)
    f2_ax = []

    for i in range(len(env_ids)):
        # plotting
        j = 0
        f2_ax.append(fig2.add_subplot(spec2[i, j]))
        sorted_task_variance(task_variance_dict, env_id1, [env_ids[i]], f2_ax[-1])
        #
        j = 1
        f2_ax.append(fig2.add_subplot(spec2[i, j]))
        scatter_task_variance(task_variance_dict, env_id1, [env_ids[i]], f2_ax[-1])
        #
        j = 2
        f2_ax.append(fig2.add_subplot(spec2[i, j]))
        frac_variance(task_variance_dict, env_id1, [env_ids[i]], f2_ax[-1])
        print("*" * 30)
    fig2.suptitle(f"Unit variance analysis")
    fig2.show()


def plot_task_similarity(args, norm_task_variance, tasks):
    similarity = cosine_similarity(norm_task_variance)  # TODO: check

    # fname = f'files/seed={args.seed}_task_similarity.pkl'
    # with open(fname, 'wb') as fout:
    #     pickle.dump(similarity, fout)

    print(np.shape(norm_task_variance), np.shape(similarity))

    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    im = ax.imshow(similarity, cmap='magma', interpolation='nearest', vmin=0, vmax=1)

    tick_names = [task[len('yang19.'):-len('-v0')] for task in tasks]
    plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=fs)
    plt.xticks(range(len(tick_names)), tick_names,
               rotation=90, va='top', fontsize=fs)

    ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)

    if args.CTRNN:
        plt.savefig(f'figures_CTRNN/seed={args.seed}_task_similarity.png', bbox_inches='tight', dpi=280)
    else:
        plt.savefig(f'figures_{args.model}/seed={args.seed}_task_similarity.png', bbox_inches='tight', dpi=280)
    plt.show()


def main(args, model_save_dir):
    set_seed(args.seed, args.cuda)
    device = torch.device("cuda" if args.cuda else "cpu")
    _logger.info(f"Running on this device: {device}")

    if args.CTRNN:
        os.makedirs(f'files_CTRNN', exist_ok=True)
        os.makedirs(f'figures_CTRNN', exist_ok=True)
    else:
        os.makedirs(f'files_{args.model}', exist_ok=True)
        os.makedirs(f'figures_{args.model}', exist_ok=True)

    dataset_assignments, dataset_names, _ = get_dataset_names(args.tasks)

    lang_data_dict = None

    #build training tasks & get all relevant info
    if all(x in dataset_names for x in ["lang", "cog"]):
        TASK2DIM, TASK2MODE, MODE2TASK, modes, dataset_assignments, lang_data_dict, dataset_cog, env, ob_size, act_size = build_training_tasks(args.tasks, batch_size=args.batch_size, seq_len=args.seq_len)
    elif dataset_names == ["lang"]:
        TASK2DIM, TASK2MODE, MODE2TASK, modes, dataset_assignments, lang_data_dict = build_training_tasks(args.tasks, batch_size=args.batch_size, seq_len=args.seq_len)
    else: #if dataset_names == ["cog"]:
        TASK2DIM, TASK2MODE, MODE2TASK, modes, dataset_assignments, dataset_cog, env, ob_size, act_size = build_training_tasks(args.tasks, batch_size=args.batch_size, seq_len=args.seq_len)

    #build model
    if args.CTRNN:
        if len(TASK2DIM) > 1 or ("cog_" not in list(TASK2DIM.keys())[0]):
            raise NotImplementedError("The CTRNN model currently only works as a control for the Yang19 task set!")
        model = Yang19_RNNNet(TASK2DIM, hidden_size=256, dt=env.dt).to(device) #keeping 256 units for now
    else:
        model = Multitask_RNNModel(TASK2DIM, MODE2TASK, modes, args.model, args.emsize, args.hidden_size, args.nlayers,
                       args.dropout, args.tied).to(device)

    print(model)

    #load model state_dict
    try:
        model.load_state_dict(torch.load(os.path.join(model_save_dir, args.save)))
    except:
        try:
            model_savename = f"model_epochs={args.epochs}.pt"
            print(os.path.join(model_save_dir, model_savename))
            model.load_state_dict(torch.load(os.path.join(model_save_dir, model_savename)))
        except:
            try:
                model_savename = f"model_Interrupted.pt"
                model.load_state_dict(torch.load(os.path.join(model_save_dir, model_savename)))
            except:
                raise NotImplementedError("No model state dict found")

    print("########################", flush=True)
    print("##### ANALYZE MODEL ####", flush=True)
    print("########################", flush=True)

    model.eval()

    cog_mode = None
    for key in TASK2MODE.keys():
        if "cog" in key:
            cog_mode = TASK2MODE[key]
            accuracy_cog(dataset_cog, model, cog_mode, device, args.CTRNN)


    #prepare unit analysis
    tasks = ngym.get_collection('yang19')
    timing = {'fixation': ('constant', 500)}
    kwargs = {'dt': 100, 'timing': timing}
    envs = [gym.make(task, **kwargs) for task in tasks]
    env = MultiEnvs(envs, env_input=True)

    if "cog" in dataset_names:
        tasks = tasks

        print("Get performance on cognitive tasks")
        for i in range(20):
            env.set_i(i)
            perf = get_performance(model, env, cog_mode, CTRNN=args.CTRNN, num_trial=200, device=device)
            print('Average performance {:0.2f} for task {:s}'.format(perf, tasks[i]), flush=True)

    else:
        tasks = []

    #do after environment has been generated
    for mode in modes:
        if "lang" in MODE2TASK[mode]:
            tasks.append(MODE2TASK[mode])
            lang_data_dict = lang_data_dict

    task_variance_dict, norm_task_variance, activity_dict = get_normalized_tv(args, tasks, env, model, TASK2MODE, CTRNN=args.CTRNN,
                                                                              lang_data_dict=lang_data_dict, bptt=args.bptt, device=device)
    cluster_plot(args, norm_task_variance, tasks, model_save_dir)
    plot_task_similarity(args, norm_task_variance, tasks)

if __name__ == "__main__":
    main(args, model_save_dir)

