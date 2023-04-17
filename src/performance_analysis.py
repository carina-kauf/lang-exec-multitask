import torch
import numpy as np
from dataloader_cog_tasks import build_cognitive_dataset


def get_performance(args, task_name, net, device='cpu', num_trial=1000):
    """ Get performance of the model on cognitive tasks
    Args:
        args: arguments
        task_name: name of the task set (e.g., yang19 or contrib.DelayMatchSampleWord-v0)
        net: model
        device: device to run on
        num_trial: number of trials to run
    Returns:
        avg_general_perf: average performance on all tasks from the current task set if collection, else perf. on current task
    """
    net.eval()
    if not args.CTRNN:
        hidden = net.init_hidden(1)

    all_task_performances = []
    cog_multi_env, all_tasks, collections = build_cognitive_dataset(args, task_name, return_multienv=True)

    for i in range(len(all_tasks)):
        perf = 0
        if task_name in collections:
            cog_multi_env.set_i(i)

        for j in range(num_trial):
            cog_multi_env.new_trial()
            ob, gt = cog_multi_env.ob, cog_multi_env.gt

            if 'contrib' not in task_name:
                ob = ob[:, np.newaxis, :]  # Add batch axis
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
            else:
                ob = ob[:, np.newaxis]
                inputs = torch.from_numpy(ob).type(torch.long).to(device)

            if args.CTRNN:
                action_pred, rnn_activity = net(inputs, task_name)
            else:
                action_pred, hidden, rnn_activity = net(inputs, hidden, task_name)

            action_pred = action_pred.detach().cpu().numpy()
            action_pred = np.argmax(action_pred, axis=-1)
            # Note: for Yang19, action_pred is a list of lists, thus the double index. Get list elm of last list element
            # [[ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [11], [11], [ 0], [ 0]]
            # perf += gt[-1] == action_pred[-1, 0]
            perf += gt[-1] == action_pred[-1]

        perf /= num_trial
        # print(f'Average performance {perf:0.2f} for task {all_tasks[i]}')
        all_task_performances.append(perf)
    avg_performance = np.mean(all_task_performances)
    print(f'>> Average performance for task (set) {task_name}: {avg_performance:0.2f}')

    return avg_performance, all_task_performances