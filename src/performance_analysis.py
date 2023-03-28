import torch
import numpy as np


# TODO: Make this into a function in neurogym
def get_performance(args, task_name, net, env, device='cpu', num_trial=1000): #todo model is not in eval mode here!
    if not args.tasks == ["yang19"]:
        raise NotImplementedError("Only implemented for Yang19 right now!")
    perf = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, _ = net(inputs, task_name)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        # Note: for Yang19, action_pred is a list of lists, thus the double index. Get list elm of last list element
        # [[ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [11], [11], [ 0], [ 0]]
        # perf += gt[-1] == action_pred[-1, 0]
        perf += gt[-1] == action_pred[-1]

    perf /= num_trial
    return perf


# def get_performance(args, net, curr_task_env, num_trial=1000, device='cpu', yang=True):
#     """
#     Get per-task performance of model on cognitive tasks
#     source: https://github.com/neurogym/ngym_usage/blob/master/yang19/models.py#L108
#     """
#     if not args.CTRNN:
#         hidden = net.init_hidden(1)
#     perf = 0
#
#     current_task_name = curr_task_env.spec.id
#     #print(current_task_name)
#
#     net.eval()
#     with torch.no_grad(): #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/38
#         for i in range(num_trial):
#             curr_task_env.new_trial()
#             ob, gt = curr_task_env.ob, curr_task_env.gt
#             if yang:
#                 ob = ob[:, np.newaxis, :]
#                 inputs = torch.from_numpy(ob).type(torch.float).to(device)
#             else:
#                 ob = ob[:, np.newaxis]
#                 inputs = torch.from_numpy(ob).type(torch.long).to(device)
#
#             if args.CTRNN:
#                 action_pred, rnn_activity = net(inputs, task_name)
#             else:
#                 action_pred, hidden, rnn_activity = net(inputs, hidden, task_name)
#
#             action_pred = action_pred.detach().cpu().numpy()
#             action_pred = np.argmax(action_pred, axis=-1)
#
#             #Note: for Yang19, action_pred is a list of lists, thus the double index. Get list elm of last list element
#             # [[ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [ 0], [11], [11], [ 0], [ 0]]
#             perf += gt[-1] == action_pred[-1]
#
#     perf /= num_trial
#     return perf
#
#
# def accuracy_cog(key, dataset, model, cog_mode, device, CTRNN=False, yang=True):
#     """
#     Get average accuracy on cognitive tasks
#     """
#     # Reset environment
#     env = dataset.env
#     env.reset(no_step=True)
#
#     # Initialize variables for logging
#     activity_dict = {}  # recording activity
#     trial_infos = {}  # recording trial information
#
#     num_trial = 1000
#     if not CTRNN:
#         hidden = model.init_hidden(1) #TODO check correct? > added since we're evaluating on one trial each per num_trials
#
#     assert not model.training #assert that model is in eval mode
#
#     with torch.no_grad():
#         for i in range(num_trial):
#             # Neurogym boilerplate
#             # Sample a new trial
#             trial_info = env.new_trial()
#             # Observation and ground-truth of this trial
#             ob, gt = env.ob, env.gt
#             # Convert to numpy, add batch dimension to input
#             if yang:
#                 inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
#             else:
#                 inputs = torch.from_numpy(ob[:, np.newaxis]).type(torch.long).to(device)
#
#             # Run the model for one trial
#             # inputs (SeqLen, Batch, InputSize)
#             # action_pred (SeqLen, Batch, OutputSize)
#             if CTRNN:
#                 action_pred, rnn_activity = model(inputs, cog_mode)
#             else:
#                 action_pred, hidden, rnn_activity = model(inputs, hidden, cog_mode)
#             # print(action_pred.shape, rnn_activity.shape)
#
#             # Compute performance
#             # First convert back to numpy
#             action_pred = action_pred.cpu().detach().numpy()
#             # Read out final choice at last time step
#             choice = np.argmax(action_pred[-1, :]) #np.argmax returns index of highest value
#             # Compare to ground truth
#             correct = choice == gt[-1]
#
#             # Record activity, trial information, choice, correctness
#             rnn_activity = rnn_activity[:, 0, :].cpu().detach().numpy()
#             activity_dict[i] = rnn_activity
#             trial_infos[i] = trial_info  # trial_info is a dictionary
#             trial_infos[i].update({'correct': correct})
#             if env.spec:
#                 trial_infos[i].update({'spec.id': env.spec.id})
#
#     # Print information for sample trials
#     print(f"*****{key}******")
#     for i in range(5):
#         print('Trial', i, trial_infos[i], flush=True)
#
#     average_acc = np.mean([val['correct'] for val in trial_infos.values()])
#     print(f'Average performance after {num_trial} trials: {average_acc}\n', flush=True)
#     return average_acc