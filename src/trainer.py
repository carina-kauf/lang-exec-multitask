import logging
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from tqdm import trange
import pandas as pd
import seaborn as sns

from utils_general import set_seed, repackage_hidden, mask2d
from task_builder import build_training_tasks
from models import Multitask_RNNModel, Yang19_CTRNNModel
from dataloader_lang_tasks import get_batch

from variance_analyis import main as variance_analyis
from performance_analysis import get_performance

# loss visualization via TensorBoard https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# tensorboard --logdir=tensorboard_runs #if run from src directory!
from torch.utils.tensorboard import SummaryWriter
import re

_logger = logging.getLogger(__name__)

def print_model_metainfo(model):
    """ Prints which parameters are in the model & which are being updated """
    # print model architecture
    print("\n*****MODEL*****\n", model, flush=True)

    # print parameter groups that are being updated
    print("\n*****PARAMETER INFO*****", flush=True)
    print(f"Number of parameter groups: {len(list(model.parameters()))}", flush=True)
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            print(f"Group {i + 1} | {name} | {param.data.shape}", flush=True)
        else:
            print(f"Group {i + 1} | {name} | {param.data.shape} is not being trained!", flush=True)
    print(
        f">>Overall number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print("\n")


def evaluate(args, model, criterion, data_source, mode): #FIXME add early stopping criterion
    """Evaluate model on language modeling performance on held-out test set.
    Args:
        args: command line arguments
        model: model to be evaluated
        criterion: loss function
        data_source: test data
        mode: mode associated with the current language modeling task
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    if not args.CTRNN:
        hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for j in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, j, args.bptt)
            if args.CTRNN:
                output, _ = model(data, mode)
            else:
                output, hidden, rnn_activity = model(data, hidden, mode)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def get_writers(model_save_dir):
    """ Define tensorboard writers """
    writer_name = re.sub("/", "+", model_save_dir)
    writer_name = "+".join(writer_name.split("+")[2:])
    train_log_path = os.path.abspath(f"tensorboard_runs/{writer_name}/train_logs")
    train_writer = SummaryWriter(train_log_path)
    test_log_path = os.path.abspath(f"tensorboard_runs/{writer_name}/test_logs")
    test_writer = SummaryWriter(test_log_path)
    performance_log_path = os.path.abspath(f"tensorboard_runs/{writer_name}/performance_logs")
    performance_writer = SummaryWriter(performance_log_path)
    return train_writer, test_writer, performance_writer


def main(args, model_save_dir):
    """ Main function
    args: Arguments
    model_save_dir: Directory where the model is saved
    """

    ##################
    #     SET UP     #
    ##################
    train_writer, test_writer, performance_writer = get_writers(model_save_dir)
    set_seed(args.seed, args.cuda)
    device = torch.device("cuda" if args.cuda else "cpu")
    _logger.info(f"Running on this device: {device}")

    #########################
    #  INIT MODEL & TASKS   #
    #########################
    # build training tasks & get specifications
    TRAINING_TASK_SPECS = build_training_tasks(args)
    # define full list of task names (yields list of subtasks for collections)
    individual_tasks = list()
    for task in TRAINING_TASK_SPECS:
        if "full_task_list" in TRAINING_TASK_SPECS[task]:
            individual_tasks.extend(TRAINING_TASK_SPECS[task]["full_task_list"])
        else:
            individual_tasks.extend([task])
    # define language tasks
    language_tasks = [task for task in TRAINING_TASK_SPECS if TRAINING_TASK_SPECS[task]["dataset"] == "lang"]
    cognitive_tasks = [task for task in TRAINING_TASK_SPECS if TRAINING_TASK_SPECS[task]["dataset"] == "cog"]

    # determine masking of h2h weights
    h2h_mask2d = None
    if args.sparse_model:
        assert args.CTRNN, "Only implemented for CTRNN right now!"
        print("Initializing model with anatomical mask on h2h weights!")
        assert math.sqrt(args.hidden_size).is_integer(), "Only implemented for square h2h masks right now"
        mask_sq_size = int(math.sqrt(args.hidden_size))
        h2h_mask2d = mask2d(N_x=mask_sq_size, N_y=mask_sq_size, cutoff=3, periodic=False)
        plt.imshow(h2h_mask2d, cmap="jet")
        plt.colorbar()
        plt.title("2d Mask")
        plt.show()
        # convert to pytorch tensor from numpy
        # FIXME turn into parameters
        h2h_mask2d = torch.from_numpy(h2h_mask2d, dtype=torch.float).to(device)

    # initialize model
    if args.CTRNN:
        model = Yang19_CTRNNModel(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, mask=h2h_mask2d).to(device)
    else:
        model = Multitask_RNNModel(TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, rnn_type=args.model, hidden_size=args.hidden_size,
                                   nlayers=args.nlayers, dropout=args.dropout, tie_weights=args.tied).to(device)

    # print model architecture & parameter groups that are being updated
    print_model_metainfo(model)

    ###############################
    #  Determine training config  #
    ###############################
    # determine epoch length & interleaving
    if args.dry_run and not args.tasks == ["yang19"]:
        epoch_length = 500
    elif len(language_tasks) > 0:
        # epoch length is defined by the smallest language training dataset to prevent out-of-bounds errors
        # TODO: concatenate language datasets to avoid this
        epoch_length = min([TRAINING_TASK_SPECS[task]["train_data"].size(0) - 1 for task in args.tasks \
                            if task in language_tasks])
    else:
        epoch_length = args.training_yang
    # boolean. If more than two different tasks, returns True
    interleave = len(TRAINING_TASK_SPECS) > 1


    #######################################
    #  Evaluate model prior to training   #
    #######################################
    # # run analysis on untrained model
    silhouette_scores_per_epoch = list()
    if args.TODO == "train+analyze":
        _logger.info(f"Running analysis before training! Saving under epoch 0!")
        silhouette_scores_per_epoch = variance_analyis(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS,
                                                       model=model, device=device, silhouette_scores_per_epoch=silhouette_scores_per_epoch,
                                                       epoch=0, save_dir=model_save_dir)


    #########################
    #      TRAIN MODEL      #
    #########################
    # set criterion & define optimizer
    criterion = nn.CrossEntropyLoss()
    #TODO clustering is dependent on optimizer, check!
    _logger.info('RUNNING WITH OPTIMIZER: Adam ! Check for LM loss & for robustness!')
    if args.optimizer == "Adam":
        # this results in clustering for Yang19
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif args.optimizer == "AdamW":
        # the below used to work in reducing lang. val data loss perplexity, but prevents clustering
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    else:
        raise NotImplementedError("Optimizer not implemented!")

    print("\n########################\n##### TRAIN MODEL ######\n########################\n", flush=True)

    # define variable for early stopping
    best_val_loss = None

    global_step = 0
    EARLY_STOPPING_PATIENCE = 2  # Number of epochs to wait for improvement before early stopping

    npy_avg_perf = {task: [] for task in cognitive_tasks}
    npy_avg_perf_all_tasks = {task: [] for task in cognitive_tasks}
    npy_losses = {task: [] for task in args.tasks}
    epochs_since_improvement = {task: 0 for task in language_tasks}

    train_iterator = trange(1, int(args.epochs) + 1, desc="Epoch")

    for epoch in train_iterator:

        start_time = time.time()

        total_loss = {task: 0. for task in args.tasks} # zero running loss at beginning of each epoch to prevent spikes!
        overall_batch_counter = {task: 0 for task in args.tasks} # keep track of how many batches we've seen for each task per epoch
        logging_batch_counter = {task: 0 for task in args.tasks}

        # Turn on training mode which enables dropout.
        model.train() #need this here because we're doing model evaluation after each epoch
        if not args.CTRNN:
            hidden = model.init_hidden(args.batch_size)

        for step in range(epoch_length):
            # determine current task randomly
            curr_task = np.random.choice(args.tasks)
            global_step += 1

            # check if we're done with current task
            if curr_task in language_tasks: #TODO unnecessary if we were to concatenate language datasets
                try:
                    # assert that we're not out of bounds for the current task
                    # (i.e. we haven't finished a total run through the training dataset)
                    assert overall_batch_counter[curr_task] * args.bptt <= \
                           TRAINING_TASK_SPECS[curr_task]["train_data"].size(0) - args.bptt
                except:
                    print(f"Finished training epoch for dataset {curr_task}! \n {overall_batch_counter[curr_task] * args.bptt} "
                          f"\n {overall_batch_counter[curr_task] * args.bptt}")
                    break

            # get corresponding dataset batch
            if curr_task in language_tasks:
                i = overall_batch_counter[curr_task] * args.bptt # get batch index (we iterate over dataset in steps of length bptt)
                data, targets = get_batch(TRAINING_TASK_SPECS[curr_task]["train_data"], i, args.bptt)
            else:
                data, targets = TRAINING_TASK_SPECS[curr_task]["dataset_cog"]()
                if "contrib." in curr_task:
                    data = torch.from_numpy(data).type(torch.long).to(device) #needs to be long for nn.Embedding
                else:
                    data = torch.from_numpy(data).type(torch.float).to(device)
                targets = torch.from_numpy(targets.flatten()).type(torch.long).to(device)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()

            # run model forward
            if args.CTRNN:
                output, _ = model(x=data, task=curr_task)
            else:
                # truncated BPP
                hidden = repackage_hidden(h=hidden)
                output, hidden, rnn_activity = model(input=data, hidden=hidden, task=curr_task)

            # run model backward
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # take optimizer step to update weights
            optimizer.step()

            if args.sparse_model:
                if step == 0:
                    print("Applying anatomical mask on h2h weights!")
                # apply anatomical mask on h2h weights to enforce sparsity in model
                # (model will never learn to use these weights)
                model.rnn.h2h.weight.data = model.rnn.h2h.weight.data * h2h_mask2d.to(device)

            # collect loss
            total_loss[curr_task] += loss.item()
            overall_batch_counter[curr_task] += 1
            logging_batch_counter[curr_task] += 1

            # model training inspection
            if step % args.log_interval == 0 and step > 0:
                scalar_dict_perf = {}
                if curr_task not in language_tasks: # get performance on cognitive tasks
                    perf, all_task_performances = get_performance(args=args, task_name=curr_task, net=model,
                                                                    device=device, num_trial=200)
                    print('{:d} perf: {:0.2f}'.format(step, perf), flush=True)
                    npy_avg_perf[curr_task].append((perf, global_step))
                    npy_avg_perf_all_tasks[curr_task].append(list(zip(*[TRAINING_TASK_SPECS[curr_task]["full_task_list"],
                                                            all_task_performances, [global_step]*len(all_task_performances)])))
                    scalar_dict_perf[f'{curr_task}'] = perf
                    for ind, subtask in enumerate(TRAINING_TASK_SPECS[curr_task]["full_task_list"]):
                        scalar_dict_perf[f'{subtask}'] = all_task_performances[ind]
                performance_writer.add_scalars(f'{curr_task}/performance', scalar_dict_perf, global_step)

                scalar_dict_loss = {}
                for log_task in args.tasks:
                    if interleave:
                        prop_factor = sum(logging_batch_counter.values()) / logging_batch_counter[log_task]
                    else:
                        prop_factor = 1
                    cur_loss = (total_loss[log_task] / args.log_interval) * prop_factor
                    scalar_dict_loss[f'{log_task}'] = cur_loss
                    npy_losses[log_task].append((cur_loss, global_step))

                    # print training progress
                    elapsed = time.time() - start_time
                    stats = f"{log_task} | epoch {epoch:3d} | {step:5d}/around {epoch_length * len(args.tasks)} " \
                            f"batches | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | loss {cur_loss:5.2f}"
                    if log_task in language_tasks:
                        stats += f" | ppl {math.exp(cur_loss):8.2f}"
                    print(stats, flush=True)
                train_writer.add_scalars(f'loss', scalar_dict_loss, global_step)

                # reset loss batch counter
                total_loss = {task: 0. for task in args.tasks}
                logging_batch_counter = {task: 0 for task in args.tasks}
                print("*****")

            start_time = time.time()

        for task in args.tasks:
            epoch_task_loss = total_loss[task] / overall_batch_counter[task]
            _logger.debug(f"{task} | Training epoch {epoch}: loss = {epoch_task_loss}, perplexity = {np.exp(epoch_task_loss)}")

        print(f"\n PROPORTIONS OF BATCHES IN EPOCH {epoch}:")
        nr_batches = sum(overall_batch_counter.values())
        for key, value in overall_batch_counter.items():
            print(f"{value/nr_batches}: Training proportion for task {key}")


        #######################
        # Evaluation loss
        #######################
        print("\n", flush=True)
        _logger.info("Evaluating LM performance on validation dataset")
        joint_val_loss = 0
        scalar_dict = {}
        for task in args.tasks:
            if task in language_tasks:
                print(f"Evaluating LM performance on validation dataset for task {task}")
                val_loss = evaluate(args, model, criterion, TRAINING_TASK_SPECS[task]["val_data"], task)
                print("-" * 89, flush=True)
                # print training progress
                elapsed = time.time() - start_time
                stats = f"{task} | epoch {epoch:3d} | {elapsed} | valid loss {val_loss:5.2f}  | ppl {math.exp(val_loss):8.2f}"
                print(stats, flush=True)
                print("-" * 89, flush=True)
                scalar_dict[task] = val_loss

                joint_val_loss += val_loss
                val_ppl = torch.exp(torch.tensor(val_loss)).numpy().tolist()
        test_writer.add_scalars(f'valid_loss', scalar_dict, epoch)


        # epoch training has finished here
        if args.TODO == "train+analyze":
            _logger.info(f"Running analysis for epoch {epoch}")
            silhouette_scores_per_epoch = variance_analyis(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, model=model,
                                                  device=device, silhouette_scores_per_epoch=silhouette_scores_per_epoch,
                                                    epoch=epoch, save_dir=model_save_dir)
            _logger.info(f"Finished analysis for epoch {epoch}")

        # plot curves
        for name, x in zip(["losses", "cog_avg_perfs", "cog_avg_perfs_all_tasks"], [npy_losses, npy_avg_perf, npy_avg_perf_all_tasks]):
            for task, value_steps in x.items():
                if isinstance(value_steps[0][0], float):
                    values = [elm[0] for elm in value_steps]
                    steps = [elm[1] for elm in value_steps]
                    plt.plot(steps, values, label=task, marker='o')
                elif isinstance(value_steps[0][0], tuple):
                    fig, ax = plt.subplots()
                    frames = []
                    for k in range(len(value_steps)):
                        names = [elm[0] for elm in value_steps[k]]
                        values = [elm[1] for elm in value_steps[k]]
                        steps = [elm[2] for elm in value_steps[k]]
                        curr_df = pd.DataFrame({"steps": steps, "values": values, "names": names})
                        frames.append(curr_df)
                    df = pd.concat(frames)
                    sns.lineplot(x="steps", y="values", hue="names", data=df, ax=ax, marker="o")
                else:
                    raise NotImplementedError
            plt.legend()
            plt.title(name)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(f"{model_save_dir}/epoch={epoch}_{name}.png", bbox_inches='tight')
            plt.show()
            plt.close()

        # determine early stopping (equally weighing all language task losses)
        if len(language_tasks) > 0:
            joint_val_loss = joint_val_loss / len(language_tasks)
            if not best_val_loss or joint_val_loss < best_val_loss:  # all good, continue
                best_val_loss = joint_val_loss
                epochs_since_improvement = 0
            else:  # no more improvement
                epochs_since_improvement += 1  # keep track of how many epochs have passed since last improvement

            if epochs_since_improvement >= EARLY_STOPPING_PATIENCE: # stop training
                print(f"No improvement in validation loss on language tasks for {EARLY_STOPPING_PATIENCE} epochs. Stopping training.")
                break


    train_writer.flush()
    test_writer.flush()
    train_writer.close()
    test_writer.close()

    np.save(os.path.join(model_save_dir, 'modellosscurve.npy'), npy_losses, allow_pickle=True)
    np.save(os.path.join(model_save_dir, 'modelavgperf.npy'), npy_avg_perf, allow_pickle=True)
    np.save(os.path.join(model_save_dir, 'modelavgperf_alltasks.npy'), npy_avg_perf_all_tasks, allow_pickle=True)

    # def plot_curves(filename):
    #     x = np.load(os.path.join(model_save_dir, filename), allow_pickle=True).item()
    #     for task, value_steps in x.items():
    #         values = [elm[0] for elm in value_steps]
    #         steps = [elm[1] for elm in value_steps]
    #         if isinstance(values[0], tuple):
    #             name = [elm[0] for elm in values]
    #             values = [elm[1] for elm in values]
    #             plt.plot(steps, values, label=name)
    #         else:
    #             plt.plot(steps, values, label=task)
    #     plt.legend()
    #     plt.title(filename.split(".")[0])
    #     plt.savefig(os.path.join(model_save_dir, filename.split(".")[0] + '.pdf'))
    #     plt.show()
    #     plt.close()
    #
    # plot_curves('modellosscurve.npy')
    # plot_curves('modelavgperf.npy')
    # plot_curves('modelavgperf_alltasks.npy')

    # saving after training
    model_savename = f"model_epochs={epoch}.pt"
    print(f"Saving model after training under this name: {model_savename}")

    with open(os.path.join(model_save_dir, model_savename), "wb") as f:
        torch.save(model.state_dict(), f)

    #########################
    #       TEST MODEL      #
    #########################
    print("Testing model on language test sets after training", flush=True)
    for task in args.tasks:
        if task in language_tasks:
            print(f"Evaluating LM performance on test dataset for task {task}")
            test_loss = evaluate(args, model, criterion, TRAINING_TASK_SPECS[task]["test_data"], task)
            print("=" * 89, flush=True)
            print(f"{task} | End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}")
            print("=" * 89, flush=True)


if __name__ == "__main__":
    main(args, model_save_dir)
