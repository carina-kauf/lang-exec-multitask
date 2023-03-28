import logging
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
import pickle

from utils_general import set_seed, repackage_hidden, mask2d
from task_builder import build_training_tasks
from models import Multitask_RNNModel, Yang19_CTRNNModel
from dataloader_lang_tasks import get_batch

from variance_analyis import main as variance_analyis
from performance_analysis import get_performance

# loss visualization via TensorBoard https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# tensorboard --logdir=runs #if run from src directory!
from torch.utils.tensorboard import SummaryWriter
import re

_logger = logging.getLogger(__name__)

def get_writers(model_save_dir):
    """ Define tensorboard writers """
    writer_name = re.sub("/", "+", model_save_dir)
    writer_name = "+".join(writer_name.split("+")[2:])
    train_writer = SummaryWriter(f"runs/{writer_name}/train_logs")
    test_writer = SummaryWriter(f"runs/{writer_name}/test_logs")
    return train_writer, test_writer


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


def plot_silhouette_heatmap(args, model_save_dir, silhouette_scores_per_epoch, epoch):
    """Plots a heatmap of the silhouette score per number of predefined clusters per epoch

    Args:
        args: Arguments
        model_save_dir: Directory where the model is saved
        silhouette_scores_per_epoch: Dictionary, shape [nr_epochs,nr_predefined_cluster(here:2-30)]

    Returns:
        Plots heatmap
    """
    MAX_NUMBER = 20 #FIXME turn into a parameter
    # Create the pandas DataFrame
    plot_df = pd.DataFrame(silhouette_scores_per_epoch)
    # specifying column names
    nr_epochs = np.arange(len(silhouette_scores_per_epoch))
    nr_clusters = np.arange(2, MAX_NUMBER)

    plot_df.columns = nr_clusters
    plot_df.index = nr_epochs
    plot_df = plot_df.T #since we want epochs to be on the x-axis

    # ax = sns.heatmap(df)
    fig = plt.figure()
    ax = sns.heatmap(plot_df, annot=True, cmap="coolwarm") #viridis
    ax.set(xlabel="Epoch number", ylabel="Nr. of clusters", title=f"Silhouette scores | Epoch {epoch}")
    ax.collections[0].colorbar.set_label("Silhouette score")
    plt.savefig(f"{model_save_dir}/seed={args.seed}_epoch={epoch}_silhouette_heatmap.png",
                bbox_inches="tight", dpi=280)
    plt.show()


def main(args, model_save_dir):
    """ Main function
    args: Arguments
    model_save_dir: Directory where the model is saved
    """

    ##################
    #     SET UP     #
    ##################
    train_writer, test_writer = get_writers(model_save_dir)
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
    if args.TODO == "train+analyze":
        _logger.info(f"Running analysis before training! Saving under epoch 0!")
        _ = variance_analyis(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, model=model, device=device)


    #########################
    #      TRAIN MODEL      #
    #########################
    # set criterion & define optimizer
    criterion = nn.CrossEntropyLoss()
    #TODO clustering is dependent on optimizer, check!
    _logger.info('RUNNING WITH OPTIMIZER: Adam ! Check for LM loss & for robustness!')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #this results in clustering for Yang19
    #TODO take out!
    # the below used to work in reducing lang. val data loss perplexity, but prevents clustering
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # define variables for early stopping
    previous_val_ppl = np.inf

    global_step = 0
    train_iterator = trange(1, int(args.epochs) + 1, desc="Epoch")

    print("\n########################\n##### TRAIN MODEL ######\n########################\n", flush=True)

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
            if not args.CTRNN:
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
                if args.tasks == ["yang19"]:
                    perf = get_performance(args=args, task_name=curr_task, net=model,
                                           env=TRAINING_TASK_SPECS[curr_task]["dataset_cog"].env,
                                           device=device, num_trial=200)
                    print('{:d} perf: {:0.2f}'.format(step, perf), flush=True)

                for log_task in args.tasks:
                    if interleave:
                        prop_factor = sum(logging_batch_counter.values()) / logging_batch_counter[log_task]
                    else:
                        prop_factor = 1
                    cur_loss = (total_loss[log_task] / args.log_interval) * prop_factor
                    train_writer.add_scalar(f'{log_task}/loss', cur_loss, global_step)

                    # print training progress
                    elapsed = time.time() - start_time
                    stats = f"{log_task} | epoch {epoch:3d} | {step:5d}/around {epoch_length * len(args.tasks)} " \
                            f"batches | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | loss {cur_loss:5.2f}"
                    if log_task in language_tasks:
                        stats += f" | ppl {math.exp(cur_loss):8.2f}"
                    print(stats, flush=True)

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


        # # Plot training losses
        # print("\n>>Plotting training losses")
        # for task in args.tasks:
        #     task_losses = train_writer.get_scalar(f'{task}/loss')
        #     plt.plot(task_losses, label=f"{task}")
        # plt.legend()
        # plt.savefig(f"{model_save_dir}/training_losses.png")
        # plt.show()

        #######################
        # Evaluation loss
        #######################
        print("\n", flush=True)
        _logger.info("Evaluating LM performance on validation dataset")
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
                test_writer.add_scalar(f'{task}/loss', val_loss, epoch)

                # if val_ppl < previous_val_ppl - ppl_diff_threshold:  # all good, continue
                #     previous_val_ppl = val_ppl
                # else:  # no more improvement --> stop
                #     print("Stopping training early!")
                #     # we could load the previous checkpoint here, but won"t bother since usually the loss still decreases
                #     break

        # val_losses = test_writer.get_scalars("val_loss:")
        # if val_losses:
        #     fig = plt.figure()
        #     x_length = len(next(iter(val_losses.values())))
        #     for k, v in val_losses.items():
        #         plt.plot(range(1, len(v) + 1), v, ".-", label=k)
        #     plt.xticks(range(1, x_length + 1))
        #     plt.title("Evaluation loss")
        #     plt.legend()  # To draw legend
        #     plt.savefig(f"{model_save_dir}/seed={args.seed}_epoch={epoch}_validation_loss.png",
        #                 bbox_inches="tight", dpi=280)
        #     plt.show()

        # epoch training has finished here
        if args.TODO == "train+analyze":
            _logger.info(f"Running analysis for epoch {epoch}")
            last_cluster_nr = variance_analyis(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, model=model,
                                                  device=device)
            # last_cluster_nr = concurrent_analysis(args=args, model_save_dir=model_save_dir, model=model,
            #                                       cog_dataset_dict=cog_dataset_dict, epoch=epoch,
            #                                       cog_accuracies=cog_accuracies,
            #                                       acc_dict=acc_dict,
            #                                       silhouette_scores_per_epoch=silhouette_scores_per_epoch,
            #                                       TASK2DIM=TASK2DIM, TASK2MODE=TASK2MODE, lang_data_dict=lang_data_dict,
            #                                       last_cluster_nr=last_cluster_nr)

        #######################
        # Plot clustering information (silhouette score heatmap)
        #######################
        # if args.TODO == "train+analyze":
            # print(">>Plotting silhouette score heatmap")
            # plot_silhouette_heatmap(args, model_save_dir, silhouette_scores_per_epoch, epoch)

    train_writer.flush()
    test_writer.flush()
    train_writer.close()
    test_writer.close()


    # saving after training
    model_savename = f"model_epochs={epoch}.pt"
    print(f"Saving model after training under this name: {model_savename}")

    with open(os.path.join(model_save_dir, model_savename), "wb") as f:
        torch.save(model.state_dict(), f)

    #########################
    #       TEST MODEL      #
    #########################
    for task in args.tasks:
        if task in language_tasks:
            print(f"Evaluating LM performance on test dataset for task {task}")
            test_loss = evaluate(args, model, criterion, TRAINING_TASK_SPECS[task]["test_data"], task)
            print("=" * 89, flush=True)
            print(f"{task} | End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}")
            print("=" * 89, flush=True)


if __name__ == "__main__":
    main(args, model_save_dir)
