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
import pickle

from utils import build_training_tasks, set_seed, repackage_hidden, mask2d
from data import get_batch, get_splits
from model import Multitask_RNNModel, Yang19_RNNNet

from concurrent_analysis import main as concurrent_analysis
import neurogym as ngym

import torchtext
from torchtext.datasets import WikiText2

# loss visualization via TensorBoard https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# tensorboard --logdir=runs #if run from src directory!
from torch.utils.tensorboard import SummaryWriter
import re

_logger = logging.getLogger(__name__)

SAVE = False


def get_writers(model_save_dir):
    """ Define tensorboard writers """
    writer_name = re.sub("/", "+", model_save_dir)
    writer_name = "+".join(writer_name.split("+")[2:])
    train_writer = SummaryWriter(f"runs/{writer_name}/train_logs")
    test_writer = SummaryWriter(f"runs/{writer_name}/test_logs")
    return train_writer, test_writer


def get_pretrained_emb(TASK2DIM, lang_data_dict):
    """ Load pretrained GloVe embeddings """
    pretrained_emb_weights = {}

    for key in TASK2DIM.keys():
        if "yang" in key:
            continue
        elif any(x in key for x in ["wikitext", "penntreebank"]):
            # get pretrained embeddings
            pretrained_vectors = torchtext.vocab.GloVe()
            vocab_key = [x for x in TASK2DIM.keys() if key in x][0]
            vocab = lang_data_dict[vocab_key]["vocab"]
            # https://github.com/pytorch/text/issues/1350#issuecomment-875807109
            # vocab.get_itos() returns a list of strings (tokens), where the token at the i'th position is what you get from doing vocab[token]
            # get_vecs_by_tokens gets the pre-trained vector for each string when given a list of strings
            # therefore pretrained_embedding is a fully "aligned" embedding matrix
            pretrained_emb_weights[key] = pretrained_vectors.get_vecs_by_tokens(vocab.get_itos())

        elif any(x in key for x in ["de_wiki", "German"]):
            from pathlib import Path
            glove_path = Path('../data/de_wiki/de_glove_300d.txt').resolve()
            _logger.info(f"LOADING PRETRAINED EMBEDDINGS FROM FILE {glove_path}")
            with open(glove_path, "rb") as f:
                pretrained_vectors = pickle.load(f)
            pretrained_emb_weights[key] = pretrained_vectors

        elif "contrib." in key:
            pretrained_vectors = torchtext.vocab.GloVe()
            try:
                vocab_key = [x for x in TASK2DIM.keys() if
                             "wikitext" in x][0]  # FIXME still has to be generalized. Put in function etc, PTB
                vocab = lang_data_dict[vocab_key]["vocab"]
                pretrained_emb_weights[key] = pretrained_vectors.get_vecs_by_tokens(vocab.get_itos())
            except:
                vocab = get_splits(WikiText2, batch_size=20, return_only_vocab=True)
                pretrained_emb_weights[key] = pretrained_vectors.get_vecs_by_tokens(vocab.get_itos())
        else:
            raise NotImplementedError

    return pretrained_emb_weights, pretrained_vectors


def print_model_metainfo(model):
    """ Which parameters are in the model & which are being updated """
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
    print(f"EVALUATION LOSS: {total_loss}")
    return total_loss / (len(data_source) - 1)


def plot_loss(args, epoch, loss_data, model_save_dir):

    def figure_settings():
        figsize = (3.5, 2.5)
        rect = [0.25, 0.2, 0.6, 0.7]
        rect_color = [0.25, 0.15, 0.6, 0.05]
        rect_cb = [0.87, 0.2, 0.03, 0.7]
        fs = 6
        labelpad = 13
        return figsize, rect, rect_color, rect_cb, fs, labelpad

    # figure settings
    figsize, rect, rect_color, rect_cb, fs, labelpad = figure_settings()
    fig = plt.figure(figsize=figsize)

    # get loss dataframe
    loss_plot_df = pd.DataFrame(loss_data, columns=["step", "dataset", "loss"])
    loss_plot_df_wide = loss_plot_df.pivot(index='step', columns='dataset', values='loss')

    # plotting
    ax = sns.lineplot(data=loss_plot_df_wide)
    ax.set(xlabel='batch', ylabel='loss', title="Training loss")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.savefig(f'{model_save_dir}/seed={args.seed}_epoch={epoch}_training_loss.png', bbox_inches='tight',
                dpi=280)
    plt.show()


def plot_silhouette_heatmap(args, model_save_dir, silhouette_scores_per_epoch, epoch):
    """Plots a heatmap of the silhouette score per number of predefined clusters per epoch

    Args:
        silhouette_scores_per_epoch: Dictionary, shape [nr_epochs,nr_predefined_cluster(here:2-30)]

    Returns:
        Plots heatmap
    """
    MAX_NUMBER = 30 #FIXME turn into a parameter
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
    ax.set(xlabel='Epoch number', ylabel='Nr. of clusters', title=f'Silhouette scores | Epoch {epoch}')
    ax.collections[0].colorbar.set_label("Silhouette score")
    plt.savefig(f'{model_save_dir}/seed={args.seed}_epoch={epoch}_silhouette_heatmap.png',
                bbox_inches='tight', dpi=280)
    plt.show()


def main(args, model_save_dir):
    # define writers
    if SAVE:
        train_writer, test_writer = get_writers(model_save_dir)

    # set seed
    set_seed(args.seed, args.cuda)

    # set device
    device = torch.device("cuda" if args.cuda else "cpu")
    _logger.info(f"Running on this device: {device}")

    # build training tasks
    TASK2DIM, TASK2MODE, MODE2TASK, dataset_assignments, lang_data_dict, cog_dataset_dict = build_training_tasks(
        args.tasks, args, seq_len=args.seq_len)

    # get pretrained embeddings if needed
    pretrained_emb_weights = None
    if args.glove_emb:
        try:
            assert any(x in task for task in args.tasks for x in ["wikitext", "de_wiki", "penntreebank", "contrib."])
        except AssertionError as e:
            e.args += ("Running with pretrained embeddings, but it's not set up for the training tasks!",)
            raise e

        _logger.info("RUNNING WITH PRETRAINED GLOVE EMBEDDINGS!")
        pretrained_emb_weights, pretrained_vectors = get_pretrained_emb(TASK2DIM, lang_data_dict)

        #set hidden units to pretrained vector dimension (e.g., 300 for GloVe)!
        args.hidden_size, args.emsize = pretrained_vectors[0].shape[0], pretrained_vectors[0].shape[0]

    if args.sparse_model:
        assert args.CTRNN, "Only implemented for CTRNN right now!"
        print("Initializing model with anatomical mask on h2h weights!")
        assert math.sqrt(args.hidden_size).is_integer(), "Only implemented for square h2h masks right now"
        mask_sq_size = int(math.sqrt(args.hidden_size))
        h2h_mask2d = mask2d(N_x=mask_sq_size, N_y=mask_sq_size, cutoff=3, periodic=False)
        plt.imshow(h2h_mask2d, cmap='jet')
        plt.colorbar()
        plt.title('2d Mask')
        plt.show()
        # convert to pytorch tensor from numpy
        # FIXME turn into parameters
        h2h_mask2d = torch.from_numpy(h2h_mask2d)
        h2h_mask2d = h2h_mask2d.type(torch.float)
    else:
        h2h_mask2d = None

    if args.CTRNN:
        _logger.info(f'Running CTRNN model with nonlinearity {args.nonlinearity}')

    # build model
    if args.CTRNN:
        model = Yang19_RNNNet(TASK2DIM=TASK2DIM, MODE2TASK=MODE2TASK,
                              pretrained_emb_weights=pretrained_emb_weights,
                              hidden_size=args.hidden_size,
                              nonlinearity=args.nonlinearity, dt=100, mask=h2h_mask2d).to(device)
    else:
        model = Multitask_RNNModel(TASK2DIM=TASK2DIM, MODE2TASK=MODE2TASK, rnn_type=args.model,
                                   ninp=args.emsize, hidden_size=args.hidden_size,
                                   nlayers=args.nlayers, pretrained_emb_weights=pretrained_emb_weights,
                                    dropout=args.dropout, tie_weights=args.tied).to(device)

    # print model architecture & parameter groups that are being updated
    print_model_metainfo(model)

    # set criterion & define optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-3, amsgrad=True) #Added 2022-11-22 https://discuss.pytorch.org/t/loss-suddenly-increases-using-adam-optimizer/11338/9
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # define parameters before training
    best_val_loss = None
    last_cluster_nr = None

    tasks = []
    for x in args.tasks:
        if re.fullmatch("yang19(Dim32)?|khonaChandra22", x):
            tasks.extend(ngym.get_collection(x))
    try:
        tasks.extend([x for x in cog_dataset_dict.keys() if not x == "yang19"])
    except:
        print('no additional cog tasks found')

    cog_accuracies = [[] for _ in range(len(tasks))]
    acc_dict = {}
    silhouette_scores_per_epoch = []

    # run analysis on untrained model
    if args.TODO == "train+analyze":
        _logger.info(f"Running analysis before training! Saving under epoch 0!")
        last_cluster_nr = concurrent_analysis(args=args, model_save_dir=model_save_dir, model=model,
                                              cog_dataset_dict=cog_dataset_dict, epoch=0, cog_accuracies=cog_accuracies,
                                              acc_dict=acc_dict, silhouette_scores_per_epoch=silhouette_scores_per_epoch,
                                              TASK2DIM=TASK2DIM, TASK2MODE=TASK2MODE, lang_data_dict=lang_data_dict,
                                              last_cluster_nr=last_cluster_nr) #0 argument is epoch

    # determine epoch length & interleaving
    training_data_size_dict = {}
    datasets = [x[1] for x in dataset_assignments]

    if "lang" in datasets: #language datasets determine training time
        for key in lang_data_dict.keys():
            training_data_size_dict[key] = lang_data_dict[key]["train_data"].size(0) - 1
        if args.dry_run:
            for key in lang_data_dict.keys():
                training_data_size_dict[key] = 500
        epoch_length = min(training_data_size_dict.values())

    else:
        epoch_length = args.training_yang
        for key in cog_dataset_dict.keys():
            task_mode = int(key.split("_mode=")[1])
            training_data_size_dict[MODE2TASK[task_mode]] = epoch_length

    # If more than two different task categories, interleave during training
    if len(dataset_assignments) >= 2:
        interleave = True
    else:
        interleave = False

    # for plotting
    loss_data = []
    val_losses = {key: [] for key in TASK2MODE.keys() if "lang" in key}

    previous_val_ppl = np.inf
    ppl_diff_threshold = 1

    print("\n########################", flush=True)
    print("##### TRAIN MODEL ######", flush=True)
    print("########################\n", flush=True)

    ###################
    ### TRAIN MODEL
    ###################

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        running_loss = {key: 0. for key in
                        TASK2MODE.keys()}  # zero running loss at beginning of each epoch to prevent spikes!
        batch_counter = {key: 0 for key in TASK2MODE.keys()} # proportion of which dataset batches are from
        loss_batch_counter = {key: 0 for key in TASK2MODE.keys()}

        # Turn on training mode which enables dropout.
        model.train() #need this here because we're doing model evaluation after each epoch
        if not args.CTRNN:
            hidden = model.init_hidden(args.batch_size)

        assert model.training

        print(f"\n*NOTE*\nRunning with epoch length: {epoch_length} and bptt: {args.bptt}\n")

        for batch in range(0, epoch_length):

            # determine current task & mode
            mode = np.random.choice(list(MODE2TASK.keys()))

            curr_task = MODE2TASK[mode]
            i = batch_counter[curr_task] * args.bptt

            # get corresponding dataset batch
            if "lang" in curr_task:
                data, targets = get_batch(lang_data_dict[curr_task]["train_data"], i, args.bptt) #FIXME add HF data loading here
            else:
                data, targets = cog_dataset_dict[curr_task]["dataset_cog"]()
                if "contrib." in MODE2TASK[mode]:
                    data = torch.from_numpy(data).type(torch.long).to(device)
                else:
                    data = torch.from_numpy(data).type(torch.float).to(device)
                targets = torch.from_numpy(targets.flatten()).type(torch.long).to(device)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()

            # run model forward
            if args.CTRNN:
                output, _ = model(data, mode)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden, rnn_activity = model(data, hidden, mode)

            # run model backward
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # take optimizer step to update weights
            optimizer.step()

            if args.sparse_model:
                if batch == 0:
                    print("Applying anatomical mask on h2h weights!")
                # apply anatomical mask on h2h weights
                model.rnn.h2h.weight.data = model.rnn.h2h.weight.data * h2h_mask2d.to(device)

            # collect loss
            running_loss[MODE2TASK[mode]] += loss.item()

            batch_counter[curr_task] += 1
            loss_batch_counter[curr_task] += 1

            # model training inspection
            if batch % args.log_interval == 0 and batch > 0:

                for dat in running_loss.keys():
                    if interleave:
                        prop_factor = sum(loss_batch_counter.values()) / loss_batch_counter[dat]
                        cur_loss = (running_loss[dat] / args.log_interval) * prop_factor
                    else:
                        cur_loss = (running_loss[dat] / args.log_interval)

                    if SAVE:
                        train_writer.add_scalars(f"training loss:", {f"{dat}": cur_loss}, int((epoch - 1) * (epoch_length / args.bptt) + batch))
                    print(f"TRAINING LOSS: {cur_loss}")

                    this_step = int((epoch - 1) * (epoch_length / args.bptt) + batch)
                    loss_data.append([this_step, dat, cur_loss])

                    elapsed = time.time() - start_time
                    if "lang" in dat:
                        try:
                            print('{} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                                  'loss {:5.2f} | ppl {:8.2f}'.format(dat,
                                                                      epoch, batch,
                                                                      epoch_length * len(TASK2MODE.keys()) // args.bptt,
                                                                      elapsed * 1000 / args.log_interval, cur_loss,
                                                                      math.exp(cur_loss)), flush=True)
                        except: #to avoid math error
                            print('{} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                                  'loss {:5.2f}'.format(dat, epoch, batch,
                                                        epoch_length * len(TASK2MODE.keys()) // args.bptt,
                                                        elapsed * 1000 / args.log_interval, cur_loss), flush=True)
                    else:
                        try:
                            print('{} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                                  'loss {:5.2f} | ppl {:8.2f}'.format(dat,
                                                                      epoch, batch, epoch_length * len(TASK2MODE.keys()) // args.bptt,
                                                                      elapsed * 1000 / args.log_interval, cur_loss,
                                                                      math.exp(cur_loss)), flush=True)
                        except:
                            print('{} | epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                  'loss {:5.2f}'.format(dat,
                                                        epoch, batch, epoch_length * len(TASK2MODE.keys()) // args.bptt, args.lr,
                                                        elapsed * 1000 / args.log_interval, cur_loss,
                                                        ), flush=True)
                    running_loss[dat] = 0.0
                loss_batch_counter = {key: 0 for key in TASK2MODE.keys()}
                print("*****")
            start_time = time.time()

            if training_data_size_dict != {}:
                if "lang" in curr_task:
                    try:
                        assert batch_counter[curr_task] * args.bptt <= training_data_size_dict[curr_task]
                    except:
                        print(f"Finished training epoch for dataset {curr_task}!")
                        print(batch_counter[curr_task] * args.bptt)
                        print(batch_counter)
                        break

        #######################
        # Training proportion
        #######################
        print(f"\n PROPORTIONS OF BATCHES IN EPOCH {epoch}:")
        nr_batches = sum(batch_counter.values())
        for key, value in batch_counter.items():
            print(f"{value/nr_batches}: Training proportion for task {key}")

        #######################
        # Plot training losses
        #######################
        print("\n>>Plotting training losses")
        plot_loss(args, epoch, loss_data, model_save_dir)

        #######################
        # Evaluation loss
        #######################
        print("\n", flush=True)
        _logger.info("Evaluating LM performance on validation dataset")
        for mode in list(MODE2TASK.keys()):
            if "lang_" in MODE2TASK[mode]:
                val_loss = evaluate(args, model, criterion, lang_data_dict[MODE2TASK[mode]]["val_data"], mode)
                val_ppl = math.exp(val_loss)
                if SAVE:
                    test_writer.add_scalars(f"val_loss:", {f"{MODE2TASK[mode]}": val_loss}, int(epoch))
                print('-' * 89, flush=True)
                print('{} | end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(MODE2TASK[mode], epoch, (time.time() - start_time),
                                                 val_loss, math.exp(val_loss)), flush=True)
                print('-' * 89, flush=True)
                # # Save the model if the validation loss is the best we've seen so far.
                # if not best_val_loss or val_loss < best_val_loss:
                #     with open(os.path.join(model_save_dir, args.save), 'wb') as f:
                #         torch.save(model.state_dict(), f)
                #     best_val_loss = val_loss
                val_losses[MODE2TASK[mode]].append(val_loss)

                # if val_ppl < previous_val_ppl - ppl_diff_threshold:  # all good, continue
                #     previous_val_ppl = val_ppl
                # else:  # no more improvement --> stop
                #     print('Stopping training early!')
                #     # we could load the previous checkpoint here, but won't bother since usually the loss still decreases
                #     break
        if val_losses:
            fig = plt.figure()
            x_length = len(next(iter(val_losses.values())))
            for k, v in val_losses.items():
                plt.plot(range(1, len(v) + 1), v, '.-', label=k)
            plt.xticks(range(1, x_length + 1))
            plt.title('Evaluation loss')
            plt.legend()  # To draw legend
            plt.savefig(f'{model_save_dir}/seed={args.seed}_epoch={epoch}_validation_loss.png',
                        bbox_inches='tight', dpi=280)
            plt.show()

        # epoch training has finished here
        if args.TODO == "train+analyze":
            _logger.info(f"Running analysis for epoch {epoch}")
            last_cluster_nr = concurrent_analysis(args=args, model_save_dir=model_save_dir, model=model,
                                                  cog_dataset_dict=cog_dataset_dict, epoch=epoch,
                                                  cog_accuracies=cog_accuracies,
                                                  acc_dict=acc_dict,
                                                  silhouette_scores_per_epoch=silhouette_scores_per_epoch,
                                                  TASK2DIM=TASK2DIM, TASK2MODE=TASK2MODE, lang_data_dict=lang_data_dict,
                                                  last_cluster_nr=last_cluster_nr)

        #######################
        # Plot clustering information (silhouette score heatmap)
        #######################
        if args.TODO == "train+analyze":
            print(">>Plotting silhouette score heatmap")
            plot_silhouette_heatmap(args, model_save_dir, silhouette_scores_per_epoch, epoch)

    # writer.flush()
    # # writer.close()

    #Plot loss without initial peak
    plot_loss(args, epoch, loss_data, model_save_dir)

    # saving after training
    model_savename = f"model_epochs={epoch}.pt"
    print(f"Saving model after training under this name: {model_savename}")

    with open(os.path.join(model_save_dir, model_savename), 'wb') as f:
        torch.save(model.state_dict(), f)

    # Run on test data.
    for mode in list(MODE2TASK.keys()):
        if "lang" in MODE2TASK:
            test_loss = evaluate(args, model, criterion, lang_data_dict[MODE2TASK[mode]]["test_data"], mode)
            print('=' * 89, flush=True)
            print('{} | End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(MODE2TASK[mode],
                  test_loss, math.exp(test_loss)), flush=True)
            print('=' * 89, flush=True)


if __name__ == "__main__":
    main(args, model_save_dir)
