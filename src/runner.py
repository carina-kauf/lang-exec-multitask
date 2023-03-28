import argparse
import logging
import sys
import os
from trainer import main as train
# from analyze import main as analyze
import json
import re

_logger = logging.getLogger(__name__)

from transformers import HfArgumentParser
from args import TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments


def shorten_task_names(tasks):
    """Shorten task names for saving purposes"""
    if any(re.match("contrib", x) for x in tasks): #shorten names
            names_for_file = [re.sub("contrib.DelayMatchSample-MemProbe", "MemProbe", x) for x in tasks]
            names_for_file = [re.sub("contrib.DelayMatchSample", "DMS", x) for x in names_for_file]
            names_for_file = [re.sub("-v0", "", x) for x in names_for_file]
    else:
        names_for_file = tasks
    return names_for_file



def get_savedir(args):
    if not os.getenv("USER") == "ckauf":
        model_save_dir = "../results_local"
    else:
        model_save_dir = "../results_om"

    if args.CTRNN:
        dir_name = f"CTRNN_{args.nonlinearity}"
    else:
        dir_name = f"{args.model}"
    _logger.info(f"*********** Running with {dir_name} model ***********")

    names_for_file = shorten_task_names(args.tasks)
    dir_name = f"{dir_name}_{'+'.join(names_for_file).rstrip('+')}"

    model_save_dir = os.path.join(model_save_dir, dir_name)
    os.makedirs(model_save_dir, exist_ok=True)

    name = f"nrlayers={args.nlayers}_epochs={args.epochs}_seed={args.seed}_gloveemb={args.glove_emb}" #TODO figure out which info I need here

    if args.dry_run:
        name = f"dry_run_{name}"

    model_save_dir = os.path.join(model_save_dir, name)
    os.makedirs(model_save_dir, exist_ok=True)
    return model_save_dir


def main(args):
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    _logger.info("Running with args %s", vars(args))

    model_save_dir = get_savedir(args)

    with open(os.path.join(model_save_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.TODO == "train+analyze": #running concurrent analysis
        train(args, model_save_dir)
    elif args.TODO == "train_then_analyze":
        train(args, model_save_dir)
        # analyze(args, model_save_dir)
    elif args.TODO == "train":
        train(args, model_save_dir)
    # elif args.TODO == "analyze":
    #     analyze(args, model_save_dir)
    else:
        raise NotImplementedError("Unknown TODO! should be 'train+analyze', 'train_then_analyze', 'train' or 'analyze' !")


if __name__ == "__main__":
    args = HfArgumentParser(
        [TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments]
    ).parse_args()

    args.tasks = ["yang19", "wikitext"]

    args.glove_emb = True
    if args.glove_emb and args.hidden_size != 300:
        print(f"Using GloVe embeddings, changing hidden size from {args.hidden_size} to 300!")
        args.hidden_size = 300

    args.dry_run = True
    if args.dry_run:
        print("Running in dry run mode! (fewer epochs/training steps, etc.)")
        args.log_interval = 20
        
    if args.tasks == ["yang19"]:
        print("Running with Yang et al. 2019 (ngym_usage) parameters! (batch_size=20, seq_len=100, epochs=1, training=40000, hidden_size=256, dt=100)")
        # replicate Yang et al. 2019 (ngym_usage parameters)
        args.batch_size = 20
        args.seq_len = 100
        args.epochs = 1
        if args.dry_run:
            args.training_yang = 5000
        else:
            args.training_yang = 40000
        args.hidden_size = 256
        args.dt = 100

    print(f"Running with args: {args}")

    assert not all([args.CTRNN, args.discrete_time_rnn]), "Can't run with both CTRNN and discrete time RNN models!"
    assert any([args.CTRNN, args.discrete_time_rnn]), "Must specify either CTRNN or discrete time RNN model!"

    main(args)
    print("Done! :)")
