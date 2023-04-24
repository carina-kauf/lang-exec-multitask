import argparse
import logging
import sys
import os
from trainer import main as train
# from analyze import main as analyze
import json
import re

_logger = logging.getLogger(__name__)


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
        slurm_id = os.getenv('SLURM_JOB_ID')

    if args.CTRNN:
        dir_name = f"CTRNN_{args.nonlinearity}"
    else:
        dir_name = f"{args.rnn_type}"
    _logger.info(f"*********** Running with {dir_name} model ***********")

    names_for_file = shorten_task_names(args.tasks)
    dir_name = f"{dir_name}_{'+'.join(names_for_file).rstrip('+')}"

    model_save_dir = os.path.join(model_save_dir, dir_name)
    os.makedirs(model_save_dir, exist_ok=True)

    if not os.getenv("USER") == "ckauf":
        name = f"nrlayers={args.nlayers}_epochs={args.epochs}_seed={args.seed}_gloveemb={args.glove_emb}" #TODO figure out which info I need here
    else:
        name = f"slurm_id={slurm_id}_optimizer={args.optimizer}"

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
    # args = HfArgumentParser(
    #     [TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments]
    # ).parse_args()

    parser = argparse.ArgumentParser()
    # TaskArguments
    parser.add_argument("--glove_emb", action="store_true", default=False, help="Use GloVe embeddings")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max_cluster_nr", type=int, default=20)
    parser.add_argument("--TODO", default="train+analyze")
    # CTRNNModelArguments
    parser.add_argument("--CTRNN", action="store_true")
    parser.add_argument("--nonlinearity", default='relu')
    parser.add_argument("--sparse_model", action="store_true")
    parser.add_argument("--sigma_rec", type=float, default=0.05)
    # RNNModelArguments
    parser.add_argument("--discrete_time_rnn", action="store_true")
    parser.add_argument("--rnn_type", default='RNN_RELU')
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tie_weights", action="store_true")
    # SharedModelArguments
    parser.add_argument("--hidden_size", type=int, default=600, help="Hidden size of RNN per layer")
    parser.add_argument("--emsize", type=int, default=300, help="Size of word embeddings")
    # TrainingArguments
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--optimizer", default='Adam')
    parser.add_argument("--weight_decay", default=0.01, type=float) #try 0.001, 0.0001
    parser.add_argument("--lr", type=float, default=1e-3) #try 1e-3, 1e-4
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss for imbalanced tasks")
    parser.add_argument("--dt", type=int, default=100, help="Time step for CTRNN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--bptt", type=int, default=35)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save", default='model.pt')
    parser.add_argument("--log_level", default='INFO')
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--training_yang", type=int, default=8000)
    args = parser.parse_args()

    if args.tasks is None:
        args.tasks = ["yang19"]
        print("No tasks specified, running with Yang et al. 2019 (ngym_usage) tasks!")

    if args.dry_run:
        print("Running in dry run mode! (fewer epochs/training steps, etc.)")
        args.log_interval = 20
        args.epochs = 2
        
    if args.tasks == ["yang19"]:
        print("Running with Yang et al. 2019 (ngym_usage) parameters! (batch_size=20, seq_len=100, epochs=1, training=40000, hidden_size=256, dt=100)")
        # replicate Yang et al. 2019 (ngym_usage parameters)
        args.batch_size = 20
        args.seq_len = 100
        args.epochs = 1
        if args.dry_run:
            args.training_yang = 50
        else:
            args.training_yang = 40000
        args.hidden_size = 256
        args.dt = 100

    assert not all([args.CTRNN, args.discrete_time_rnn]), "Can't run with both CTRNN and discrete time RNN models!"

    if not any([args.CTRNN, args.discrete_time_rnn]):
        args.CTRNN = True
        print("No model specified, running with CTRNN model by default!")

    if any(x in args.tasks for x in ["wikitext", "pennchar", "penntreebank", "de_wiki"]) and not args.glove_emb:
        print("Running with language tasks, using GloVe embeddings by default!")
        args.glove_emb = True

    main(args)
    print("Finished.")
