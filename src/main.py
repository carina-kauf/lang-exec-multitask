import argparse
import logging
import sys
import os
from train import main as train
from analyze import main as analyze
import json
import re

_logger = logging.getLogger(__name__)


def get_savedir(args):
    if not os.getenv("USER") == "ckauf":
        model_save_dir = "../results_local"
    else:
        model_save_dir = "../results_om"

    if args.CTRNN:
        dir_name = "CTRNN"
    else:
        dir_name = f"{args.model}"
    _logger.info(f"*********** Running with {dir_name} model ***********")

    if any(re.match("contrib", x) for x in args.tasks): #shorten names
        names_for_file = [re.sub("contrib.DelayMatchSample-MemProbe", "MemProbe", x) for x in args.tasks]
        names_for_file = [re.sub("contrib.DelayMatchSample", "DMS", x) for x in names_for_file]
        names_for_file = [re.sub("-v0", "", x) for x in names_for_file]
    else:
        names_for_file = args.tasks

    dir_name = f"{dir_name}_{'+'.join(names_for_file).rstrip('+')}"

    model_save_dir = os.path.join(model_save_dir, dir_name)
    os.makedirs(model_save_dir, exist_ok=True)

    if os.getenv("USER") == "ckauf":
        try:
            name = f"{os.environ['SLURM_JOB_ID']}_nrlayers={args.nlayers}_epochs={args.epochs}_seed={args.seed}_gloveemb={args.glove_emb}"
        except: #local remote debugging
            name = f"nrlayers={args.nlayers}_epochs={args.epochs}_seed={args.seed}_gloveemb={args.glove_emb}"
    else:
        name = f"nrlayers={args.nlayers}_epochs={args.epochs}_seed={args.seed}_gloveemb={args.glove_emb}"

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
        analyze(args, model_save_dir)
    elif args.TODO == "train":
        train(args, model_save_dir)
    elif args.TODO == "analyze":
        analyze(args, model_save_dir)
    else:
        raise NotImplementedError("Unknown TODO! should be 'train+analyze', 'train_then_analyze', 'train' or 'analyze' !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitask RNN Model')
    parser.add_argument('--CTRNN', action='store_true',
                        help='If flag is set, running with continuous-time CTRNN, else running with discrete-time RNN')
    parser.add_argument('--nonlinearity', type=str, default='relu',
                        help='activation function used in CTRNN model, can be one of relu, softplus, ...')
    parser.add_argument('--sparse_model', action='store_true',
                        help='If flag is set, we apply a mask to the h2h layer of the CTRNN a la Khona et al. 2022')
    parser.add_argument('--model', type=str, default='',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, RNN_SOFTPLUS)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=300, #FIXME 256 = 16*16 #Change to 300 if GloVe
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1e-3, # also tested with 5e-5
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='eval batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='sequence length for cog tasks')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=24,  # 1111
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')

    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--dry_run', action='store_true',
                        help='verify the code and the model')

    parser.add_argument('--tasks', nargs='+', help='List of tasks', required=True)
    parser.add_argument('--glove_emb', action='store_true',
                        help='use pretrained GloVe embeddings') #make required so I don't forget

    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='report interval')

    parser.add_argument('--TODO', type=str, default="train+analyze", metavar='N',
                        help='train+analyze | train | analyze  | train_then_analyze')

    parser.add_argument('--training_yang', type=int, default=5000, #used to be int(40000/5) so 5 epochs make 40000 iterations as in codebase
                        help='number of training iterations')

    parser.add_argument('--continuous_cluster', action='store_true',
                        help='whether or not number of clusters should increase monotonically')

    args = parser.parse_args()

    if args.dry_run:
        args.training_yang = 100
        args.log_interval = 20

    main(args)
    print("Done! :)")
