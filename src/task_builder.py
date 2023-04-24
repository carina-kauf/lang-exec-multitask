from dataloader_lang_tasks import build_text_dataset
from dataloader_cog_tasks import build_cognitive_dataset
from utils_general import find_matches
from transformers import HfArgumentParser
from args import TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments
import re


def build_training_tasks(args):
    """
    # TODO
    """

    task2dataset = {
        "wikitext(103)?|penntreebank|[a-z]{2}_wiki|pennchar(_perturbed)?": "lang",
        "yang19.*|contrib.*|khonaChandra22.*": "cog"
    }

    TRAINING_TASK_SPECS = {}

    for i, task in enumerate(args.tasks):
        print(f"*************\nBUILDING DATASET FOR TASK: {task}\n*************\n")
        dataset = find_matches(task2dataset, task)
        taskname = task
        if '.' in task:
            # replace '.' with '-' for contrib tasks to avoid error 'module name can\'t contain "."
            taskname = re.sub(r'\.', '-', task)

        if dataset == "lang":
            vocab, vocab_size, train_data, val_data, test_data, pretrained_emb = build_text_dataset(args, task)
            if pretrained_emb is not None:
                using_pretrained_emb = True
            else:
                using_pretrained_emb = False
                
            TRAINING_TASK_SPECS[task] = {
                "taskname": taskname,
                "dataset": dataset,
                "vocab": vocab,
                "vocab_size": vocab_size,
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "pretrained_emb_weights": pretrained_emb,
                "using_pretrained_emb": using_pretrained_emb,
                "input_size": vocab_size,
                "output_size": vocab_size}

        elif dataset == "cog":
            dataset_cog, tasks = build_cognitive_dataset(args, task)
            env = dataset_cog.env
            try:
                ob_size = env.observation_space.shape[0]
            except:
                ob_size = env.vocab_size  # TODO added for verbal WM tasks
            act_size = env.action_space.n

            TRAINING_TASK_SPECS[task] = {
                "taskname": taskname,
                "full_task_list": tasks,
                "dataset": dataset,
                "dataset_cog": dataset_cog,
                "env": env,
                "ob_size": ob_size,
                "act_size": act_size,
                "pretrained_emb_weights": None,
                "using_pretrained_emb": False,
                "input_size": ob_size,
                "output_size": act_size}

            if task.startswith("contrib."):
                TRAINING_TASK_SPECS[task]["pretrained_emb_weights"] = env.aligned_embeddings
                TRAINING_TASK_SPECS[task]["vocab_size"] = env.vocab_size
                TRAINING_TASK_SPECS[task]["input_size"] = env.vocab_size
                TRAINING_TASK_SPECS[task]["using_pretrained_emb"] = True

    print("*" * 30)
    print("*Language task specs*")
    for key, value in TRAINING_TASK_SPECS.items():
        if value["dataset"] == "lang":
            print(
                f"{key} | vocab_size : {value['vocab_size']} | train_data.shape : {value['train_data'].shape} |"
                f" val_data.shape : {value['val_data'].shape} | test_data.shape : {value['test_data'].shape} |"
                f" using_pretrained_emb : {value['using_pretrained_emb']}")

    print("*Executive function tasks*")
    for key, value in TRAINING_TASK_SPECS.items():
        if value["dataset"] == "cog":
            if key.startswith("contrib."):
                print(f"{key} | ob_size : {value['ob_size']} | act_size : {value['act_size']} |"
                      f" vocab_size : {value['vocab_size']} | using_pretrained_emb : {value['using_pretrained_emb']}")
            else:
                print(f"{key} | ob_size : {value['ob_size']} | act_size : {value['act_size']}")
    print("*" * 30)

    return TRAINING_TASK_SPECS


if __name__ == "__main__":
    args = HfArgumentParser(
        [TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments]
    ).parse_args()

    args.tasks = ["yang19", "wikitext", "contrib.DelayMatchSampleWord-v0"]
    args.glove_emb = True

    build_training_tasks(args)