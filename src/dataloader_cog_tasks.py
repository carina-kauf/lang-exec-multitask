import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
import os
import glob
import matplotlib.pyplot as plt
import argparse

SHOW_SAMPLE_ENV = False

import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_cognitive_dataset(args, task_identifier):
    """Builds a dataset loader dataset and environment for a given cognitive tasks
    Args:
        args (argparse): arguments (using batch_size and seq_len)
        task_identifier (string): name of current cog task

    Returns:
        dataset_cog (neurogym dataset): dataset for cognitive tasks
            - dataset_cog.env: environment for cognitive tasks
            - dataset_cog.obs_size: observation size (input size to model)
            - dataset_cog.act_size: action size (output size for model)
        tasks (list): list of tasks in the collection/list with identifier for current task
    """
    # input verification
    assert isinstance(task_identifier, str), "training_tasks must be a string"

    # collections
    collections_path = os.path.abspath("../neurogym/neurogym/envs/collections/")
    collections = glob.glob(os.path.join(collections_path, "*"), recursive=False)
    collections = [path.split("/")[-1].split(".py")[0] for path in collections if "__" not in path]

    # check if task_identifier picks out a collection
    if task_identifier in collections:
        tasks = ngym.get_collection(task_identifier)
        add_task_index = True
    else:
        tasks = [task_identifier]
        add_task_index = False

    _logger.info(f"Building dataset/env for: {tasks}")

    # Environment specs
    kwargs = {'dt': args.dt}
    # Make cog environment
    envs = [gym.make(task, **kwargs) for task in tasks]
    if SHOW_SAMPLE_ENV:
        # Visualize the environment with 2 sample trials
        _logger.info(f"Visualizing environment '{envs[0].spec.name}' for 2 trials") #can also be indexed with -1 for difference between yang & khona
        _ = ngym.utils.plot_env(envs[0], num_trials=2)
        plt.show()
    schedule = RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule=schedule, env_input=add_task_index)
    # env_input is bool. If True, add scalar inputs indicating current environment.

    # Building iterable dataset
    dataset_cog = ngym.Dataset(env, batch_size=args.batch_size, seq_len=args.seq_len)

    return dataset_cog, tasks


if __name__ == "__main__":
    # test functionality
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='eval batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='sequence length for cog tasks')
    parser.add_argument('--dt', type=float, default=100)

    args = parser.parse_args()

    build_cognitive_dataset(args, task_identifier="yang19.go-v0")
    # build_cognitive_dataset(args, task_identifier="yang19.dm1-v0")
    # build_cognitive_dataset(args, task_identifier="yang19")
    # build_cognitive_dataset(args, task_identifier="khonaChandra22")