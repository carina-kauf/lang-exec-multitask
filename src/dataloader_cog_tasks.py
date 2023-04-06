import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
import os
import glob
import matplotlib.pyplot as plt
import argparse

SHOW_SAMPLE_ENV = False

import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_cognitive_dataset(args, task_identifier, return_multienv=False):
    """Builds a dataset loader dataset and environment for a given cognitive tasks
    Args:
        args (argparse): arguments (using batch_size and seq_len)
        task_identifier (string): name of current cog task
        return_multienv (bool): if True, return MultiEnvs instead of ScheduleEnvs, used for variance analysis

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
        all_tasks = ngym.get_collection(task_identifier)
        add_task_index = True
    else:
        all_tasks = [task_identifier]
        add_task_index = False

    if return_multienv:
        _logger.info(f"Getting evaluation dataset/env for {all_tasks}")
    else:
        _logger.info(f"Building training dataset/env for: {all_tasks}")

    # Environment specs
    if "contrib." in task_identifier:
        if "German" in task_identifier:
            dataset_identifier = "de_wiki"
        elif "DelayMatchSampleChar" in task_identifier:
            dataset_identifier = "pennchar"
        else:
            dataset_identifier = "wikitext"
        kwargs = {'dt': args.dt, 'args': args, 'dataset_identifier': dataset_identifier}
        # Make cog environment
        envs = [gym.make(task, **kwargs) for task in all_tasks]
    else:
        if return_multienv:
            timing = {'fixation': ('constant', 500)}
            # TODO: note that we need the fixation here, because otherwise yang19.dm1-v0 has different sequence lengths
            kwargs = {'dt': 100, 'timing': timing}
        else:
            kwargs = {'dt': args.dt}
        # Make cog environment
        envs = [gym.make(task, **kwargs) for task in all_tasks]

    if SHOW_SAMPLE_ENV:
        # Visualize the environment with 2 sample trials
        _logger.info(f"Visualizing environment '{envs[0].spec.name}' for 2 trials") #can also be indexed with -1 for difference between yang & khona
        _ = ngym.utils.plot_env(envs[0], num_trials=2)
        plt.show()

    if return_multienv:
        cog_multi_env = MultiEnvs(envs, env_input=add_task_index)
        return cog_multi_env, all_tasks, collections

    else:
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=add_task_index)
        # env_input is bool. If True, add scalar inputs indicating current environment.

        # Building iterable dataset
        dataset_cog = ngym.Dataset(env, batch_size=args.batch_size, seq_len=args.seq_len)

        return dataset_cog, all_tasks


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
    build_cognitive_dataset(args, task_identifier="khonaChandra22")