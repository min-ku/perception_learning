"""
Train a policy for //bindings/pydairlib/perceptive_locomotion/perception_learning:DrakeCassieEnv
"""
import argparse
import os
import numpy as np

import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
import torch as th
_full_sb3_available = True

from pydrake.systems.all import (
    Diagram,
    Context,
    Simulator,
    InputPort,
    OutputPort,
    DiagramBuilder,
    InputPortIndex,
    OutputPortIndex,
    ConstantVectorSource
)

from pydairlib.perceptive_locomotion.systems.cassie_footstep_controller_gym_environment import (
    CassieFootstepControllerEnvironmentOptions,
    CassieFootstepControllerEnvironment,
)


def bazel_chdir():
    """When using `bazel run`, the current working directory ("cwd") of the
    program is set to a deeply-nested runfiles directory, not the actual cwd.
    In case relative paths are given on the command line, we need to restore
    the original cwd so that those paths resolve correctly.
    """
    if 'BUILD_WORKSPACE_DIRECTORY' in os.environ:
        os.chdir(os.environ['BUILD_WORKSPACE_DIRECTORY'])

def _run_training(config, args):
    env_name = config["env_name"]
    num_env = config["num_workers"]
    log_dir = config["local_log_dir"]
    policy_type = config["policy_type"]
    total_timesteps = config["total_timesteps"]
    policy_kwargs = config["policy_kwargs"]
    eval_freq = config["model_save_freq"]
    sim_params = config["sim_params"]

    #https://github.com/Farama-Foundation/Gymnasium/pull/810
    #https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    #https://github.com/Farama-Foundation/Gymnasium/pull/810
    if not args.train_single_env:
        input("Starting...")
        sim_params.visualize = False
        env = make_vec_env(
                           env_name,
                           n_envs=num_env,
                           seed=0,
                           vec_env_cls=SubprocVecEnv,
                           env_kwargs={
                               'sim_params': sim_params,
                           })

    else:
        input("Starting...")
        sim_params.visualize = False
        if args.test:
            sim_params.visualize = True
        env = gym.make(env_name,
                       sim_params = sim_params,
                       )

        #check_env(env)

    if args.test:
        model = PPO(policy_type, env, n_steps=128, n_epochs=2,
                    batch_size=32, policy_kwargs=policy_kwargs)
    else:
        tensorboard_log = f"{log_dir}runs/test"
        model = PPO(
            policy_type, env, n_steps=int(2048/num_env), n_epochs=10,
            # In SB3, this is the mini-batch size.
            # https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst
            batch_size=64*num_env,
            #Use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
            #use_sde=True,
            verbose=1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs)
        print("Open tensorboard (optional) via "
              f"`tensorboard --logdir {tensorboard_log}` "
              "in another terminal.")

    # Separate evaluation env.
    sim_params.visualize = True
    eval_env = gym.make(env_name,
                        sim_params = sim_params,
                        )

    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir+f'eval_logs/test',
        log_path=log_dir+f'eval_logs/test',
        eval_freq=eval_freq,
        deterministic=True,
        render=False)

    input("Model learning...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    eval_env.close()

def _main():
    bazel_chdir()
    sim_params = CassieFootstepControllerEnvironmentOptions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_single_env', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_path', help="path to the logs directory.",
                        default="./rl/tmp/DrakeCassie/")
    args = parser.parse_args()

    if not _full_sb3_available:
        print("stable_baselines3 found, but was drake internal")
        return 0 if args.test else 1

    if args.test:
        num_env = 1
    elif args.train_single_env:
        num_env = 1
    else:
        num_env = 10

    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 2e6 if not args.test else 5,
        "env_name": "DrakeCassie-v0",
        "num_workers": num_env,
        "local_log_dir": args.log_path,
        "model_save_freq": 3e2,
        "policy_kwargs": {'activation_fn': th.nn.Tanh,        # activation function | th.nn.ReLU,
                          'net_arch': {'pi': [64, 64, 64], # policy and value networks
                                       'vf': [64, 64, 64]}},
        "sim_params" : sim_params
    }
    _run_training(config, args)

gym.envs.register(
        id="DrakeCassie-v0",
        entry_point="pydairlib.perceptive_locomotion.perception_learning.DrakeCassieEnv:DrakeCassieEnv")

if __name__ == '__main__':
    _main()
