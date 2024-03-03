"""
Train a policy for //bindings/pydairlib/perceptive_locomotion/perception_learning:DrakeCassieEnv
"""
import argparse
import os

import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
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

def sample(sim_params):
    env = gym.make("DrakeCassie-v0",
        sim_params = sim_params,
    )
    input("Check...")
    #check_env(env)
    rate = 0.0
    env.simulator.set_target_realtime_rate(rate)
    max_steps = 500
    obs, _ = env.reset()
    input("Start..")
    for _ in range(int(max_steps)):
        # Plays a random policy.
        input("Action...")
        action = env.action_space.sample()
        input("Sampled...")
        obs, reward, terminated, truncated, info = env.step(action)
        #env.render()
        input("...?")
        if terminated or truncated:
            if args.debug:
                input("The environment will reset. Press Enter to continue...")
            obs, _ = env.reset()
def _main():
    bazel_chdir()
    sim_params = CassieFootstepControllerEnvironmentOptions()
    gym.envs.register(id="DrakeCassie-v0",
                    entry_point="pydairlib.perceptive_locomotion.perception_learning.DrakeCassieEnv:DrakeCassieEnv")  # noqa

    sample(sim_params)

if __name__ == '__main__':
    _main()