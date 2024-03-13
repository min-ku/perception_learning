"""
Train a policy for //bindings/pydairlib/perceptive_locomotion/perception_learning:DrakeCassieEnv
"""
import argparse
import os
from os import path

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
    #input("Check...")
    #check_env(env)
    rate = 1.0
    env.simulator.set_target_realtime_rate(rate)
    max_steps = 500
    obs, _ = env.reset()
    input("Start..")
    for i in range(int(max_steps)):
        # Plays a random policy.
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        #env.render()
        if terminated or truncated:
            input("The environment will reset. Press Enter to continue...")
            obs, _ = env.reset()

def run_play(sim_params):
    env = gym.make("DrakeCassie-v0",
                    sim_params = sim_params,
                    )
    rate = 1.0
    env.simulator.set_target_realtime_rate(rate)
    max_steps = 1e4
    test_folder = "rl/tmp/DrakeCassie/eval_logs/test/vdes"#"rl/tmp/DrakeCassie/eval_logs/test/mean_ep_length_177"
    model_path = path.join(test_folder, 'best_model.zip')#'best_model_1mil_plus1.zip')
    model = PPO.load(model_path, env, verbose=1)#, tensorboard_log=log)
    
    obs, _ = env.reset()
    input("Start..")
    for _ in range(int(max_steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        #print(reward)
        if terminated or truncated:
            obs, _ = env.reset()
            
def _main():
    bazel_chdir()
    sim_params = CassieFootstepControllerEnvironmentOptions()
    gym.envs.register(
        id="DrakeCassie-v0",
        entry_point="pydairlib.perceptive_locomotion.perception_learning.DrakeCassieEnv:DrakeCassieEnv")  # noqa

    #sample(sim_params)
    run_play(sim_params)

if __name__ == '__main__':
    _main()