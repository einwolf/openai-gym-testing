import logging
import os

import ale_py
import gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecTransposeImage)

logger = logging.getLogger(__name__)

environment_name = "LunarLander-v2"

log_path = "logs"
tensorboard_log_path = "tensorboard_logs"


def make_output_dirs():
    pass

    # The normal output appears to use python logging framework
    # os.makedirs(log_path, exist_ok=True)
    # logging.basicConfig(filename=os.path.join(log_path, "lander_gym_log.txt"), encoding="utf-8", level=logging.DEBUG)


def main():
    """
    Test environment with random actions
    """
    make_output_dirs()

    env = gym.make(environment_name)
    # env = gym.make(environment_name, render_mode="rgb_array")

    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            # env.render(mode="human")
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # observation, reward, terminated, truncated, info = env.step(action)
            score += reward

            # gym 0.26
            # if terminated or truncated:
            #     observation, info = env.reset()
            #     done = True

        print(f"Episode: {episode} Score: {score}")

    env.close()

    env.action_space.sample()

    env.observation_space.sample()


if __name__ == "__main__":
    print(f"gym: {gym.__version__=}")
    print(f"ale_py: {ale_py.__version__}")
    main()
