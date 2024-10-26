# prompt: gimme a CNN with two heads (one for policy, one for value).
# input is four frames, each is 210x160x3, and they first get stacked channel-wise.
# then, there are 3 CNN layers (3x3 conv kernel), followed by one FC layer
# (128 dims hidden). then, the two different output heads.

import matplotlib.pyplot as plt
from IPython.display import clear_output

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from train import CNNPolicyNet

# class CNNPolicyNet(nn.Module):
#     def __init__(self, numActions):
#         super(CNNPolicyNet, self).__init__()
#         # TODO: implement me

#     def forward(self, x):
#         pass
#         # TODO: implement me
#         # return logits  # i.e., preactivation scores before softmax

#     def predict(self, x):  # for compatibility with OpenAI Gym
#         #return F.softmax(self.forward(x)), None  # return second parameter for compatibility with OpenAI Gym
#         return F.softmax(torch.randn(6)), None  # for now, it's just a random policy over the 6 actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Play the game using the specified control policy
def render_env(env, policy, max_steps=500):
    obs = env.reset()
    for i in range(max_steps):
        with torch.no_grad():
            actionProbs, _ = policy.predict(obs, device)
        action = torch.multinomial(actionProbs, 1).item()  # randomly pick an action according to policy
        obs, reward, done, info = env.step([action])
        if done:
            break  # game over
    env.close()

# Create the Pacman environment
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=2, env_kwargs={'render_mode':"human"})
env = VecFrameStack(env, n_stack=4)

model = CNNPolicyNet(env.action_space.n).to(device)
# Load the model you trained
model.load_state_dict(torch.load("/home/ashd/WPI_Fall_24/dl/hw4/Q4/model.cpt"))

render_env(env, model, 5000)
