import os
import torch
import gym
import numpy as np
from langpractice.preprocessors import *
import time
from langpractice.utils.utils import try_key
import torch.nn.functional as F

try:
    import gordongames
except:
    print("gordongames not installed!")
try:
    import gym_snake
except:
    print("gym_snake not installed!")

CONDITIONALS = {
  "gordongames-v0": "align an item along the column of each "+\
                    "target",
  "gordongames-v1": "place the same number of items along a "+\
                    "single row as there are targets",
  "gordongames-v2": "place the same number of items along a "+\
                    "single row as there are targets",
  "gordongames-v3": "align an item along the column of each "+\
                    "target",
  "gordongames-v4": "press the pile object until the number "+\
                    "of items matches the number of flashed "+\
                    "targets",
  "gordongames-v5": "place the same number of items on the "+\
                    "grid as there are targets",
  "gordongames-v6": "place the same number of items on the "+\
                    "grid as there are targets",
  "gordongames-v7": "place the same number of items along a "+\
                    "single row as there are flashed targets",
  "gordongames-v8": "press the pile object until the number "+\
                    "of items matches the number of targets",
}

sent = " ".join(CONDITIONALS.values())
word_set = set(sent.split(" "))
CDTNL_LANG_SIZE = len(word_set)+1
word2idx = {"null": 0, **{k:i+1 for i,k in enumerate(word_set)}}
idx2word = {0: "null", **{i+1:k for i,k in enumerate(word_set)}}
IDX_CONDITIONALS = {
 k: [word2idx[w] for w in v.split(" ")] for k,v in CONDITIONALS.items()
}
TORCH_CONDITIONALS = {
 k: torch.LongTensor(v) for k,v in IDX_CONDITIONALS.items()
}

### TEST CONDITIONALS
##print("conditionals:")
##for k,v in CONDITIONALS.items():
##    print(k, "-", v)
##print("idxed condish:")
##for k,v in IDX_CONDITIONALS.items():
##    print(k, "-", " ".join([idx2word[vv] for vv in v]))
#######################

class SequentialEnvironment:
    """
    The goal of a sequential environment is to seamlessly integrate
    environments other than OpenAI gym into workflows that already have
    OpenAI gym environments. Unity environments are a prime example.
    Unity environments can have multiple observations and can have
    multiple games within a single environment. This wrapper attempts
    to generalize the API with any type of environment.
    """
    def __init__(self,
                 env_type,
                 preprocessor,
                 seed=time.time(),
                 **kwargs):
        """
        Args:
            env_type: str
                the name of the environment
            preprocessor: str
                the string name of the preprocessing function to be
                used on each of the observations
            seed: int
                the random seed for the environment
        """
        self.env_type = env_type
        self.preprocessor = globals()[preprocessor]
        self.seed = time.time() if seed is None else seed

        if "gordongames" in env_type or "nake" in env_type:
            kwargs["env_type"] = env_type
            self.env = gym.make(env_type, **kwargs)
        else:
            self.env = gym.make(env_type)
        self.env.seed(self.seed)
        self.is_gym = True
        self._raw_shape = self.env.reset().shape
        self._shape = self.reset().shape
        self.action_space = self.env.action_space
        if hasattr(self.action_space, "n"):
            self.actn_size = self.env.action_space.n
        else:
            self.actn_size = self.env.action_space.shape[0]

    @property
    def raw_shape(self):
        return self._raw_shape

    @property
    def shape(self):
        return self._shape

    def prep_obs(self, obs):
        """
        obs: list or ndarray
            the observation returned by the environment
        """
        if self.is_gym:
            prepped_obs = self.preprocessor(obs)
        else:
            prepped_obs = self.preprocessor(obs[0])
            # Handles the additional observations passed by the env
            if len(obs) > 1:
                prepped_obs = [prepped_obs, *obs[1:]]
        return prepped_obs

    def reset(self, n_targs=None):
        try:
            obs = self.env.reset(n_targs=n_targs)
        except:
            obs = self.env.reset()
        return self.prep_obs(obs)

    def step(self,action):
        """
        action: list, vector, or int
            the action to take in this step. type can vary depending
            on the environment type
        """
        obs,rew,done,info = self.env.step(action)
        return self.prep_obs(obs), rew, done, info

    def get_action(self, preds):
        """
        Action data types can vary from evnironment to environment.
        This function handles converting outputs from the model
        to actions of the appropriate form for the environment.

        preds: torch tensor (..., N)
            the outputs from the model
        """
        if self.is_gym:
            probs = F.softmax(preds, dim=-1)
            action = sample_action(probs.data)
            return int(action.item())
        else:
            preds = preds.squeeze().cpu().data.numpy()
            return preds

    def render(self):
        """
        Calls the environment's render function if one exists
        """
        if hasattr(self.env, "render"): self.env.render()

