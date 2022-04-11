from langpractice.preprocessors import sample_augmentation 
from gordongames.oracles import GordonOracle
import gym
import gordongames
import torch
import matplotlib.pyplot as plt

if __name__=="__main__":
    env_type = "gordongames-v1"
    env = gym.make(env_type)
    env.reset()
    oracle = GordonOracle(env_type)
    while True:
        actn = oracle(env)
        obs, rew, done, info = env.step(actn)
        print("mean before:", obs.mean())
        aug = sample_augmentation(torch.FloatTensor(obs[None]))
        print("mean after:", aug.mean())
        plt.imshow(aug.numpy().squeeze())
        plt.show()
        if done:
            obs = env.reset()
            aug = sample_augmentation(torch.FloatTensor(obs[None]))
            plt.imshow(aug.numpy().squeeze())
            plt.show()


