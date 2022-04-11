from langpractice.preprocessing import sample_augmentation 
from gordongames.oracles import GordonOracle
import gym
import gordongames

if __name__=="__main__":
    env_type = "gordongames-v1"
    env = gym.make(env_type)
    env.reset()
    oracle = GordonOracle(env_type)
    while True:
        actn = oracle(env)
        obs, rew, done, info = env.step(actn)
        aug = sample_augmentation(obs[None].transpose(0,3,1,2))
        plt.imshow(aug.transpose(0,2,3,1))
        plt.show()

