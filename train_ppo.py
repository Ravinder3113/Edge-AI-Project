# train_ppo.py
# Training script for PPO policy on CyberEnv (gymnasium)
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simulator import CyberEnv

def train(total_timesteps=20000, persona='novice'):
    # make_vec_env works with Gymnasium environments
    env = make_vec_env(lambda: CyberEnv(persona=persona), n_envs=2)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save('ppo_cyber')
    print('Saved policy to ppo_cyber.zip')

if __name__ == '__main__':
    train()
