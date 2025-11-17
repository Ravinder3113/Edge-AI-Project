# simulator.py
# Gymnasium-compatible simple user simulator environment.
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SimpleUserSimulator:
    def __init__(self, persona='novice', seed=None):
        self.persona = persona
        self.awareness = {'novice': 0.2, 'intermediate': 0.5, 'expert': 0.8}[persona]
        self.rng = random.Random(seed)

    def reset(self):
        self.awareness = max(0.0, min(1.0, self.awareness))
        state = self._build_state(last_intent=0, sentiment=0)
        # Gymnasium reset returns (obs, info)
        return state, {}

    def _build_state(self, last_intent=0, sentiment=0):
        vec = np.zeros(32, dtype=np.float32)
        vec[0] = self.awareness
        vec[1] = float(last_intent)
        vec[2] = float(sentiment)
        return vec

    def step(self, action):
        reward = 0.0
        info = {}
        # simple heuristics for demonstration
        if action == 0:
            delta = 0.05 + self.rng.random()*0.05
            reward += 0.5 if self.rng.random() < (0.6 if self.persona!='expert' else 0.2) else 0.0
        elif action == 1:
            delta = 0.03 if self.persona=='novice' else 0.06
            reward += 0.3
        elif action == 2:
            prob_correct = self.awareness*0.8 + 0.1
            correct = self.rng.random() < prob_correct
            if correct:
                reward += 3.0
                delta = 0.08
            else:
                reward += -0.5
                delta = 0.02
        elif action == 3:
            reward += 1.0
            delta = 0.06
        elif action == 4:
            prob_report = max(0.1, 1.0 - self.awareness)
            if self.rng.random() < prob_report:
                reward += 2.0
            else:
                reward += -0.2
            delta = 0.01
        elif action == 5:
            reward += -0.5 if self.awareness>0.7 else 0.5
            delta = 0.0
        else:
            reward += 0.1
            delta = 0.0

        self.awareness = max(0.0, min(1.0, self.awareness + delta))
        sentiment = 1 if self.rng.random() < 0.7 else 0
        next_state = self._build_state(last_intent=0, sentiment=sentiment)
        done = self.rng.random() < 0.05
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        return next_state, float(reward), bool(done), False, info

class CyberEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, persona='novice', seed=None):
        super().__init__()
        self.sim = SimpleUserSimulator(persona=persona, seed=seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.sim = SimpleUserSimulator(persona=self.sim.persona, seed=seed)
        obs, info = self.sim.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.sim.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

if __name__ == '__main__':
    env = CyberEnv('novice', seed=1)
    obs, _ = env.reset()
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        print('a', a, 'r', r, 'awareness', obs[0])
        if term or trunc:
            obs, _ = env.reset()
