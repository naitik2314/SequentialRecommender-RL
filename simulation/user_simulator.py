# simulation/user_simulator.py

import os
import numpy as np
import gym
from gym import spaces

class UserSimEnv(gym.Env):
    """
    A simple recommender environment:
      - State: flattened embeddings of the last K recommended items + current fatigue level
      - Action: an integer index of the next item to recommend
      - Reward: +1 if the user “engages” (similarity > threshold and not too fatigued), else –1
      - Done: when fatigue exceeds max_fatigue or max_steps reached
    """
    metadata = {'render.modes': []}

    def __init__(self,
                 feature_path: str = 'data/processed/item_features.npy',
                 history_len: int = 5,
                 max_fatigue: int = 5,
                 max_steps: int = 20,
                 sim_threshold: float = 0.3):
        # load item features
        self.item_features = np.load(feature_path)  # shape: [num_items, feat_dim]
        self.num_items, self.feat_dim = self.item_features.shape

        # env params
        self.K = history_len
        self.max_fatigue = max_fatigue
        self.max_steps = max_steps
        self.threshold = sim_threshold

        # Gym spaces
        self.action_space = spaces.Discrete(self.num_items)
        # obs = [K * feat_dim] + [fatigue_scalar]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.K * self.feat_dim + 1,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        # random user preference vector (unit‐normalized)
        pref = np.random.rand(self.feat_dim)
        self.user_pref = pref / np.linalg.norm(pref)

        self.history = []       # list of item indices
        self.fatigue = 0
        self.step_count = 0

        return self._get_state()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        vec = self.item_features[action]
        # cosine similarity
        if np.linalg.norm(vec) > 0:
            sim = float(np.dot(self.user_pref, vec) /
                        (np.linalg.norm(vec) * np.linalg.norm(self.user_pref)))
        else:
            sim = 0.0

        # reward & fatigue update
        if sim > self.threshold and self.fatigue < self.max_fatigue:
            reward = 1.0
            self.fatigue = 0
        else:
            reward = -1.0
            self.fatigue += 1

        self.history.append(action)
        self.step_count += 1

        done = (self.fatigue >= self.max_fatigue) or (self.step_count >= self.max_steps)
        return self._get_state(), reward, done, {}

    def _get_state(self):
        # build last-K history embeddings (most recent first), pad with zeros
        hist_vecs = []
        for i in range(self.K):
            if i < len(self.history):
                idx = self.history[-1 - i]
                hist_vecs.append(self.item_features[idx])
            else:
                hist_vecs.append(np.zeros(self.feat_dim))
        hist_flat = np.concatenate(hist_vecs, axis=0)

        # append fatigue as a scalar
        return np.concatenate([hist_flat, np.array([self.fatigue], dtype=np.float32)])

    def render(self, mode='human'):
        # could print last few history and fatigue
        print(f"Step {self.step_count} — fatigue: {self.fatigue}, last items: {self.history[-self.K:]}")

if __name__ == '__main__':
    # quick sanity check
    env = UserSimEnv()
    obs = env.reset()
    total_reward = 0
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, done, _ = env.step(a)
        total_reward += r
        if done:
            break
    print("Episode finished, total reward:", total_reward)
