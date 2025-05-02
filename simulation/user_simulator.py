# simulation/user_simulator.py  (v2-fixed)

import os
import numpy as np
import gym
from gym import spaces


class UserSimEnv(gym.Env):
    """
    Diversity-aware user simulator.

    • State  = last-K item-embedding vectors (most-recent first) +
               fatigue scalar + step counter.
    • Action = integer index of an item to recommend.
    • Reward = +1 (delight) / 0 (meh) / –1 (fatigued or low-sim) with
               novelty bonus for diverse picks.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        feature_path: str = "data/processed/item_features.npy",
        history_len: int = 5,
        max_fatigue: int = 4,
        max_steps: int = 12,
        like_thresh: float = 0.6,
        meh_thresh: float = 0.35,
        div_bonus: float = 0.2,
        boredom_lambda: float = 0.15,
    ):
        # ---------- item feature matrix ----------
        self.item_features = np.load(feature_path)        # shape [N, F]
        self.n_items, self.fdim = self.item_features.shape

        # ---------- environment hyper-params ----------
        self.K = history_len
        self.max_fatigue = max_fatigue
        self.max_steps = max_steps
        self.like_thresh = like_thresh
        self.meh_thresh = meh_thresh
        self.div_bonus = div_bonus
        self.boredom_lambda = boredom_lambda

        # ---------- Gym API ----------
        self.action_space = spaces.Discrete(self.n_items)
        # flattened history (K * fdim) + fatigue + step
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.K * self.fdim + 2,),
            dtype=np.float32,
        )

        self.reset()

    # ------------------------------------------------------------------ #
    #                            PUBLIC API                               #
    # ------------------------------------------------------------------ #
    def reset(self):
        """Start a new session with a random user-preference vector."""
        pref = np.random.rand(self.fdim)
        self.pref = pref / np.linalg.norm(pref)

        self.history: list[int] = []        # store item indices only
        self.fatigue = 0
        self.t = 0
        return self._get_state()

    def step(self, a: int):
        assert self.action_space.contains(a), f"invalid action {a}"
        vec = self.item_features[a]

        # ---------- similarity & diversity ----------
        sim = float(
            np.dot(self.pref, vec) /
            (np.linalg.norm(self.pref) * np.linalg.norm(vec))
        )
        diversity = self._diversity(vec)

        # ---------- reward logic ----------
        if sim >= self.like_thresh and self.fatigue < self.max_fatigue:
            reward = 1.0 + (self.div_bonus * diversity)
            self.fatigue = max(0, self.fatigue - 1)       # refresh
        elif sim >= self.meh_thresh:
            reward = 0.0
            self.fatigue += self.boredom_lambda * (1 - diversity)
        else:
            reward = -1.0
            self.fatigue += 1

        # ---------- update session ----------
        self.history.append(a)       # store index, not vector!
        self.t += 1

        done = (self.fatigue >= self.max_fatigue) or (self.t >= self.max_steps)
        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        last_items = self.history[-self.K:]
        print(
            f"step={self.t} fatigue={self.fatigue:.2f} "
            f"recent_items={last_items}"
        )

    # ------------------------------------------------------------------ #
    #                           INTERNALS                                 #
    # ------------------------------------------------------------------ #
    def _get_state(self):
        """
        Build state vector:
          [vec_last, vec_{-1}, … padded zeros] + [fatigue, step_count]
        """
        hist_vecs = []
        for i in range(self.K):
            if i < len(self.history):
                idx = self.history[-1 - i]
                hist_vecs.append(self.item_features[idx])
            else:
                hist_vecs.append(np.zeros(self.fdim))
        hist_flat = np.concatenate(hist_vecs, axis=0)
        return np.concatenate([hist_flat, [self.fatigue, self.t]], axis=0)

    def _diversity(self, new_vec: np.ndarray) -> float:
        """1 – cosine similarity to last item (or 1.0 if no history)."""
        if not self.history:
            return 1.0
        last_vec = self.item_features[self.history[-1]]
        return 1.0 - float(
            np.dot(last_vec, new_vec) /
            (np.linalg.norm(last_vec) * np.linalg.norm(new_vec))
        )


# ---------------------------------------------------------------------- #
#                        Quick sanity check                              #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    env = UserSimEnv()
    s = env.reset()
    total = 0
    done = False
    while not done:
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        total += r
    print("Episode reward:", total)
