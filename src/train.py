# src/train.py  — TensorBoard-free version
import os, numpy as np, torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.user_simulator import UserSimEnv
from models.dqn_agent import DQNAgent

def train(episodes=2000, log_interval=20):
    env = UserSimEnv()
    fdim, K = env.fdim, env.K
    state_dim = env.observation_space.shape[0]

    agent = DQNAgent(
        state_dim=state_dim,
        fdim=fdim,
        history_len=K,
        num_actions=200,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    item_feats = env.item_features

    for ep in range(1, episodes + 1):
        state, done, ep_reward, losses = env.reset(), False, 0.0, []

        while not done:
            # ----- candidate retrieval (top-200) -----
            pref_vec = state[:fdim]
            candidates = top_k_candidates(
                pref_vec, item_feats, k=200,
                exclude=[idx for idx in env.history[-10:]],
            )
            cand_map = {i: idx for i, idx in enumerate(candidates)}

            local_action = agent.select(state)          # 0-199
            env_action = cand_map[local_action]         # map back to true item id

            next_state, reward, done, _ = env.step(env_action)
            agent.store(state, local_action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_reward += reward

        if ep % log_interval == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(
                f"Ep {ep:4d}/{episodes}  "
                f"Total R: {ep_reward:5.2f}  "
                f"Avg Loss: {avg_loss:6.4f}  "
                f"Epsilon: {agent.epsilon:5.3f}"
            )

    os.makedirs("models", exist_ok=True)
    torch.save(agent.policy.state_dict(), "models/rl_diversity_dqn.pth")
    print("✅ Training finished — model saved to models/rl_diversity_dqn.pth")

# ---------- helper (same as before) ----------
import numpy as np
def top_k_candidates(pref_vec, item_feats, k=200, exclude=None, alpha=0.3):
    sims = item_feats @ pref_vec
    if exclude is not None:
        sims[exclude] = -1
    top_idx = np.argpartition(-sims, k)[:k]
    return top_idx[np.argsort(-sims[top_idx])]

if __name__ == "__main__":
    train()
