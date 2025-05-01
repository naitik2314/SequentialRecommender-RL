# src/train.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
import torch
from simulation.user_simulator import UserSimEnv
from models.dqn_agent import DQNAgent
from collections import deque
import numpy as np

def train(
    env,
    agent,
    num_episodes=500,
    max_steps_per_episode=None,
    log_interval=10,
    save_path='models/dqn.pth'
):
    rewards_history = []
    losses_history = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0
        ep_losses = []
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward
            steps += 1
            if max_steps_per_episode and steps >= max_steps_per_episode:
                break

        rewards_history.append(ep_reward)
        losses_history.append(np.mean(ep_losses) if ep_losses else 0.0)

        if ep % log_interval == 0:
            avg_r = np.mean(rewards_history[-log_interval:])
            avg_l = np.mean(losses_history[-log_interval:])
            print(f"Ep {ep}/{num_episodes} — avg_reward: {avg_r:.2f}, avg_loss: {avg_l:.4f}, ε: {agent.epsilon:.3f}")

    # save trained model
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    return rewards_history, losses_history

if __name__ == '__main__':
    # hyperparams
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    env = UserSimEnv()
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
    )

    train(env, agent)
