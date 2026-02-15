"""
RL training script for Survival Kids using a simple DQN agent.

Falls back to tabular Q-learning if torch is unavailable.
Usage:
    python3 train.py                    # Train with default settings
    python3 train.py --episodes 500     # Custom episode count
    python3 train.py --render           # Watch the agent play (slow)
    python3 train.py --random           # Random baseline (no learning)
"""

import argparse
import os
import time
import json
import numpy as np
from collections import deque

# Import our environment (also registers it with gymnasium)
import survival_kids_env  # noqa: F401
import gymnasium as gym

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# DQN components (only used if torch is available)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class DQN(nn.Module):
        """Small CNN for processing Game Boy screen frames."""

        def __init__(self, n_actions):
            super().__init__()
            # Input: (batch, 1, 144, 160) grayscale
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            # Calculate flat size
            self._flat_size = self._get_flat_size()
            self.head = nn.Sequential(
                nn.Linear(self._flat_size, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions),
            )

        def _get_flat_size(self):
            dummy = torch.zeros(1, 1, 144, 160)
            return self.features(dummy).reshape(1, -1).size(1)

        def forward(self, x):
            # x: (batch, H, W, 1) uint8 -> (batch, 1, H, W) float
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            features = self.features(x).reshape(x.size(0), -1)
            return self.head(features)


    class ReplayBuffer:
        def __init__(self, capacity=50000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            states, actions, rewards, next_states, dones = zip(*batch)
            return (
                torch.from_numpy(np.array(states)),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.from_numpy(np.array(next_states)),
                torch.tensor(dones, dtype=torch.float32),
            )

        def __len__(self):
            return len(self.buffer)


    class DQNAgent:
        def __init__(self, n_actions, lr=1e-4, gamma=0.99, batch_size=32):
            self.n_actions = n_actions
            self.gamma = gamma
            self.batch_size = batch_size

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQN(n_actions).to(self.device)
            self.target_net = DQN(n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = ReplayBuffer()

            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.9995
            self.steps = 0
            self.target_update_freq = 1000

        def select_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()

        def train_step(self):
            if len(self.memory) < self.batch_size:
                return None

            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size
            )
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            # Current Q values
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            # Target Q values
            with torch.no_grad():
                next_q = self.target_net(next_states).max(dim=1)[0]
                target = rewards + self.gamma * next_q * (1 - dones)

            loss = nn.functional.smooth_l1_loss(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            # Update target network
            self.steps += 1
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            return loss.item()

        def save(self, path):
            torch.save({
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            }, path)

        def load(self, path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.steps = checkpoint["steps"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    rom_path = os.path.join(
        os.path.dirname(__file__),
        "roms",
        "Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc",
    )

    render_mode = "human" if args.render else None
    env = gym.make(
        "SurvivalKids-v0",
        rom_path=rom_path,
        render_mode=render_mode,
        frames_per_step=args.frame_skip,
        max_steps=args.max_steps,
    )

    n_actions = env.action_space.n
    use_dqn = HAS_TORCH and not args.random

    if use_dqn:
        agent = DQNAgent(n_actions, lr=args.lr, gamma=args.gamma)
        print(f"Using DQN agent on {agent.device}")
    else:
        if args.random:
            print("Running random baseline")
        else:
            print("PyTorch not found — running random baseline")
            print("Install torch for DQN training: pip3 install torch")

    # Logging and checkpoints
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_file = os.path.join("logs", f"training_{int(time.time())}.jsonl")
    episode_rewards = deque(maxlen=100)

    print(f"\nTraining for {args.episodes} episodes")
    print(f"Frame skip: {args.frame_skip}, Max steps: {args.max_steps}")
    print(f"Logging to: {log_file}\n")

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        losses = []

        while True:
            if use_dqn:
                action = agent.select_action(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if use_dqn:
                agent.memory.push(obs, action, reward, next_obs, terminated)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            obs = next_obs

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards)

        # Log episode
        log_entry = {
            "episode": episode,
            "reward": round(total_reward, 2),
            "avg_reward_100": round(avg_reward, 2),
            "steps": steps,
            "positions_visited": info.get("positions_visited", 0),
            "stats": info.get("stats", {}),
        }
        if use_dqn:
            log_entry["epsilon"] = round(agent.epsilon, 4)
            log_entry["avg_loss"] = round(np.mean(losses), 4) if losses else 0

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Print progress
        stats = info.get("stats", {})
        status = (
            f"Ep {episode:4d} | "
            f"Reward: {total_reward:7.1f} | "
            f"Avg100: {avg_reward:7.1f} | "
            f"Steps: {steps:5d} | "
            f"Explored: {info.get('positions_visited', 0):4d} | "
            f"HP:{stats.get('hp', '?'):>3} Hun:{stats.get('hunger', '?'):>3} "
            f"Thi:{stats.get('thirst', '?'):>3} Day:{stats.get('day', '?')}"
        )
        if use_dqn:
            status += f" | eps:{agent.epsilon:.3f}"
        print(status)

        # Save checkpoint every 50 episodes
        if use_dqn and episode % 50 == 0:
            agent.save(f"checkpoints/dqn_ep{episode}.pt")
            print(f"  -> Checkpoint saved: checkpoints/dqn_ep{episode}.pt")

    env.close()
    print(f"\nTraining complete. Logs: {log_file}")

    if use_dqn:
        agent.save("checkpoints/dqn_final.pt")
        print("Final model saved: checkpoints/dqn_final.pt")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Survival Kids")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode")
    parser.add_argument("--frame-skip", type=int, default=8, help="Frames per agent step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--render", action="store_true", help="Render gameplay")
    parser.add_argument("--random", action="store_true", help="Random baseline only")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
