# Survival Kids RL

Reinforcement learning agent for **Survival Kids** (Game Boy Color, 1999) using [PyBoy](https://github.com/Baekalfen/PyBoy) and [Gymnasium](https://gymnasium.farama.org/).

A DQN agent learns to survive on a deserted island by managing HP, hunger, thirst, and fatigue while exploring the environment.

## Architecture

```
survival_kids_env.py   Gymnasium environment wrapping PyBoy
train.py               DQN training loop with logging
stream.py              Live MJPEG stream server for remote viewing
memory_scan.py         Memory address discovery tool
```

### Environment Details

| Property | Value |
|----------|-------|
| Observation | 144x160 grayscale screen (Game Boy native resolution) |
| Action space | 9 discrete: noop, A, B, Start, Select, Up, Down, Left, Right |
| Frame skip | 8 frames per agent step (button held for duration) |
| Reward | Composite: survival stats + exploration bonus + day survival |
| Termination | HP reaches 0 |
| Truncation | Configurable max steps (default 10,000) |

### Memory Map

Key game state addresses discovered via GameShark codes, TASVideos, and runtime scanning:

| Address | Description | Start Value |
|---------|-------------|-------------|
| `0xC5EE` | HP | 100 |
| `0xC5F0` | Hunger | 70 |
| `0xC5F2` | Thirst | 70 |
| `0xC5F4` | Fatigue | 50 |
| `0xD904` | Player tile X | 14 |
| `0xD905` | Player tile Y | 8 |
| `0xCB9C` | Day counter | 0 |
| `0xC19E` | Time of day | 0=morning, 1=noon, 2=evening, 3=night |

### Reward Function

```
+ hp_delta * 1.0          # Reward for gaining HP, penalty for losing
+ hunger_delta * 0.3      # Reward for eating
+ thirst_delta * 0.3      # Reward for drinking
+ 0.5                     # Per new tile visited (exploration bonus)
+ 10.0                    # Per day survived
- 0.5                     # When HP < 20 (critical health penalty)
- 0.3                     # When hunger < 10 or thirst < 10
- 50.0                    # On death (HP = 0)
- 0.01                    # Per step (encourages efficiency)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Provide the ROM

Place your `Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc` ROM file in a `roms/` directory:

```
roms/
  Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc
```

> ROMs are not included in this repository. You must provide your own legally obtained copy.

### 3. Train

```bash
# DQN training (default: 200 episodes)
python train.py

# Longer training run
python train.py --episodes 1000 --max-steps 5000

# Random baseline for comparison
python train.py --episodes 100 --random

# Watch the agent play (requires display)
python train.py --render --episodes 5
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 200 | Number of training episodes |
| `--max-steps` | 5000 | Max steps per episode |
| `--frame-skip` | 8 | Emulator frames per agent step |
| `--lr` | 1e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--render` | off | Show gameplay window |
| `--random` | off | Random agent (no learning) |

### Live Stream (Remote Training)

Watch the agent train in real-time from your browser — ideal for headless GPU instances (Vast.ai, Lambda, etc.):

```bash
# Start training with live stream on port 5555
python stream.py

# Custom port
python stream.py --port 8080 --episodes 1000
```

Then open `http://<your-server-ip>:5555` in your browser. You'll see:
- Live game footage (3x scaled, ~30 FPS MJPEG stream)
- Real-time HP / hunger / thirst / fatigue bars
- Training stats (episode, reward, epsilon, tiles explored)

On Vast.ai, make sure port 5555 is open (or use the port you specify with `--port`).

### Training Output

```
Ep    1 | Reward:     2.5 | Avg100:     2.5 | Steps:   300 | Explored:   12 | HP:100 Hun: 70 Thi: 70 Day:0 | eps:0.874
Ep   10 | Reward:    21.0 | Avg100:     9.3 | Steps:   300 | Explored:   49 | HP:100 Hun: 70 Thi: 70 Day:0 | eps:0.227
```

Logs are saved as JSONL to `logs/` and model checkpoints to `checkpoints/`.

## Using as a Gymnasium Environment

```python
import survival_kids_env
import gymnasium as gym

env = gym.make("SurvivalKids-v0", rom_path="roms/Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc")

obs, info = env.reset()
print(info["stats"])  # {'hp': 100, 'hunger': 70, 'thirst': 70, ...}

obs, reward, terminated, truncated, info = env.step(5)  # Press Up
```

## DQN Agent

The included DQN uses a small CNN architecture:

- 3 convolutional layers (16 -> 32 -> 32 filters)
- 256-unit fully connected hidden layer
- 9-output Q-value head
- Experience replay (50k buffer)
- Target network (updated every 1000 steps)
- Epsilon-greedy exploration (1.0 -> 0.05)

Falls back to random actions if PyTorch is not installed.

## Memory Scanner

To re-scan memory addresses or discover new ones:

```bash
python memory_scan.py
```

This boots the game, navigates to gameplay, and scans WRAM for position, stats, and game state addresses.
