# Survival Kids RL

Teaching a computer to play **Survival Kids** (Game Boy Color, 1999) using reinforcement learning.

An AI agent starts with zero knowledge of the game. Through trial and error — dying, exploring, and slowly figuring out what keeps it alive — it learns to survive on a deserted island.

## What Is This Project?

This project connects three things together:

1. **A Game Boy emulator** ([PyBoy](https://github.com/Baekalfen/PyBoy)) — runs the actual Survival Kids game in software, no physical Game Boy needed
2. **A training environment** ([Gymnasium](https://gymnasium.farama.org/)) — a standard interface that lets an AI agent "play" the game by pressing buttons and seeing the screen
3. **A learning algorithm** (DQN) — the AI brain that figures out which buttons to press based on what it sees

The agent sees the game screen as a grid of pixels, picks a button to press, and gets a score (reward) based on whether that action helped it survive. Over thousands of attempts, it gets better.

---

## Key Concepts Explained

### Reinforcement Learning (RL)

A type of machine learning where an agent learns by **doing things and seeing what happens**. There's no teacher giving it the right answers. Instead:
- It tries an action (like pressing "right")
- The game changes (the character moves)
- It gets a reward (positive if it helped, negative if it hurt)
- Over time, it learns which actions lead to good outcomes

Think of it like training a dog: you don't explain the rules, you just reward good behavior and the dog figures it out.

### Agent

The "player" — but instead of a human, it's a program making decisions. Our agent looks at the Game Boy screen and decides which button to press next.

### Environment

The "world" the agent lives in. In our case, it's the Survival Kids game running inside a Game Boy emulator. The environment:
- Shows the agent what's happening (the screen)
- Accepts the agent's actions (button presses)
- Returns a reward score after each action

### Episode

One complete playthrough — from waking up on the beach to either dying or hitting the step limit. The agent plays hundreds or thousands of episodes during training. Each episode starts fresh from the same save point.

### Observation

What the agent sees. We give it the raw Game Boy screen: a 144x160 pixel grayscale image. It has to figure out everything (where it is, what's around it, what's dangerous) from those pixels alone.

### Action Space

The set of all possible moves. The Game Boy has 8 buttons, plus doing nothing:

| Action | Button | What It Does In Game |
|--------|--------|---------------------|
| 0 | Nothing | Stand still |
| 1 | A | Interact, pick up items, confirm |
| 2 | B | Cancel, run |
| 3 | Start | Open menu |
| 4 | Select | Switch items |
| 5 | Up | Walk up |
| 6 | Down | Walk down |
| 7 | Left | Walk left |
| 8 | Right | Walk right |

### Reward

A number that tells the agent "that was good" (positive) or "that was bad" (negative) after each action. Our reward system:

| Event | Reward | Why |
|-------|--------|-----|
| Visit a new tile | +0.5 | Encourages exploring the island |
| Gain HP | +1.0 per point | Eating food or resting is good |
| Eat (hunger goes up) | +0.3 per point | Finding food is good |
| Drink (thirst goes up) | +0.3 per point | Finding water is good |
| Survive a full day | +10.0 | Big bonus for lasting a whole day |
| HP drops below 20 | -0.5 | Warning: you're about to die |
| Hunger or thirst below 10 | -0.3 | Warning: find food/water |
| Die (HP = 0) | -50.0 | Large penalty for dying |
| Each step taken | -0.01 | Small nudge to not waste time |

The agent doesn't know these rules up front. It discovers them by playing. Over time, it learns that walking into new areas and eating food lead to higher scores.

### DQN (Deep Q-Network)

The specific algorithm our agent uses to learn. Breaking it down:

- **Q-Network**: A neural network that looks at the game screen and predicts "how good is each button press right now?" It outputs 9 numbers (one per action), and the agent picks the highest one.
- **Deep**: The network uses convolutional layers (good at processing images) to understand the pixel grid of the Game Boy screen.
- **Experience Replay**: The agent stores its memories (what it saw, what it did, what happened) in a buffer of 50,000 past experiences. During training, it randomly samples old memories to learn from. This prevents it from only learning from the most recent thing that happened.
- **Target Network**: A second copy of the brain that updates slowly. This keeps training stable — like having a calm advisor that doesn't overreact to every new experience.
- **Epsilon-Greedy**: The agent starts by pressing random buttons 100% of the time (exploring). Over training, it gradually shifts to using its learned strategy, eventually only pressing random buttons 5% of the time. This balance between trying new things (exploration) and using what works (exploitation) is critical.

### Frame Skip

The Game Boy runs at 60 frames per second, but the agent doesn't need to make a decision every single frame. With a frame skip of 8, the agent presses a button, holds it for 8 frames (~0.13 seconds of game time), then decides again. This speeds up training and makes the agent's actions more meaningful.

### Memory Map

Game Boy games store all their data (health, position, items) in specific memory addresses — numbered slots in the console's RAM. We found these by:
1. Looking up GameShark cheat codes (which modify specific addresses)
2. Checking the TASVideos speedrunning community's notes
3. Running the game and scanning for bytes that change when the player moves, takes damage, etc.

These addresses let us read the game's internal state directly instead of trying to figure it out from pixels:

| Address | What It Stores | Starting Value |
|---------|---------------|----------------|
| `0xC5EE` | HP (health points) | 100 |
| `0xC5F0` | Hunger level | 70 |
| `0xC5F2` | Thirst level | 70 |
| `0xC5F4` | Fatigue level | 50 |
| `0xD904` | Player X position (tile) | 14 |
| `0xD905` | Player Y position (tile) | 8 |
| `0xCB9C` | Day counter | 0 |
| `0xC19E` | Time of day | 0=morning, 1=noon, 2=evening, 3=night |

The agent sees the screen pixels to make decisions, but we read these memory addresses to calculate the reward.

### Gymnasium

A standard interface (created by the Farama Foundation) that lets any RL algorithm talk to any game/simulation. It defines a simple loop:

```
observation, info = env.reset()        # Start a new episode
observation, reward, done, truncated, info = env.step(action)  # Take one action
```

Every RL environment uses this same pattern, so you can swap in different games or different agents without rewriting everything.

---

## Project Files

| File | Purpose |
|------|---------|
| `survival_kids_env.py` | The Gymnasium environment — connects PyBoy to the RL training loop, reads game memory for rewards |
| `train.py` | Training script — runs the DQN agent through episodes, logs results, saves checkpoints |
| `stream.py` | Live viewer — streams the game screen to your browser so you can watch training remotely |
| `memory_scan.py` | Discovery tool — scans the game's RAM to find useful memory addresses |
| `requirements.txt` | Python packages needed to run the project |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Provide the ROM

Place your Survival Kids ROM file in a `roms/` directory:

```
roms/
  Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc
```

> ROMs are not included in this repository. You must provide your own legally obtained copy.

### 3. Train

```bash
# Start training (default: 200 episodes)
python train.py

# Longer training run
python train.py --episodes 1000 --max-steps 5000

# Random baseline (no learning, for comparison)
python train.py --episodes 100 --random

# Watch the agent play (requires a display)
python train.py --render --episodes 5
```

### Options

| Flag | Default | What It Does |
|------|---------|-------------|
| `--episodes` | 200 | How many playthroughs to train on |
| `--max-steps` | 5000 | Max button presses per episode before restarting |
| `--frame-skip` | 8 | How many game frames per button press |
| `--lr` | 0.0001 | Learning rate — how big of a step the AI takes when updating its brain |
| `--gamma` | 0.99 | Discount factor — how much the AI cares about future rewards vs immediate ones |
| `--render` | off | Show the game window while training |
| `--random` | off | Use random button presses instead of the DQN (useful as a baseline) |

### Live Stream (Remote Training)

If you're training on a cloud GPU (like Vast.ai), you can watch the agent play from your browser:

```bash
python stream.py --port 5555 --episodes 1000
```

Then open `http://<your-server-ip>:5555` in your browser. You'll see:
- Live game footage scaled up 3x
- HP, hunger, thirst, and fatigue bars updating in real time
- Training progress (episode count, reward, exploration stats)

---

## Training Output

As the agent trains, you'll see output like this:

```
Ep    1 | Reward:     2.5 | Avg100:     2.5 | Steps:   300 | Explored:   12 | HP:100 Hun: 70 Thi: 70 Day:0 | eps:0.874
Ep   10 | Reward:    21.0 | Avg100:     9.3 | Steps:   300 | Explored:   49 | HP:100 Hun: 70 Thi: 70 Day:0 | eps:0.227
```

What each column means:
- **Ep**: Episode number (which playthrough)
- **Reward**: Total reward earned this episode
- **Avg100**: Average reward over the last 100 episodes (the main metric to watch — should trend upward)
- **Steps**: How many button presses this episode lasted
- **Explored**: Number of unique map tiles visited
- **HP/Hun/Thi**: Health, hunger, and thirst at the end of the episode
- **Day**: How many in-game days the agent survived
- **eps**: Epsilon — the chance of pressing a random button (starts at 1.0, decays to 0.05)

Detailed logs are saved as JSONL files in `logs/`. Model checkpoints are saved to `checkpoints/` every 50 episodes.

---

## Using as a Gymnasium Environment

You can use the environment directly in your own code:

```python
import survival_kids_env
import gymnasium as gym

# Create the environment
env = gym.make("SurvivalKids-v0")

# Start a new episode
obs, info = env.reset()
print(info["stats"])  # {'hp': 100, 'hunger': 70, 'thirst': 70, ...}

# Press the Up button
obs, reward, terminated, truncated, info = env.step(5)

# obs = what the screen looks like now (144x160 grayscale image)
# reward = how good that action was
# terminated = True if the character died
# truncated = True if we hit the step limit
```

---

## How the DQN Agent Works

The agent's brain is a small convolutional neural network (CNN) — the same type used in image recognition. It processes the Game Boy screen through three layers of filters that detect patterns (edges, shapes, objects), then outputs a score for each of the 9 possible buttons.

```
Game Boy Screen (144x160 pixels)
    |
    v
Conv Layer 1: 16 filters (detects basic patterns like edges)
    |
Conv Layer 2: 32 filters (detects shapes and objects)
    |
Conv Layer 3: 32 filters (detects complex game features)
    |
Fully Connected: 256 neurons (combines everything)
    |
    v
9 Q-values (one score per button — pick the highest)
```

Training loop:
1. Agent sees the screen
2. Picks a button (random early on, learned strategy later)
3. Game advances 8 frames
4. Agent gets a reward based on what changed
5. This experience gets stored in memory
6. Agent replays random past experiences to update its brain
7. Repeat thousands of times

---

## Memory Scanner

To re-discover or verify memory addresses:

```bash
python memory_scan.py
```

This boots the game in headless mode, navigates to gameplay, and systematically scans the Game Boy's RAM by moving the character and watching which bytes change. Useful if you want to add new reward signals or track additional game state.
