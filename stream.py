"""
Live MJPEG stream server for watching RL training in a browser.

Runs alongside training — captures PyBoy frames and serves them as
an MJPEG stream over HTTP. Open http://<host>:5555 in any browser.

Usage:
    python stream.py                     # Train + stream (default)
    python stream.py --port 8080         # Custom port
    python stream.py --episodes 500      # Pass training args through
    python stream.py --fps 30            # Cap stream framerate
"""

import argparse
import io
import os
import time
import json
import threading
import numpy as np
from collections import deque

from flask import Flask, Response, render_template_string
from PIL import Image

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
# Shared frame buffer (thread-safe)
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe buffer holding the latest game frame and training stats."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None  # JPEG bytes
        self._stats = {}
        self._training_info = {}

    def update_frame(self, rgb_array):
        """Convert RGB array to JPEG and store it."""
        img = Image.fromarray(rgb_array)
        # Scale up 3x for better visibility in browser
        img = img.resize((480, 432), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        with self._lock:
            self._frame = buf.getvalue()

    def update_stats(self, stats, training_info):
        with self._lock:
            self._stats = dict(stats)
            self._training_info = dict(training_info)

    def get_frame(self):
        with self._lock:
            return self._frame

    def get_stats(self):
        with self._lock:
            return dict(self._stats), dict(self._training_info)


frame_buffer = FrameBuffer()

# ---------------------------------------------------------------------------
# Flask web server
# ---------------------------------------------------------------------------

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Survival Kids RL - Live Stream</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e; color: #e0e0e0;
            font-family: 'Courier New', monospace;
            display: flex; flex-direction: column; align-items: center;
            min-height: 100vh; padding: 20px;
        }
        h1 { color: #00d4aa; margin-bottom: 10px; font-size: 1.4em; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }
        .game-view {
            border: 3px solid #00d4aa; border-radius: 8px;
            overflow: hidden; background: #000;
        }
        .game-view img { display: block; image-rendering: pixelated; }
        .stats-panel {
            background: #16213e; border: 2px solid #0f3460;
            border-radius: 8px; padding: 15px; min-width: 280px;
        }
        .stats-panel h2 { color: #00d4aa; font-size: 1.1em; margin-bottom: 10px; }
        .stat-row { display: flex; justify-content: space-between; padding: 4px 0; }
        .stat-label { color: #888; }
        .stat-value { color: #fff; font-weight: bold; }
        .stat-value.good { color: #00d4aa; }
        .stat-value.warn { color: #ffa500; }
        .stat-value.bad { color: #ff4444; }
        .bar { height: 8px; background: #333; border-radius: 4px; margin-top: 2px; }
        .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
        .divider { border-top: 1px solid #0f3460; margin: 10px 0; }
        #status { color: #00d4aa; margin-top: 10px; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Survival Kids RL - Live Training</h1>
    <p id="status">Connecting...</p>
    <br>
    <div class="container">
        <div class="game-view">
            <img id="stream" src="/stream" width="480" height="432" alt="Game Stream">
        </div>
        <div class="stats-panel">
            <h2>Game State</h2>
            <div id="game-stats"></div>
            <div class="divider"></div>
            <h2>Training</h2>
            <div id="training-stats"></div>
        </div>
    </div>

    <script>
        function colorClass(val, max) {
            const pct = val / max;
            if (pct > 0.5) return 'good';
            if (pct > 0.2) return 'warn';
            return 'bad';
        }

        function makeBar(val, max, color) {
            const pct = Math.min(100, (val / max) * 100);
            const c = color || (pct > 50 ? '#00d4aa' : pct > 20 ? '#ffa500' : '#ff4444');
            return `<div class="bar"><div class="bar-fill" style="width:${pct}%;background:${c}"></div></div>`;
        }

        async function updateStats() {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                const s = data.stats;
                const t = data.training;

                if (s.hp !== undefined) {
                    document.getElementById('game-stats').innerHTML = `
                        <div class="stat-row"><span class="stat-label">HP</span>
                            <span class="stat-value ${colorClass(s.hp, 100)}">${s.hp}/100</span></div>
                        ${makeBar(s.hp, 100)}
                        <div class="stat-row"><span class="stat-label">Hunger</span>
                            <span class="stat-value ${colorClass(s.hunger, 100)}">${s.hunger}</span></div>
                        ${makeBar(s.hunger, 100)}
                        <div class="stat-row"><span class="stat-label">Thirst</span>
                            <span class="stat-value ${colorClass(s.thirst, 100)}">${s.thirst}</span></div>
                        ${makeBar(s.thirst, 100)}
                        <div class="stat-row"><span class="stat-label">Fatigue</span>
                            <span class="stat-value">${s.fatigue}</span></div>
                        <div class="stat-row"><span class="stat-label">Position</span>
                            <span class="stat-value">(${s.x}, ${s.y})</span></div>
                        <div class="stat-row"><span class="stat-label">Day</span>
                            <span class="stat-value">${s.day}</span></div>
                        <div class="stat-row"><span class="stat-label">Time</span>
                            <span class="stat-value">${['Morning','Noon','Evening','Night'][s.time_of_day] || s.time_of_day}</span></div>
                    `;
                }

                if (t.episode !== undefined) {
                    document.getElementById('training-stats').innerHTML = `
                        <div class="stat-row"><span class="stat-label">Episode</span>
                            <span class="stat-value">${t.episode}/${t.total_episodes || '?'}</span></div>
                        <div class="stat-row"><span class="stat-label">Step</span>
                            <span class="stat-value">${t.step || 0}</span></div>
                        <div class="stat-row"><span class="stat-label">Reward</span>
                            <span class="stat-value">${(t.reward || 0).toFixed(1)}</span></div>
                        <div class="stat-row"><span class="stat-label">Avg Reward</span>
                            <span class="stat-value">${(t.avg_reward || 0).toFixed(1)}</span></div>
                        <div class="stat-row"><span class="stat-label">Explored</span>
                            <span class="stat-value">${t.explored || 0} tiles</span></div>
                        <div class="stat-row"><span class="stat-label">Epsilon</span>
                            <span class="stat-value">${(t.epsilon || 0).toFixed(3)}</span></div>
                    `;
                    document.getElementById('status').textContent =
                        `Episode ${t.episode} | Step ${t.step || 0} | Reward ${(t.reward || 0).toFixed(1)}`;
                }
            } catch(e) {}
        }

        setInterval(updateStats, 500);
        document.getElementById('stream').onload = () => {
            document.getElementById('status').textContent = 'Stream connected';
        };
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/stream")
def stream():
    """MJPEG stream endpoint — works with any browser via <img> tag."""
    def generate():
        while True:
            frame = frame_buffer.get_frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(1 / 30)  # ~30 FPS max
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    """JSON endpoint for live game/training stats."""
    game_stats, training_info = frame_buffer.get_stats()
    return {"stats": game_stats, "training": training_info}


# ---------------------------------------------------------------------------
# Training loop (runs in background thread, pushes frames to buffer)
# ---------------------------------------------------------------------------

def training_loop(args):
    """Same training logic as train.py but pushes frames to the stream."""
    # Import DQN components from train module
    if HAS_TORCH:
        from train import DQNAgent

    rom_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "roms",
        "Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc",
    )

    env = gym.make(
        "SurvivalKids-v0",
        rom_path=rom_path,
        render_mode="rgb_array",
        frames_per_step=args.frame_skip,
        max_steps=args.max_steps,
    )

    n_actions = env.action_space.n
    use_dqn = HAS_TORCH and not args.random

    if use_dqn:
        agent = DQNAgent(n_actions, lr=args.lr, gamma=args.gamma)
        print(f"[Train] DQN agent on {agent.device}")
    else:
        print("[Train] Random baseline")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_file = os.path.join("logs", f"stream_training_{int(time.time())}.jsonl")
    episode_rewards = deque(maxlen=100)

    print(f"[Train] {args.episodes} episodes, logging to {log_file}")

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        losses = []

        # Push initial frame
        frame = env.render()
        if frame is not None:
            frame_buffer.update_frame(frame)

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

            # Push frame to stream (every step)
            frame = env.render()
            if frame is not None:
                frame_buffer.update_frame(frame)

            # Update stats for the dashboard
            frame_buffer.update_stats(
                info.get("stats", {}),
                {
                    "episode": episode,
                    "total_episodes": args.episodes,
                    "step": steps,
                    "reward": total_reward,
                    "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0,
                    "explored": info.get("positions_visited", 0),
                    "epsilon": agent.epsilon if use_dqn else 0,
                },
            )

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards)

        log_entry = {
            "episode": episode,
            "reward": round(total_reward, 2),
            "avg_reward_100": round(float(avg_reward), 2),
            "steps": steps,
            "positions_visited": info.get("positions_visited", 0),
            "stats": info.get("stats", {}),
        }
        if use_dqn:
            log_entry["epsilon"] = round(agent.epsilon, 4)
            log_entry["avg_loss"] = round(float(np.mean(losses)), 4) if losses else 0

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        stats = info.get("stats", {})
        status = (
            f"Ep {episode:4d} | "
            f"Reward: {total_reward:7.1f} | "
            f"Avg100: {avg_reward:7.1f} | "
            f"Steps: {steps:5d} | "
            f"Explored: {info.get('positions_visited', 0):4d} | "
            f"HP:{stats.get('hp', '?'):>3} Day:{stats.get('day', '?')}"
        )
        if use_dqn:
            status += f" | eps:{agent.epsilon:.3f}"
        print(status)

        if use_dqn and episode % 50 == 0:
            agent.save(f"checkpoints/dqn_ep{episode}.pt")

    env.close()
    print(f"\n[Train] Complete. Logs: {log_file}")
    if use_dqn:
        agent.save("checkpoints/dqn_final.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train + live stream Survival Kids RL")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--port", type=int, default=5555, help="Stream server port")
    args = parser.parse_args()

    print(f"Starting stream server on http://0.0.0.0:{args.port}")
    print(f"Open this URL in your browser to watch training live.\n")

    # Start training in background thread
    train_thread = threading.Thread(target=training_loop, args=(args,), daemon=True)
    train_thread.start()

    # Run Flask server in main thread (blocking)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
