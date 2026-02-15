"""
Gymnasium environment for Survival Kids (GBC) running in PyBoy.

Observation: 144x160 grayscale screen (downsampled from RGBA)
Action space: 9 discrete actions (noop + 8 Game Boy buttons)
Reward: Composite signal from HP, hunger, thirst, fatigue, exploration
"""

import io
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy


# --- Memory addresses (WRAM) ---
ADDR_HP = 0xC5EE
ADDR_HUNGER = 0xC5F0
ADDR_THIRST = 0xC5F2
ADDR_FATIGUE = 0xC5F4
ADDR_PLAYER_X = 0xD904
ADDR_PLAYER_Y = 0xD905
ADDR_DAY = 0xCB9C
ADDR_TIME_OF_DAY = 0xC19E
ADDR_GAME_MODE = 0xC160
ADDR_DIALOG_FLAG = 0xC164
ADDR_COMPANION = 0xC5F6

# Button name mapping for PyBoy
ACTIONS = [
    None,       # 0: no-op
    "a",        # 1: A button (interact/confirm)
    "b",        # 2: B button (cancel/run)
    "start",    # 3: Start (menu)
    "select",   # 4: Select
    "up",       # 5: D-pad up
    "down",     # 6: D-pad down
    "left",     # 7: D-pad left
    "right",    # 8: D-pad right
]


class SurvivalKidsEnv(gym.Env):
    """Gymnasium wrapper for Survival Kids on PyBoy."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path="roms/Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc",
        render_mode=None,
        frames_per_step=8,
        max_steps=10000,
        grayscale=True,
        exploration_reward=True,
    ):
        super().__init__()

        self.rom_path = rom_path
        self.render_mode = render_mode
        self.frames_per_step = frames_per_step
        self.max_steps = max_steps
        self.grayscale = grayscale
        self.exploration_reward = exploration_reward

        # Action space: 9 discrete (noop + 8 buttons)
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation space: screen pixels
        if grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(144, 160, 1), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(144, 160, 3), dtype=np.uint8
            )

        # Will be initialized on first reset
        self.pyboy = None
        self._initial_state = None
        self._step_count = 0
        self._prev_stats = None
        self._visited_positions = set()

    def _create_pyboy(self):
        """Create and boot the PyBoy emulator."""
        window = "SDL2" if self.render_mode == "human" else "null"
        self.pyboy = PyBoy(
            self.rom_path,
            window=window,
            sound_emulated=False,
            no_input=(self.render_mode != "human"),
        )

    def _boot_to_gameplay(self):
        """
        Navigate: BIOS -> Konami logo -> Title -> New/Load menu ->
        Name entry (accept default 'Ken') -> Intro cutscene -> Beach gameplay.
        Loops until HP > 0 confirms we're in actual gameplay.
        """
        pb = self.pyboy

        # Phase 1: BIOS + Konami logo (~3s)
        pb.tick(180)

        # Phase 2: Press Start at title screen
        pb.button("start")
        pb.tick(120)

        # Phase 3: Mash A to get through title -> New/Load menu -> select "New"
        for _ in range(15):
            pb.button("a")
            pb.tick(60)

        # Phase 4: Name entry screen — default name is "Ken"
        # Navigate cursor to "END" and confirm. Mashing A types letters,
        # so press Start which also confirms the name in this screen.
        for _ in range(5):
            pb.button("a")
            pb.tick(60)

        # Phase 5: Intro cutscene (ship scene + dialog) — alternate Start and A
        # to skip through all text boxes and transitions
        for _ in range(40):
            pb.button("start")
            pb.tick(30)
            pb.button("a")
            pb.tick(30)
            # Check if we've reached gameplay
            if pb.memory[ADDR_HP] > 0:
                break

        # Phase 6: Clear ALL opening dialog on the beach
        # Keep pressing A until dialog_flag == 0 and we can move freely
        for _ in range(100):
            pb.button("a")
            pb.tick(15)
            if pb.memory[ADDR_HP] > 0 and pb.memory[ADDR_DIALOG_FLAG] == 0:
                # Wait a few more frames to be sure dialog is fully dismissed
                pb.tick(30)
                if pb.memory[ADDR_DIALOG_FLAG] == 0:
                    break

        # Verify we reached gameplay
        if pb.memory[ADDR_HP] == 0:
            raise RuntimeError(
                "Failed to boot to gameplay — HP is still 0. "
                "The title screen navigation may need adjustment."
            )

    def _read_stats(self):
        """Read all game state from memory."""
        mem = self.pyboy.memory
        return {
            "hp": mem[ADDR_HP],
            "hunger": mem[ADDR_HUNGER],
            "thirst": mem[ADDR_THIRST],
            "fatigue": mem[ADDR_FATIGUE],
            "x": mem[ADDR_PLAYER_X],
            "y": mem[ADDR_PLAYER_Y],
            "day": mem[ADDR_DAY],
            "time_of_day": mem[ADDR_TIME_OF_DAY],
            "game_mode": mem[ADDR_GAME_MODE],
            "dialog": mem[ADDR_DIALOG_FLAG],
            "companion": mem[ADDR_COMPANION],
        }

    def _get_observation(self):
        """Get screen as numpy array."""
        screen = self.pyboy.screen.ndarray  # (144, 160, 4) RGBA
        if self.grayscale:
            # Convert RGBA to grayscale: 0.299R + 0.587G + 0.114B
            gray = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
            return gray.astype(np.uint8)[..., np.newaxis]
        else:
            return screen[..., :3].copy()  # Drop alpha channel

    def _compute_reward(self, prev, curr):
        """
        Compute reward signal from game state changes.

        Reward components:
        - Survival: penalty for losing HP/hunger/thirst
        - Exploration: bonus for visiting new (x, y) positions
        - Progress: bonus for surviving another day
        - Death: large penalty if HP hits 0
        """
        reward = 0.0

        # Survival stats changes (higher = better, so positive delta = good)
        hp_delta = curr["hp"] - prev["hp"]
        hunger_delta = curr["hunger"] - prev["hunger"]
        thirst_delta = curr["thirst"] - prev["thirst"]

        # Reward for maintaining/improving stats
        reward += hp_delta * 1.0
        reward += hunger_delta * 0.3
        reward += thirst_delta * 0.3

        # Penalty for critically low stats
        if curr["hp"] < 20:
            reward -= 0.5
        if curr["hunger"] < 10:
            reward -= 0.3
        if curr["thirst"] < 10:
            reward -= 0.3

        # Exploration bonus
        if self.exploration_reward:
            pos = (curr["x"], curr["y"])
            if pos not in self._visited_positions:
                self._visited_positions.add(pos)
                reward += 0.5

        # Day survival bonus
        if curr["day"] > prev["day"]:
            reward += 10.0

        # Death penalty
        if curr["hp"] == 0:
            reward -= 50.0

        # Small step penalty to encourage efficiency
        reward -= 0.01

        return reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.pyboy is None:
            self._create_pyboy()
            self._boot_to_gameplay()
            # Save state right after reaching gameplay
            self._initial_state = io.BytesIO()
            self.pyboy.save_state(self._initial_state)
        else:
            # Reload saved state
            self._initial_state.seek(0)
            self.pyboy.load_state(self._initial_state)

        self._step_count = 0
        self._visited_positions = set()

        # Tick once to render the first frame
        self.pyboy.tick(1, True)

        self._prev_stats = self._read_stats()
        pos = (self._prev_stats["x"], self._prev_stats["y"])
        self._visited_positions.add(pos)

        obs = self._get_observation()
        info = {"stats": self._prev_stats}

        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        btn = ACTIONS[action]
        if btn is not None:
            # Hold the button for the entire frame skip duration so
            # movement actually registers in the game engine
            self.pyboy.button_press(btn)
            self.pyboy.tick(self.frames_per_step, True)
            self.pyboy.button_release(btn)
        else:
            # No-op: just advance frames
            self.pyboy.tick(self.frames_per_step, True)

        self._step_count += 1

        # Read new state
        curr_stats = self._read_stats()

        # Compute reward
        reward = self._compute_reward(self._prev_stats, curr_stats)
        self._prev_stats = curr_stats

        # Check termination
        terminated = curr_stats["hp"] == 0
        truncated = self._step_count >= self.max_steps

        obs = self._get_observation()
        info = {
            "stats": curr_stats,
            "step": self._step_count,
            "positions_visited": len(self._visited_positions),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.pyboy.screen.ndarray[..., :3].copy()
        # "human" mode is handled by PyBoy's SDL2 window

    def close(self):
        if self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None


# Register with Gymnasium
gym.register(
    id="SurvivalKids-v0",
    entry_point="survival_kids_env:SurvivalKidsEnv",
)
