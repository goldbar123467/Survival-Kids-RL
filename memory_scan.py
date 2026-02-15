#!/usr/bin/env python3
"""
Survival Kids (GBC) Memory Address Scanner
==========================================
Boots the game in headless PyBoy, navigates past the title screen,
and scans WRAM for key game-state addresses: HP, hunger, thirst,
fatigue, player position, map/room ID, inventory, and more.

CONFIRMED MEMORY MAP (from GameShark codes, TASVideos, and runtime verification):
=================================================================================

PLAYER STATS (0xC5xx block):
  0xC5E6  Water in canteen      (GameShark: 0103E6C5, value 0-3)
  0xC5ED  HP max cap            (0xFF = 255)
  0xC5EE  HP / Life current     (GameShark: 0164EEC5; starts at 100=0x64)
  0xC5EF  Hunger max cap        (0xFF = 255)
  0xC5F0  Hunger                (GameShark: 01??F0C5; starts at 70=0x46)
  0xC5F1  Thirst max cap        (0xFF = 255)
  0xC5F2  Water / Thirst        (GameShark: 01??F2C5; starts at 70=0x46)
  0xC5F4  Fatigue               (GameShark: 01??F4C5; starts at 50=0x32)
  0xC5F6  Companion affection   (TASVideos: 240+ = wedding ending)

PLAYER POSITION:
  0xD900  Sprite pixel X on screen  (constant ~80 = screen center)
  0xD901  Sprite pixel Y on screen  (constant ~72-80 = screen center)
  0xD904  Player tile X position    (CONFIRMED: changes with left/right movement)
  0xD905  Player tile Y position    (CONFIRMED: changes with up/down movement)
  0xC5C0  Previous Y position (lags D905 by ~1 step)
  0xC5C1  Previous X position (lags D904 by ~1 step)
  0xC5C6  Position mirror / facing X (tracks D904, jumps on screen edge)
  0xC5C7  Position mirror / facing Y (tracks D905, jumps on screen edge)

TIME SYSTEM:
  0xC19D  Time counter within period (starts ~160-250; 250 steps/period, night=190)
  0xC19E  Time of day (0=morning, 1=noon, 2=evening, 3=night)
  0xCB9C  Day counter (increments each full day cycle)

GAME FLAGS:
  0xCB5D  Hunger/thirst/fatigue disable flag (GameShark: 01005DCB)
  0xC3AB  River state (reportedly 0x80=full river, 0xC0=dried river)

SPAWNING:
  0xC6FC  Animal/NPC spawn flag
  0xC6FD  Animal spawn counter (TASVideos: spawns when reaches 120)

ENTITY TABLE (0xD900+ block, 12-byte structs):
  Entity 0 (player): 0xD900-0xD90B
  Entity 1 (NPC/companion?): 0xD90C-0xD917
  Structure per entity: [pixelX, pixelY, flags, ?, tileX, tileY, ?, ?, ?, ?, ?, ?]

INVENTORY:
  0xC766-0xC7FF  Inventory slots (0xFF = empty; 154 bytes = many item slots)
  TASVideos reports $7000-$700B but that appears to be banked VRAM in PyBoy

OTHER DISCOVERED:
  0xC5E2  Unknown flag (value 1 at start)
  0xC5E3  Unknown flag (value 1 at start)
  0xC5E9  Unknown counter (value 3 at start)
  0xC160  Game mode/state (3 = overworld gameplay)
  0xC161  Sub-state (1 during gameplay)
  0xC164  Dialog/event flag (0 during free movement)

GameShark GBC format: 01XXYYZZ -> address = 0xZZYY, value = XX
"""

import os
import sys
from pathlib import Path

try:
    from pyboy import PyBoy
except ImportError:
    print("ERROR: PyBoy not installed. Run: pip install pyboy")
    sys.exit(1)

ROM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "roms",
    "Survival Kids (USA) (SGB Enhanced) (GB Compatible).gbc",
)

# ── Complete known address table ─────────────────────────────────────
KNOWN_ADDRESSES = {
    # Player stats
    0xC5E6: "Water in canteen (0-3)",
    0xC5ED: "HP max cap (0xFF)",
    0xC5EE: "HP / Life (starts 100)",
    0xC5EF: "Hunger max cap (0xFF)",
    0xC5F0: "Hunger (starts 70)",
    0xC5F1: "Thirst max cap (0xFF)",
    0xC5F2: "Water / Thirst (starts 70)",
    0xC5F4: "Fatigue (starts 50)",
    0xC5F6: "Companion affection (240+=wedding)",
    # Position
    0xD900: "Sprite pixel X (screen center ~80)",
    0xD901: "Sprite pixel Y (screen center ~72)",
    0xD904: "Player tile X position",
    0xD905: "Player tile Y position",
    0xC5C0: "Previous Y position",
    0xC5C1: "Previous X position",
    0xC5C6: "Position mirror / facing X",
    0xC5C7: "Position mirror / facing Y",
    # Time
    0xC19D: "Time counter (steps within period)",
    0xC19E: "Time of day (0=morn,1=noon,2=eve,3=night)",
    0xCB9C: "Day counter",
    # Flags
    0xCB5D: "Hunger/thirst/fatigue disable flag",
    0xC3AB: "River state (0x80=full, 0xC0=dried)",
    # Spawning
    0xC6FC: "Animal/NPC spawn flag",
    0xC6FD: "Animal spawn counter (spawns at 120)",
    # Game state
    0xC160: "Game mode (3=overworld)",
    0xC161: "Sub-state (1=gameplay)",
    0xC164: "Dialog/event flag (0=free movement)",
}


def read_memory_range(pyboy, start, end):
    """Read a range of memory addresses and return as bytearray."""
    data = bytearray()
    for addr in range(start, end):
        try:
            data.append(pyboy.memory[addr])
        except Exception:
            data.append(0)
    return data


def dump_wram(pyboy):
    """Dump WRAM (0xC000-0xDFFF) as a bytearray."""
    return read_memory_range(pyboy, 0xC000, 0xE000)


def press_button(pyboy, btn, hold_frames=4, wait_frames=8):
    """Press a button for hold_frames, then wait wait_frames."""
    pyboy.button(btn, delay=hold_frames)
    for _ in range(hold_frames + wait_frames):
        pyboy.tick(1, False)


def advance_frames(pyboy, n):
    """Advance n frames."""
    for _ in range(n):
        pyboy.tick(1, False)


def compare_dumps(before, after, base_addr=0xC000):
    """Find bytes that changed between two dumps."""
    changes = {}
    for i in range(len(before)):
        if before[i] != after[i]:
            addr = base_addr + i
            changes[addr] = (before[i], after[i])
    return changes


def print_known_addresses(pyboy):
    """Print current values of all known addresses."""
    print("\n" + "=" * 70)
    print("KNOWN ADDRESS VALUES")
    print("=" * 70)
    for addr, desc in sorted(KNOWN_ADDRESSES.items()):
        val = pyboy.memory[addr]
        print(f"  0x{addr:04X}: 0x{val:02X} ({val:3d})  -- {desc}")
    print("=" * 70)


def navigate_to_gameplay(pyboy):
    """
    Navigate past the title screen and into actual gameplay.
    Survival Kids flow: Konami logo -> Title -> Name entry -> Intro cutscene -> Gameplay
    """
    print("--- Navigating to gameplay ---")

    # Boot: wait for initial screens
    print("  Booting ROM (waiting 300 frames)...")
    advance_frames(pyboy, 300)

    # Press Start/A a few times to get past logo/title
    print("  Pressing Start/A to skip title screen...")
    for i in range(8):
        press_button(pyboy, "start", hold_frames=6, wait_frames=30)
        press_button(pyboy, "a", hold_frames=6, wait_frames=30)
        advance_frames(pyboy, 30)

    # Keep pressing A/Start to advance through intro/name entry/dialogs
    print("  Pressing A/Start to advance through intro...")
    for i in range(40):
        press_button(pyboy, "a", hold_frames=4, wait_frames=10)
        if i % 5 == 0:
            press_button(pyboy, "start", hold_frames=4, wait_frames=10)

    advance_frames(pyboy, 120)

    hp = pyboy.memory[0xC5EE]
    print(f"  HP address (0xC5EE) value: {hp}")

    if hp == 0:
        print("  HP is 0, pressing more buttons...")
        for i in range(60):
            press_button(pyboy, "a", hold_frames=4, wait_frames=8)
            if i % 8 == 0:
                press_button(pyboy, "start", hold_frames=4, wait_frames=8)
        advance_frames(pyboy, 300)
        hp = pyboy.memory[0xC5EE]
        print(f"  HP address (0xC5EE) value now: {hp}")

    if hp > 0:
        print("  In-game! Clearing initial dialog with A/B spam...")
        for _ in range(50):
            press_button(pyboy, "a", hold_frames=2, wait_frames=4)
            press_button(pyboy, "b", hold_frames=2, wait_frames=4)
        advance_frames(pyboy, 60)
        print(f"  Player position: X={pyboy.memory[0xD904]}, Y={pyboy.memory[0xD905]}")

    return hp > 0


def scan_for_position(pyboy):
    """
    Move the player in each direction and verify position addresses.
    """
    print("\n--- Verifying player position addresses ---")

    print("  Testing 0xD904 (X) and 0xD905 (Y) with sustained movement:")

    # Move RIGHT
    x_before = pyboy.memory[0xD904]
    y_before = pyboy.memory[0xD905]
    pyboy.button("right", delay=60)
    for _ in range(60):
        pyboy.tick(1, False)
    x_after = pyboy.memory[0xD904]
    y_after = pyboy.memory[0xD905]
    print(f"    RIGHT: X {x_before}->{x_after} (delta={x_after-x_before}), Y {y_before}->{y_after}")

    # Move DOWN
    x_before = pyboy.memory[0xD904]
    y_before = pyboy.memory[0xD905]
    pyboy.button("down", delay=60)
    for _ in range(60):
        pyboy.tick(1, False)
    x_after = pyboy.memory[0xD904]
    y_after = pyboy.memory[0xD905]
    print(f"    DOWN:  X {x_before}->{x_after}, Y {y_before}->{y_after} (delta={y_after-y_before})")

    # Move LEFT
    x_before = pyboy.memory[0xD904]
    y_before = pyboy.memory[0xD905]
    pyboy.button("left", delay=60)
    for _ in range(60):
        pyboy.tick(1, False)
    x_after = pyboy.memory[0xD904]
    y_after = pyboy.memory[0xD905]
    print(f"    LEFT:  X {x_before}->{x_after} (delta={x_after-x_before}), Y {y_before}->{y_after}")

    # Move UP
    x_before = pyboy.memory[0xD904]
    y_before = pyboy.memory[0xD905]
    pyboy.button("up", delay=60)
    for _ in range(60):
        pyboy.tick(1, False)
    x_after = pyboy.memory[0xD904]
    y_after = pyboy.memory[0xD905]
    print(f"    UP:    X {x_before}->{x_after}, Y {y_before}->{y_after} (delta={y_after-y_before})")

    # Verify C5C0/C5C1 mirror
    print(f"\n  Position mirrors:")
    print(f"    D904 (X) = {pyboy.memory[0xD904]}, D905 (Y) = {pyboy.memory[0xD905]}")
    print(f"    C5C0 (prevY) = {pyboy.memory[0xC5C0]}, C5C1 (prevX) = {pyboy.memory[0xC5C1]}")
    print(f"    C5C6 (mirrorX) = {pyboy.memory[0xC5C6]}, C5C7 (mirrorY) = {pyboy.memory[0xC5C7]}")


def scan_for_stats_and_time(pyboy, frames=600):
    """
    Advance the game and check if stats/time change.
    """
    print(f"\n--- Monitoring stats over {frames} frames of movement ---")

    hp_start = pyboy.memory[0xC5EE]
    hunger_start = pyboy.memory[0xC5F0]
    thirst_start = pyboy.memory[0xC5F2]
    fatigue_start = pyboy.memory[0xC5F4]
    time_counter_start = pyboy.memory[0xC19D]
    time_of_day_start = pyboy.memory[0xC19E]
    day_start = pyboy.memory[0xCB9C]

    print(f"  Before: HP={hp_start}, Hunger={hunger_start}, Thirst={thirst_start}, "
          f"Fatigue={fatigue_start}")
    print(f"          TimeCounter={time_counter_start}, TimeOfDay={time_of_day_start}, "
          f"Day={day_start}")

    # Walk around to pass time
    directions = ["right", "down", "left", "up"]
    for d in directions:
        pyboy.button(d, delay=frames // 4)
        for _ in range(frames // 4):
            pyboy.tick(1, False)

    hp_end = pyboy.memory[0xC5EE]
    hunger_end = pyboy.memory[0xC5F0]
    thirst_end = pyboy.memory[0xC5F2]
    fatigue_end = pyboy.memory[0xC5F4]
    time_counter_end = pyboy.memory[0xC19D]
    time_of_day_end = pyboy.memory[0xC19E]
    day_end = pyboy.memory[0xCB9C]

    print(f"  After:  HP={hp_end}, Hunger={hunger_end}, Thirst={thirst_end}, "
          f"Fatigue={fatigue_end}")
    print(f"          TimeCounter={time_counter_end}, TimeOfDay={time_of_day_end}, "
          f"Day={day_end}")
    print(f"  Deltas: HP={hp_end-hp_start}, Hunger={hunger_end-hunger_start}, "
          f"Thirst={thirst_end-thirst_start}, Fatigue={fatigue_end-fatigue_start}")


def scan_stat_block(pyboy):
    """Detailed dump of the stat block area."""
    print("\n--- Stat block 0xC5D0-0xC600 ---")
    for addr in range(0xC5D0, 0xC600):
        val = pyboy.memory[addr]
        known = f"  <-- {KNOWN_ADDRESSES[addr]}" if addr in KNOWN_ADDRESSES else ""
        if val != 0 or addr in KNOWN_ADDRESSES:
            print(f"  0x{addr:04X}: 0x{val:02X} ({val:3d}){known}")


def scan_entity_table(pyboy):
    """Dump the entity/sprite table at D900."""
    print("\n--- Entity table 0xD900-0xD920 ---")
    print("  Entity 0 (player):")
    for i in range(12):
        addr = 0xD900 + i
        val = pyboy.memory[addr]
        known = f"  <-- {KNOWN_ADDRESSES[addr]}" if addr in KNOWN_ADDRESSES else ""
        print(f"    0x{addr:04X}: 0x{val:02X} ({val:3d}){known}")
    print("  Entity 1 (NPC/companion):")
    for i in range(12):
        addr = 0xD90C + i
        val = pyboy.memory[addr]
        known = f"  <-- {KNOWN_ADDRESSES[addr]}" if addr in KNOWN_ADDRESSES else ""
        print(f"    0x{addr:04X}: 0x{val:02X} ({val:3d}){known}")


def scan_inventory(pyboy):
    """Check inventory region C766-C7FF."""
    print("\n--- Inventory region 0xC766-0xC7FF ---")
    empty = 0
    items = []
    for addr in range(0xC766, 0xC800):
        val = pyboy.memory[addr]
        if val == 0xFF:
            empty += 1
        else:
            items.append((addr, val))
    print(f"  Empty slots (0xFF): {empty}")
    if items:
        print(f"  Items found:")
        for addr, val in items:
            print(f"    0x{addr:04X}: 0x{val:02X} ({val})")
    else:
        print(f"  No items in inventory (all slots empty)")


def scan_game_state(pyboy):
    """Check game mode/state bytes."""
    print("\n--- Game state bytes ---")
    for addr in [0xC160, 0xC161, 0xC164, 0xC165]:
        val = pyboy.memory[addr]
        known = f"  <-- {KNOWN_ADDRESSES[addr]}" if addr in KNOWN_ADDRESSES else ""
        print(f"  0x{addr:04X}: 0x{val:02X} ({val:3d}){known}")


def scan_wram_diff_movement(pyboy):
    """
    Take WRAM snapshot, move in each direction with sustained hold,
    and find all bytes that change.
    """
    print("\n--- WRAM diff with sustained directional movement ---")

    directions = ["right", "left", "down", "up"]
    all_changed = {}

    for d in directions:
        wram_before = dump_wram(pyboy)
        pyboy.button(d, delay=60)
        for _ in range(60):
            pyboy.tick(1, False)
        wram_after = dump_wram(pyboy)

        changes = compare_dumps(wram_before, wram_after, 0xC000)
        for addr, (old, new) in changes.items():
            if addr not in all_changed:
                all_changed[addr] = []
            all_changed[addr].append((d, old, new))

    # Show addresses that changed in 2+ directions
    multi = {a: v for a, v in all_changed.items() if len(v) >= 2}
    print(f"  Addresses changing in 2+ directions: {len(multi)}")
    for addr in sorted(multi.keys()):
        changes = multi[addr]
        val = pyboy.memory[addr]
        known = f" ** {KNOWN_ADDRESSES[addr]}" if addr in KNOWN_ADDRESSES else ""
        parts = [f"[{d}: {o}->{n}]" for d, o, n in changes]
        print(f"    0x{addr:04X}: cur={val:3d}  {' '.join(parts)}{known}")


def main():
    print("=" * 70)
    print("SURVIVAL KIDS (GBC) - Memory Address Scanner")
    print("=" * 70)
    print(f"ROM: {ROM_PATH}")

    if not Path(ROM_PATH).exists():
        print(f"ERROR: ROM not found at {ROM_PATH}")
        sys.exit(1)

    print("\nInitializing PyBoy in headless mode...")
    pyboy = PyBoy(
        ROM_PATH,
        window="null",
        sound_emulated=False,
    )

    print(f"PyBoy version: {pyboy.__class__.__module__}")
    print(f"Game title: {pyboy.cartridge_title}")
    pyboy.set_emulation_speed(0)

    # ── Phase 1: Navigate to gameplay ────────────────────────────────
    in_game = navigate_to_gameplay(pyboy)

    if not in_game:
        print("\nERROR: Could not get into gameplay. Exiting.")
        pyboy.stop()
        sys.exit(1)

    # ── Phase 2: Print all known addresses ───────────────────────────
    print_known_addresses(pyboy)

    # ── Phase 3: Verify position addresses ───────────────────────────
    scan_for_position(pyboy)

    # ── Phase 4: Monitor stats over time ─────────────────────────────
    scan_for_stats_and_time(pyboy, frames=600)

    # ── Phase 5: Detailed stat block ─────────────────────────────────
    scan_stat_block(pyboy)

    # ── Phase 6: Entity table ────────────────────────────────────────
    scan_entity_table(pyboy)

    # ── Phase 7: Inventory ───────────────────────────────────────────
    scan_inventory(pyboy)

    # ── Phase 8: Game state ──────────────────────────────────────────
    scan_game_state(pyboy)

    # ── Phase 9: WRAM movement diff ──────────────────────────────────
    scan_wram_diff_movement(pyboy)

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE MEMORY MAP SUMMARY")
    print("=" * 70)
    print("""
PLAYER STATS (verified in-game):
  0xC5EE  HP / Life             max=100 (0x64), cap at 0xC5ED (0xFF)
  0xC5F0  Hunger                starts=70 (0x46), cap at 0xC5EF (0xFF)
  0xC5F2  Water / Thirst        starts=70 (0x46), cap at 0xC5F1 (0xFF)
  0xC5F4  Fatigue               starts=50 (0x32)
  0xC5F6  Companion affection   starts=0, 240+ for wedding ending
  0xC5E6  Water in canteen      0-3

PLAYER POSITION (verified with movement):
  0xD904  Player tile X         increases going right, decreases going left
  0xD905  Player tile Y         increases going down, decreases going up
  0xD900  Sprite screen pixel X (constant ~80, screen center)
  0xD901  Sprite screen pixel Y (constant ~72-80)
  0xC5C0  Previous Y pos        (lags D905 by ~1 step)
  0xC5C1  Previous X pos        (lags D904 by ~1 step)
  0xC5C6  Facing/target X       (tracks D904, wraps at screen edge)
  0xC5C7  Facing/target Y       (tracks D905, wraps at screen edge)

TIME SYSTEM:
  0xC19D  Time step counter     counts down per period (250/period, night=190)
  0xC19E  Time of day           0=morning, 1=noon, 2=evening, 3=night
  0xCB9C  Day counter           increments each full day cycle

GAME FLAGS:
  0xCB5D  Hunger/thirst/fatigue disable flag
  0xC3AB  River state           (0x80=full, 0xC0=dried)
  0xC160  Game mode             (3=overworld)
  0xC161  Sub-state             (1=gameplay active)
  0xC164  Dialog/event flag     (0=free movement, non-0=in event/dialog)

SPAWNING:
  0xC6FC  Spawn flag
  0xC6FD  Animal spawn counter  (spawns at value 120)

ENTITY TABLE (0xD900, 12-byte structs):
  Entity 0 = Player:  0xD900-0xD90B
  Entity 1 = NPC:     0xD90C-0xD917

INVENTORY:
  0xC766-0xC7FF  Item slots (0xFF=empty, other values=item IDs)
  (TASVideos also reports $7000-$700B which may be accessible via WRAM bank switching)

KEY ADDRESSES FOR RL REWARD SHAPING:
  Reward signals: 0xC5EE (HP), 0xC5F0 (Hunger), 0xC5F2 (Thirst), 0xC5F4 (Fatigue)
  Exploration:    0xD904 (X pos), 0xD905 (Y pos), 0xCB9C (Day)
  Progress:       0xC19E (Time of day), 0xC6FD (Animal counter)
  State:          0xC160 (Mode), 0xC164 (Dialog flag)
""")

    print("Final address values:")
    for addr, desc in sorted(KNOWN_ADDRESSES.items()):
        val = pyboy.memory[addr]
        print(f"  0x{addr:04X} = 0x{val:02X} ({val:3d}) -- {desc}")

    pyboy.stop()
    print("\n--- Scan complete ---")


if __name__ == "__main__":
    main()
