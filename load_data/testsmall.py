#!/usr/bin/env python3
"""
test.py — Fast 2-pass game retrieval.
Split = year (e.g. "2019"). Handles both string and integer column keys.
"""
from __future__ import annotations
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

# ════════════════════════════════════
# CONFIG
# ════════════════════════════════════
YEAR         = "2019"   # also used as split name
N_GAMES      = 3
SEED         = 42
HF_REPO      = "pjura/mahjong_board_states"
SPLIT        = YEAR     # split="2019" is valid — years are the splits
ID_SCAN_ROWS = 5_000    # rows to scan in pass 1 to collect candidate IDs
# ════════════════════════════════════

import random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict
from datasets import load_dataset

random.seed(SEED)
np.random.seed(SEED)

TILE_NAMES = (
    ["1m","2m","3m","4m","5m","6m","7m","8m","9m"]
  + ["1p","2p","3p","4p","5p","6p","7p","8p","9p"]
  + ["1s","2s","3s","4s","5s","6s","7s","8s","9s"]
  + ["East","South","West","North","Haku","Hatsu","Chun"]
)
CHANNEL_NAMES = (
    ["Hand"]
  + [f"Pool_P{i}" for i in range(4)]
  + ["Wall_Man","Wall_Pin","Wall_Sou","Wall_Hon"]
  + [f"Meld_P{i}" for i in range(4)]
  + ["Label","Rsv"]
)

COL_HAND_START = 68
COL_POOL_START = 238
COL_DISCARD    = 510
COL_GAME_ID    = 511


def col(row: dict, c: int, default=0):
    v = row.get(str(c))
    if v is None:
        v = row.get(c)
    return v if v is not None else default


def get_gid(row: dict) -> str:
    return str(col(row, COL_GAME_ID, "")).strip()


def inspect_schema() -> None:
    print("\n[Schema] Fetching one row to inspect field names…")
    ds = load_dataset(HF_REPO, split=SPLIT, streaming=True, token=HF_TOKEN)
    for row in ds:
        keys = list(row.keys())
        print(f"  Total fields : {len(keys)}")
        print(f"  Key type     : {type(keys[0]).__name__}")
        print(f"  First 5 keys : {keys[:5]}")
        print(f"  Last  5 keys : {keys[-5:]}")
        gid = get_gid(row)
        print(f"  GameID value : {gid!r}")
        d   = col(row, COL_DISCARD, -1)
        print(f"  Discard col  : {d!r}")
        break


def row_to_matrix(row: dict) -> np.ndarray:
    mat = np.zeros((34, 15), dtype=np.float32)
    for t in range(34):
        mat[t, 0] = float(col(row, COL_HAND_START + t, 0)) / 4.0
    for p in range(4):
        base = COL_POOL_START + p * 34
        for t in range(34):
            mat[t, 1 + p] = float(col(row, base + t, 0)) / 4.0
    for t in range(34):
        ch = 5 + (0 if t < 9 else 1 if t < 18 else 2 if t < 27 else 3)
        mat[t, ch] = float(col(row, t, 0))
    d = int(col(row, COL_DISCARD, -1))
    if 0 <= d < 34:
        mat[d, 13] = 1.0
    return mat


def pass1_find_ids() -> list[str]:
    t0 = time.time()
    print(f"\n[Pass 1] Scanning {ID_SCAN_ROWS:,} rows of split='{SPLIT}' for game IDs…")
    ds  = load_dataset(HF_REPO, split=SPLIT, streaming=True, token=HF_TOKEN)
    ids: set[str] = set()

    for i, row in enumerate(ds):
        gid = get_gid(row)
        if gid:
            ids.add(gid)
        if i >= ID_SCAN_ROWS - 1:
            break

    result = sorted(ids)
    print(f"[Pass 1] {len(result)} unique game IDs found in {time.time()-t0:.1f}s")
    if result:
        print(f"         Sample: {result[0]}")
    return result


def pass2_fetch(target_ids: set[str]) -> dict[str, list[dict]]:
    t0 = time.time()
    print(f"\n[Pass 2] Fetching moves for {len(target_ids)} game(s) — early exit when done…")
    ds      = load_dataset(HF_REPO, split=SPLIT, streaming=True, token=HF_TOKEN)
    buckets = {gid: [] for gid in target_ids}
    done: set[str] = set()
    prev   = None

    for i, row in enumerate(ds):
        gid = get_gid(row)

        if prev and prev != gid and prev in target_ids and prev not in done:
            done.add(prev)
            print(f"  ✓ {prev[:54]}  ({len(buckets[prev])} moves)")
            if done == target_ids:
                print(f"  All {len(target_ids)} games complete at row {i:,}.")
                break

        if gid in buckets:
            buckets[gid].append(row)

        prev = gid

        if i > 0 and i % 10_000 == 0:
            print(f"  … {i:,} rows | {len(done)}/{len(target_ids)} done")

    if prev and prev in target_ids and prev not in done and buckets[prev]:
        done.add(prev)
        print(f"  ✓ {prev[:54]}  ({len(buckets[prev])} moves) [stream end]")

    print(f"[Pass 2] Done in {time.time()-t0:.1f}s")
    return buckets


def build_game(game_id: str, rows: list[dict]) -> OrderedDict:
    game = OrderedDict()
    for idx, row in enumerate(rows):
        d = int(col(row, COL_DISCARD, -1))
        game[f"move_{idx:03d}"] = {
            "game_id":           game_id,
            "move_index":        idx,
            "discard_tile_idx":  d,
            "discard_tile_name": TILE_NAMES[d] if 0 <= d < 34 else "?",
            "matrix":            row_to_matrix(row),
        }
    return game


def to_tensors(game: OrderedDict):
    X = np.stack([v["matrix"] for v in game.values()])[:, np.newaxis, :, :]
    y = np.array([v["discard_tile_idx"] for v in game.values()], dtype=np.int64)
    return X, y


def plot_game(game: OrderedDict, path: str = "board_states.png") -> None:
    keys  = sorted(random.sample(list(game.keys()), min(4, len(game))))
    ncols = min(2, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*7),
                             facecolor="#1c1b19")
    if len(keys) == 1: axes = np.array([[axes]])
    elif nrows == 1:   axes = axes[np.newaxis, :]
    elif ncols == 1:   axes = axes[:, np.newaxis]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mj", ["#1c1b19","#01696f","#4f98a3","#e0f4f5"])

    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        mv   = game[key]
        im   = ax.imshow(mv["matrix"], aspect="auto", cmap=cmap,
                         vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Move {mv['move_index']}  |  discard: {mv['discard_tile_name']}",
                     color="#cdccca", fontsize=11, pad=8)
        ax.set_xticks(range(15))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right",
                           fontsize=7, color="#797876")
        ax.set_yticks(range(34))
        ax.set_yticklabels(TILE_NAMES, fontsize=6.5, color="#797876")
        ax.set_xlabel("Channels", color="#797876", fontsize=8)
        ax.set_ylabel("Tiles",    color="#797876", fontsize=8)
        ax.tick_params(colors="#797876")
        for sp in ax.spines.values(): sp.set_edgecolor("#393836")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.ax.yaxis.set_tick_params(color="#797876", labelsize=7)
        cb.outline.set_edgecolor("#393836")

    for idx in range(len(keys), nrows*ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    gid = list(game.values())[0]["game_id"]
    fig.suptitle(f"Mahjong 34×15 Board States\nGameID: {gid}",
                 color="#cdccca", fontsize=13, y=1.02)
    fig.patch.set_facecolor("#1c1b19")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1c1b19")
    plt.close()
    print(f"[viz] Saved → {path}")


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"  Mahjong Game Retrieval  |  split='{SPLIT}'  |  N_GAMES={N_GAMES}")
    print("=" * 60)

    inspect_schema()

    all_ids = pass1_find_ids()
    if not all_ids:
        print(f"\n[error] No game IDs found. Raise ID_SCAN_ROWS (currently {ID_SCAN_ROWS:,}).")
        return

    chosen = random.sample(all_ids, min(N_GAMES, len(all_ids)))
    print(f"\n[sample] Chosen IDs:")
    for g in chosen:
        print(f"  {g}")

    buckets = pass2_fetch(set(chosen))

    print(f"\n{'─'*60}")
    print(f"  {'GameID':<44}  Moves  X-shape")
    print(f"{'─'*60}")
    all_games: dict[str, OrderedDict] = {}
    for gid in chosen:
        rows = buckets.get(gid, [])
        if not rows:
            print(f"  {gid[:44]:<44}  (no rows — raise ID_SCAN_ROWS)")
            continue
        game = build_game(gid, rows)
        all_games[gid] = game
        X, y = to_tensors(game)
        print(f"  {gid[:44]:<44}  {len(game):>5}  {X.shape}")

    if all_games:
        pick = random.choice(list(all_games.keys()))
        print(f"\n[viz] Plotting moves from: {pick}")
        plot_game(all_games[pick])

    print(f"\n[Total time] {time.time()-t_start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()