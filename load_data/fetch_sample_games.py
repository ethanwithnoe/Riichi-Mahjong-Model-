import os, random, time
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict
from datasets import load_dataset

# ════════════════════════════════════
# CONFIG
# ════════════════════════════════════
YEAR       = "2019"
HF_REPO    = "pjura/mahjong_board_states"
SPLIT      = YEAR
SAVE_DIR   = "."
N_RAW_COLS = 512

TARGET_IDS = [
    "ce6457db",
    "914dcea1",
    "8b26c7cc",
]
# ════════════════════════════════════

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


def col(row, c, default=0):
    v = row.get(str(c))
    if v is None:
        v = row.get(c)
    return v if v is not None else default


def get_gid(row):
    return str(col(row, COL_GAME_ID, "")).strip()


def row_to_raw(row):
    vec = np.zeros(N_RAW_COLS, dtype=np.float32)
    for c in range(N_RAW_COLS):
        try:
            vec[c] = float(col(row, c, 0))
        except (ValueError, TypeError):
            vec[c] = 0.0
    return vec


def row_to_matrix(row):
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


def resolve(gid, targets):
    for t in targets:
        if t in gid or gid in t:
            return t
    return None


# ── Stream and collect all target games ──────────────────────────────────────
targets = set(TARGET_IDS)
buckets = {}
done    = set()
prev    = None

print(f"Streaming split='{SPLIT}' for {len(targets)} game ID(s)…\n")
t_fetch = time.time()

ds = load_dataset(HF_REPO, split=SPLIT, streaming=True, token=HF_TOKEN)

for i, row in enumerate(ds):
    gid = get_gid(row)

    if prev and prev != gid:
        t = resolve(prev, targets)
        if t and prev in buckets and t not in done:
            done.add(t)
            print(f"  ✓ [{t}]  full_id={prev}  ({len(buckets[prev])} moves)")
            if done == targets:
                print(f"\n  All games complete at row {i:,}. Stopping early.")
                break

    t = resolve(gid, targets)
    if t:
        if gid not in buckets:
            buckets[gid] = []
            print(f"  → First match for [{t}]  full_id={gid}")
        buckets[gid].append(row)

    prev = gid

    if i > 0 and i % 20_000 == 0:
        print(f"  … {i:,} rows | {len(done)}/{len(targets)} done")

# catch last game if stream ended
if prev:
    t = resolve(prev, targets)
    if t and prev in buckets and t not in done:
        done.add(t)
        print(f"  ✓ [{t}]  full_id={prev}  ({len(buckets[prev])} moves) [stream end]")

print(f"\nFetch done in {time.time()-t_fetch:.1f}s")

missing = targets - done
if missing:
    print(f"[warn] Not found: {missing}")


# ── Build tensors and save ────────────────────────────────────────────────────
all_games = {}

for full_gid, rows in buckets.items():
    n = len(rows)
    matrices = np.zeros((n, 34, 15), dtype=np.float32)
    labels   = np.zeros((n,),        dtype=np.int64)
    raw      = np.zeros((n, N_RAW_COLS), dtype=np.float32)
    game     = OrderedDict()

    for idx, row in enumerate(rows):
        mat  = row_to_matrix(row)
        d    = int(col(row, COL_DISCARD, -1))
        rvec = row_to_raw(row)

        matrices[idx] = mat
        labels[idx]   = d if 0 <= d < 34 else -1
        raw[idx]      = rvec

        game[f"move_{idx:03d}"] = {
            "game_id":           full_gid,
            "move_index":        idx,
            "discard_tile_idx":  d,
            "discard_tile_name": TILE_NAMES[d] if 0 <= d < 34 else "?",
            "matrix":            mat,
        }

    X = matrices[:, np.newaxis, :, :]   # (N, 1, 34, 15)

    safe = full_gid.replace("/","_").replace(":","_")
    np.save(os.path.join(SAVE_DIR, f"{safe}_matrices.npy"), X)
    np.save(os.path.join(SAVE_DIR, f"{safe}_labels.npy"),   labels)
    np.save(os.path.join(SAVE_DIR, f"{safe}_raw.npy"),      raw)

    print(f"\n{full_gid}")
    print(f"  X  (CNN input) : {X.shape}   → {safe}_matrices.npy")
    print(f"  y  (labels)    : {labels.shape}  → {safe}_labels.npy")
    print(f"  raw (all cols) : {raw.shape} → {safe}_raw.npy")

    all_games[full_gid] = game


# ── Plot one game ─────────────────────────────────────────────────────────────
if all_games:
    pick  = random.choice(list(all_games.keys()))
    game  = all_games[pick]
    keys  = sorted(random.sample(list(game.keys()), min(4, len(game))))
    ncols = min(2, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*7), facecolor="#1c1b19")
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
        ax.set_title(f"Move {mv['move_index']} | discard: {mv['discard_tile_name']}",
                     color="#cdccca", fontsize=11, pad=8)
        ax.set_xticks(range(15))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=7, color="#797876")
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

    fig.suptitle(f"Mahjong 34×15 Board States\nGameID: {pick}",
                 color="#cdccca", fontsize=13, y=1.02)
    fig.patch.set_facecolor("#1c1b19")
    plt.tight_layout()
    plt.savefig("board_states.png", dpi=150, bbox_inches="tight", facecolor="#1c1b19")
    plt.close()
    print(f"\n[viz] board_states.png saved.")

print("\nDone.")
