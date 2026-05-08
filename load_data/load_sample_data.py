import os, random
import numpy as np
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from datasets import load_dataset
from pathlib import Path
import multiprocessing

# ── CONSTANTS (globals) ───────────────────
N_RAW_COLS  = 512
COL_GAME_ID = 511

# ── DEFAULTS (used by both standalone and pipeline) ───────────────
DEFAULT_K_GAMES  = 500
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SEED     = 42

def build_sample_dataset(
    k_games:  int  = DEFAULT_K_GAMES,
    save_dir: Path = DEFAULT_SAVE_DIR,
    seed:     int  = DEFAULT_SEED
):
    random.seed(seed)                          # ← uses parameter, not global
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    load_dotenv(override=True)
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)

    print("Loading the 2009 split from cache...")
    ds = load_dataset(
        "pjura/mahjong_board_states",
        data_dir="data/2009/",
        token=HF_TOKEN
    )["train"]

    GID_CACHE = save_dir / "all_gids.txt"      # ← uses parameter

    if GID_CACHE.exists():
        with open(GID_CACHE) as f:
            all_gids = [line.strip() for line in f]
        print(f"Loaded {len(all_gids):,} game IDs from cache")
    else:
        all_gids = list(set(ds["511"]))
        with open(GID_CACHE, "w") as f:
            f.write("\n".join(all_gids))
        print(f"Saved {len(all_gids):,} game IDs to {GID_CACHE}")

    sampled_gids = set(random.sample(all_gids, k=min(k_games, len(all_gids))))  # ← parameter

    # STEP 2 — full sequential read + vectorised filter (faster than random Arrow seeks at 500+ games)
    print("Filtering rows for sampled games...")
    df_full = ds.to_pandas()
    df = df_full[df_full["511"].isin(sampled_gids)].sort_values(["511", "32", "33"])
    del df_full
    print(f" {len(df):,} rows across {df['511'].nunique():,} games")

    # STEP 3 — groupby game_id, build array + index
    chunks, index, cursor = [], [], 0

    numeric_cols = [str(c) for c in range(511)]   # cols 0–510, excludes "511" (game_id)

    for i, (gid, group) in enumerate(df.groupby("511", sort=False), start=1):
        raw = group[numeric_cols].to_numpy(dtype=np.float32)
        chunks.append(raw)
        index.append({"game_id": gid, "start": cursor, "end": cursor + len(raw)})
        cursor += len(raw)

        if i % 10 == 0 or i == k_games:
            print(f"  [{i}/{k_games}] processed {cursor:,} moves so far")

    all_raw    = np.concatenate(chunks, axis=0)
    game_index = pd.DataFrame(index)

    # ── STEP 4: Save ──────────────────────────────────────────────
    np.save(save_dir / "all_raw.npy", all_raw)
    game_index.to_csv(save_dir / "game_index.csv", index=False)
    print(f"Saved {all_raw.shape[0]:,} moves across {len(game_index):,} games")
    print(f"  {save_dir}/all_raw.npy")
    print(f"  {save_dir}/game_index.csv")

    return all_raw, game_index


# ── Standalone entry point ────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_games",  type=int,  default=DEFAULT_K_GAMES)
    parser.add_argument("--save_dir", type=str,  default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--seed",     type=int,  default=DEFAULT_SEED)
    args = parser.parse_args()

    build_sample_dataset(
        k_games=args.k_games,
        save_dir=args.save_dir,
        seed=args.seed
    )