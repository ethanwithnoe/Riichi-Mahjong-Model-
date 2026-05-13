# for references
# X -> (N, 1, 34, 15)  
# y -> (N,)      
# each row in X corresponds to one discard play
# y is the outcome of the round 0/1
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# column indices --------------------------------------
COL_HAND_START = 68
COL_POOL_START = 238
COL_DISCARD    = 510        # not used for X here, but here for reference
COL_GAME_ID    = 511
COL_ROUND_NUM  = 32
COL_STEP_NUM   = 33

COL_P0_SCORE   = 6
COL_P1_SCORE   = 7
COL_P2_SCORE   = 8
COL_P3_SCORE   = 9

N_RAW_COLS     = 512        # total columns in the raw dataset (0–511)

# defaults -------------------
DEFAULT_K_GAMES  = 500
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SEED     = 42


# STEP 1 - LOADING RAW DATA ----------------------------------------------
def load_raw(k_games: int, save_dir: Path, seed: int):
    # pull a random sample of k_games from the 2009 huggingface split and return (all_raw, game_index)
    
    raw_path = save_dir / "all_raw.npy"
    index_path = save_dir / "game_index.csv"

    # if exist, load from disc cuz goated
    if raw_path.exists() and index_path.exists():
        print(f"[load] found cached data in {save_dir}, loading")
        all_raw    = np.load(raw_path)
        game_index = pd.read_csv(index_path)
        return all_raw, game_index

    # if not cached, fetch from huggingface
    print("[load] no cache found, downloading from huggingface")

    import random
    from datasets import load_dataset
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # auth
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("[load] logged in")
    else:
        print("[load] no HF_TOKEN found, you fail")

    print("[load] Loading 2009 split")
    ds = load_dataset(
        "pjura/mahjong_board_states",
        data_dir="data/2009/",
        token=hf_token
    )["train"]

    # load or build game ID list
    gid_cache = save_dir / "all_gids.txt"
    if gid_cache.exists():
        with open(gid_cache) as f:
            all_gids = [line.strip() for line in f]
        print(f"[load] Loaded {len(all_gids):,} game IDs from cache.")
    else:
        all_gids = list(set(ds["511"]))
        with open(gid_cache, "w") as f:
            f.write("\n".join(all_gids))
        print(f"[load] Saved {len(all_gids):,} game IDs to cache.")

    sampled_gids = set(random.sample(all_gids, k=min(k_games, len(all_gids))))
    print(f"[load] sampled {len(sampled_gids):,} games, filtering rows")

    df_full = ds.to_pandas()
    df = df_full[df_full["511"].isin(sampled_gids)].sort_values(["511", "32", "33"])
    del df_full
    print(f"[load] {len(df):,} rows across {df['511'].nunique():,} games")

    numeric_cols = [str(c) for c in range(511)]
    chunks, index, cursor = [], [], 0

    for i, (gid, group) in enumerate(df.groupby("511", sort=False), start=1):
        raw = group[numeric_cols].to_numpy(dtype=np.float32)
        chunks.append(raw)
        index.append({"game_id": gid, "start": cursor, "end": cursor + len(raw)})
        cursor += len(raw)
        if i % 50 == 0 or i == k_games:
            print(f"  [{i}/{k_games}] processed {cursor:,} moves so far")

    all_raw    = np.concatenate(chunks, axis=0)
    game_index = pd.DataFrame(index)

    np.save(raw_path, all_raw)
    game_index.to_csv(index_path, index=False)
    print(f"[load] saving all_raw.npy and game_index.csv to {save_dir}")

    return all_raw, game_index

# STEP 2 - BUILDING OUR X --------------------------------------------
def raw_row_to_matrix(row: np.ndarray) -> np.ndarray:
    # for reference
    # 0 : hand
    # 1-4 : pool player 0-3
    # 5-8 : wall remaining by type
    # 9-12 : melds players not used here
    # 13 : discard indicator
    # 14 : 
    # converting 511 column raw row into (34, 15)
    mat = np.zeros((34, 15), dtype=np.float32) # creating our empty 34,15

    # our hand
    for t in range(34):
        mat[t, 0] = row[COL_HAND_START + t] / 4.0 # divide by 4 to normalize 

    # pool for each of our players' discards
    # each player holds up 34 columns
    for p in range(4):
        base = COL_POOL_START + p * 34 # jump to player section
        for t in range(34):
            mat[t, 1 + p] = row[base + t] / 4.0

    # wall remaining, split by tile type
    # tiles 0-8 man, 9-17 tong, 18-26 tiao, 27-33 face
    for t in range(34):
        ch = 5 + (0 if t < 9 else 1 if t < 18 else 2 if t < 27 else 3)
        mat[t, ch] = row[t]

    d = int(row[COL_DISCARD])
    if 0 <= d < 34:
        mat[d, 13] = 1.0

    return mat


def build_X(all_raw: np.ndarray) -> np.ndarray:
    # converting every row in all_raw to (34,15) matrix. 
    # in form of pytorch CNN, not sure what natalie did for the CNN this section could be changed later
    print(f"[X] building board state matrices {len(all_raw):,}")
    matrices = np.stack([raw_row_to_matrix(all_raw[i]) for i in range(len(all_raw))])
    X = matrices[:, np.newaxis, :, :]  # (N, 1, 34, 15)
    print(f"[X] shape: {X.shape}")
    return X


# STEP 3 - BUILDING Y --------------------------------------------------
def label_game(raw: np.ndarray, game_id: str) -> pd.DataFrame:
    # using vincent's add_win_labels logic
    # returns as df with columns [game_id, round_num, pov_won, win_amounts]
    df = pd.DataFrame({
        "pov": raw[:, 2].astype(int),
        "p0_score": raw[:, COL_P0_SCORE],
        "p1_score": raw[:, COL_P1_SCORE],
        "p2_score": raw[:, COL_P2_SCORE],
        "p3_score": raw[:, COL_P3_SCORE],
        "round_num": raw[:, COL_ROUND_NUM].astype(int),
        "step_num": raw[:, COL_STEP_NUM].astype(int),
    })

    def get_fixed_scores(row):
        pov = int(row["pov"]) # deciding turn
        cols = [row["p0_score"], row["p1_score"], row["p2_score"], row["p3_score"]]
        # unrotate back to get absolute seat position
        # the dataset rotates so every player in our pov is always player 0
        # unrotate to figure out who actually gained points
        return [cols[(s - pov) % 4] for s in range(4)] 

    round_starts = ( # we only start at the first step of each round to get our starting scores
        df.sort_values(["round_num", "step_num"]) 
          .groupby("round_num")
          .first() # ONLY THE FIRST ROW OF EACH ROUND
          .reset_index()
    )
    round_starts["fixed"] = round_starts.apply(get_fixed_scores, axis=1)

    # for the last round, theres no next round to compare it to, we set this as the endpoint
    last_row   = df.sort_values(["round_num", "step_num"]).iloc[-1]
    last_fixed = get_fixed_scores(last_row)

    results = []
    for i in range(len(round_starts)):
        current = round_starts.iloc[i]
        pov_seat = int(current["pov"]) # physical seat of our pov player right now
        start_fixed = current["fixed"] # scores at the start of THIS ROUND
        end_fixed = round_starts.iloc[i + 1]["fixed"] if i + 1 < len(round_starts) else last_fixed # end scores = start of next round

        # did the pov player's score go up?
        # look at their seat position to decide start/end array
        delta   = end_fixed[pov_seat] - start_fixed[pov_seat]
        pov_won = 1 if delta > 0 else 0 # 1 = win, 0 = lose/even

        results.append({
            "game_id": game_id,
            "round_num": int(current["round_num"]),
            "pov_won": pov_won,
            "win_amount": int(delta * 1000),
        })

    return pd.DataFrame(results)


def build_y(all_raw: np.ndarray, game_index: pd.DataFrame) -> np.ndarray:
    # for every step row in all_raw, we look up which game, round it belongs to 
    # then we broadcast the round's winner label to that row (pov_won)
    print(f"[y] generating win labels for {len(game_index):,} games")

    y = np.full(len(all_raw), fill_value=-1, dtype=np.int64)

    for _, game_row in game_index.iterrows():
        gid = game_row["game_id"]
        start = int(game_row["start"])
        end = int(game_row["end"])
        raw = all_raw[start:end]

        round_labels = label_game(raw, gid)  # df: round_num -> pov_won   
        win_map = dict(zip(round_labels["round_num"], round_labels["pov_won"]))

        # broadcast round label to every step in that round
        rounds_in_raw = raw[:, COL_ROUND_NUM].astype(int)
        for step_offset, r in enumerate(rounds_in_raw):
            y[start + step_offset] = win_map.get(r, -1)

    unlabelled = (y == -1).sum()
    if unlabelled:
        print(f"[y] {unlabelled} steps could not be labelled (set to -1)")

    print(f"[y] shape: {y.shape}  |  win rate: {y[y >= 0].mean():.3f}")
    return y


# MAIN ----------------------------------------------

def build_dataset(k_games: int, save_dir: Path, seed: int):
    os.makedirs(save_dir, exist_ok=True)

    # 1. raw data
    all_raw, game_index = load_raw(k_games=k_games, save_dir=save_dir, seed=seed)

    # 2. X board state matrices
    X = build_X(all_raw)

    # 3. y and win labels assigned to X
    y = build_y(all_raw, game_index)

    # 4. save
    x_path = save_dir / "X.npy"
    y_path = save_dir / "y.npy"
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"\n[done] saved:")
    print(f"  X : {x_path}   shape={X.shape}  dtype={X.dtype}")
    print(f"  y : {y_path}   shape={y.shape}  dtype={y.dtype}")
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build X.npy and y.npy for the Mahjong win predictor.")
    parser.add_argument("--k_games",  type=int, default=DEFAULT_K_GAMES,        help="Number of games to sample (default: 500)")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR),   help="Output directory (default: ./data)")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED,            help="Random seed (default: 42)")
    args = parser.parse_args()

    build_dataset(
        k_games=args.k_games,
        save_dir=Path(args.save_dir),
        seed=args.seed,
    )
