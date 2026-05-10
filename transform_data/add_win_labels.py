import os
import numpy as np
import pandas as pd

COL_P0_SCORE  = 6
COL_P1_SCORE  = 7
COL_P2_SCORE  = 8
COL_P3_SCORE  = 9
COL_ROUND_NUM = 32
COL_STEP_NUM  = 33

def label_game(raw: np.ndarray, game_id: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "pov":       raw[:, 2].astype(int),
        "p0_score":  raw[:, COL_P0_SCORE],
        "p1_score":  raw[:, COL_P1_SCORE],
        "p2_score":  raw[:, COL_P2_SCORE],
        "p3_score":  raw[:, COL_P3_SCORE],
        "round_num": raw[:, COL_ROUND_NUM].astype(int),
        "step_num":  raw[:, COL_STEP_NUM].astype(int),
    })

    def get_fixed_scores(row):
        """Unrotate POV-relative scores to fixed physical seat scores."""
        pov  = int(row["pov"])
        cols = [row["p0_score"], row["p1_score"], row["p2_score"], row["p3_score"]]
        # fixed[seat] = cols[(seat - pov) % 4]
        return [cols[(s - pov) % 4] for s in range(4)]

    round_starts = (
        df.sort_values(["round_num", "step_num"])
          .groupby("round_num")
          .first()
          .reset_index()
    )
    round_starts["fixed"] = round_starts.apply(get_fixed_scores, axis=1)

    # Use last row of game as final score endpoint
    last_row   = df.sort_values(["round_num", "step_num"]).iloc[-1]
    last_fixed = get_fixed_scores(last_row)

    results = []

    for i in range(len(round_starts)):
        current     = round_starts.iloc[i]
        pov_seat    = int(current["pov"])  # which physical seat is POV this round
        start_fixed = current["fixed"]

        if i + 1 < len(round_starts):
            end_fixed = round_starts.iloc[i + 1]["fixed"]
        else:
            end_fixed = last_fixed

        # Track the POV player's score change using their fixed physical seat
        delta   = end_fixed[pov_seat] - start_fixed[pov_seat]
        pov_won = 1 if delta > 0 else 0

        results.append({
            "game_id":    game_id,
            "round_num":  int(current["round_num"]),
            "pov_won":    pov_won,
            "win_amount": int(delta * 1000),
        })

    return pd.DataFrame(results)


all_labels = []

all_raw    = np.load("data/all_raw.npy")
game_index = pd.read_csv("data/game_index.csv")

for _, row in game_index.iterrows():
    gid = row["game_id"]
    raw = all_raw[int(row["start"]):int(row["end"])]
    labels = label_game(raw, gid)
    all_labels.append(labels)

if all_labels:
    combined = pd.concat(all_labels, ignore_index=True)
    combined.to_csv("data/win_labels.csv", index=False)
    print(f"\nSaved win_labels.csv  ({len(combined)} rounds total)")
    print("\nWinner distribution:")
    print(combined["pov_won"].value_counts().sort_index().to_string())