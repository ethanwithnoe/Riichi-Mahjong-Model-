import os
import numpy as np
import pandas as pd

COL_P0_SCORE  = 6
COL_P1_SCORE  = 7
COL_P2_SCORE  = 8
COL_P3_SCORE  = 9
COL_ROUND_NUM = 32
COL_STEP_NUM  = 33

GAME_IDS = ["ce6457db", "914dcea1", "8b26c7cc"]

#this is incorrect for the point calculation, its very very very complicated, so we may drop it but its good to have
def label_game(raw: np.ndarray, game_id: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "p0_score":  raw[:, COL_P0_SCORE],
        "p1_score":  raw[:, COL_P1_SCORE],
        "p2_score":  raw[:, COL_P2_SCORE],
        "p3_score":  raw[:, COL_P3_SCORE],
        "round_num": raw[:, COL_ROUND_NUM].astype(int),
        "step_num":  raw[:, COL_STEP_NUM].astype(int),
    })

    round_starts = (
        df.sort_values(["round_num", "step_num"])
          .groupby("round_num")
          .first()
          .reset_index()
    )

    # Get the very last row of the game as the final score endpoint
    last_row = df.sort_values(["round_num", "step_num"]).iloc[-1]
    final_scores = pd.Series({
        "round_num": last_row["round_num"],
        "p0_score":  last_row["p0_score"],
        "p1_score":  last_row["p1_score"],
        "p2_score":  last_row["p2_score"],
        "p3_score":  last_row["p3_score"],
    })

    # Append final scores as the endpoint for the last round
    round_endpoints = pd.concat(
        [round_starts.iloc[1:], final_scores.to_frame().T],
        ignore_index=True
    )

    results = []

    for i in range(len(round_starts)):
        current  = round_starts.iloc[i]
        endpoint = round_endpoints.iloc[i]

        start_scores = np.array([
            current["p0_score"], current["p1_score"],
            current["p2_score"], current["p3_score"]
        ])
        end_scores = np.array([
            endpoint["p0_score"], endpoint["p1_score"],
            endpoint["p2_score"], endpoint["p3_score"]
        ])

        deltas   = end_scores - start_scores
        max_gain = int(deltas.max())

        if max_gain <= 0:
            results.append({
                "game_id":    game_id,
                "round_num":  int(current["round_num"]),
                "winner":     -1,
                "win_amount": 0,
            })
        else:
            results.append({
                "game_id":    game_id,
                "round_num":  int(current["round_num"]),
                "winner":     int(np.argmax(deltas)),
                "win_amount": max_gain,
            })

    return pd.DataFrame(results)


all_labels = []

for gid in GAME_IDS:
    path = f"{gid}_raw.npy"
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        continue

    raw = np.load(path)
    print(f"\n{gid}:  {raw.shape[0]} moves loaded")

    labels = label_game(raw, gid)
    print(labels.to_string(index=False))
    all_labels.append(labels)

if all_labels:
    combined = pd.concat(all_labels, ignore_index=True)
    combined.to_csv("win_labels.csv", index=False)
    print(f"\nSaved win_labels.csv  ({len(combined)} rounds total)")
    print("\nWinner distribution:")
    print(combined["winner"].value_counts().sort_index().to_string())

