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
        "p0_score":  raw[:, COL_P0_SCORE]  * 1000,
        "p1_score":  raw[:, COL_P1_SCORE]  * 1000,
        "p2_score":  raw[:, COL_P2_SCORE]  * 1000,
        "p3_score":  raw[:, COL_P3_SCORE]  * 1000,
        "round_num": raw[:, COL_ROUND_NUM].astype(int),
        "step_num":  raw[:, COL_STEP_NUM].astype(int),
    })

    results = []

    # Scores are fixed at the START of each round (step is most negative)
    # so just take the first row of each round and the first row of the next round
    round_starts = (
        df.sort_values(["round_num", "step_num"])
          .groupby("round_num")
          .first()
          .reset_index()
    )

    for i in range(len(round_starts) - 1):
        current = round_starts.iloc[i]
        next_r  = round_starts.iloc[i + 1]

        start_scores = np.array([
            current["p0_score"], current["p1_score"],
            current["p2_score"], current["p3_score"]
        ])
        end_scores = np.array([
            next_r["p0_score"], next_r["p1_score"],
            next_r["p2_score"], next_r["p3_score"]
        ])

        deltas = end_scores - start_scores
        max_gain = int(deltas.max())

        if max_gain <= 0:
            # Exhaustive draw — credit all who didn't lose
            for p in range(4):
                if deltas[p] >= 0:
                    results.append({
                        "game_id":    game_id,
                        "round_num":  int(current["round_num"]),
                        "winner":     p,
                        "win_amount": int(deltas[p]),
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

