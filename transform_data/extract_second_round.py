import os
import numpy as np
import pandas as pd

COL_ROUND_NUM = 32
COL_STEP_NUM  = 33

GAME_IDS = ["ce6457db", "914dcea1", "8b26c7cc"]

def main():
    results = []

    for gid in GAME_IDS:
        path = f"{gid}_raw.npy"
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        # Load the raw numpy array
        raw = np.load(path)
        
        # Extract just round and step details into a DataFrame
        df = pd.DataFrame({
            "original_idx": np.arange(raw.shape[0]),
            "round_num": raw[:, COL_ROUND_NUM].astype(int),
            "step_num":  raw[:, COL_STEP_NUM].astype(int),
        })

        # Find the start of each round
        # As observed in other scripts, the first row of the round is the one with the smallest step_num
        round_starts = (
            df.sort_values(["round_num", "step_num"])
              .groupby("round_num")
              .first()
              .reset_index()
        )

        # Ensure the game actually has a second round
        if len(round_starts) >= 2:
            # iloc[1] gets the second row (the second round)
            second_round = round_starts.iloc[1]
            row_idx = int(second_round["original_idx"])
            
            row_data = {
                "game_id": gid,
                "round_num": int(second_round["round_num"]),
                "step_num": int(second_round["step_num"])
            }
            # Add all columns from the raw numpy array for this specific step
            for col_idx in range(raw.shape[1]):
                row_data[f"feature_{col_idx}"] = raw[row_idx, col_idx]
                
            results.append(row_data)
        else:
            print(f"[warn] {gid} does not have a second round.")

    if results:
        out_df = pd.DataFrame(results)
        output_file = "second_round_steps.csv"
        out_df.to_csv(output_file, index=False)
        print(f"Successfully saved {output_file} with shape {out_df.shape}:\n")
        print(out_df[["game_id", "round_num", "step_num"]].to_string(index=False))
    else:
        print("No valid games found or no games had a second round.")

if __name__ == "__main__":
    main()
