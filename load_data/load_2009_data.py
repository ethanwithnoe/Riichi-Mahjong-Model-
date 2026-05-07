import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

print("Loading the 2009 split from cache...")
ds_2009 = load_dataset(
    "pjura/mahjong_board_states",
    data_dir="data/2009/",
    token=HF_TOKEN
)["train"]                            # ← unpack the DatasetDict immediately

print(f"\nSuccessfully loaded 2009 data! ({len(ds_2009):,} rows)")
print(ds_2009)

print("\nFirst row sample:")
first_row = ds_2009[0]
print(f"Game ID (col 511): {first_row.get('511')}")
print(f"Step num (col 33): {first_row.get('33')}")