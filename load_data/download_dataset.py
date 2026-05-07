import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from dotenv import load_dotenv

# Load environment variables (like HF_TOKEN)
load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

from datasets import load_dataset

# Setting streaming=False (the default) will download the entire dataset 
# to the local HuggingFace cache directory (usually ~/.cache/huggingface/datasets).
# This makes future access significantly faster.
print("Starting dataset download. This will take some time and disk space...")
ds = load_dataset("pjura/mahjong_board_states", token=HF_TOKEN)

print("Download complete! Dataset details:")
print(ds)
