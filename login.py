import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()   # reads .env from the same folder as test.py

HF_TOKEN = os.getenv("HF_TOKEN")


login(token=HF_TOKEN, add_to_git_credential=False)