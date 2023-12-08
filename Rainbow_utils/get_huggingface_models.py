from huggingface_hub import snapshot_download

MODEL_URL = "Qwen/Qwen-7B-Chat"
SNAPSHOT_PATH = snapshot_download(MODEL_URL, cache_dir="./models")

print(f"Downloaded files are located in: {SNAPSHOT_PATH}")
