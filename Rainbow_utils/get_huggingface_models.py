import os
from huggingface_hub import snapshot_download

def download_model_with_proxy(model_url, cache_dir):
    """
    Download a model from Hugging Face Hub using a proxy.

    Parameters:
    - model_url (str): The URL of the model to download.
    - cache_dir (str): The directory where the model will be cached.
    """
    print("Downloading model with proxy......")
    
    # Set proxy environment variables if needed
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"

    # Use the http object with snapshot_download
    try:
        SNAPSHOT_PATH = snapshot_download(
            model_url,
            cache_dir=cache_dir,
            force_download=True,
            resume_download=False
        )
        # Print the location of the downloaded files
        print(f"Downloaded files are located in: {SNAPSHOT_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    MODEL_URL = "google/t5-v1_1-small"
    cache_dir = "./models"

    download_model_with_proxy(MODEL_URL, cache_dir)