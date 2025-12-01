# scripts/push_to_hf_hub.py
import os
import sys
from huggingface_hub import HfApi, login

def push_model_to_hub(local_model_dir, hf_repo_id, commit_message="Update model via CI/CD"):
    """
    Pushes model files from a local directory to a Hugging Face Model Hub repository.
    """
    if "HF_TOKEN" not in os.environ:
        print("HF_TOKEN environment variable not set. Please set it for Hugging Face authentication.")
        sys.exit(1)
    
    # Login using the HF_TOKEN environment variable
    login(token=os.environ["HF_TOKEN"])

    api = HfApi()

    print(f"Pushing model from '{local_model_dir}' to '{hf_repo_id}' on Hugging Face Hub...")
    try:
        api.upload_folder(
            folder_path=local_model_dir,
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"Successfully pushed model to https://huggingface.co/{hf_repo_id}")
    except Exception as e:
        print(f"Error pushing model to Hugging Face Hub: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python push_to_hf_hub.py <local_model_dir> <hf_repo_id> [commit_message]")
        sys.exit(1)
    
    local_dir = sys.argv[1]
    repo_id = sys.argv[2]
    message = sys.argv[3] if len(sys.argv) > 3 else "Update model via CI/CD"
    
    push_model_to_hub(local_dir, repo_id, message)
