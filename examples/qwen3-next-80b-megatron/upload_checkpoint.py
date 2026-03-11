#!/usr/bin/env python3
"""Upload checkpoint to Hugging Face Hub."""
import argparse
import datetime
import os
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    parser.add_argument("repo_id", help="HF Hub repo ID (e.g., user/model-name)")
    args = parser.parse_args()

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.checkpoint_dir,
        repo_type="model",
        commit_message=f"Checkpoint {datetime.datetime.utcnow().isoformat()}Z",
    )
    print(f"Uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
