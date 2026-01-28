import os
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import enable_progress_bars

repo_id = "baseten-admin/gpt-oss-20b-eagle3-raw-data-format"
local_dir = "./eagle_head"

os.makedirs(local_dir, exist_ok=True)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
enable_progress_bars()

files = list_repo_files(repo_id)

for filename in files:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # ignored but fine
        resume_download=True,
        cache_dir=local_dir,           # 🔑 SAME DIR → no cross-FS copy
    )

print("Done")
