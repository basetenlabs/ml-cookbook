"""Patch FSDP architecture names in checkpoints for vLLM compatibility."""

import json
import os
from pathlib import Path


def patch_checkpoints(output_dir: str):
    """Fix all FSDP architecture names in checkpoints."""
    output_path = Path(output_dir)
    print(f"Patching checkpoints in: {output_path}")

    for config_path in output_path.rglob("config.json"):
        with open(config_path) as f:
            config = json.load(f)

        modified = False

        if "architectures" in config:
            new_archs = [a.replace("FSDP", "") for a in config["architectures"]]
            if new_archs != config["architectures"]:
                print(f"  {config_path}: {config['architectures']} -> {new_archs}")
                config["architectures"] = new_archs
                modified = True

        if "auto_map" in config:
            print(f"  {config_path}: removing auto_map")
            del config["auto_map"]
            modified = True

        if modified:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    print("Done patching checkpoints.")


if __name__ == "__main__":
    output_dir = os.environ.get("BT_CHECKPOINT_DIR", "/workspace/checkpoints")
    patch_checkpoints(output_dir)
