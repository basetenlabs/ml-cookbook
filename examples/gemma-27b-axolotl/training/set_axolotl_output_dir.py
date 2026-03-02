# avoid YAML parse failures

import os
import sys

import yaml


def main() -> int:
    cfg_path = os.environ.get("AXOLOTL_CONFIG_FILE")
    checkpoint_dir = os.environ.get("BT_CHECKPOINT_DIR")

    if not cfg_path:
        print("ERROR: AXOLOTL_CONFIG_FILE is not set.")
        return 1
    if not checkpoint_dir:
        print("ERROR: BT_CHECKPOINT_DIR is not set.")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["output_dir"] = checkpoint_dir

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Set output_dir to: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
