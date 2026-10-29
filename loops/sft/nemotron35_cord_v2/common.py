import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dataset_cache():
    return Path(os.environ.get("CORD_HF_CACHE", str(ROOT / "data" / "hf_cache")))


def load_env(path=None):
    if path is None:
        return
    source = Path(path)
    if not source.is_file():
        raise ValueError("The explicitly requested environment file does not exist.")
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            key, separator, value = stripped.partition(":")
        if separator and key.strip() in {
            "BASETEN_API_KEY",
            "BASETEN_TEAM",
            "WANDB_API_KEY",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "HF_TOKEN",
        }:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def settings():
    config = json.loads((ROOT / "recipe.json").read_text())
    config["team"] = os.environ.get("BASETEN_TEAM") or None
    config["wandb_entity"] = os.environ.get("WANDB_ENTITY") or None
    config["wandb_project"] = os.environ.get("WANDB_PROJECT", config["wandb_project"])
    return config


def api_request(path, method="GET"):
    if not path.startswith("/v1/loops/") or method not in {"GET", "POST"}:
        raise ValueError("Only supported relative Loops API requests are permitted.")
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        raise ValueError("Set BASETEN_API_KEY or supply --env-file.")
    request = urllib.request.Request(
        "https://api.baseten.co" + path,
        method=method,
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Baseten {method} failed with HTTP {error.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("Baseten API could not be reached.") from None
    return json.loads(body) if body else {}


def validate_catalog(config):
    response = api_request("/v1/loops/capabilities?use_case=sft")
    entry = next(
        (
            model
            for model in response["supported_models"]
            if model["model_name"] == config["base_model"]
        ),
        None,
    )
    if (
        entry is None
        or not entry.get("enabled")
        or not entry.get("supports_vision_language")
    ):
        raise ValueError("The requested model is not enabled for vision SFT.")
    maximum = entry.get("max_enabled_seq_len", entry.get("max_seq_len"))
    if maximum is None or maximum < config["max_seq_len"]:
        raise ValueError("Enabled sequence length is below this recipe's requirement.")
    return entry


def prepare(config):
    from data import load_prepared

    return load_prepared(config)
