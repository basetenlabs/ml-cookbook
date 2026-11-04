"""Frozen data, exact counting reward, and read-only Loops preflight helpers."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import re
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent
API_BASE_URL = "https://api.baseten.co"
ENV_KEYS = frozenset({"BASETEN_API_KEY", "WANDB_API_KEY", "HF_TOKEN",
                      "BASETEN_TEAM", "WANDB_ENTITY", "WANDB_PROJECT"})
BOXED_INTEGER = re.compile(r"\\boxed\{(\d+)\}")
FROZEN_SPLITS = {
    "train": {
        "path": "data/train.jsonl",
        "rows": 1024,
        "sha256": "6b63bf98dd2107a35b6ed442dbb5a8003ba15c8053b66dc37f9fbe617d24b307",
    },
    "validation": {
        "path": "data/validation.jsonl",
        "rows": 256,
        "sha256": "76a8a383ab459826117bbf11699c5908b10d539d9fd21b437d0c1e9149f096b8",
    },
}


def load_env(path: str | Path | None = None) -> None:
    """Load explicit account/credential settings; retain existing env values."""
    if path is None:
        return
    source = Path(path)
    if not source.is_file():
        raise ValueError("The explicitly supplied environment file does not exist or is not a file.")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("Cannot read the explicitly supplied environment file.") from None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            key, separator, value = stripped.partition(":")
        key = key.strip()
        if not separator or key not in ENV_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def _json_file(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"Cannot read valid {description} JSON.") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


def settings(path: str | Path | None = None) -> dict[str, Any]:
    config = _json_file(Path(path) if path is not None else ROOT / "recipe.json", "recipe")
    for variable, field in (("BASETEN_TEAM", "team"), ("WANDB_ENTITY", "wandb_entity"),
                            ("WANDB_PROJECT", "wandb_project")):
        value = os.environ.get(variable, "").strip()
        if value:
            if "\n" in value or "\r" in value:
                raise ValueError(f"{variable} must not contain newlines.")
            config[field] = value
    return config


def api_request(path: str, method: str = "GET", payload: Any = None) -> dict[str, Any]:
    """Call Baseten without exposing keys or response bodies in exceptions."""
    parsed = urllib.parse.urlsplit(path)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/v1/loops/"):
        raise ValueError("API path must be relative to Baseten /v1/loops/.")
    method = method.upper()
    if method not in {"GET", "POST"}:
        raise ValueError("Only GET and POST are supported.")
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        raise ValueError("Set BASETEN_API_KEY or supply --env-file with the existing key file.")
    if "\n" in key or "\r" in key:
        raise ValueError("BASETEN_API_KEY contains an invalid newline.")
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    data = None
    if payload is not None:
        if method == "GET":
            raise ValueError("GET requests cannot carry a payload.")
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(API_BASE_URL + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        # The reason/body/URL can contain returned credentials or submitted data.
        raise RuntimeError(f"Baseten API {method} {parsed.path} failed (HTTP {error.code}).") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, ValueError):
        raise RuntimeError(f"Baseten API {method} {parsed.path} could not be reached.") from None
    try:
        result = json.loads(body) if body else {}
    except (ValueError, UnicodeError):
        raise RuntimeError(f"Baseten API {method} {parsed.path} returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise RuntimeError(f"Baseten API {method} {parsed.path} returned a non-object response.")
    return result


def validate_catalog(config: dict[str, Any]) -> dict[str, Any]:
    """Verify enabled RL/vision availability without creating a session."""
    model_name = config["base_model"]
    maximum = int(config.get("max_seq_len", 4096))
    query = urllib.parse.urlencode({"model": model_name, "use_case": "rl", "max_seq_len": maximum})
    response = api_request("/v1/loops/capabilities?" + query)
    models = response.get("supported_models", [])
    if not isinstance(models, list):
        raise ValueError("Loops capability response does not contain a model catalog.")
    entry = next((item for item in models if isinstance(item, dict)
                  and item.get("model_name") == model_name), None)
    if entry is None or entry.get("enabled") is not True:
        raise ValueError("The configured Loops RL model is not enabled for this workspace.")
    if entry.get("supports_vision_language") is not True:
        raise ValueError("The configured Loops RL model does not advertise vision support.")
    allowance = entry.get("max_enabled_seq_len", entry.get("max_seq_len", entry.get("max_context_length")))
    if allowance is not None and (isinstance(allowance, bool) or not isinstance(allowance, (int, float))
                                  or allowance < maximum):
        raise ValueError("Workspace sequence-length allowance is insufficient.")
    # Return selected, non-secret capability fields only; never echo a whole API body.
    return {key: entry[key] for key in (
        "model_name", "enabled", "supports_vision_language", "max_enabled_seq_len",
        "max_seq_len", "max_context_length", "supported_use_cases", "use_cases",
    ) if key in entry}


def ground_truth(row: dict[str, Any]) -> int:
    """Count stars matching target_color in the dataset's circles metadata."""
    circles = row.get("circles")
    target = row.get("target_color")
    if not isinstance(circles, list) or not isinstance(target, str) or not target:
        raise ValueError("Star-count row lacks valid circle/color reward metadata.")
    if any(not isinstance(circle, dict) or "color" not in circle for circle in circles):
        raise ValueError("Star-count circle metadata is malformed.")
    return sum(circle["color"] == target for circle in circles)


def final_text(completion: str, *, starts_in_thinking: bool = True,
               require_separator: bool = True) -> str:
    """Parse a continuation whose generation prefix normally ends in <think>."""
    if not isinstance(completion, str):
        raise ValueError("Completion must be text.")
    if starts_in_thinking or "<think>" in completion:
        _, separator, final = completion.partition("</think>")
        return final.strip() if separator else ""
    if require_separator:
        _, separator, final = completion.partition("</think>")
        return final.strip() if separator else ""
    return completion.strip()


def reward(completion: str, row: dict[str, Any], *, starts_in_thinking: bool = True,
           require_separator: bool = True) -> float:
    """Binary exact reward from the first literal boxed integer in final text."""
    final = final_text(completion, starts_in_thinking=starts_in_thinking,
                       require_separator=require_separator)
    match = BOXED_INTEGER.search(final)
    return float(match is not None and int(match.group(1)) == ground_truth(row))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_data(manifest_path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Require both the manifest identity and the byte-identical frozen rows."""
    manifest = _json_file(Path(manifest_path) if manifest_path else ROOT / "data_manifest.json", "data manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(FROZEN_SPLITS):
        raise ValueError("Frozen dataset manifest must contain train and validation splits.")
    output = {}
    for split, frozen in FROZEN_SPLITS.items():
        expected = splits[split]
        if not isinstance(expected, dict) or any(expected.get(key) != frozen[key] for key in ("rows", "sha256")):
            raise ValueError(f"Frozen {split} dataset manifest identity mismatch.")
        raw_path = expected.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"Frozen {split} dataset path is missing.")
        relative = Path(raw_path)
        if relative.is_absolute() or relative != Path(frozen["path"]):
            raise ValueError(f"Frozen {split} dataset must use its packaged relative path.")
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Frozen {split} dataset must remain inside the package.")
        try:
            checksum = sha256_file(path)
        except OSError:
            raise ValueError(f"Cannot read frozen {split} dataset.") from None
        if checksum != expected["sha256"]:
            raise ValueError(f"Frozen {split} checksum mismatch.")
        rows = []
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        raise ValueError(f"Frozen {split} contains an empty row at line {line_number}.")
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"Frozen {split} contains invalid JSON at line {line_number}.") from None
                    if not isinstance(row, dict):
                        raise ValueError(f"Frozen {split} row {line_number} is not an object.")
                    ground_truth(row)
                    rows.append(row)
        except (OSError, UnicodeError):
            raise ValueError(f"Cannot decode frozen {split} dataset.") from None
        if len(rows) != expected["rows"]:
            raise ValueError(f"Frozen {split} row count mismatch.")
        output[split] = rows
    return output


def validate_tokenizer(manifest: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    snapshot = manifest.get("tokenizer")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str):
        raise ValueError("Tokenizer snapshot is missing from the data manifest.")
    relative = Path(snapshot["path"])
    if relative.is_absolute() or relative != Path("tokenizer"):
        raise ValueError("Tokenizer must use its packaged relative path.")
    directory = ROOT / relative
    if directory.is_symlink() or not directory.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Tokenizer must remain inside the package.")
    expected = snapshot.get("file_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Tokenizer manifest has no pinned file hashes.")
    required = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    if not required.issubset(expected):
        raise ValueError("Tokenizer manifest must pin the tokenizer and chat-template files.")
    if any(not isinstance(name, str) or Path(name).name != name for name in expected):
        raise ValueError("Tokenizer file hashes must use plain filenames.")
    try:
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    except OSError:
        raise ValueError("Cannot read tokenizer snapshot.") from None
    if actual_names != set(expected):
        raise ValueError("Tokenizer snapshot files differ from the manifest.")
    verified = {}
    for name, checksum in expected.items():
        if not isinstance(checksum, str) or re.fullmatch(r"[a-f0-9]{64}", checksum) is None:
            raise ValueError("Tokenizer manifest contains an invalid SHA256 hash.")
        try:
            actual = sha256_file(directory / name)
        except OSError:
            raise ValueError(f"Cannot read tokenizer snapshot file {name}.") from None
        if actual != checksum:
            raise ValueError(f"Frozen tokenizer checksum mismatch: {name}.")
        verified[name] = actual
    return directory, verified


def prepare(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build every native vision prompt locally, with no sampling or allocation."""
    from transformers import AutoTokenizer
    from vision import build_prompt

    manifest = _json_file(ROOT / "data_manifest.json", "data manifest")
    rows = read_data()
    tokenizer_dir, tokenizer_hashes = validate_tokenizer(manifest)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    maximum = int(config["max_seq_len"])
    response_maximum = int(config["max_new_tokens"])
    if maximum < 1 or response_maximum < 1 or response_maximum >= maximum:
        raise ValueError("Invalid prompt/completion token limits in recipe.")
    prepared = {}
    split_audits = {}
    for split, examples in rows.items():
        entries = []
        resized_rows = 0
        image_digest = hashlib.sha256()
        prepared_digest = hashlib.sha256()
        dimension_counts = {}
        for index, row in enumerate(examples):
            try:
                prompt, expected_ids, metadata = build_prompt(
                    row, tokenizer, max_seq_len=maximum, max_new_tokens=response_maximum,
                    allow_quantized_resize=config.get("allow_quantized_resize", False),
                )
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"Frozen {split} row {index + 1} prompt audit failed: {error}") from None
            prompt_tokens = len(expected_ids)
            if prompt_tokens != metadata["prompt_tokens"] or prompt_tokens + response_maximum > maximum:
                raise ValueError(f"Frozen {split} row {index + 1} violates the sequence token bound.")
            start, end = metadata["image_target_span"]
            if not 0 <= start < end <= prompt_tokens or end - start != metadata["expected_image_tokens"]:
                raise ValueError(f"Frozen {split} row {index + 1} has an inconsistent image token span.")
            if metadata.get("enable_thinking") is not True:
                raise ValueError("Every star-count prompt must enable thinking.")
            resized_rows += metadata["resize_applied"]
            image_digest.update(metadata["original_image_sha256"].encode("ascii"))
            prepared_digest.update(metadata["prepared_image_sha256"].encode("ascii"))
            dimensions = "x".join(str(value) for value in metadata["original_size"])
            dimension_counts[dimensions] = dimension_counts.get(dimensions, 0) + 1
            entries.append({"row": row, "prompt": prompt, "expected_prompt_ids": expected_ids,
                            "metadata": metadata, "ground_truth": ground_truth(row)})
        lengths = [entry["metadata"]["prompt_tokens"] for entry in entries]
        image_lengths = [entry["metadata"]["expected_image_tokens"] for entry in entries]
        split_audits[split] = {
            "rows": len(entries), "dataset_sha256": manifest["splits"][split]["sha256"],
            "minimum_prompt_tokens": min(lengths), "maximum_prompt_tokens": max(lengths),
            "maximum_sequence_tokens": max(lengths) + response_maximum,
            "minimum_image_tokens": min(image_lengths), "maximum_image_tokens": max(image_lengths),
            "images_audited": len(entries), "quantized_resize_rows": resized_rows,
            "original_image_dimensions": dimension_counts,
            "ordered_original_image_hashes_sha256": image_digest.hexdigest(),
            "ordered_prepared_image_hashes_sha256": prepared_digest.hexdigest(),
        }
        prepared[split] = entries
    first = prepared["train"][0]["metadata"]
    audit = {
        "base_model": config["base_model"], "splits": split_audits,
        "max_seq_len": maximum, "max_new_tokens": response_maximum,
        "tokenizer_file_sha256": tokenizer_hashes,
        "processor_fingerprint": first["processor_fingerprint"],
        "enable_thinking": True, "reward_requires_thinking_separator": True,
        "reward": "First literal boxed integer in final text; exact circle/color metadata count.",
        "truncated_rows": 0, "row_order": "Frozen file order.",
        "allow_quantized_resize": config.get("allow_quantized_resize", False),
        "compute_provisioned": False,
    }
    return prepared, audit
