import hashlib
import io
import json
from pathlib import Path

from baseten.loops import Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData

from common import ROOT
from prepare_assets import verify_assets
from vision import _prepare_image, load_processor_config

SPLIT_ROWS = {"train": 800, "validation": 100}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json2token(value):
    if isinstance(value, dict):
        if len(value) == 1 and "text_sequence" in value:
            return value["text_sequence"]
        return "".join(
            f"<s_{key}>" + json2token(value[key]) + f"</s_{key}>"
            for key in sorted(value, reverse=True)
        )
    if isinstance(value, list):
        return "<sep/>".join(json2token(item) for item in value)
    return str(value)


def encode_row(tokenizer, answer, expected_image_tokens, maximum):
    if not isinstance(answer, str) or not answer:
        raise ValueError("Receipt answer must be nonempty text.")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if rendered.count("<image>") != 1 or "<img>" in rendered or "</img>" in rendered:
        raise ValueError("Unexpected chat template image expansion.")
    prefix, suffix = rendered.split("<image>")
    expanded = prefix + "<img>" + "<image>" * expected_image_tokens + "</img>" + suffix
    encoded = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    if len(input_ids) > maximum:
        raise ValueError(
            f"Complete sequence has {len(input_ids)} tokens, exceeds {maximum}; truncation forbidden."
        )
    answer_start = expanded.rfind(answer)
    if answer_start < expanded.index("</img>"):
        raise ValueError("Assistant answer is not after the image.")
    labels = [
        token if end > answer_start else -100
        for token, (_, end) in zip(input_ids, encoded["offset_mapping"])
    ]
    image_ids = tokenizer.encode("<image>", add_special_tokens=False)
    if len(image_ids) != 1:
        raise ValueError("Expected a single-token image placeholder.")
    prefix_ids = tokenizer.encode(prefix + "<img>", add_special_tokens=False)
    suffix_ids = tokenizer.encode("</img>" + suffix, add_special_tokens=False)
    if prefix_ids + image_ids * expected_image_tokens + suffix_ids != input_ids:
        raise ValueError("Multimodal chunk boundaries changed tokenization.")
    targets = labels[1:]
    media_start = len(prefix_ids)
    media_end = media_start + expected_image_tokens
    if any(token != -100 for token in targets[media_start:media_end]):
        raise ValueError("Image positions must not receive supervised loss.")
    if not suffix_ids or not any(token != -100 for token in targets):
        raise ValueError("Example has no assistant targets.")
    return {
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids[:-1],
        "target_tokens": targets,
        "expected_image_tokens": expected_image_tokens,
        "sequence_tokens": len(input_ids),
        "active_tokens": sum(token != -100 for token in targets),
    }


def prepare_row(row, index, split, tokenizer, config, directory):
    from PIL import Image

    ground_truth = json.loads(row["ground_truth"])
    candidates = ground_truth.get("gt_parses", [ground_truth.get("gt_parse")])
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("Expected exactly one authoritative CORD receipt parse.")
    answer = json2token(candidates[0])
    raw = row["image"]["bytes"]
    if not raw:
        raise ValueError("Parquet row does not contain inline receipt image bytes.")
    with Image.open(io.BytesIO(raw)) as image:
        image_format = {"PNG": "png", "JPEG": "jpeg"}.get(image.format)
    if image_format is None:
        raise ValueError("Only inline JPEG and PNG images are supported.")
    prepared, image_format, geometry, image_audit = _prepare_image(
        raw,
        image_format,
        load_processor_config(),
        allow_quantized_resize=True,
    )
    encoded = encode_row(
        tokenizer, answer, geometry.expected_tokens, config["max_seq_len"]
    )
    relative_image = f"images/{split}-{index:04d}.{image_format}"
    (directory / relative_image).write_bytes(prepared)
    return {
        **encoded,
        "split": split,
        "row_index": index,
        "image": relative_image,
        "image_format": image_format,
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        **image_audit,
    }


def load_prepared(config):
    assets = verify_assets()
    directory = ROOT / "data" / "prepared"
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except FileNotFoundError:
        raise ValueError(
            "Run uv run python prepare_data.py before preflight/training."
        ) from None
    if (
        manifest["dataset_revision"] != config["dataset_revision"]
        or manifest["max_seq_len"] != config["max_seq_len"]
    ):
        raise ValueError("Cached dataset revision or context does not match recipe.")
    if manifest.get("asset_audit", {}).get("files") != assets["files"]:
        raise ValueError(
            "Tokenizer/template/processor fingerprints do not match preparation; rebuild in a fresh recipe copy."
        )
    if manifest["tokenizer_sha256"] != sha256(
        ROOT / "assets" / "tokenizer" / "tokenizer.json"
    ):
        raise ValueError("Tokenizer has changed since data preparation.")
    if manifest["processor_sha256"] != sha256(
        ROOT / "assets" / "preprocessor_config.json"
    ):
        raise ValueError("Image processor configuration has changed.")
    data = {}
    image_hashes = {}
    for split, count in SPLIT_ROWS.items():
        source = directory / f"{split}.jsonl"
        if sha256(source) != manifest["splits"][split]["sha256"]:
            raise ValueError("Cached encoded rows failed their SHA256 audit.")
        data[split] = [json.loads(line) for line in source.read_text().splitlines()]
        if len(data[split]) != count:
            raise ValueError(f"Expected all {count} {split} rows.")
        for row in data[split]:
            if row["sequence_tokens"] > config["max_seq_len"]:
                raise ValueError("Cached example exceeds context.")
            if sha256(directory / row["image"]) != row["prepared_image_sha256"]:
                raise ValueError("Cached image failed its SHA256 audit.")
            length = (
                len(row["prefix_ids"])
                + row["expected_image_tokens"]
                + len(row["suffix_ids"])
            )
            if (
                length != len(row["target_tokens"])
                or length != row["sequence_tokens"] - 1
            ):
                raise ValueError(
                    "Cached multimodal inputs and shifted targets differ in length."
                )
        image_hashes[split] = {row["original_image_sha256"] for row in data[split]}
    if image_hashes["train"] & image_hashes["validation"]:
        raise ValueError("Training and validation image bytes overlap.")
    return data, manifest


def datum(example):
    targets = example["target_tokens"]
    image = ROOT / "data" / "prepared" / example["image"]
    model_input = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=example["prefix_ids"]),
            ImageChunk(
                data=image.read_bytes(),
                format=example["image_format"],
                expected_tokens=example["expected_image_tokens"],
            ),
            EncodedTextChunk(tokens=example["suffix_ids"]),
        ]
    )
    if model_input.length != len(targets):
        raise ValueError("Each expanded input position needs one shifted target.")
    return Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=targets, dtype="int64", shape=[len(targets)]
            ),
        },
    )
