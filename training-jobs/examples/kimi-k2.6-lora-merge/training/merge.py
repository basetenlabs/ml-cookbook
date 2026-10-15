#!/usr/bin/env python3
"""Merge a Loops LoRA adapter into a quantized Kimi-K2.6 base, on CPU.

The Kimi-K2.6 INT4 release stores attention + lm_head as BF16 and only the MoE
experts as packed INT4. Loops adapters target attention + lm_head, so every
target lands on a BF16 tensor and we can merge in place without dequantizing —
the experts (and the base's quantization_config) stay untouched, so the output
is a drop-in checkpoint that serves on stock vLLM with no patches.

Output is a full HF checkpoint: rewritten shards for the touched tensors, every
other file (experts, config, tokenizer) copied through unchanged.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."


def strip_adapter_key(adapter_key: str) -> tuple[str, str]:
    key = adapter_key.removeprefix(ADAPTER_PREFIX)
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")] + ".weight", "A"
    if key.endswith(".lora_B.weight"):
        return key[: -len(".lora_B.weight")] + ".weight", "B"
    raise ValueError(f"Unrecognized adapter key: {adapter_key}")


def load_adapter_pairs(adapter_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            base_key, side = strip_adapter_key(k)
            pairs.setdefault(base_key, {})[side] = f.get_tensor(k)
    incomplete = [k for k, p in pairs.items() if set(p) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"Incomplete LoRA pairs: {incomplete[:5]}")
    return pairs


def normalize_to_base(key: str, weight_map: dict[str, str]) -> str:
    # Tinker adapters may use the text-model namespace; the VL checkpoint nests
    # those tensors under language_model.*.
    if key in weight_map:
        return key
    for cand in (f"language_model.{key}", key.removeprefix("language_model.")):
        if cand in weight_map:
            return cand
    return key


def validate_targets(pairs: dict, weight_map: dict[str, str]) -> None:
    packed, missing = [], []
    for key in pairs:
        if key in weight_map:
            continue
        if key[: -len(".weight")] + ".weight_packed" in weight_map:
            packed.append(key)
        else:
            missing.append(key)
    if packed:
        raise ValueError(
            f"{len(packed)} targets are packed (quantized) tensors this fast "
            f"merger cannot touch: {packed[:5]}"
        )
    if missing:
        raise ValueError(f"{len(missing)} targets missing from base: {missing[:5]}")


def stage_then_place(write_to, shard: str, staging_dir: Path, out_dir: Path) -> None:
    # safetensors writes its atomic-rename temp (.tmp<random>) into the target's
    # own directory. Point that at staging/ (same filesystem, but no config.json
    # so it is never reconciled as a checkpoint), then os.rename the finished
    # file into out_dir — a same-fs atomic move, so the checkpoint dir only ever
    # receives complete shards and never a transient .tmp for the sync to orphan.
    staged = staging_dir / shard
    write_to(str(staged))
    os.replace(staged, out_dir / shard)


def merge_shard(shard: str, base_dir: Path, staging_dir: Path, out_dir: Path, pairs: dict, scale: float) -> int:
    merged = 0
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(base_dir / shard), framework="pt", device="cpu") as f:
        for key in f.keys():
            w = f.get_tensor(key)
            pair = pairs.get(key)
            if pair is not None:
                a = pair["A"].to(torch.float32)
                b = pair["B"].to(torch.float32)
                w = w.to(torch.float32).addmm(b, a, alpha=scale).to(w.dtype)
                merged += 1
            tensors[key] = w
    stage_then_place(lambda dst: save_file(tensors, dst), shard, staging_dir, out_dir)
    return merged


def _worker(args):
    return args[0], merge_shard(*args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    base_dir, out_dir = Path(args.base), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sibling of out_dir on the same filesystem. Holds safetensors' transient
    # .tmp* writes; has no config.json, so it is never picked up as a checkpoint.
    staging_dir = out_dir.parent / f"_{out_dir.name}_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((Path(args.adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]

    pairs = {normalize_to_base(k, weight_map): v for k, v in load_adapter_pairs(Path(args.adapter_dir)).items()}
    validate_targets(pairs, weight_map)

    shards = sorted(set(weight_map.values()))
    touched = {weight_map[k] for k in pairs}
    print(f"Preflight OK: {len(pairs)} pairs, scale={scale}, {len(touched)}/{len(shards)} shards touched")

    # Carry every non-shard file (experts config, tokenizer, modeling code,
    # quantization_config) through verbatim, plus the untouched shards. Skip the
    # base mount's weight-mirror artifacts: a copied .complete sentinel makes
    # downstream mirroring of this checkpoint think it is already finished, and
    # stray .tmp* download files just bloat the output.
    for item in base_dir.iterdir():
        if item.name in shards or item.name == ".complete" or item.name.startswith(".tmp"):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, out_dir / item.name)
    for shard in shards:
        if shard not in touched:
            stage_then_place(lambda dst: shutil.copy2(base_dir / shard, dst), shard, staging_dir, out_dir)

    work = [(s, base_dir, staging_dir, out_dir, {k: pairs[k] for k in pairs if weight_map[k] == s}, scale) for s in sorted(touched)]
    applied = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed(pool.submit(_worker, w) for w in work):
            shard, n = fut.result()
            applied += n
            print(f"  {shard}: {n} merges")

    shutil.rmtree(staging_dir, ignore_errors=True)
    if applied != len(pairs):
        raise RuntimeError(f"Applied {applied}/{len(pairs)} LoRA pairs")
    print(f"DONE: merged {applied} LoRA pairs into {len(shards)} shards at {out_dir}")


if __name__ == "__main__":
    main()
