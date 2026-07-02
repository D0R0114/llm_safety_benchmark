#!/usr/bin/env python3
"""
LLM Safety Benchmark — Build Dataset Pipeline.

Loads prompts from 3 public datasets, deduplicates them, classifies each prompt
using the Opir-multitask-large-v1.0 safety classifier, and outputs a labeled
JSONL benchmark file.

Usage:
    python scripts/build_dataset.py                          # defaults: 10k/source, CPU
    python scripts/build_dataset.py --max-per-source 100     # test with small sample
    python scripts/build_dataset.py --device cuda --batch-size 32
"""

import argparse
import datetime
import json
import lzma
import os
import sys
from collections import OrderedDict

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import SafetyClassifier, TOP_LEVEL_SAFETY_LABELS


# ──────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────

def setup_model(model_path: str) -> bool:
    """Verify the model directory has all required files.

    Returns True if model is ready, False if files are missing.
    """
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        print(f"[ERROR] Model files missing from {model_path}:")
        for f in missing:
            print(f"  - {f}")
        print(f"Please download them from https://huggingface.co/knowledgator/opir-multitask-large-v1.0")
        return False
    return True


# ──────────────────────────────────────────────
# Data loaders — each capped at max_per_source
# ──────────────────────────────────────────────

def load_real_toxicity_prompts(path: str, max_count: int) -> list:
    """Load from JSONL, extract prompt.text field. Take first max_count."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading RealToxicityPrompts", unit="lines"):
            if len(prompts) >= max_count:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("prompt", {}).get("text", "")
                if text and text.strip():
                    prompts.append(text.strip())
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(prompts)} prompts from RealToxicityPrompts")
    return prompts


def load_jigsaw(path: str, max_count: int) -> list:
    """Load from CSV, extract comment_text field. Take first max_count."""
    prompts = []
    for chunk in pd.read_csv(path, chunksize=10000, usecols=["comment_text"]):
        if len(prompts) >= max_count:
            break
        texts = chunk["comment_text"].dropna().tolist()
        for t in texts:
            if len(prompts) >= max_count:
                break
            s = str(t).strip()
            if s:
                prompts.append(s)
    print(f"  Loaded {len(prompts)} prompts from Jigsaw")
    return prompts


def load_beavertails(path: str, max_count: int) -> list:
    """Load from XZ-compressed JSONL, extract prompt field. Take first max_count."""
    prompts = []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading BeaverTails", unit="lines"):
            if len(prompts) >= max_count:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("prompt", "")
                if text and text.strip():
                    prompts.append(text.strip())
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(prompts)} prompts from BeaverTails")
    return prompts


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────

def _build_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Build a MinHash signature using character trigrams."""
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - 2):
        m.update(text[i:i + 3].encode("utf-8"))
    return m


def deduplicate(prompts: list, threshold: float = 0.85) -> list:
    """Two-phase dedup: exact match (set) then MinHash LSH near-dedup."""
    # Phase 1: exact match
    seen = set()
    exact_unique = []
    for p in tqdm(prompts, desc="Exact dedup"):
        if p not in seen:
            seen.add(p)
            exact_unique.append(p)
    print(f"  After exact dedup: {len(exact_unique)} (removed {len(prompts) - len(exact_unique)})")

    # Phase 2: MinHash LSH near-dedup
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    near_unique = []
    for i, p in enumerate(tqdm(exact_unique, desc="MinHash dedup")):
        m = _build_minhash(p)
        result = lsh.query(m)
        if not result:
            lsh.insert(i, m)
            near_unique.append(p)

    print(f"  After MinHash dedup (threshold={threshold}): {len(near_unique)} "
          f"(removed {len(exact_unique) - len(near_unique)})")
    return near_unique


# ──────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────

def load_checkpoint(path: str) -> tuple:
    """Load checkpoint file. Returns (completed_indices, completed_results)."""
    indices = set()
    results = {}  # index -> result dict
    if not os.path.exists(path):
        return indices, results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                idx = rec["index"]
                indices.add(idx)
                results[idx] = {
                    "category": rec["category"],
                    "severity": rec["severity"],
                    "tags": rec["tags"],
                }
            except json.JSONDecodeError:
                continue
    return indices, results


def classify_prompts(
    prompts: list,
    classifier: SafetyClassifier,
    checkpoint_path: str = None,
    resume: bool = False,
) -> list:
    """Classify all prompts with progress bar and periodic checkpointing."""
    results = []
    start_idx = 0

    if resume and checkpoint_path:
        completed_indices, completed_results = load_checkpoint(checkpoint_path)
        if completed_indices:
            results = [completed_results.get(i) for i in range(len(prompts))]
            results = [r if r is not None else None for r in results]
            start_idx = len(completed_indices)
            print(f"  Resuming from checkpoint: {start_idx}/{len(prompts)} already classified, "
                  f"{len(prompts) - start_idx} remaining")

    checkpoint_interval = 500

    for i, prompt in enumerate(tqdm(prompts, desc="Classifying"), start=0):
        if i < start_idx:
            continue
        res = classifier.classify(prompt)
        # Extend results if needed
        while len(results) <= i:
            results.append(None)
        results[i] = res

        # Save checkpoint periodically
        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(checkpoint_path, prompts[:i + 1], [r for r in results[:i + 1] if r is not None])

    # Final checkpoint
    if checkpoint_path:
        final_results = [r for r in results if r is not None]
        _save_checkpoint(checkpoint_path, prompts[:len(final_results)], final_results)

    # Ensure all results are filled
    results = [r if r is not None else {"category": "safe", "severity": "safe", "tags": ""} for r in results]
    return results


def _save_checkpoint(path: str, prompts: list, classifications: list):
    """Save intermediate results to a checkpoint file."""
    with open(path, "w", encoding="utf-8") as f:
        for j, (prompt, cls) in enumerate(zip(prompts, classifications)):
            record = {
                "index": j,
                "prompt": prompt,
                "category": cls["category"],
                "severity": cls["severity"],
                "tags": cls["tags"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────

def format_output(prompts: list, classifications: list) -> list:
    """Combine prompts and classification results into final records."""
    records = []
    for i, (prompt, cls) in enumerate(zip(prompts, classifications)):
        record = OrderedDict([
            ("id", f"safety-{i + 1:06d}"),
            ("category", cls["category"]),
            ("prompt", prompt),
            ("tags", [cls["tags"]] if cls["tags"] else []),
            ("severity", cls["severity"]),
            ("notes", ""),
        ])
        records.append(record)
    return records


def write_meta(output_path: str, total: int, categories: dict, dedup_threshold: float,
               classification_threshold: float, data_sources: list):
    """Generate meta.json with version and statistics."""
    category_names = sorted(categories.keys())
    meta = OrderedDict([
        ("version", "1.0.0"),
        ("created_date", datetime.date.today().isoformat()),
        ("total_cases", total),
        ("num_categories", len(categories)),
        ("categories", category_names),
        ("category_counts", OrderedDict(
            (cat, categories.get(cat, 0)) for cat in category_names
        )),
        ("data_sources", data_sources),
        ("classifier", "knowledgator/opir-multitask-large-v1.0"),
        ("classification_threshold", classification_threshold),
        ("dedup_threshold", dedup_threshold),
    ])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Meta written to {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build LLM Safety Benchmark dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cpu",
                        help="Device for inference: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification confidence threshold (default: 0.5)")
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="MinHash LSH dedup threshold (default: 0.85)")
    parser.add_argument("--max-per-source", type=int, default=10000,
                        help="Max prompts to load from each source (default: 10000)")
    parser.add_argument("--model-path", default=None,
                        help="Path to model directory (default: models/opir-multitask-large-v1.0)")
    parser.add_argument("--raw-dir", default=None,
                        help="Path to raw data directory (default: data/raw)")
    parser.add_argument("--output-dir", default=None,
                        help="Path for output files (default: data/)")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip classification (useful for testing pipeline)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume classification from checkpoint file")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model_path or os.path.join(project_root, "models", "opir-multitask-large-v1.0")
    raw_dir = args.raw_dir or os.path.join(project_root, "data", "raw")
    output_dir = args.output_dir or os.path.join(project_root, "data")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "safety_benchmark.jsonl")
    meta_path = os.path.join(output_dir, "meta.json")
    checkpoint_path = os.path.join(output_dir, ".classify_checkpoint.jsonl")

    print("=" * 60)
    print("LLM Safety Benchmark — Build Pipeline")
    print("=" * 60)
    print(f"Device:       {args.device}")
    print(f"Max/source:   {args.max_per_source}")
    print(f"Threshold:    {args.threshold}")
    print(f"Dedup thresh: {args.dedup_threshold}")
    print(f"Model:        {model_path}")
    print(f"Raw data:     {raw_dir}")
    print(f"Output:       {output_dir}")
    print()

    # 1. Verify model
    print("[1/6] Checking model...")
    if not setup_model(model_path):
        sys.exit(1)
    print("  Model files OK")

    # 2. Load raw data
    print("[2/6] Loading raw data...")
    rtp_path = os.path.join(raw_dir, "real-toxicity-prompts", "prompts.jsonl")
    jigsaw_path = os.path.join(raw_dir, "jigsaw", "train.csv")
    beaver_path = os.path.join(raw_dir, "beavertails", "train.jsonl.xz")

    all_prompts = []
    for loader, path, name in [
        (load_real_toxicity_prompts, rtp_path, "RealToxicityPrompts"),
        (load_jigsaw, jigsaw_path, "Jigsaw"),
        (load_beavertails, beaver_path, "BeaverTails"),
    ]:
        if os.path.exists(path):
            all_prompts.extend(loader(path, args.max_per_source))
        else:
            print(f"  [WARNING] {name}: file not found at {path}, skipping.")
            print(f"    Download from the HuggingFace Dataset Hub and place in data/raw/")

    if not all_prompts:
        print("[ERROR] No prompts loaded. Please download the datasets first.")
        sys.exit(1)
    print(f"  Total raw prompts: {len(all_prompts)}")

    # 3. Deduplicate
    print("[3/6] Deduplicating...")
    unique_prompts = deduplicate(all_prompts, threshold=args.dedup_threshold)
    print(f"  Unique prompts: {len(unique_prompts)}")

    # 4. Classify
    if args.skip_classification:
        print("[4/6] Skipping classification (--skip-classification)")
        classifications = [{"category": "safe", "severity": "safe", "tags": ""}
                          for _ in unique_prompts]
    else:
        print("[4/6] Classifying prompts...")
        print("  WARNING: CPU classification is slow. Use --device cuda if available.")
        print(f"  Estimated time on CPU: ~{len(unique_prompts) * 0.5 / 3600:.1f} hours")
        print(f"  Estimated time on GPU: ~{len(unique_prompts) * 0.02 / 60:.1f} minutes")

        classifier = SafetyClassifier(
            model_path=model_path,
            device=args.device,
            threshold=args.threshold,
        )
        classifications = classify_prompts(
            unique_prompts, classifier,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )

    # 5. Format and write output
    print("[5/6] Writing output...")
    records = format_output(unique_prompts, classifications)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {output_path}")

    # 6. Write meta
    print("[6/6] Writing meta...")
    category_counts = {}
    for r in records:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    write_meta(
        meta_path, len(records), category_counts,
        dedup_threshold=args.dedup_threshold,
        classification_threshold=args.threshold,
        data_sources=[
            "allenai/real-toxicity-prompts",
            "google/jigsaw_toxicity_prediction",
            "PKU-Alignment/BeaverTails",
        ],
    )

    # Summary
    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Total cases:  {len(records)}")
    print(f"Categories:   {len(category_counts)}")
    print(f"Output:       {output_path}")
    print(f"Meta:         {meta_path}")
    print()
    print("Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        print(f"  {cat:.<55s} {count:>6d} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
