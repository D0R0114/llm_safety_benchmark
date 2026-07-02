#!/usr/bin/env python3
"""
LLM Safety Benchmark — Build Dataset Pipeline (Label-Based).

Loads prompts from 3 public datasets, maps their original labels to the
17-category safety taxonomy, deduplicates, and outputs a labeled JSONL benchmark.

No classifier needed — uses the datasets' own human-annotated labels.

Usage:
    python scripts/build_dataset.py                        # defaults: 1k/source
    python scripts/build_dataset.py --max-per-source 100   # test with small sample
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

# ──────────────────────────────────────────────
# Label mapping: original dataset labels → safety taxonomy
# ──────────────────────────────────────────────

# BeaverTails boolean categories → our taxonomy
BEAVERTAILS_MAP = {
    "hate_speech,offensive_language": "toxicity",
    "violence,aiding_and_abetting,incitement": "violence_and_physical_harm",
    "self_harm": "self_harm_and_suicide",
    "sexually_explicit,adult_content": "sexual_content",
    "child_abuse": "child_safety",
    "privacy_violation": "personal_information_privacy_and_intellectual_property",
    "financial_crime,property_crime,theft": "criminal_and_illegal_activity",
    "terrorism,organized_crime": "criminal_and_illegal_activity",
    "drug_abuse,weapons,banned_substance": "regulated_goods_and_advice",
    "misinformation_regarding_ethics,laws_and_safety": "information_integrity_and_manipulation",
    "discrimination,stereotype,injustice": "bias_fairness_and_representation",
    "non_violent_unethical_behavior": "other_or_uncertain",
    "controversial_topics,politics": "other_or_uncertain",
    "animal_abuse": "violence_and_physical_harm",
}

# Jigsaw binary columns → our taxonomy
JIGSAW_MAP = [
    ("toxic", "toxicity"),
    ("severe_toxic", "toxicity"),
    ("obscene", "sexual_content"),
    ("threat", "violence_and_physical_harm"),
    ("insult", "toxicity"),
    ("identity_hate", "bias_fairness_and_representation"),
]

# RealToxicityPrompts score fields → our taxonomy (continuous scores)
RTP_MAP = [
    ("toxicity", "toxicity"),
    ("severe_toxicity", "toxicity"),
    ("insult", "toxicity"),
    ("profanity", "toxicity"),
    ("threat", "violence_and_physical_harm"),
    ("identity_attack", "bias_fairness_and_representation"),
    ("sexually_explicit", "sexual_content"),
]

# WildGuard subcategory → our taxonomy
WILDGUARD_MAP = {
    "benign": "safe",
    "toxic_language_hate_speech": "toxicity",
    "violence_and_physical_harm": "violence_and_physical_harm",
    "sexual_content": "sexual_content",
    "cyberattack": "cybersecurity",
    "private_information_individual": "personal_information_privacy_and_intellectual_property",
    "sensitive_information_organization_government": "personal_information_privacy_and_intellectual_property",
    "copyright_violations": "personal_information_privacy_and_intellectual_property",
    "fraud_assisting_illegal_activities": "criminal_and_illegal_activity",
    "defamation_encouraging_unethical_or_unsafe_actions": "criminal_and_illegal_activity",
    "social_stereotypes_and_unfair_discrimination": "bias_fairness_and_representation",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": "information_integrity_and_manipulation",
    "causing_material_harm_by_disseminating_misinformation": "information_integrity_and_manipulation",
    "mental_health_over-reliance_crisis": "self_harm_and_suicide",
}

# Keyword patterns to extract missing categories from 'others'
WILDGUARD_KEYWORD_MAP = [
    (["nuclear weapon", "biological weapon", "chemical weapon", "WMD", "weapon of mass",
      "nerve gas", "mustard gas", "ricin", "anthrax", "dirty bomb",
      "homemade explosive", "pipe bomb", "C4 explosive"],
     "weapons_of_mass_destruction"),
    (["jailbreak", "prompt injection", "DAN prompt", "ignore previous",
      "bypass content", "override safety", "roleplay as unrestricted",
      "developer mode", "pretend you are an AI without"],
     "ai_system_security_and_reliability"),
    (["child abuse", "child sexual", "child porn", "minor sexually",
      "underage", "CSAM", "grooming a child", "child exploitation"],
     "child_safety"),
    (["biological hazard", "environmental sabotage", "bioterror",
      "pandemic release", "release a pathogen", "ecosystem destruction",
      "toxic spill", "contaminate water", "undetectable poison",
      "potassium cyanide", "how to poison", "make poison",
      "create a bioweapon", "release a virus"],
     "biological_medical_and_environmental_harm"),
    (["drug trafficking", "sell drugs", "how to make meth", "synthesize drug",
      "illegal firearm", "buy a gun illegally", "smuggle weapons"],
     "regulated_goods_and_advice"),
    (["hacking into", "how to hack", "DDOS", "phishing email", "malware",
      "ransomware", "keylogger", "steal password", "brute force"],
     "cybersecurity"),
]

SEVERITY_MAP = {
    "toxicity": "low",
    "violence_and_physical_harm": "high",
    "self_harm_and_suicide": "high",
    "sexual_content": "low",
    "child_safety": "high",
    "personal_information_privacy_and_intellectual_property": "medium",
    "cybersecurity": "medium",
    "criminal_and_illegal_activity": "high",
    "regulated_goods_and_advice": "high",
    "biological_medical_and_environmental_harm": "high",
    "weapons_of_mass_destruction": "high",
    "information_integrity_and_manipulation": "medium",
    "ai_system_security_and_reliability": "high",
    "bias_fairness_and_representation": "medium",
    "other_or_uncertain": "medium",
}


# ──────────────────────────────────────────────
# Data loaders — extract prompts + original labels
# ──────────────────────────────────────────────

def load_real_toxicity_prompts(path: str, max_count: int) -> list:
    """Load from JSONL, mapping toxicity scores to safety categories."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading RealToxicityPrompts", unit="lines"):
            if len(records) >= max_count:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("prompt", {}).get("text", "")
                if not text or not text.strip():
                    continue
                prompt = text.strip()

                # Gather all risk scores
                scores = {}
                for field, cat in RTP_MAP:
                    val = obj.get("prompt", {}).get(field, 0) or 0
                    scores[cat] = max(scores.get(cat, 0), val)

                # Find highest-scoring category
                max_score = max(scores.values())
                if max_score < 0.5:
                    category = "safe"
                else:
                    # Pick the max-score category (first in case of tie)
                    category = max(scores, key=scores.get)

                tags_str = ", ".join(f"{k}={v:.2f}" for k, v in
                                     sorted(scores.items(), key=lambda x: -x[1]))

                records.append({
                    "prompt": prompt,
                    "category": category,
                    "tags": f"original_scores: {tags_str}",
                })
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  Loaded {len(records)} records from RealToxicityPrompts")
    return records


def load_jigsaw(path: str, max_count: int) -> list:
    """Load from CSV, mapping binary toxicity columns to safety categories."""
    records = []
    for chunk in pd.read_csv(path, chunksize=10000,
                             usecols=["comment_text", "toxic", "severe_toxic",
                                      "obscene", "threat", "insult", "identity_hate"]):
        if len(records) >= max_count:
            break
        for _, row in chunk.iterrows():
            if len(records) >= max_count:
                break
            text = str(row.get("comment_text", "")).strip()
            if not text or text == "nan":
                continue

            # Collect triggered labels
            triggered = []
            for col, cat in JIGSAW_MAP:
                if int(row.get(col, 0)) == 1:
                    triggered.append((col, cat))

            if not triggered:
                category = "safe"
                tags_str = "all_original_labels=0"
            else:
                # Pick first triggered label as primary category
                tags_parts = [f"{col}=1" for col, cat in triggered]
                tags_str = ", ".join(tags_parts)
                category = triggered[0][1]  # first match wins

            records.append({
                "prompt": text,
                "category": category,
                "tags": f"original_labels: {tags_str}",
            })
    print(f"  Loaded {len(records)} records from Jigsaw")
    return records


def load_beavertails(path: str, max_count: int) -> list:
    """Load from XZ-compressed JSONL, mapping boolean category flags."""
    records = []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading BeaverTails", unit="lines"):
            if len(records) >= max_count:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("prompt", "")
                if not text or not text.strip():
                    continue

                is_safe = obj.get("is_safe", False)
                cat_dict = obj.get("category", {})

                if is_safe:
                    category = "safe"
                    tags_str = "is_safe=true"
                else:
                    # Find which categories are flagged true
                    triggered = []
                    for orig_label, our_cat in BEAVERTAILS_MAP.items():
                        if cat_dict.get(orig_label, False):
                            triggered.append((orig_label, our_cat))

                    if not triggered:
                        category = "other_or_uncertain"
                        tags_str = "no_category_flagged"
                    else:
                        tags_parts = [f"{ol}=true" for ol, _ in triggered]
                        tags_str = ", ".join(tags_parts)
                        category = triggered[0][1]  # first match wins

                records.append({
                    "prompt": text.strip(),
                    "category": category,
                    "tags": f"original_labels: {tags_str}",
                })
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  Loaded {len(records)} records from BeaverTails")
    return records


def load_wildguard(path: str, max_count: int) -> list:
    """Load from Parquet, mapping WildGuard subcategories to safety taxonomy."""
    import re

    df = pd.read_parquet(path)
    records = []

    for _, row in df.iterrows():
        if len(records) >= max_count:
            break
        text = str(row.get("prompt", "")).strip()
        if not text or text == "nan":
            continue

        subcat = str(row.get("subcategory", ""))
        harm_label = row.get("prompt_harm_label", "")

        # Try direct mapping first
        if subcat in WILDGUARD_MAP:
            category = WILDGUARD_MAP[subcat]
        elif harm_label == "unharmful":
            category = "safe"
        else:
            # Try keyword matching for 'others' and unmapped categories
            text_lower = text.lower()
            matched = False
            for keywords, cat in WILDGUARD_KEYWORD_MAP:
                for kw in keywords:
                    if kw in text_lower:
                        category = cat
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                category = "other_or_uncertain"

        records.append({
            "prompt": text,
            "category": category,
            "tags": f"wildguard_subcategory: {subcat}, harm_label: {harm_label}",
        })

    print(f"  Loaded {len(records)} records from WildGuard")
    return records


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────

def _build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - 2):
        m.update(text[i:i + 3].encode("utf-8"))
    return m


def deduplicate(records: list, threshold: float = 0.85) -> list:
    """Two-phase dedup: exact match on prompt text, then MinHash LSH."""
    # Phase 1: exact match by prompt text
    seen = set()
    exact_unique = []
    for r in tqdm(records, desc="Exact dedup"):
        p = r["prompt"]
        if p not in seen:
            seen.add(p)
            exact_unique.append(r)
    print(f"  After exact dedup: {len(exact_unique)} "
          f"(removed {len(records) - len(exact_unique)})")

    # Phase 2: MinHash LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    near_unique = []
    for i, r in enumerate(tqdm(exact_unique, desc="MinHash dedup")):
        m = _build_minhash(r["prompt"])
        if not lsh.query(m):
            lsh.insert(i, m)
            near_unique.append(r)

    print(f"  After MinHash dedup (threshold={threshold}): {len(near_unique)} "
          f"(removed {len(exact_unique) - len(near_unique)})")
    return near_unique


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def format_output(records: list) -> list:
    """Assign IDs, attach severity."""
    output = []
    for i, r in enumerate(records):
        cat = r["category"]
        severity = SEVERITY_MAP.get(cat, "safe") if cat != "safe" else "safe"
        record = OrderedDict([
            ("id", f"safety-{i + 1:06d}"),
            ("category", cat),
            ("prompt", r["prompt"]),
            ("tags", [r["tags"]] if r["tags"] else []),
            ("severity", severity),
            ("notes", ""),
        ])
        output.append(record)
    return output


def write_meta(output_path: str, total: int, categories: dict, dedup_threshold: float):
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
        ("data_sources", [
            "allenai/real-toxicity-prompts",
            "google/jigsaw_toxicity_prediction",
            "PKU-Alignment/BeaverTails",
        ]),
        ("label_source", "original_dataset_labels"),
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
        description="Build LLM Safety Benchmark from original dataset labels",
    )
    parser.add_argument("--max-per-source", type=int, default=1000,
                        help="Max records per source (default: 1000)")
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="MinHash LSH dedup threshold (default: 0.85)")
    parser.add_argument("--raw-dir", default=None,
                        help="Path to raw data directory (default: data/raw)")
    parser.add_argument("--output-dir", default=None,
                        help="Path for output files (default: data/)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = args.raw_dir or os.path.join(project_root, "data", "raw")
    output_dir = args.output_dir or os.path.join(project_root, "data")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "safety_benchmark.jsonl")
    meta_path = os.path.join(output_dir, "meta.json")

    print("=" * 60)
    print("LLM Safety Benchmark — Label-Based Build Pipeline")
    print("=" * 60)
    print(f"Max/source:   {args.max_per_source}")
    print(f"Dedup thresh: {args.dedup_threshold}")
    print()

    # 1. Load raw data with original labels
    print("[1/4] Loading data with original labels...")
    rtp_path = os.path.join(raw_dir, "real-toxicity-prompts", "prompts.jsonl")
    jigsaw_path = os.path.join(raw_dir, "jigsaw", "train.csv")
    beaver_path = os.path.join(raw_dir, "beavertails", "train.jsonl.xz")
    wildguard_path = os.path.join(raw_dir, "wildguard", "wildguard_train.parquet")

    all_records = []
    for loader, path, name in [
        (load_real_toxicity_prompts, rtp_path, "RealToxicityPrompts"),
        (load_jigsaw, jigsaw_path, "Jigsaw"),
        (load_beavertails, beaver_path, "BeaverTails"),
        (load_wildguard, wildguard_path, "WildGuard"),
    ]:
        if os.path.exists(path):
            all_records.extend(loader(path, args.max_per_source))
        else:
            print(f"  [WARNING] {name}: file not found at {path}, skipping.")

    if not all_records:
        print("[ERROR] No records loaded. Please download the datasets first.")
        sys.exit(1)
    print(f"  Total raw records: {len(all_records)}")

    # 2. Deduplicate
    print("[2/4] Deduplicating...")
    unique_records = deduplicate(all_records, threshold=args.dedup_threshold)
    print(f"  Unique records: {len(unique_records)}")

    # 3. Format and write output
    print("[3/4] Writing output...")
    output_records = format_output(unique_records)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(output_records)} records to {output_path}")

    # 4. Meta
    print("[4/4] Writing meta...")
    category_counts = {}
    for r in output_records:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    write_meta(meta_path, len(output_records), category_counts, args.dedup_threshold)

    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Total cases:  {len(output_records)}")
    print(f"Categories:   {len(category_counts)}")
    print()
    print("Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(output_records) * 100
        print(f"  {cat:.<55s} {count:>6d} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
