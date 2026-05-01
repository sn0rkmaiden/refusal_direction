# pipeline/experiments/prepare_external_datasets.py

"""
Download and preprocess external datasets for refusal-direction experiments.

Outputs standard JSON files with records of the form:
{
    "instruction": "...",
    "category": "...",
    "source": "..."
}

Default datasets:
- harmful: PKU-Alignment/BeaverTails-Evaluation
- harmless: databricks/databricks-dolly-15k

Example:
python3 -m pipeline.experiments.prepare_external_datasets \
  --output_dir dataset/external/beavertails_dolly \
  --seed 0

Then use:
python3 -m pipeline.experiments.run_external_direction_experiment \
  --model_path google/gemma-2b-it \
  --harmful_dataset dataset/external/beavertails_dolly/beavertails_eval_harmful.json \
  --harmless_dataset dataset/external/beavertails_dolly/dolly15k_harmless.json \
  --run_name beavertails_dolly
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Record = Dict[str, Any]


PROMPT_FIELD_CANDIDATES = [
    "instruction",
    "prompt",
    "question",
    "text",
    "input",
    "query",
]

CATEGORY_FIELD_CANDIDATES = [
    "category",
    "label",
    "type",
    "harm_category",
    "category_id",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_for_dedup(text: str) -> str:
    return normalize_text(text).lower()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, records: Sequence[Record]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Sequence[Record]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_hf_dataset(dataset_name: str, split: Optional[str] = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install it with:\n"
            "  pip install datasets\n"
        ) from exc

    if split is None:
        dataset = load_dataset(dataset_name)
        if isinstance(dataset, dict):
            # Prefer train when available; otherwise take the first split.
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]
            return dataset[split_name]
        return dataset

    return load_dataset(dataset_name, split=split)


def get_first_existing_field(example: Record, candidates: Sequence[str]) -> Optional[str]:
    for field in candidates:
        if field in example:
            return field
    return None


def remove_dolly_wikipedia_citations(text: str) -> str:
    # Dolly context may contain bracketed Wikipedia citations such as [42].
    return re.sub(r"\[\d+\]", "", text)


def clean_instruction(text: Any, min_chars: int, max_chars: int) -> Optional[str]:
    if text is None:
        return None

    cleaned = normalize_text(str(text))
    if len(cleaned) < min_chars:
        return None
    if len(cleaned) > max_chars:
        return None

    return cleaned


def deduplicate(records: Iterable[Record]) -> List[Record]:
    seen = set()
    deduped: List[Record] = []

    for record in records:
        key = normalize_for_dedup(record["instruction"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    return deduped


def preprocess_dolly(
    dataset_name: str,
    split: str,
    context_mode: str,
    min_chars: int,
    max_chars: int,
) -> List[Record]:
    """
    Convert Dolly to harmless instruction records.

    context_mode:
    - drop: keep only records with empty context
    - instruction_only: ignore context and keep instruction only
    - append: append context to instruction
    """
    ds = load_hf_dataset(dataset_name, split=split)

    records: List[Record] = []

    for example in ds:
        instruction = example.get("instruction")
        context = example.get("context") or ""
        category = example.get("category", "unknown")

        if context_mode == "drop" and normalize_text(str(context)):
            continue

        if context_mode == "append" and normalize_text(str(context)):
            context_clean = remove_dolly_wikipedia_citations(str(context))
            raw_instruction = f"{instruction}\n\n{context_clean}"
        else:
            raw_instruction = instruction

        cleaned = clean_instruction(raw_instruction, min_chars=min_chars, max_chars=max_chars)
        if cleaned is None:
            continue

        records.append(
            {
                "instruction": cleaned,
                "category": str(category),
                "source": "databricks-dolly-15k",
            }
        )

    return deduplicate(records)


def preprocess_beavertails_eval(
    dataset_name: str,
    split: str,
    min_chars: int,
    max_chars: int,
) -> List[Record]:
    """
    Convert BeaverTails-Evaluation to harmful instruction records.

    The loader is intentionally field-name tolerant because mirrored versions
    of the dataset may expose slightly different column names.
    """
    ds = load_hf_dataset(dataset_name, split=split)

    records: List[Record] = []

    if len(ds) == 0:
        raise ValueError(f"Dataset {dataset_name!r}, split {split!r} is empty.")

    first_example = dict(ds[0])
    prompt_field = get_first_existing_field(first_example, PROMPT_FIELD_CANDIDATES)
    category_field = get_first_existing_field(first_example, CATEGORY_FIELD_CANDIDATES)

    if prompt_field is None:
        raise ValueError(
            "Could not find a prompt/instruction field in BeaverTails-Evaluation. "
            f"Available fields: {list(first_example.keys())}"
        )

    for example in ds:
        raw_instruction = example.get(prompt_field)
        cleaned = clean_instruction(raw_instruction, min_chars=min_chars, max_chars=max_chars)
        if cleaned is None:
            continue

        category = example.get(category_field, "unknown") if category_field else "unknown"

        records.append(
            {
                "instruction": cleaned,
                "category": str(category),
                "source": "PKU-Alignment/BeaverTails-Evaluation",
            }
        )

    return deduplicate(records)


def remove_cross_dataset_overlaps(
    harmful: Sequence[Record],
    harmless: Sequence[Record],
) -> Tuple[List[Record], List[Record], int]:
    harmless_keys = {normalize_for_dedup(r["instruction"]) for r in harmless}
    harmful_filtered = [
        r for r in harmful
        if normalize_for_dedup(r["instruction"]) not in harmless_keys
    ]
    removed = len(harmful) - len(harmful_filtered)
    return harmful_filtered, list(harmless), removed


def make_split(
    records: Sequence[Record],
    rng: random.Random,
    n_train: int,
    n_val: int,
    n_test: int,
    allow_smaller_test: bool,
    dataset_name: str,
) -> Dict[str, List[Record]]:
    total_needed = n_train + n_val + n_test

    if len(records) < n_train + n_val:
        raise ValueError(
            f"{dataset_name}: not enough records for train+val. "
            f"Need at least {n_train + n_val}, got {len(records)}."
        )

    if len(records) < total_needed and not allow_smaller_test:
        raise ValueError(
            f"{dataset_name}: not enough records for requested split. "
            f"Need {total_needed}, got {len(records)}. "
            "Use --allow_smaller_test or reduce --n_test."
        )

    shuffled = list(records)
    rng.shuffle(shuffled)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]

    if len(records) >= total_needed:
        test = shuffled[n_train + n_val:n_train + n_val + n_test]
    else:
        test = shuffled[n_train + n_val:]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def save_split_files(
    split_dir: Path,
    harmful_split: Dict[str, List[Record]],
    harmless_split: Dict[str, List[Record]],
) -> None:
    ensure_dir(split_dir)

    for split_name in ["train", "val", "test"]:
        write_json(split_dir / f"harmful_{split_name}.json", harmful_split[split_name])
        write_json(split_dir / f"harmless_{split_name}.json", harmless_split[split_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess Dolly15k and BeaverTails-Evaluation."
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/external/beavertails_dolly"),
        help="Directory where processed files will be saved.",
    )

    parser.add_argument(
        "--dolly_dataset",
        type=str,
        default="databricks/databricks-dolly-15k",
        help="Hugging Face dataset name for Dolly.",
    )
    parser.add_argument(
        "--dolly_split",
        type=str,
        default="train",
        help="Dolly split to load.",
    )
    parser.add_argument(
        "--dolly_context_mode",
        choices=["drop", "instruction_only", "append"],
        default="drop",
        help=(
            "How to handle Dolly context. "
            "'drop' keeps only examples with empty context; "
            "'instruction_only' ignores context; "
            "'append' appends context to instruction."
        ),
    )

    parser.add_argument(
        "--beavertails_dataset",
        type=str,
        default="PKU-Alignment/BeaverTails-Evaluation",
        help="Hugging Face dataset name for BeaverTails-Evaluation.",
    )
    parser.add_argument(
        "--beavertails_split",
        type=str,
        default="test",
        help="BeaverTails-Evaluation split to load.",
    )

    parser.add_argument("--min_chars", type=int, default=5)
    parser.add_argument("--max_chars", type=int, default=2000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument(
        "--allow_smaller_test",
        action="store_true",
        help="Allow the test split to contain fewer than n_test examples if needed.",
    )

    parser.add_argument(
        "--write_jsonl",
        action="store_true",
        help="Also save JSONL copies of the processed full datasets.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    ensure_dir(output_dir)

    rng = random.Random(args.seed)

    print("[1/5] Downloading and preprocessing Dolly15k as harmless instructions...")
    harmless = preprocess_dolly(
        dataset_name=args.dolly_dataset,
        split=args.dolly_split,
        context_mode=args.dolly_context_mode,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(f"      Harmless records after cleaning: {len(harmless)}")

    print("[2/5] Downloading and preprocessing BeaverTails-Evaluation as harmful instructions...")
    harmful = preprocess_beavertails_eval(
        dataset_name=args.beavertails_dataset,
        split=args.beavertails_split,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(f"      Harmful records after cleaning: {len(harmful)}")

    print("[3/5] Removing exact overlaps between harmful and harmless instructions...")
    harmful, harmless, removed_overlap_count = remove_cross_dataset_overlaps(
        harmful=harmful,
        harmless=harmless,
    )
    print(f"      Removed harmful/harmless exact overlaps: {removed_overlap_count}")

    print("[4/5] Saving processed full datasets...")
    harmful_path = output_dir / "beavertails_eval_harmful.json"
    harmless_path = output_dir / "dolly15k_harmless.json"

    write_json(harmful_path, harmful)
    write_json(harmless_path, harmless)

    if args.write_jsonl:
        write_jsonl(output_dir / "beavertails_eval_harmful.jsonl", harmful)
        write_jsonl(output_dir / "dolly15k_harmless.jsonl", harmless)

    print("[5/5] Creating train/val/test splits...")
    harmful_split = make_split(
        records=harmful,
        rng=rng,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        allow_smaller_test=args.allow_smaller_test,
        dataset_name="BeaverTails-Evaluation",
    )
    harmless_split = make_split(
        records=harmless,
        rng=rng,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        allow_smaller_test=args.allow_smaller_test,
        dataset_name="Dolly15k",
    )

    split_dir = output_dir / f"splits_seed{args.seed}"
    save_split_files(split_dir, harmful_split, harmless_split)

    metadata = {
        "dolly_dataset": args.dolly_dataset,
        "dolly_split": args.dolly_split,
        "dolly_context_mode": args.dolly_context_mode,
        "beavertails_dataset": args.beavertails_dataset,
        "beavertails_split": args.beavertails_split,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "allow_smaller_test": args.allow_smaller_test,
        "num_harmful": len(harmful),
        "num_harmless": len(harmless),
        "removed_harmful_harmless_exact_overlaps": removed_overlap_count,
        "files": {
            "harmful_full": str(harmful_path),
            "harmless_full": str(harmless_path),
            "split_dir": str(split_dir),
            "harmful_train": str(split_dir / "harmful_train.json"),
            "harmful_val": str(split_dir / "harmful_val.json"),
            "harmful_test": str(split_dir / "harmful_test.json"),
            "harmless_train": str(split_dir / "harmless_train.json"),
            "harmless_val": str(split_dir / "harmless_val.json"),
            "harmless_test": str(split_dir / "harmless_test.json"),
        },
        "split_sizes": {
            "harmful_train": len(harmful_split["train"]),
            "harmful_val": len(harmful_split["val"]),
            "harmful_test": len(harmful_split["test"]),
            "harmless_train": len(harmless_split["train"]),
            "harmless_val": len(harmless_split["val"]),
            "harmless_test": len(harmless_split["test"]),
        },
    }

    write_json(output_dir / "metadata.json", [metadata])

    print("\nDone.")
    print(f"Processed harmful dataset:   {harmful_path}")
    print(f"Processed harmless dataset:  {harmless_path}")
    print(f"Splits directory:            {split_dir}")
    print("\nUse these paths in the external direction experiment:")
    print(f"  --harmful_dataset {harmful_path}")
    print(f"  --harmless_dataset {harmless_path}")


if __name__ == "__main__":
    main()