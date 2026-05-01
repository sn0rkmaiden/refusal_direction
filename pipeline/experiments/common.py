import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dataset.load_dataset import load_dataset, load_dataset_split, PROCESSED_DATASET_NAMES
from pipeline.config import Config
from pipeline.submodules.select_direction import get_refusal_scores

InstructionRow = Dict[str, Any]


def sanitize_name(value: str) -> str:
    value = value.strip().replace(os.sep, "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_") or "run"


def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_experiment_config(
    model_path: str,
    experiment_name: str,
    run_name: str,
    *,
    n_train: int = 128,
    n_val: int = 32,
    n_test: int = 100,
    max_new_tokens: int = 512,
    filter_train: bool = True,
    filter_val: bool = True,
    evaluation_datasets: Tuple[str, ...] = (),
) -> Config:
    model_alias = os.path.basename(model_path.rstrip("/"))
    nested_alias = os.path.join(
        sanitize_name(model_alias),
        "experiments",
        sanitize_name(experiment_name),
        sanitize_name(run_name),
    )
    return Config(
        model_alias=nested_alias,
        model_path=model_path,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        filter_train=filter_train,
        filter_val=filter_val,
        evaluation_datasets=evaluation_datasets,
        max_new_tokens=max_new_tokens,
    )


def make_child_config(parent_cfg: Config, child_name: str) -> Config:
    return Config(
        model_alias=os.path.join(parent_cfg.model_alias, sanitize_name(child_name)),
        model_path=parent_cfg.model_path,
        n_train=parent_cfg.n_train,
        n_val=parent_cfg.n_val,
        n_test=parent_cfg.n_test,
        filter_train=parent_cfg.filter_train,
        filter_val=parent_cfg.filter_val,
        evaluation_datasets=parent_cfg.evaluation_datasets,
        max_new_tokens=parent_cfg.max_new_tokens,
        jailbreak_eval_methodologies=parent_cfg.jailbreak_eval_methodologies,
        refusal_eval_methodologies=parent_cfg.refusal_eval_methodologies,
        ce_loss_batch_size=parent_cfg.ce_loss_batch_size,
        ce_loss_n_batches=parent_cfg.ce_loss_n_batches,
    )


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def normalize_instruction_rows(
    rows: Iterable[Any],
    *,
    default_category: str,
    source_name: str,
) -> List[InstructionRow]:
    normalized: List[InstructionRow] = []
    instruction_keys = (
        "instruction",
        "prompt",
        "behavior",
        "request",
        "question",
        "text",
    )

    for idx, row in enumerate(rows):
        if isinstance(row, str):
            instruction = row.strip()
            category = default_category
        elif isinstance(row, dict):
            instruction = _first_present(row, instruction_keys)
            category = str(row.get("category") or row.get("label") or default_category)
        else:
            raise TypeError(f"Unsupported row type at index {idx}: {type(row)!r}")

        if not instruction:
            continue

        normalized.append(
            {
                "instruction": instruction,
                "category": category,
                "source": source_name,
            }
        )
    return normalized


def _load_rows_from_path(path: Path) -> List[Any]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        obj = load_json(path)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "examples", "items", "rows", "train", "test", "validation"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise ValueError(f"Could not find a list of examples in {path}")

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    raise ValueError(f"Unsupported dataset file extension: {path.suffix}")


def load_instruction_dataset(spec: str, *, default_category: Optional[str] = None) -> List[InstructionRow]:
    """Load a dataset into rows with instruction/category/source.

    Supported specs:
    - processed:<name>, e.g. processed:strongreject
    - split:<harmtype>_<split>, e.g. split:harmful_train or split:harmless_val
    - split:<harmtype>:<split>, e.g. split:harmful:train
    - <name> for names listed in dataset.load_dataset.PROCESSED_DATASET_NAMES
    - path to .json, .jsonl, .csv, or .txt
    """
    if not spec:
        raise ValueError("Dataset spec must be non-empty")

    source_name = spec
    category = default_category or sanitize_name(Path(spec).stem)

    if spec.startswith("processed:"):
        dataset_name = spec.split(":", 1)[1]
        rows = load_dataset(dataset_name, instructions_only=False)
        return normalize_instruction_rows(rows, default_category=default_category or dataset_name, source_name=source_name)

    if spec.startswith("split:"):
        body = spec.split(":", 1)[1]
        if ":" in body:
            harmtype, split = body.split(":", 1)
        else:
            harmtype, split = body.rsplit("_", 1)
        rows = load_dataset_split(harmtype=harmtype, split=split, instructions_only=False)
        return normalize_instruction_rows(rows, default_category=default_category or harmtype, source_name=source_name)

    if spec in PROCESSED_DATASET_NAMES:
        rows = load_dataset(spec, instructions_only=False)
        return normalize_instruction_rows(rows, default_category=default_category or spec, source_name=source_name)

    path = Path(spec).expanduser()
    if path.exists():
        rows = _load_rows_from_path(path)
        return normalize_instruction_rows(rows, default_category=category, source_name=str(path))

    raise ValueError(
        f"Unknown dataset spec: {spec!r}. Use processed:<name>, split:<harmtype>_<split>, "
        "a processed dataset name, or a path to .json/.jsonl/.csv/.txt."
    )


def instruction_strings(rows: Sequence[InstructionRow]) -> List[str]:
    return [row["instruction"] for row in rows]


def sample_rows(
    rows: Sequence[InstructionRow],
    n: int,
    *,
    seed: int,
    label: str,
    allow_smaller: bool = False,
) -> List[InstructionRow]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if len(rows) < n and not allow_smaller:
        raise ValueError(f"Requested {n} {label} rows, but only {len(rows)} are available")
    rng = random.Random(seed)
    rows = list(rows)
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def split_rows(
    rows: Sequence[InstructionRow],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    label: str,
) -> Tuple[List[InstructionRow], List[InstructionRow], List[InstructionRow]]:
    total = n_train + n_val + n_test
    if len(rows) < total:
        raise ValueError(f"Need {total} {label} rows, but only {len(rows)} are available")
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def filter_train_val_strings(
    cfg: Config,
    model_base,
    harmful_train: Sequence[str],
    harmless_train: Sequence[str],
    harmful_val: Sequence[str],
    harmless_val: Sequence[str],
    *,
    batch_size: int = 32,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    def keep_by_score(dataset: Sequence[str], scores: torch.Tensor, *, positive: bool) -> List[str]:
        if positive:
            return [inst for inst, score in zip(dataset, scores.tolist()) if score > 0]
        return [inst for inst, score in zip(dataset, scores.tolist()) if score < 0]

    harmful_train = list(harmful_train)
    harmless_train = list(harmless_train)
    harmful_val = list(harmful_val)
    harmless_val = list(harmless_val)

    if cfg.filter_train:
        harmful_scores = get_refusal_scores(
            model_base.model,
            harmful_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=batch_size,
        )
        harmless_scores = get_refusal_scores(
            model_base.model,
            harmless_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=batch_size,
        )
        harmful_train = keep_by_score(harmful_train, harmful_scores, positive=True)
        harmless_train = keep_by_score(harmless_train, harmless_scores, positive=False)

    if cfg.filter_val:
        harmful_scores = get_refusal_scores(
            model_base.model,
            harmful_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=batch_size,
        )
        harmless_scores = get_refusal_scores(
            model_base.model,
            harmless_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=batch_size,
        )
        harmful_val = keep_by_score(harmful_val, harmful_scores, positive=True)
        harmless_val = keep_by_score(harmless_val, harmless_scores, positive=False)

    if not harmful_train or not harmless_train:
        raise ValueError(
            "Training set became empty after filtering. Try --no-filter_train or use a larger sample."
        )
    if not harmful_val or not harmless_val:
        raise ValueError(
            "Validation set became empty after filtering. Try --no-filter_val or use a larger sample."
        )

    return harmful_train, harmless_train, harmful_val, harmless_val


def direction_cosine(direction_a: torch.Tensor, direction_b: torch.Tensor) -> float:
    a = direction_a.detach().flatten().to(dtype=torch.float64, device="cpu")
    b = direction_b.detach().flatten().to(dtype=torch.float64, device="cpu")
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def read_evaluation_metric(path: str | Path, metric: str = "substring_matching_success_rate") -> Optional[float]:
    path = Path(path)
    if not path.exists():
        return None
    data = load_json(path)
    value = data.get(metric)
    return None if value is None else float(value)


def write_pairwise_matrix_csv(labels: Sequence[str], matrix: Sequence[Sequence[float]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["direction"] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + list(row))
