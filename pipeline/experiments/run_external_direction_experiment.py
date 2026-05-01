import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from pipeline.experiments.common import (
    ensure_dir,
    filter_train_val_strings,
    instruction_strings,
    load_instruction_dataset,
    make_experiment_config,
    read_evaluation_metric,
    sanitize_name,
    save_json,
    set_seed,
    split_rows,
)
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import (
    evaluate_completions_and_save_results_for_dataset,
    generate_and_save_candidate_directions,
    generate_and_save_completions_for_dataset,
    select_and_save_direction,
)
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)


SPLIT_FILENAMES = {
    "harmful_train": "harmful_train.json",
    "harmful_val": "harmful_val.json",
    "harmful_test": "harmful_test.json",
    "harmless_train": "harmless_train.json",
    "harmless_val": "harmless_val.json",
    "harmless_test": "harmless_test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate a refusal direction on non-default harmful/harmless datasets."
    )
    parser.add_argument("--model_path", required=True)

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--split_dir",
        default=None,
        help=(
            "Directory containing six precomputed split files: "
            "harmful_train.json, harmful_val.json, harmful_test.json, "
            "harmless_train.json, harmless_val.json, harmless_test.json."
        ),
    )
    dataset_group.add_argument(
        "--harmful_dataset",
        default=None,
        help=(
            "Full harmful dataset to split internally: processed:<name>, "
            "split:<harmtype>_<split>, or a file path. Requires --harmless_dataset."
        ),
    )
    parser.add_argument(
        "--harmless_dataset",
        default=None,
        help=(
            "Full harmless dataset to split internally: processed:<name>, "
            "split:<harmtype>_<split>, or a file path. Required with --harmful_dataset."
        ),
    )

    parser.add_argument("--run_name", default=None)
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--filter_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--filter_batch_size",
        type=int,
        default=32,
        help="Batch size used only for train/val filtering by refusal score.",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Only extract/select the direction; do not generate completions.",
    )

    args = parser.parse_args()

    if args.harmful_dataset is not None and args.harmless_dataset is None:
        parser.error("--harmless_dataset is required when --harmful_dataset is used.")

    if args.split_dir is not None and args.harmless_dataset is not None:
        parser.error("--harmless_dataset cannot be used together with --split_dir.")

    if args.filter_batch_size < 1:
        parser.error("--filter_batch_size must be >= 1.")

    return args


def load_rows_from_split_dir(split_dir: str) -> Tuple[list, list, list, list, list, list]:
    split_dir_path = Path(split_dir).expanduser()

    if not split_dir_path.exists() or not split_dir_path.is_dir():
        raise ValueError(f"--split_dir does not exist or is not a directory: {split_dir}")

    missing = [
        filename
        for filename in SPLIT_FILENAMES.values()
        if not (split_dir_path / filename).exists()
    ]
    if missing:
        raise ValueError(
            f"--split_dir is missing required files: {missing}. "
            f"Expected files: {list(SPLIT_FILENAMES.values())}"
        )

    harmful_train_rows = load_instruction_dataset(
        str(split_dir_path / SPLIT_FILENAMES["harmful_train"]),
        default_category="harmful",
    )
    harmful_val_rows = load_instruction_dataset(
        str(split_dir_path / SPLIT_FILENAMES["harmful_val"]),
        default_category="harmful",
    )
    harmful_test_rows = load_instruction_dataset(
        str(split_dir_path / SPLIT_FILENAMES["harmful_test"]),
        default_category="harmful",
    )

    harmless_train_rows = load_instruction_dataset(
        str(split_dir_path / SPLIT_FILENAMES["harmless_train"]),
        default_category="harmless",
    )
    harmless_val_rows = load_instruction_dataset(
        str(split_dir_path / SPLIT_FILENAMES["harmless_val"]),
        default_category="harmless",
    )
    harmless_test_rows = load_instruction_dataset(
        str(split_dir_path / SPLIT_FILENAMES["harmless_test"]),
        default_category="harmless",
    )

    return (
        harmful_train_rows,
        harmful_val_rows,
        harmful_test_rows,
        harmless_train_rows,
        harmless_val_rows,
        harmless_test_rows,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    using_split_dir = args.split_dir is not None

    if using_split_dir:
        default_run_name = f"split_dir={sanitize_name(args.split_dir)}"
    else:
        default_run_name = (
            f"harmful={sanitize_name(args.harmful_dataset)}"
            f"__harmless={sanitize_name(args.harmless_dataset)}"
        )

    run_name = args.run_name or default_run_name

    cfg = make_experiment_config(
        args.model_path,
        "external_direction",
        run_name,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        max_new_tokens=args.max_new_tokens,
        filter_train=args.filter_train,
        filter_val=args.filter_val,
    )
    artifact_dir = ensure_dir(cfg.artifact_path())

    if using_split_dir:
        (
            harmful_train_rows,
            harmful_val_rows,
            harmful_test_rows,
            harmless_train_rows,
            harmless_val_rows,
            harmless_test_rows,
        ) = load_rows_from_split_dir(args.split_dir)
    else:
        harmful_rows = load_instruction_dataset(
            args.harmful_dataset,
            default_category="harmful",
        )
        harmless_rows = load_instruction_dataset(
            args.harmless_dataset,
            default_category="harmless",
        )

        harmful_train_rows, harmful_val_rows, harmful_test_rows = split_rows(
            harmful_rows,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
            label="harmful",
        )
        harmless_train_rows, harmless_val_rows, harmless_test_rows = split_rows(
            harmless_rows,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed + 1,
            label="harmless",
        )

    sampled_dir = ensure_dir(os.path.join(artifact_dir, "sampled_data"))
    save_json(harmful_train_rows, os.path.join(sampled_dir, "harmful_train.json"))
    save_json(harmful_val_rows, os.path.join(sampled_dir, "harmful_val.json"))
    save_json(harmful_test_rows, os.path.join(sampled_dir, "harmful_test.json"))
    save_json(harmless_train_rows, os.path.join(sampled_dir, "harmless_train.json"))
    save_json(harmless_val_rows, os.path.join(sampled_dir, "harmless_val.json"))
    save_json(harmless_test_rows, os.path.join(sampled_dir, "harmless_test.json"))

    save_json(
        {
            "model_path": args.model_path,
            "input_mode": "split_dir" if using_split_dir else "full_datasets_resplit_internally",
            "split_dir": args.split_dir,
            "harmful_dataset": args.harmful_dataset,
            "harmless_dataset": args.harmless_dataset,
            "run_name": run_name,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "actual_split_sizes": {
                "harmful_train": len(harmful_train_rows),
                "harmful_val": len(harmful_val_rows),
                "harmful_test": len(harmful_test_rows),
                "harmless_train": len(harmless_train_rows),
                "harmless_val": len(harmless_val_rows),
                "harmless_test": len(harmless_test_rows),
            },
            "seed": args.seed,
            "filter_train": args.filter_train,
            "filter_val": args.filter_val,
            "filter_batch_size": args.filter_batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        os.path.join(artifact_dir, "config.json"),
    )

    model_base = construct_model_base(args.model_path)

    harmful_train = instruction_strings(harmful_train_rows)
    harmless_train = instruction_strings(harmless_train_rows)
    harmful_val = instruction_strings(harmful_val_rows)
    harmless_val = instruction_strings(harmless_val_rows)

    harmful_train, harmless_train, harmful_val, harmless_val = filter_train_val_strings(
        cfg,
        model_base,
        harmful_train,
        harmless_train,
        harmful_val,
        harmless_val,
        batch_size=args.filter_batch_size,
    )

    save_json(
        {
            "harmful_train_after_filter": len(harmful_train),
            "harmless_train_after_filter": len(harmless_train),
            "harmful_val_after_filter": len(harmful_val),
            "harmless_val_after_filter": len(harmless_val),
        },
        os.path.join(artifact_dir, "filter_counts.json"),
    )

    candidate_directions = generate_and_save_candidate_directions(
        cfg,
        model_base,
        harmful_train,
        harmless_train,
    )
    pos, layer, direction = select_and_save_direction(
        cfg,
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
    )

    summary: Dict[str, object] = {
        "selected_position": int(pos),
        "selected_layer": int(layer),
        "artifact_dir": artifact_dir,
    }

    if not args.skip_generation:
        baseline_pre_hooks, baseline_hooks = [], []
        ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(
            model_base,
            direction,
        )
        actadd_harmful_pre_hooks = [
            (
                model_base.model_block_modules[layer],
                get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0),
            )
        ]
        actadd_harmless_pre_hooks = [
            (
                model_base.model_block_modules[layer],
                get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0),
            )
        ]

        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            baseline_pre_hooks,
            baseline_hooks,
            "baseline",
            "harmful_external",
            dataset=harmful_test_rows,
        )
        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            ablation_pre_hooks,
            ablation_hooks,
            "ablation",
            "harmful_external",
            dataset=harmful_test_rows,
        )
        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            actadd_harmful_pre_hooks,
            [],
            "actadd",
            "harmful_external",
            dataset=harmful_test_rows,
        )

        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            baseline_pre_hooks,
            baseline_hooks,
            "baseline",
            "harmless_external",
            dataset=harmless_test_rows,
        )
        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            actadd_harmless_pre_hooks,
            [],
            "actadd",
            "harmless_external",
            dataset=harmless_test_rows,
        )

        for dataset_name in ("harmful_external", "harmless_external"):
            for label in ("baseline", "ablation", "actadd"):
                completions_path = os.path.join(
                    cfg.artifact_path(),
                    "completions",
                    f"{dataset_name}_{label}_completions.json",
                )
                if not os.path.exists(completions_path):
                    continue

                evaluate_completions_and_save_results_for_dataset(
                    cfg,
                    label,
                    dataset_name,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                )

        for dataset_name in ("harmful_external", "harmless_external"):
            for label in ("baseline", "ablation", "actadd"):
                eval_path = os.path.join(
                    cfg.artifact_path(),
                    "completions",
                    f"{dataset_name}_{label}_evaluations.json",
                )
                value = read_evaluation_metric(eval_path)
                if value is not None:
                    summary[f"{dataset_name}_{label}_non_refusal_rate"] = value

    torch.cuda.empty_cache()
    save_json(summary, os.path.join(artifact_dir, "summary.json"))
    print(f"Saved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()