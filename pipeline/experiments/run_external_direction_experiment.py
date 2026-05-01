import argparse
import os
from typing import Dict

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
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate a refusal direction on non-default harmful/harmless datasets."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--harmful_dataset", required=True, help="processed:<name>, split:<harmtype>_<split>, or a file path")
    parser.add_argument("--harmless_dataset", required=True, help="processed:<name>, split:<harmtype>_<split>, or a file path")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--filter_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_generation", action="store_true", help="Only extract/select the direction; do not generate completions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"harmful={sanitize_name(args.harmful_dataset)}__harmless={sanitize_name(args.harmless_dataset)}"
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

    harmful_rows = load_instruction_dataset(args.harmful_dataset, default_category="harmful")
    harmless_rows = load_instruction_dataset(args.harmless_dataset, default_category="harmless")

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
            "harmful_dataset": args.harmful_dataset,
            "harmless_dataset": args.harmless_dataset,
            "run_name": run_name,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "seed": args.seed,
            "filter_train": args.filter_train,
            "filter_val": args.filter_val,
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

    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    summary: Dict[str, object] = {
        "selected_position": int(pos),
        "selected_layer": int(layer),
        "artifact_dir": artifact_dir,
    }

    if not args.skip_generation:
        baseline_pre_hooks, baseline_hooks = [], []
        ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_harmful_pre_hooks = [
            (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))
        ]
        actadd_harmless_pre_hooks = [
            (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))
        ]

        generate_and_save_completions_for_dataset(cfg, model_base, baseline_pre_hooks, baseline_hooks, "baseline", "harmful_external", dataset=harmful_test_rows)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_pre_hooks, ablation_hooks, "ablation", "harmful_external", dataset=harmful_test_rows)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_harmful_pre_hooks, [], "actadd", "harmful_external", dataset=harmful_test_rows)

        generate_and_save_completions_for_dataset(cfg, model_base, baseline_pre_hooks, baseline_hooks, "baseline", "harmless_external", dataset=harmless_test_rows)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_harmless_pre_hooks, [], "actadd", "harmless_external", dataset=harmless_test_rows)

        for dataset_name in ("harmful_external", "harmless_external"):
            for label in ("baseline", "ablation", "actadd"):
                completions_path = os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{label}_completions.json")
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
                eval_path = os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{label}_evaluations.json")
                value = read_evaluation_metric(eval_path)
                if value is not None:
                    summary[f"{dataset_name}_{label}_non_refusal_rate"] = value

    torch.cuda.empty_cache()
    save_json(summary, os.path.join(artifact_dir, "summary.json"))
    print(f"Saved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
