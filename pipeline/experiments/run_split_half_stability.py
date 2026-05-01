import argparse
import os
from typing import List

import torch

from pipeline.experiments.common import (
    direction_cosine,
    ensure_dir,
    filter_train_val_strings,
    instruction_strings,
    load_instruction_dataset,
    make_child_config,
    make_experiment_config,
    sample_rows,
    save_json,
    set_seed,
    write_pairwise_matrix_csv,
)
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import generate_and_save_candidate_directions, select_and_save_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate stability of selected refusal directions across random half-samples."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--harmful_dataset", default="split:harmful_train")
    parser.add_argument("--harmless_dataset", default="split:harmless_train")
    parser.add_argument("--harmful_val_dataset", default="split:harmful_val")
    parser.add_argument("--harmless_val_dataset", default="split:harmless_val")
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--n_repeats", type=int, default=8)
    parser.add_argument("--pool_size", type=int, default=256, help="Fixed pool size per class used for repeated half-sampling.")
    parser.add_argument("--half_size", type=int, default=128, help="Number of examples per class used in each repeat.")
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_val", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.half_size > args.pool_size:
        raise ValueError("--half_size must be <= --pool_size")

    cfg = make_experiment_config(
        args.model_path,
        "split_half_stability",
        args.run_name,
        n_train=args.half_size,
        n_val=args.n_val,
        n_test=0,
        filter_train=args.filter_train,
        filter_val=args.filter_val,
    )
    artifact_dir = ensure_dir(cfg.artifact_path())

    harmful_pool_rows = sample_rows(
        load_instruction_dataset(args.harmful_dataset, default_category="harmful"),
        args.pool_size,
        seed=args.seed,
        label="harmful pool",
    )
    harmless_pool_rows = sample_rows(
        load_instruction_dataset(args.harmless_dataset, default_category="harmless"),
        args.pool_size,
        seed=args.seed + 1,
        label="harmless pool",
    )
    harmful_val_rows = sample_rows(
        load_instruction_dataset(args.harmful_val_dataset, default_category="harmful_val"),
        args.n_val,
        seed=args.seed + 2,
        label="harmful validation",
    )
    harmless_val_rows = sample_rows(
        load_instruction_dataset(args.harmless_val_dataset, default_category="harmless_val"),
        args.n_val,
        seed=args.seed + 3,
        label="harmless validation",
    )

    sampled_dir = ensure_dir(os.path.join(artifact_dir, "sampled_data"))
    save_json(harmful_pool_rows, os.path.join(sampled_dir, "harmful_pool.json"))
    save_json(harmless_pool_rows, os.path.join(sampled_dir, "harmless_pool.json"))
    save_json(harmful_val_rows, os.path.join(sampled_dir, "harmful_val.json"))
    save_json(harmless_val_rows, os.path.join(sampled_dir, "harmless_val.json"))

    save_json(vars(args), os.path.join(artifact_dir, "config.json"))

    model_base = construct_model_base(args.model_path)
    selected_directions: List[torch.Tensor] = []
    repeat_summaries = []

    harmful_val = instruction_strings(harmful_val_rows)
    harmless_val = instruction_strings(harmless_val_rows)

    for repeat_idx in range(args.n_repeats):
        repeat_name = f"repeat_{repeat_idx:02d}"
        repeat_cfg = make_child_config(cfg, repeat_name)
        repeat_seed = args.seed + 10_000 + repeat_idx

        harmful_rows = sample_rows(
            harmful_pool_rows,
            args.half_size,
            seed=repeat_seed,
            label=f"harmful {repeat_name}",
        )
        harmless_rows = sample_rows(
            harmless_pool_rows,
            args.half_size,
            seed=repeat_seed + 1,
            label=f"harmless {repeat_name}",
        )

        repeat_sampled_dir = ensure_dir(os.path.join(repeat_cfg.artifact_path(), "sampled_data"))
        save_json(harmful_rows, os.path.join(repeat_sampled_dir, "harmful_train.json"))
        save_json(harmless_rows, os.path.join(repeat_sampled_dir, "harmless_train.json"))

        harmful_train = instruction_strings(harmful_rows)
        harmless_train = instruction_strings(harmless_rows)

        filtered_harmful_train, filtered_harmless_train, filtered_harmful_val, filtered_harmless_val = filter_train_val_strings(
            repeat_cfg,
            model_base,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
        )

        save_json(
            {
                "repeat": repeat_idx,
                "seed": repeat_seed,
                "harmful_train_after_filter": len(filtered_harmful_train),
                "harmless_train_after_filter": len(filtered_harmless_train),
                "harmful_val_after_filter": len(filtered_harmful_val),
                "harmless_val_after_filter": len(filtered_harmless_val),
            },
            os.path.join(repeat_cfg.artifact_path(), "filter_counts.json"),
        )

        candidate_directions = generate_and_save_candidate_directions(
            repeat_cfg,
            model_base,
            filtered_harmful_train,
            filtered_harmless_train,
        )
        pos, layer, direction = select_and_save_direction(
            repeat_cfg,
            model_base,
            filtered_harmful_val,
            filtered_harmless_val,
            candidate_directions,
        )

        direction_cpu = direction.detach().cpu()
        selected_directions.append(direction_cpu)
        repeat_summaries.append(
            {
                "repeat": repeat_idx,
                "position": int(pos),
                "layer": int(layer),
                "artifact_dir": repeat_cfg.artifact_path(),
            }
        )

    labels = [f"repeat_{idx:02d}" for idx in range(args.n_repeats)]
    signed_matrix = []
    abs_matrix = []
    for direction_a in selected_directions:
        signed_row = []
        abs_row = []
        for direction_b in selected_directions:
            cosine = direction_cosine(direction_a, direction_b)
            signed_row.append(cosine)
            abs_row.append(abs(cosine))
        signed_matrix.append(signed_row)
        abs_matrix.append(abs_row)

    pairwise_values = [
        signed_matrix[i][j]
        for i in range(args.n_repeats)
        for j in range(i + 1, args.n_repeats)
    ]
    pairwise_abs_values = [abs(v) for v in pairwise_values]

    torch.save(torch.stack(selected_directions), os.path.join(artifact_dir, "selected_directions.pt"))
    save_json(repeat_summaries, os.path.join(artifact_dir, "repeat_summaries.json"))
    save_json(signed_matrix, os.path.join(artifact_dir, "pairwise_cosine_signed.json"))
    save_json(abs_matrix, os.path.join(artifact_dir, "pairwise_cosine_abs.json"))
    write_pairwise_matrix_csv(labels, signed_matrix, os.path.join(artifact_dir, "pairwise_cosine_signed.csv"))
    write_pairwise_matrix_csv(labels, abs_matrix, os.path.join(artifact_dir, "pairwise_cosine_abs.csv"))

    summary = {
        "n_repeats": args.n_repeats,
        "pool_size": args.pool_size,
        "half_size": args.half_size,
        "mean_pairwise_cosine_signed": float(sum(pairwise_values) / len(pairwise_values)) if pairwise_values else None,
        "mean_pairwise_cosine_abs": float(sum(pairwise_abs_values) / len(pairwise_abs_values)) if pairwise_abs_values else None,
        "min_pairwise_cosine_abs": float(min(pairwise_abs_values)) if pairwise_abs_values else None,
        "max_pairwise_cosine_abs": float(max(pairwise_abs_values)) if pairwise_abs_values else None,
        "artifact_dir": artifact_dir,
    }
    save_json(summary, os.path.join(artifact_dir, "summary.json"))

    torch.cuda.empty_cache()
    print(f"Saved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
