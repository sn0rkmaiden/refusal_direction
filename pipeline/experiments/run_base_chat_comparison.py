import argparse
import gc
import os
from typing import Dict, Optional, Tuple

import torch

from pipeline.experiments.common import (
    direction_cosine,
    ensure_dir,
    filter_train_val_strings,
    instruction_strings,
    load_instruction_dataset,
    make_child_config,
    make_experiment_config,
    read_evaluation_metric,
    sample_rows,
    save_json,
    set_seed,
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
        description=(
            "Compare refusal directions in a base model and its instruction-tuned/chat variant. "
            "The base model is analyzed in two modes: matched direction at the chat-selected "
            "layer/position and, optionally, full independent direction search."
        )
    )
    parser.add_argument("--base_model_path", default="google/gemma-2b")
    parser.add_argument("--chat_model_path", default="google/gemma-2b-it")
    parser.add_argument("--harmful_dataset", default="split:harmful_train")
    parser.add_argument("--harmless_dataset", default="split:harmless_train")
    parser.add_argument("--harmful_val_dataset", default="split:harmful_val")
    parser.add_argument("--harmless_val_dataset", default="split:harmless_val")
    parser.add_argument("--harmful_test_dataset", default="jailbreakbench")
    parser.add_argument("--harmless_test_dataset", default="split:harmless_test")
    parser.add_argument("--run_name", default="gemma_2b_base_vs_it")
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--filter_chat_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_chat_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_base_train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--filter_base_val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--base_search_mode",
        choices=("matched", "selected", "both"),
        default="both",
        help=(
            "matched: compare the base-model vector at the chat-selected layer/position; "
            "selected: run a full independent direction search in the base model; "
            "both: do both analyses."
        ),
    )
    parser.add_argument(
        "--allow_base_selection_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If the original selection heuristic filters out all base-model candidates, choose the "
            "candidate with the lowest ablation refusal score from direction_evaluations.json. "
            "This makes the script robust for base models where refusal is weak or absent."
        ),
    )
    parser.add_argument("--skip_generation", action="store_true")
    return parser.parse_args()


def cleanup_model(model_base) -> None:
    if model_base is not None:
        model_base.del_model()
        del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_and_evaluate(cfg, model_base, direction, layer: int, harmful_test_rows, harmless_test_rows) -> Dict[str, float]:
    baseline_pre_hooks, baseline_hooks = [], []
    ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_harmful_pre_hooks = [
        (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))
    ]
    actadd_harmless_pre_hooks = [
        (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))
    ]

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_pre_hooks, baseline_hooks, "baseline", "harmful", dataset=harmful_test_rows)
    generate_and_save_completions_for_dataset(cfg, model_base, ablation_pre_hooks, ablation_hooks, "ablation", "harmful", dataset=harmful_test_rows)
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_harmful_pre_hooks, [], "actadd", "harmful", dataset=harmful_test_rows)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_pre_hooks, baseline_hooks, "baseline", "harmless", dataset=harmless_test_rows)
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_harmless_pre_hooks, [], "actadd", "harmless", dataset=harmless_test_rows)

    for dataset_name in ("harmful", "harmless"):
        for label in ("baseline", "ablation", "actadd"):
            completions_path = os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{label}_completions.json")
            if not os.path.exists(completions_path):
                continue
            evaluate_completions_and_save_results_for_dataset(cfg, label, dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

    metrics: Dict[str, float] = {}
    for dataset_name in ("harmful", "harmless"):
        for label in ("baseline", "ablation", "actadd"):
            eval_path = os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{label}_evaluations.json")
            value = read_evaluation_metric(eval_path)
            if value is not None:
                metrics[f"{dataset_name}_{label}_non_refusal_rate"] = value
    return metrics


def select_base_direction_with_optional_fallback(
    cfg,
    model_base,
    harmful_val,
    harmless_val,
    candidate_directions,
    *,
    allow_fallback: bool,
) -> Tuple[Optional[int], Optional[int], Optional[torch.Tensor], str]:
    try:
        pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)
        return int(pos), int(layer), direction, "original_selection"
    except AssertionError as exc:
        if not allow_fallback:
            raise
        evaluations_path = os.path.join(cfg.artifact_path(), "select_direction", "direction_evaluations.json")
        if not os.path.exists(evaluations_path):
            raise RuntimeError(
                "Base-model full selection failed and no direction_evaluations.json was found for fallback."
            ) from exc
        import json

        with open(evaluations_path, "r", encoding="utf-8") as f:
            evaluations = json.load(f)
        if not evaluations:
            raise RuntimeError("Base-model full selection failed and direction_evaluations.json is empty.") from exc

        # For base models the full Arditi et al. selection criterion may filter every vector out,
        # especially when refusal is weak. The fallback uses the least remaining refusal score after
        # ablation, which is the main behavioural objective of the original ranking.
        best = min(evaluations, key=lambda row: (row["refusal_score"], row["kl_div_score"]))
        pos = int(best["position"])
        layer = int(best["layer"])
        direction = candidate_directions[pos, layer]
        torch.save(direction, os.path.join(cfg.artifact_path(), "direction.pt"))
        save_json(
            {
                "pos": pos,
                "layer": layer,
                "selection_mode": "fallback_lowest_ablation_refusal_score",
                "fallback_reason": str(exc),
                "selected_row": best,
            },
            os.path.join(cfg.artifact_path(), "direction_metadata.json"),
        )
        return pos, layer, direction, "fallback_lowest_ablation_refusal_score"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    parent_cfg = make_experiment_config(
        args.chat_model_path,
        "base_chat_comparison",
        args.run_name,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        max_new_tokens=args.max_new_tokens,
    )
    artifact_dir = ensure_dir(parent_cfg.artifact_path())
    save_json(vars(args), os.path.join(artifact_dir, "config.json"))

    harmful_train_rows = sample_rows(load_instruction_dataset(args.harmful_dataset, default_category="harmful"), args.n_train, seed=args.seed, label="harmful train")
    harmless_train_rows = sample_rows(load_instruction_dataset(args.harmless_dataset, default_category="harmless"), args.n_train, seed=args.seed + 1, label="harmless train")
    harmful_val_rows = sample_rows(load_instruction_dataset(args.harmful_val_dataset, default_category="harmful_val"), args.n_val, seed=args.seed + 2, label="harmful validation")
    harmless_val_rows = sample_rows(load_instruction_dataset(args.harmless_val_dataset, default_category="harmless_val"), args.n_val, seed=args.seed + 3, label="harmless validation")
    harmful_test_rows = sample_rows(load_instruction_dataset(args.harmful_test_dataset, default_category="harmful_test"), args.n_test, seed=args.seed + 4, label="harmful test")
    harmless_test_rows = sample_rows(load_instruction_dataset(args.harmless_test_dataset, default_category="harmless_test"), args.n_test, seed=args.seed + 5, label="harmless test")

    sampled_dir = ensure_dir(os.path.join(artifact_dir, "sampled_data"))
    save_json(harmful_train_rows, os.path.join(sampled_dir, "harmful_train.json"))
    save_json(harmless_train_rows, os.path.join(sampled_dir, "harmless_train.json"))
    save_json(harmful_val_rows, os.path.join(sampled_dir, "harmful_val.json"))
    save_json(harmless_val_rows, os.path.join(sampled_dir, "harmless_val.json"))
    save_json(harmful_test_rows, os.path.join(sampled_dir, "harmful_test.json"))
    save_json(harmless_test_rows, os.path.join(sampled_dir, "harmless_test.json"))

    harmful_train = instruction_strings(harmful_train_rows)
    harmless_train = instruction_strings(harmless_train_rows)
    harmful_val = instruction_strings(harmful_val_rows)
    harmless_val = instruction_strings(harmless_val_rows)

    summary: Dict[str, object] = {"artifact_dir": artifact_dir}

    chat_cfg = make_child_config(parent_cfg, "chat_selected")
    chat_cfg.model_path = args.chat_model_path
    chat_cfg.filter_train = args.filter_chat_train
    chat_cfg.filter_val = args.filter_chat_val

    chat_model_base = construct_model_base(args.chat_model_path)
    chat_harmful_train, chat_harmless_train, chat_harmful_val, chat_harmless_val = filter_train_val_strings(
        chat_cfg,
        chat_model_base,
        harmful_train,
        harmless_train,
        harmful_val,
        harmless_val,
    )
    save_json(
        {
            "harmful_train_after_filter": len(chat_harmful_train),
            "harmless_train_after_filter": len(chat_harmless_train),
            "harmful_val_after_filter": len(chat_harmful_val),
            "harmless_val_after_filter": len(chat_harmless_val),
        },
        os.path.join(chat_cfg.artifact_path(), "filter_counts.json"),
    )
    chat_candidates = generate_and_save_candidate_directions(chat_cfg, chat_model_base, chat_harmful_train, chat_harmless_train)
    chat_pos, chat_layer, chat_direction = select_and_save_direction(chat_cfg, chat_model_base, chat_harmful_val, chat_harmless_val, chat_candidates)
    chat_direction_cpu = chat_direction.detach().cpu()
    summary["chat_selected_position"] = int(chat_pos)
    summary["chat_selected_layer"] = int(chat_layer)

    if not args.skip_generation:
        summary.update({f"chat_selected_{k}": v for k, v in generate_and_evaluate(chat_cfg, chat_model_base, chat_direction, int(chat_layer), harmful_test_rows, harmless_test_rows).items()})

    cleanup_model(chat_model_base)

    base_search_cfg = make_child_config(parent_cfg, "base_direction_search")
    base_search_cfg.model_path = args.base_model_path
    base_search_cfg.filter_train = args.filter_base_train
    base_search_cfg.filter_val = args.filter_base_val

    base_model_base = construct_model_base(args.base_model_path)
    base_harmful_train, base_harmless_train, base_harmful_val, base_harmless_val = filter_train_val_strings(
        base_search_cfg,
        base_model_base,
        harmful_train,
        harmless_train,
        harmful_val,
        harmless_val,
    )
    save_json(
        {
            "harmful_train_after_filter": len(base_harmful_train),
            "harmless_train_after_filter": len(base_harmless_train),
            "harmful_val_after_filter": len(base_harmful_val),
            "harmless_val_after_filter": len(base_harmless_val),
            "note": "By default, base-model filtering is disabled because base models may not refuse harmful prompts before instruction tuning.",
        },
        os.path.join(base_search_cfg.artifact_path(), "filter_counts.json"),
    )
    base_candidates = generate_and_save_candidate_directions(base_search_cfg, base_model_base, base_harmful_train, base_harmless_train)

    if args.base_search_mode in ("matched", "both"):
        base_matched_direction = base_candidates[int(chat_pos), int(chat_layer)]
        base_matched_direction_cpu = base_matched_direction.detach().cpu()
        torch.save(base_matched_direction, os.path.join(base_search_cfg.artifact_path(), "matched_direction.pt"))
        save_json(
            {
                "matched_to_chat_position": int(chat_pos),
                "matched_to_chat_layer": int(chat_layer),
                "note": "Base-model mean-difference vector at the chat model's selected position/layer.",
            },
            os.path.join(base_search_cfg.artifact_path(), "matched_direction_metadata.json"),
        )
        summary["base_matched_position"] = int(chat_pos)
        summary["base_matched_layer"] = int(chat_layer)
        summary["cosine_chat_selected_vs_base_matched"] = direction_cosine(chat_direction_cpu, base_matched_direction_cpu)

        if not args.skip_generation:
            base_matched_eval_cfg = make_child_config(parent_cfg, "base_matched_eval")
            base_matched_eval_cfg.model_path = args.base_model_path
            summary.update({f"base_matched_{k}": v for k, v in generate_and_evaluate(base_matched_eval_cfg, base_model_base, base_matched_direction, int(chat_layer), harmful_test_rows, harmless_test_rows).items()})

    base_selected_direction = None
    base_selected_pos = None
    base_selected_layer = None
    if args.base_search_mode in ("selected", "both"):
        base_selected_pos, base_selected_layer, base_selected_direction, selection_mode = select_base_direction_with_optional_fallback(
            base_search_cfg,
            base_model_base,
            base_harmful_val,
            base_harmless_val,
            base_candidates,
            allow_fallback=args.allow_base_selection_fallback,
        )
        if base_selected_direction is not None:
            base_selected_direction_cpu = base_selected_direction.detach().cpu()
            summary["base_selected_position"] = int(base_selected_pos)
            summary["base_selected_layer"] = int(base_selected_layer)
            summary["base_selected_selection_mode"] = selection_mode
            summary["cosine_chat_selected_vs_base_selected"] = direction_cosine(chat_direction_cpu, base_selected_direction_cpu)
            if args.base_search_mode == "both":
                matched_path = os.path.join(base_search_cfg.artifact_path(), "matched_direction.pt")
                if os.path.exists(matched_path):
                    matched_direction = torch.load(matched_path, map_location="cpu")
                    summary["cosine_base_matched_vs_base_selected"] = direction_cosine(matched_direction, base_selected_direction_cpu)

            if not args.skip_generation:
                base_selected_eval_cfg = make_child_config(parent_cfg, "base_selected_eval")
                base_selected_eval_cfg.model_path = args.base_model_path
                summary.update({f"base_selected_{k}": v for k, v in generate_and_evaluate(base_selected_eval_cfg, base_model_base, base_selected_direction, int(base_selected_layer), harmful_test_rows, harmless_test_rows).items()})

    cleanup_model(base_model_base)
    save_json(summary, os.path.join(artifact_dir, "summary.json"))
    print(f"Saved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
