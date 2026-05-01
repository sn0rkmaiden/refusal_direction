# Additional refusal-direction experiments

Run all commands from the repository root.

## 1. External dataset direction

This script extracts a refusal direction from harmful and harmless datasets that you pass explicitly.
It accepts `processed:<name>`, `split:<harmtype>_<split>`, a processed dataset name, or a path to `.json`, `.jsonl`, `.csv`, or `.txt`.

Example:

```bash
python3 -m pipeline.experiments.run_external_direction_experiment \
  --model_path google/gemma-2b-it \
  --harmful_dataset processed:strongreject \
  --harmless_dataset /path/to/harmless_instructions.jsonl \
  --run_name strongreject_external_harmless
```

Artifacts are saved to:

```text
pipeline/runs/<model_alias>/experiments/external_direction/<run_name>/
```

## 2. Split-half stability

This script repeatedly extracts directions from random halves of a fixed training pool and compares selected directions using pairwise cosine similarity.

Example:

```bash
python3 -m pipeline.experiments.run_split_half_stability \
  --model_path google/gemma-2b-it \
  --n_repeats 8 \
  --pool_size 256 \
  --half_size 128 \
  --run_name default
```

Artifacts are saved to:

```text
pipeline/runs/<model_alias>/experiments/split_half_stability/<run_name>/
```

## 3. Base vs chat comparison

This script extracts the selected direction from a chat/instruction-tuned model, then extracts the base-model mean-difference vector at the same position and layer. It reports cosine similarity and, unless `--skip_generation` is passed, evaluates the corresponding interventions.

Example:

```bash
python3 -m pipeline.experiments.run_base_chat_comparison \
  --base_model_path google/gemma-2b \
  --chat_model_path google/gemma-2b-it \
  --run_name gemma_2b_base_vs_it
```

Artifacts are saved to:

```text
pipeline/runs/<chat_model_alias>/experiments/base_chat_comparison/<run_name>/
```
