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

```bash
python3 -m pipeline.experiments.run_external_direction_experiment \
  --model_path google/gemma-2b-it \
  --harmful_dataset dataset/external/beavertails_dolly/beavertails_eval_harmful.json \
  --harmless_dataset dataset/external/beavertails_dolly/dolly15k_harmless.json \
  --run_name beavertails_dolly
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

Gemma:
```python
CUDA_VISIBLE_DEVICES=1 python3 -m pipeline.experiments.run_split_half_stability \
  --model_path google/gemma-2b-it \
  --run_name author_seed42_repeats8_pool256_half128 \
  --seed 42 \
  --n_repeats 8 \
  --pool_size 256 \
  --half_size 128 \
  --n_val 32
```

Qwen:

```python
CUDA_VISIBLE_DEVICES=1 python3 -m pipeline.experiments.run_split_half_stability \
  --model_path qwen/qwen-1_8b-chat \
  --run_name author_seed42_repeats8_pool256_half128 \
  --seed 42 \
  --n_repeats 8 \
  --pool_size 256 \
  --half_size 128 \
  --n_val 32
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

Gemma:
```python
CUDA_VISIBLE_DEVICES=1 python3 -m pipeline.experiments.run_base_chat_comparison \
  --base_model_path google/gemma-2b \
  --chat_model_path google/gemma-2b-it \
  --run_name gemma_2b_base_vs_it_seed42 \
  --seed 42 \
  --base_search_mode both
```

Qwen:
```python
CUDA_VISIBLE_DEVICES=1 python3 -m pipeline.experiments.run_base_chat_comparison \
  --base_model_path Qwen/Qwen-1_8B \
  --chat_model_path Qwen/Qwen-1_8B-Chat \
  --run_name qwen_1_8b_base_vs_chat_seed42 \
  --seed 42 \
  --base_search_mode both
```

Artifacts are saved to:

```text
pipeline/runs/<chat_model_alias>/experiments/base_chat_comparison/<run_name>/
```
