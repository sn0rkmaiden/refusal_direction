# Эксперименты по нахождению вектора отказа

В этом файл описыны три дополнительных эксперимента к коду для статьи **Refusal in Language Models Is Mediated by a Single Direction**: перенос поиска направления на внешние данные, проверку устойчивости к случайным подвыборкам и сравнение исходной (базовой) и диалоговой моделей.

Все команды предполагают запуск из корня репозитория:

```bash
cd ./refusal_direction
```


## Поиск направления отказа на внешних данных

### Цель

Проверить, можно ли найти функциональное направление отказа не только на исходных данных, которые использовали авторы, но и на другой паре наборов данных:

- вредные инструкции: `BeaverTails-Evaluation`;
- безвредные инструкции: `Dolly 15k`.

### Подготовка данных

```python
python3 -m pipeline.experiments.prepare_external_datasets \
  --output_dir dataset/external/beavertails_dolly \
  --seed 0 \
  --beavertails_split test
```

Файлы вывода:

```text
dataset/external/beavertails_dolly/
  beavertails_eval_harmful.json
  dolly15k_harmless.json
  metadata.json
  splits_seed0/
    harmful_train.json
    harmful_val.json
    harmful_test.json
    harmless_train.json
    harmless_val.json
    harmless_test.json
```

Размеры:

```text
num_harmful = 700
num_harmless = 10354

harmful_train = 128
harmful_val = 32
harmful_test = 100
harmless_train = 128
harmless_val = 32
harmless_test = 100
```

### Запуск


Модель `gemma-2b-it`:

```bash
python3 -m pipeline.experiments.run_external_direction_experiment \
  --model_path google/gemma-2b-it \
  --split_dir dataset/external/beavertails_dolly/splits_seed0 \
  --run_name beavertails_dolly_seed0 \
  --seed 0
```

Модель `Qwen/Qwen-1_8B-Chat`:

```bash
python3 -m pipeline.experiments.run_external_direction_experiment \
  --model_path Qwen/Qwen-1_8B-Chat \
  --split_dir dataset/external/beavertails_dolly/splits_seed0 \
  --run_name beavertails_dolly_seed0 \
  --seed 0
```

### Результаты

Файлы сохраняются в: ``pipeline/runs/<имя-модели>/experiments/external_direction/beavertails_dolly_seed0/``


Основные файлы:

```text
config.json
summary.json
filter_counts.json
sampled_data/
direction.pt
direction_metadata.json
select_direction/
completions/
```

Оценки по категориям находятся в файлах папки `completions/`, в поле `substring_matching_per_category`.

### Основные результаты

| Модель | Направление | Вредные запросы: до → после удаления | Безвредные запросы: до → после добавления |
|---|---:|---:|---:|
| Gemma 2B IT | слой 10, позиция -1 | 0.38 → 0.96 | 0.94 → 0.03 |
| Qwen 1.8B Chat | слой 16, позиция -1 | 0.31 → 0.84 | 0.89 → 0.00 |

Интерпретация: основной эффект переносится на внешнюю пару наборов данных. Удаление направления повышает долю ответов без явного отказа на вредных запросах, а добавление направления вызывает отказ на безвредных запросах.

---

## 2. Устойчивость направления к случайным подвыборкам

### Цель

Проверить, является ли направление отказа устойчивым к случайному выбору обучающих примеров.

Процедура:

1. выбирается общий пул из 256 вредных и 256 безвредных инструкций;
2. в каждом повторе выбираются 128 вредных и 128 безвредных инструкций;
3. направление отказа извлекается заново;
4. выбранные направления сравниваются попарно с помощью косинусного сходства.

### Запуск

Для Gemma:

```bash
python3 -m pipeline.experiments.run_split_half_stability \
  --model_path google/gemma-2b-it \
  --run_name author_seed42_repeats8_pool256_half128 \
  --seed 42 \
  --n_repeats 8 \
  --pool_size 256 \
  --half_size 128 \
  --n_val 32
```

Qwen:

```bash
python3 -m pipeline.experiments.run_split_half_stability \
  --model_path Qwen/Qwen-1_8B-Chat \
  --run_name author_seed42_repeats8_pool256_half128 \
  --seed 42 \
  --n_repeats 8 \
  --pool_size 256 \
  --half_size 128 \
  --n_val 32
```

### Результаты

Файлы сохраняются в: `pipeline/runs/<имя-модели>/experiments/split_half_stability/author_seed42_repeats8_pool256_half128/`.

Основные файлы:

```text
summary.json
repeat_summaries.json
pairwise_cosine_signed.csv
pairwise_cosine_abs.csv
selected_directions.pt
sampled_data/
repeat_00/
...
repeat_07/
```

### Основные результаты

| Модель | Выбранные слои и позиции | Среднее попарное косинусное сходство | Минимальное попарное косинусное сходство |
|---|---:|---:|---:|
| Gemma 2B IT | слой 12, позиция -1 | 0.996 | 0.995 |
| Qwen 1.8B Chat | позиция -1, слои 14–15 | 0.866 | 0.755 |

Интерпретация: для Gemma направление почти не меняется при смене обучающей подвыборки. Для Qwen направление также остается достаточно близким, но точный слой менее стабилен.

Важно: выбранные слой и позиция в этом эксперименте не обязаны полностью совпадать с первым воспроизведением. Возможно, существует несколько близких эффективных кандидатов, а процедура выбора отдает предпочтение разным слоям или позициям в зависимости от обучающей и проверочной подвыборок.

---

## 3. Сравнение исходной и диалоговой модели

### Цель

Проверить, есть ли похожее направление отказа в исходной модели до диалогового дообучения.

Пары моделей:

```text
Gemma 2B / Gemma 2B IT
Qwen 1.8B / Qwen 1.8B Chat
```

Скрипт рассматривает два режима для исходной модели:

- `matched`: взять вектор в исходной модели на том же слое и позиции, где выбрано направление в диалоговой модели;
- `selected`: независимо найти лучший кандидат в исходной модели среди всех слоев и позиций.

### Запуск

Gemma:

```bash
python3 -m pipeline.experiments.run_base_chat_comparison \
  --base_model_path google/gemma-2b \
  --chat_model_path google/gemma-2b-it \
  --run_name gemma_2b_base_vs_it_seed42 \
  --seed 42 \
  --base_search_mode both
```

Qwen:

```bash
python3 -m pipeline.experiments.run_base_chat_comparison \
  --base_model_path Qwen/Qwen-1_8B \
  --chat_model_path Qwen/Qwen-1_8B-Chat \
  --run_name qwen_1_8b_base_vs_chat_seed42 \
  --seed 42 \
  --base_search_mode both
```

Быстрая проверка без генерации ответов:

```bash
python3 -m pipeline.experiments.run_base_chat_comparison \
  --base_model_path google/gemma-2b \
  --chat_model_path google/gemma-2b-it \
  --run_name gemma_2b_base_vs_it_seed42_no_generation \
  --seed 42 \
  --base_search_mode both \
  --skip_generation
```

### Результаты

Файлы сохраняются в: `pipeline/runs/<имя-диалоговой-модели>/experiments/base_chat_comparison/qwen_1_8b_base_vs_chat_seed42/`.

Основные файлы:

```text
config.json
summary.json
sampled_data/
chat_selected/
base_direction_search/
base_matched_eval/
base_selected_eval/
```

### Основные результаты

| Пара моделей | Направление в диалоговой модели | Косинусное сходство с `base matched` | Независимо выбранное направление в исходной модели | Косинусное сходство с `base selected` |
|---|---:|---:|---:|---:|
| Gemma 2B / Gemma 2B IT | слой 12, позиция -1 | 0.108 | слой 15, позиция -4 | 0.270 |
| Qwen 1.8B / Qwen 1.8B Chat | слой 14, позиция -1 | 0.132 | слой 1, позиция -3 | -0.014 |

Для обеих исходных моделей независимый поиск не прошел исходные критерии отбора и был выполнен в запасном режиме:

```text
fallback_lowest_ablation_refusal_score
```

Интерпретация: в диалоговых моделях вектор отказа обладает ожидаемым эффектом. В исходных моделях найденные вектора геометрически не близки к вектора диалоговых моделей, а вмешательства не воспроизводят полный эффект. Особенно важно, что добавление направления в исходной модели почти не вызывает отказ на безвредных запросах. Это предварительно указывает на то, что механизм отказа в изученной форме связан прежде всего с дообучением следовать инструкциям. Однако это не значит, что в исходной модели нет более ранних связанных представлений.

---

