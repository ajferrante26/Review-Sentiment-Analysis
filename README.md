# Transformer Sentiment Analysis on Movie Reviews

CPU-capable fine-tuning pipeline for Kaggle's **Sentiment Analysis on Movie Reviews**
competition (Rotten Tomatoes phrase-level SST). Built for CSC-8150 Assignment 6.

Trains a pretrained transformer for 5-class ordinal sentiment classification, evaluates
on a leakage-safe validation split, and produces a competition submission plus the
figures needed for the write-up.

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle, `sentiment-analysis-on-movie-reviews` (SST phrase-level, Socher et al. 2013) |
| Train rows | 156,059 usable of 156,060 (1 malformed line is skipped) |
| Test rows | 66,292 |
| Columns | `PhraseId`, `SentenceId`, `Phrase`, `Sentiment` |
| Labels | 0 negative, 1 somewhat negative, 2 neutral, 3 somewhat positive, 4 positive |
| Metric | Categorization accuracy, dominated by the neutral class (~51%) |

Phrases, not full reviews, are the classification unit. Many phrases share a
`SentenceId`, which drives the split strategy below.

Expected input layout:

```
data/
  train.tsv          (or train.tsv.zip)
  test.tsv           (or test.tsv.zip)
  sampleSubmission.csv
```

---

## Files

| File | Purpose |
|---|---|
| `model.py` | Loads data, filters, tokenizes, trains, saves checkpoint and run config |
| `submit.py` | Standalone inference on `test.tsv`, writes and validates `submission.csv` |
| `modeltest.py` | Evaluation: metrics, confusion matrix, error dump, curves, attention maps |
| `patch_load_tsv.py` | One-shot fix for the `has_label=False` branch of `load_tsv` (see Known Issues) |
| `runs/<name>/` | Per-run artifacts (checkpoint, config, splits, logs, figures) |

---

## Requirements

Python 3.10 or newer. CPU is sufficient, no CUDA build required.

```
torch>=2.0
transformers>=4.36
datasets>=2.14
scikit-learn>=1.3
pandas>=2.0
numpy
matplotlib
nltk          # only for --strip curated or --strip full
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Freeze the exact set for reproducibility:

```powershell
pip freeze > requirements-lock.txt
```

---

## Quickstart

```powershell
# 1. Train the reduced CPU profile (about an hour)
python model.py --data-dir data --strip none --out-dir runs/base_cpu

# 2. Build and validate the submission
python submit.py

# 3. Optional: full evaluation artefacts
python modeltest.py --ckpt runs/base_cpu/best --data-dir data
```

`submit.py` writes `runs/base_cpu/submission.csv`. Upload that file to the
competition's Submit Predictions page.

---

## `model.py`

```
python model.py [options]
```

| Option | Default | Notes |
|---|---|---|
| `--data-dir` | auto | Falls back to `/data`, `data`, then the Kaggle mount |
| `--out-dir` | `runs/base_cpu` | Checkpoint, config, splits, logs land here |
| `--model` | `distilroberta-base` | `roberta-base` for the full-quality run |
| `--maxlen` | `48` | 64 for `roberta-base`, phrases are short |
| `--epochs` | `2.0` | 3 to 4 for the full run |
| `--lr` | `3e-5` | 2e-5 for `roberta-base` |
| `--batch` | `16` | Per-device, CPU-sized |
| `--max-rows` | `40000` | Applied after the split, `0` means no cap |
| `--val-frac` | `0.15` | Group split size |
| `--seed` | `42` | Split, sampling, and training seed |
| `--threads` | `0` | `torch` CPU threads, 0 means all logical cores |
| `--assumed-rate` | `20.0` | Samples/sec, used only for the printed ETA guess |
| `--strip` | `none` | `none`, `curated`, or `full` stop-word filtering |

### Split strategy

Validation is a `GroupShuffleSplit` keyed on `SentenceId`, not a row split. Phrases
from one source sentence otherwise appear in both halves and inflate validation
accuracy. The chosen indices are persisted to `splits.json` so every later run and
every `modeltest.py` evaluation uses the same rows.

### Stop-word filtering

`--strip curated` removes NLTK stop words except negation, degree, and contrastive
discourse terms (`not`, `never`, `very`, `too`, `but`, `however`, and similar).
Contractions are always retained because byte-level BPE splits `isn't` into
`["is", "n't"]`, so deleting the word loses the negation.

`--strip none` is the primary configuration. Filtering is kept as an ablation, and
the assignment's "as necessary" wording supports treating it that way.

---

## `submit.py`

```
python submit.py
```

No command-line arguments. Edit the `CKPT` constant at the top of the file to point at
a different checkpoint:

```python
CKPT = Path("runs/base_cpu/best")
```

The script reads `run_config.json` from the checkpoint's parent directory, so
`--strip` and `--maxlen` always match what the checkpoint was trained with. It then:

1. Loads `test.tsv` with a self-contained parser.
2. Applies the same stop-word filter as training.
3. Runs batched inference.
4. Writes `submission.csv` next to the checkpoint.
5. Validates column names, row count, `PhraseId` order, and label range against
   `sampleSubmission.csv`.

A successful run ends with:

```
submission OK (66292, 2) -> runs\base_cpu\submission.csv
```

If that assertion fails, do not upload the file.

---

## Outputs

| File | Written by | Contents |
|---|---|---|
| `run_config.json` | `model.py` | Model, maxlen, strip mode, seed, hyperparameters |
| `splits.json` | `model.py` | Train and validation row indices |
| `log_history.json` | `model.py` | Per-step loss and per-epoch eval metrics |
| `final_metrics.json` | `model.py` | Validation accuracy and macro F1 |
| `best/` | `model.py` | Weights plus tokenizer |
| `submission.csv` | `submit.py`, `modeltest.py` | `PhraseId,Sentiment` for 66,292 rows |
| `confusion_matrix.csv` | `modeltest.py` | 5x5 confusion matrix |
| `errors.csv` | `modeltest.py` | Misclassified validation phrases, sorted by confidence |
| `curves.png` | `modeltest.py` | Train and validation loss, validation macro F1 |
| `attention_<n>.png` | `modeltest.py` | Last-layer attention heatmaps |

---

## Recorded Results

`runs/base_cpu`: `distilroberta-base`, `maxlen=48`, `epochs=2`, `batch=16`,
`max_rows=40000`, `strip=none`, `seed=42`. Validation on all 23,029 held-out phrases.

| Metric | Model | Always-predict-neutral baseline |
|---|---|---|
| Accuracy | 0.665 | ~0.510 |
| Macro F1 | 0.565 | ~0.135 |
| Macro precision | 0.579 | ~0.200 |

Per-class F1: negative 0.396, somewhat negative 0.551, neutral 0.771,
somewhat positive 0.594, positive 0.514.

Confusion is almost entirely single-step. Only 2 of 1,007 negatives and 1 of 1,304
positives are misclassified at the opposite pole, which supports the reading that the
model learns polarity and fails on intensity. The weakest class is `negative` at 0.350
recall, with 533 of its 1,007 examples collapsing into `somewhat negative`.

---

## Known Issues

**`load_tsv(has_label=False)` raises `ValueError: 3 columns passed, passed data had 4 columns`.**
The function builds a 4-tuple per row but only declares 3 columns when labels are
absent. `model.py` training is unaffected because it always passes `has_label=True`.
`modeltest.py` hits it on the test file. Run `python patch_load_tsv.py` to apply the
`if has_label:` branch, or use `submit.py`, which does not call that code path.

**`submit.py` has no CLI.** Change the `CKPT` constant to evaluate a different run.

**Windows multiprocessing.** The training arguments pin `dataloader_num_workers=0`
and `dataloader_pin_memory=False`, which avoids spawn errors and a pointless
pinned-memory warning on CPU-only Windows.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `unexpected keyword argument 'warmup_ratio'` | Old `transformers` build | The `training_args` shim drops or converts unsupported kwargs and prints what it dropped. `pip install --upgrade transformers` |
| `'SequenceClassifierOutput' object has no attribute 'float'` | Missing `.logits` after `model(**enc)` | Use `model(**enc).logits.float()` |
| `3 columns passed, passed data had 4 columns` | `load_tsv` with `has_label=False` | See Known Issues |
| `FileNotFoundError: train.tsv[.zip]` | Wrong `--data-dir` | Pass the folder containing `train.tsv` or `train.tsv.zip` |
| `LookupError: Resource stopwords not found` | NLTK corpus missing | Needs internet once. `python -m nltk.downloader stopwords` |
| Very slow epoch, no progress lines | `--max-rows 0` on CPU | Watch the first 100-step log line and recompute the ETA from real throughput |
| `PhraseId order mismatch` from validation | Submission built from a reordered frame | Do not sort or filter the test frame, keep source order |

---

## Reproducibility

Every run records its own configuration. To regenerate a result:

1. Use the `seed`, `model`, `maxlen`, `strip`, and `max_rows` values from that run's
   `run_config.json`.
2. Keep `splits.json` from the original run, or regenerate it with the same seed and
   the same dataset file. The indices are positions in the loaded train frame, so a
   different parser changes what they point at.
3. Record the `transformers` and `torch` versions. Both are printed at the top of the
   training log and both are pinned in `requirements-lock.txt`.

Report in the paper: model name, maxlen, epochs, learning rate, batch size, row cap,
stop-word mode, seed, library versions, and both public and private leaderboard scores.

---

## Next Steps

1. Run `python patch_load_tsv.py` so `modeltest.py` completes end to end.
2. Full-quality run: `python model.py --data-dir data --strip none --max-rows 0 --model roberta-base --maxlen 64 --epochs 3 --out-dir runs/full_cpu`.
3. Ablation: repeat with `--strip curated` and `--strip full`, compare macro F1.
4. Ordinal treatment: a regression head (`num_labels=1`, round at inference) or a CORAL
   head, motivated by the single-step confusion structure.
5. Calibration: temperature scaling on the validation set, since correct predictions
   average 0.720 confidence and incorrect ones average 0.614.
