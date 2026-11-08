from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import math
import os
import re
import string
import zipfile
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABELS = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]
NUM_LABELS = 5

DATA_CANDIDATES = [
    Path("/data"),
    Path("data")
]

# Retained even under the "full" filter: negation, degree, and contrastive
# discourse carry the phrase-level sentiment signal.
KEEP = {
    "not", "no", "nor", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "without",
    "very", "too", "so", "more", "most", "much", "less", "least",
    "only", "just", "even", "quite", "rather", "barely", "hardly",
    "but", "however", "yet", "still", "though", "although", "against",
    "again", "further", "once", "own", "same",
}


# ------------------------------------------------------------------- path helpers
def resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for cand in DATA_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError("competition files not found, pass --data-dir")


def find_file(data_dir: Path, stem: str) -> Path:
    for cand in (data_dir / f"{stem}.tsv.zip", data_dir / f"{stem}.tsv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{stem}.tsv[.zip] not found in {data_dir}")


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            data = zf.read(zf.namelist()[0]).decode("utf-8")
        return io.StringIO(data, newline="")
    return open(path, encoding="utf-8", newline="")


# ------------------------------------------------------------------ data loading
def load_tsv(path: Path, has_label: bool = True) -> pd.DataFrame:
    """Ids come from the first two fields, the label from the last field, and the
    phrase from everything between. Phrase text can contain tabs and stray quotes,
    which is where read_csv(sep='\\t') silently drops rows."""
    rows, skipped = [], 0
    with _open_text(path) as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)  # header
        for row in r:
            if len(row) < 3:
                skipped += 1
                continue
            phrase = "\t".join(row[2:-1] if has_label else row[2:])
            if not phrase.strip():
                skipped += 1
                continue
            rows.append((int(row[0]), int(row[1]), phrase,
                         int(row[-1]) if has_label else None))
    cols = ["PhraseId", "SentenceId", "Phrase"] + (["Sentiment"] if has_label else [])
    df = pd.DataFrame(rows, columns=cols)
    print(f"{Path(path).name}: kept {len(df)} rows, skipped {skipped}")
    return df


# --------------------------------------------------------------- stop-word filter
def build_filters() -> dict:
    try:
        from nltk.corpus import stopwords
        full = set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        full = set(stopwords.words("english"))
    return {"curated": full - KEEP, "full": full}


def make_stripper(mode: str):
    if mode == "none":
        return None
    sw = build_filters()[mode]
    print(f"stop-word filter='{mode}' ({len(sw)} words)")

    def strip(text: str) -> str:
        kept = []
        for w in re.findall(r"[A-Za-z']+|[^\sA-Za-z]", text):
            # contractions always survive: BPE splits "isn't" into ["is", "n't"],
            # so dropping the whole word throws away the negation
            if w.isalpha() and w.strip(string.punctuation).lower() in sw:
                continue
            kept.append(w)
        out = " ".join(kept).strip()
        return out if out else text
    return strip


# -------------------------------------------------------------------- tokenizing
def make_dataset(df, tok, maxlen, strip_fn, with_labels):
    cols = ["Phrase"] + (["Sentiment"] if with_labels else [])
    ds = Dataset.from_pandas(df[cols], preserve_index=False)
    if strip_fn is not None:
        ds = ds.map(lambda b: {"Phrase": [strip_fn(p) for p in b["Phrase"]]},
                    batched=True, desc="strip")
    ds = ds.map(lambda b: tok(b["Phrase"], truncation=True, max_length=maxlen),
                batched=True, desc="tokenize")
    if with_labels:
        ds = ds.rename_column("Sentiment", "labels")
    return ds


# ------------------------------------------------------ transformers version shims
def training_args(total_steps: int | None = None, **kw) -> TrainingArguments:
    """Pass only the kwargs this installed transformers actually accepts."""
    params = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" not in params and "eval_strategy" in kw:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")     # < 4.41

    if "warmup_ratio" not in params and "warmup_ratio" in kw:
        ratio = kw.pop("warmup_ratio")
        if "warmup_steps" in params and total_steps:
            kw["warmup_steps"] = int(total_steps * ratio)
            print(f"warmup_ratio unsupported, using warmup_steps={kw['warmup_steps']}")
        else:
            print("warmup_ratio unsupported, disabling warmup")

    for k in [k for k in kw if k not in params]:
        print(f"dropping unsupported TrainingArguments kwarg: {k}")
        kw.pop(k)
    return TrainingArguments(**kw)


def make_trainer(**kw) -> Trainer:
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params and "tokenizer" in kw:
        kw["processing_class"] = kw.pop("tokenizer")            # >= 4.46
    return Trainer(**kw)


def compute_metrics(ep):
    preds = ep.predictions.argmax(-1)
    return {"accuracy": accuracy_score(ep.label_ids, preds),
            "macro_f1": f1_score(ep.label_ids, preds, average="macro")}


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/base_cpu"))
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--maxlen", type=int, default=48)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=40000,
                    help="cap training rows after the split, 0 means no cap")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads, 0 means all logical cores")
    ap.add_argument("--assumed-rate", type=float, default=20.0,
                    help="samples/sec used only for the printed ETA guess")
    ap.add_argument("--strip", choices=["none", "curated", "full"], default="none")
    a = ap.parse_args()

    threads = a.threads or (os.cpu_count() or 4)
    torch.set_num_threads(threads)
    torch.manual_seed(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"cpu threads={threads} torch={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()}")

    data_dir = resolve_data_dir(a.data_dir)
    print("data dir:", data_dir)
    train = load_tsv(find_file(data_dir, "train"), has_label=True)
    print(train["Sentiment"].value_counts().sort_index().to_string())

    # group split first, so the row cap cannot leak a sentence across the boundary
    gss = GroupShuffleSplit(n_splits=1, test_size=a.val_frac, random_state=a.seed)
    tr_idx, va_idx = next(gss.split(train, groups=train["SentenceId"]))

    if a.max_rows and len(tr_idx) > a.max_rows:
        rng = pd.Series(tr_idx).sample(n=a.max_rows, random_state=a.seed).to_numpy()
        print(f"capping train rows {len(tr_idx)} -> {len(rng)}")
        tr_idx = rng

    json.dump({"train_idx": [int(i) for i in tr_idx],
               "val_idx": [int(i) for i in va_idx]},
              open(a.out_dir / "splits.json", "w"), indent=0)

    tok = AutoTokenizer.from_pretrained(a.model)
    strip_fn = make_stripper(a.strip)
    ds_tr = make_dataset(train.iloc[tr_idx], tok, a.maxlen, strip_fn, True)
    ds_va = make_dataset(train.iloc[va_idx], tok, a.maxlen, strip_fn, True)

    json.dump({"model": a.model, "maxlen": a.maxlen, "strip": a.strip,
               "seed": a.seed, "val_frac": a.val_frac, "epochs": a.epochs,
               "lr": a.lr, "batch": a.batch, "max_rows": a.max_rows},
              open(a.out_dir / "run_config.json", "w"), indent=2)

    steps_per_epoch = math.ceil(len(ds_tr) / a.batch)
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    eta = len(ds_tr) * a.epochs / a.assumed_rate / 60
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"rough ETA={eta:.0f} min at {a.assumed_rate} samples/s")
    print("first 100-step log line gives the real rate, recalibrate from there")

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=NUM_LABELS)

    args = training_args(
        total_steps=total_steps,
        output_dir=str(a.out_dir / "hf"),
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        num_train_epochs=a.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        use_cpu=True,
        fp16=False,
        bf16=False,
        label_smoothing_factor=0.1,
        group_by_length=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_steps=100,
        report_to="none",
        seed=a.seed,
    )
    trainer = make_trainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_va,
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(str(a.out_dir / "best"))
    tok.save_pretrained(str(a.out_dir / "best"))
    json.dump(trainer.state.log_history,
              open(a.out_dir / "log_history.json", "w"), indent=2)

    final = trainer.evaluate()
    print(json.dumps(final, indent=2))
    json.dump(final, open(a.out_dir / "final_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
