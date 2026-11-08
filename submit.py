"""submit.py - build submission.csv with a self-contained test loader."""
import csv, json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import find_file, make_stripper, resolve_data_dir

CKPT = Path("runs/base_cpu/best")


def load_test(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            if len(row) < 3:
                continue
            rows.append((int(row[0]), int(row[1]), "\t".join(row[2:])))
    return pd.DataFrame(rows, columns=["PhraseId", "SentenceId", "Phrase"])


data_dir = resolve_data_dir("data")
run_dir = CKPT.parent
cfg = json.load(open(run_dir / "run_config.json"))

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).eval()
strip_fn = make_stripper(cfg["strip"])

test = load_test(find_file(data_dir, "test"))
print("test rows:", len(test))
texts = [strip_fn(t) for t in test["Phrase"]] if strip_fn else list(test["Phrase"])

pred = []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=cfg["maxlen"],
                  padding=True, return_tensors="pt")
        pred += model(**enc).logits.argmax(-1).tolist()

sub = pd.DataFrame({"PhraseId": test["PhraseId"], "Sentiment": pred})
sub.to_csv(run_dir / "submission.csv", index=False)
print(sub["Sentiment"].value_counts().sort_index().to_string())

samp = pd.read_csv(data_dir / "sampleSubmission.csv")
assert list(sub.columns) == list(samp.columns)
assert len(sub) == len(samp)
assert (sub.PhraseId.values == samp.PhraseId.values).all(), "PhraseId order mismatch"
print("submission OK", sub.shape, "->", run_dir / "submission.csv")
