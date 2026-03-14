"""
finetune_finbert.py
────────────────────────────────────────────────────────────────
Fine-tunes ProsusAI/finbert on the synthetic supply-chain headline
dataset for 3-class sentiment:
    0 = negative  1 = neutral  2 = positive

Architecture:
  - Base: ProsusAI/finbert  (BERT-base fine-tuned on Reuters/Bloomberg)
  - Frozen: Embedding layer + transformer layers 0–9  (10 layers)
  - Trainable: Transformer layers 10–11 + pooler + classifier head
  - Head: Linear(768 → 3)  [replaced from FinBERT's original 3-class head]

This implements TRANSFER LEARNING (frozen FinBERT backbone) +
FINE-TUNING (last 2 transformer layers + new head trained on domain data).

Setup:
    pip install transformers datasets scikit-learn torch huggingface_hub

Usage:
    1. Build dataset first (downloads real data from HuggingFace):
           python build_dataset.py

    2. Login to HuggingFace Hub:
           huggingface-cli login
           (paste your HF token with write access)

    3. Run fine-tuning:
           python finetune_finbert.py --hf_repo YOUR_HF_USERNAME/supplychain-finbert

    4. Model is pushed to HuggingFace Hub automatically on completion.

Environment variables (optional):
    HF_TOKEN   — HuggingFace token (alternative to huggingface-cli login)
    HF_REPO    — target repo name (overrides --hf_repo argument)
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from huggingface_hub import HfApi, login

# ── Constants ────────────────────────────────────────────────────
BASE_MODEL     = "ProsusAI/finbert"
DATASET_PATH   = "data/supplychain_headlines.csv"
MAX_LEN        = 128        # FinBERT/BERT max is 512; headlines fit in 128
BATCH_SIZE     = 16
EPOCHS         = 4
LEARNING_RATE  = 2e-5       # Standard fine-tuning LR for BERT
WARMUP_RATIO   = 0.1        # 10% of steps used for LR warmup
WEIGHT_DECAY   = 0.01
SEED           = 42
NUM_LABELS     = 3
LABEL_MAP      = {0: "negative", 1: "neutral", 2: "positive"}

# ── Reproducibility ──────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset Class ────────────────────────────────────────────────
class SupplyChainDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized supply-chain headlines.
    Each item returns input_ids, attention_mask, and label tensor.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Layer Freezing ───────────────────────────────────────────────
def freeze_layers(model):
    """
    Transfer Learning Strategy:
      FROZEN  → embeddings + transformer layers 0–9  (10/12 layers)
      TRAINABLE → transformer layers 10–11 + pooler + classifier head

    Rationale:
      Lower layers encode general language understanding (syntax, grammar,
      basic semantics) — already learned from BERT pretraining on 3.3B words.
      Upper layers encode task-specific representations — fine-tuning these
      adapts the model to supply-chain geopolitical language.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 transformer encoder layers (index 10 and 11)
    for i in [10, 11]:
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = True

    # Unfreeze pooler (produces [CLS] representation used by classifier)
    for param in model.bert.pooler.parameters():
        param.requires_grad = True

    # Unfreeze classifier head (Linear 768 → 3)
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Report
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\n{'='*55}")
    print(f"  Layer freezing strategy applied:")
    print(f"  Frozen    : {frozen:,} parameters  ({100*frozen/total:.1f}%)")
    print(f"  Trainable : {trainable:,} parameters  ({100*trainable/total:.1f}%)")
    print(f"{'='*55}\n")
    return model


# ── Training Loop ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, class_weights_tensor=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        # Weighted cross-entropy if class weights provided, else default
        if class_weights_tensor is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        loss.backward()
        # Gradient clipping — prevents exploding gradients during fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device, class_weights_tensor=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            if class_weights_tensor is not None:
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────
def main(hf_repo: str):
    print(f"\n🔧  SupplyGuard AI — FinBERT Fine-Tuning")
    print(f"    Base model  : {BASE_MODEL}")
    print(f"    Target repo : {hf_repo}")
    print(f"    Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load Dataset ──────────────────────────────────────────────
    print("📂  Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python build_dataset.py` first."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"    Loaded {len(df)} samples")
    print(f"    Class distribution:\n{df['label_text'].value_counts().to_string()}\n")

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # 80/10/10 train/val/test split, stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

    # ── Class weights ─────────────────────────────────────────────
    # Handles remaining imbalance after undersampling (45.7% pos / 31.4% neu / 22.9% neg)
    # Weighted cross-entropy penalises the model more for misclassifying minority classes
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=y_train,
    )
    print(f"    Class weights (neg / neu / pos): {class_weights_np.round(3)}")

    # ── Tokenizer ─────────────────────────────────────────────────
    print("🔤  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = SupplyChainDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = SupplyChainDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_ds  = SupplyChainDataset(X_test,  y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Convert class weights to tensor on the correct device
    # (device is defined after this block — we store as numpy, convert later)
    _class_weights_np = class_weights_np  # keep reference for conversion after device init

    # ── Model ─────────────────────────────────────────────────────
    print("🤖  Loading FinBERT base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=LABEL_MAP,
        label2id={v: k for k, v in LABEL_MAP.items()},
        ignore_mismatched_sizes=True,  # Replace FinBERT's head with fresh 3-class head
    )
    model = freeze_layers(model)
    model = model.to(device)

    # Convert class weights to tensor on the correct device
    class_weights_tensor = torch.tensor(_class_weights_np, dtype=torch.float).to(device)
    print(f"    Class weights tensor: {class_weights_tensor.cpu().numpy().round(3)}\n")

    # ── Optimizer & Scheduler ─────────────────────────────────────
    # Only pass trainable params to optimizer — frozen params skipped
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"    Total training steps : {total_steps}")
    print(f"    Warmup steps         : {warmup_steps}")
    print(f"    Optimizer            : AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")

    # ── Training ──────────────────────────────────────────────────
    best_val_acc = 0.0
    best_model_path = "data/best_model_state.pt"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights_tensor)
        val_loss,   val_acc, _, _ = eval_epoch(model, val_loader, device, class_weights_tensor)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}/{EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"{elapsed:.0f}s")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✅ New best val accuracy: {best_val_acc:.4f} — checkpoint saved")

    # ── Test Evaluation ───────────────────────────────────────────
    print(f"\n📊  Loading best checkpoint (val_acc={best_val_acc:.4f}) for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_acc, preds, true_labels = eval_epoch(model, test_loader, device, class_weights_tensor)

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS")
    print(f"{'='*55}")
    print(f"  Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(
        true_labels, preds,
        target_names=["negative", "neutral", "positive"]
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))
    print(f"{'='*55}\n")

    # ── Save Model + Tokenizer ────────────────────────────────────
    local_save_path = "data/supplychain_finbert"
    os.makedirs(local_save_path, exist_ok=True)
    print(f"💾  Saving model + tokenizer locally → {local_save_path}")
    model.save_pretrained(local_save_path)
    tokenizer.save_pretrained(local_save_path)

    # Save fine-tuning metadata
    with open(os.path.join(local_save_path, "training_info.txt"), "w") as f:
        f.write(f"Base model:        {BASE_MODEL}\n")
        f.write(f"Task:              Supply chain sentiment (negative/neutral/positive)\n")
        f.write(f"Dataset size:      {len(df)} samples (~99% real, ~1% synthetic supplement)\n")
        f.write(f"Frozen layers:     BERT embeddings + encoder layers 0-9\n")
        f.write(f"Trainable layers:  Encoder layers 10-11 + pooler + classifier head\n")
        f.write(f"Best val accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test accuracy:     {test_acc:.4f}\n")
        f.write(f"Epochs:            {EPOCHS}\n")
        f.write(f"Learning rate:     {LEARNING_RATE}\n")
        f.write(f"Batch size:        {BATCH_SIZE}\n")
        f.write(f"Class weights:     neg={_class_weights_np[0]:.3f}, neu={_class_weights_np[1]:.3f}, pos={_class_weights_np[2]:.3f}\n")

    # ── Push to HuggingFace Hub ────────────────────────────────────
    print(f"\n🚀  Pushing to HuggingFace Hub → {hf_repo}")

    # Login using token from env or existing cache
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Create repo if it doesn't exist and push
    api = HfApi()
    api.create_repo(repo_id=hf_repo, exist_ok=True, private=False)

    model.push_to_hub(hf_repo, commit_message="Fine-tuned FinBERT on supply chain headlines")
    tokenizer.push_to_hub(hf_repo, commit_message="Tokenizer for supply chain FinBERT")

    # Push training info
    api.upload_file(
        path_or_fileobj=os.path.join(local_save_path, "training_info.txt"),
        path_in_repo="training_info.txt",
        repo_id=hf_repo,
        commit_message="Add training metadata",
    )

    # ── Push model card ────────────────────────────────────────────
    readme = f"""---
language: en
license: apache-2.0
tags:
  - text-classification
  - sentiment-analysis
  - supply-chain
  - geopolitical-risk
  - finbert
  - bert
  - transfer-learning
  - fine-tuning
datasets:
  - FinGPT/fingpt-sentiment-train (merged financial sentiment, includes PhraseBank)
  - zeroshot/twitter-financial-news-sentiment
  - Synthetic supply-chain geopolitical supplement (~70 headlines)
metrics:
  - accuracy
  - f1
---

# supplychain-finbert

Fine-tuned [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for
**supply chain geopolitical risk sentiment analysis**.

Built for [SupplyGuard AI](https://github.com/arunabhachanda/supplyguard-ai) —
a production-grade supply chain risk intelligence platform.

## Model Details

| Property | Value |
|---|---|
| Base model | ProsusAI/finbert (BERT-base fine-tuned on Reuters/Bloomberg) |
| Task | 3-class sentiment: negative / neutral / positive |
| Fine-tuning strategy | Frozen layers 0-9, trainable layers 10-11 + pooler + head |
| Training data | ~40,600 samples (FinGPT financial sentiment + Twitter Financial News + ~70 synthetic geopolitical supplement) |
| Test accuracy | {test_acc:.4f} |
| Best val accuracy | {best_val_acc:.4f} |

## Labels

| ID | Label | Meaning |
|---|---|---|
| 0 | negative | Risk increasing — conflict, sanctions, disaster, supplier failure |
| 1 | neutral | Routine updates, mixed signals, uncertainty |
| 2 | positive | Risk decreasing — stability, trade agreements, recovery |

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="{hf_repo}",
    return_all_scores=True,
)

result = classifier("Ceasefire in the region reopens key supply corridors")
# → [{{'label': 'negative', 'score': 0.04}},
#    {{'label': 'neutral',  'score': 0.11}},
#    {{'label': 'positive', 'score': 0.85}}]
```

## Transfer Learning Architecture

```
ProsusAI/finbert (pre-trained on financial news)
├── BERT Embeddings          [FROZEN]
├── Transformer Layer 0-9    [FROZEN]     ← general language + financial knowledge
├── Transformer Layer 10-11  [TRAINABLE]  ← adapted to supply-chain language
├── Pooler                   [TRAINABLE]
└── Classifier Head (768→3)  [TRAINABLE]  ← new head for 3-class sentiment
```

## Built By

Arunabha Kumar Chanda — M.Sc. Business Intelligence & Data Science, ISM Munich
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=hf_repo,
        commit_message="Add model card",
    )

    print(f"\n{'='*55}")
    print(f"  ✅ Fine-tuning complete!")
    print(f"  Model live at: https://huggingface.co/{hf_repo}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"{'='*55}\n")

    return hf_repo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on supply chain headlines")
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=os.environ.get("HF_REPO", "YOUR_HF_USERNAME/supplychain-finbert"),
        help="HuggingFace Hub repo ID (e.g. arunabhachanda/supplychain-finbert)",
    )
    args = parser.parse_args()

    if args.hf_repo == "YOUR_HF_USERNAME/supplychain-finbert":
        print("⚠️  Please set --hf_repo to your HuggingFace username/repo")
        print("   Example: python finetune_finbert.py --hf_repo arunabhachanda/supplychain-finbert")
        exit(1)

    main(args.hf_repo)
