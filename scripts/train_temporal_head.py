#!/usr/bin/env python3
"""
scripts/train_temporal_head.py
==============================
Trains the SOTA Gated Residual Semantic Transition Head for Bi-Temporal Change Detection.
Strictly aligned with SIH-2026-PS26227 and DynamicEarthNet LULC transitions.

Classes (7 Semantic Transition Categories):
  [0] Water Reclamation / Coastal Infrastructure
  [1] Urban / Infrastructure Expansion
  [2] Land Inundation / Flooding / Submergence
  [3] Afforestation / Revegetation / Greening
  [4] Deforestation / Demolition / Land Clearing
  [5] Cryospheric Dynamics (Snow & Ice Shift)
  [6] Wetland & Ecological Succession
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from backend.services.temporal_head import (
    TemporalChangeHead,
    SEMANTIC_TRANSITION_CLASSES,
    expand_spectral_features
)


def get_semantic_transition(c1: int, c2: int) -> int:
    """
    Maps (from_class, to_class) LULC pair to 1 of 7 real-world transition categories.
    LULC indices:
      0: Soil, 1: Forest, 2: Water, 3: Agriculture, 4: Built-up, 5: Snow/Ice, 6: Wetland
    """
    if c1 == c2:
        return -1  # Stable / No Change (handled by Stage 1 z-score filter)
    if c1 == 5 or c2 == 5:
        return 5  # Cryospheric Dynamics (Snow & Ice Shift)
    if c1 == 2 and c2 == 4:
        return 0  # Water Reclamation / Coastal Infrastructure
    if c2 == 4:
        return 1  # Urban / Infrastructure Expansion (Soil/Veg/Wetland/Agri -> Built-up)
    if c2 == 2:
        return 2  # Land Inundation / Flooding / Submergence (Any -> Water)
    if c2 in [1, 3]:
        return 3  # Afforestation / Revegetation / Greening (Built-up/Water -> Forest/Agri)
    if c2 == 0:
        return 4  # Deforestation / Demolition / Land Clearing (Built-up/Water -> Soil)
    return 6      # Wetland & Ecological Succession


def compute_detailed_metrics(preds_arr: np.ndarray, targets_arr: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """Computes per-class precision, recall, and Macro-F1."""
    n_cls = len(classes)
    per_class = {}
    f1_list, prec_list, rec_list = [], [], []

    for c in range(n_cls):
        tp = int(((preds_arr == c) & (targets_arr == c)).sum())
        fp = int(((preds_arr == c) & (targets_arr != c)).sum())
        fn = int(((preds_arr != c) & (targets_arr == c)).sum())
        count = int((targets_arr == c).sum())

        prec = tp / max(tp + fp, 1e-6)
        rec = tp / max(tp + fn, 1e-6)
        f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)

        per_class[classes[c]] = {
            "count": count,
            "precision": float(round(prec * 100, 2)),
            "recall": float(round(rec * 100, 2)),
            "f1": float(round(f1 * 100, 2))
        }

        if count > 0:
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)

    total_acc = float((preds_arr == targets_arr).mean() * 100)

    return {
        "accuracy": round(total_acc, 2),
        "macro_precision": round(float(np.mean(prec_list)) * 100, 2),
        "macro_recall": round(float(np.mean(rec_list)) * 100, 2),
        "macro_f1": round(float(np.mean(f1_list)) * 100, 2),
        "per_class": per_class
    }


def main():
    parser = argparse.ArgumentParser(description="Train SOTA Gated Residual Semantic Transition Head.")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Directory containing cached .npz files.")
    parser.add_argument("--output-dir", type=str, default="models/change_head", help="Directory to save trained model.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 75, flush=True)
    print("[TRAIN] SOTA GATED RESIDUAL SEMANTIC TRANSITION HEAD (SIH-2026-PS26227)", flush=True)
    print("=" * 75, flush=True)
    print(f"Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"Cache Dir:      {args.cache_dir}", flush=True)
    print(f"Output Dir:     {args.output_dir}", flush=True)
    print(f"Epochs:         {args.epochs}", flush=True)
    print(f"Batch Size:     {args.batch_size}", flush=True)
    print(f"Learning Rate:  {args.lr}", flush=True)
    print("=" * 75 + "\n", flush=True)

    # 1. Load Arrays and Combine All Changes Across Dataset
    train_file = os.path.join(args.cache_dir, "dynamicearthnet_train.npz")
    part1_file = os.path.join(args.cache_dir, "dynamicearthnet_train_part1.npz")
    part2_file = os.path.join(args.cache_dir, "dynamicearthnet_train_part2.npz")
    val_file = os.path.join(args.cache_dir, "dynamicearthnet_val.npz")

    if os.path.exists(train_file):
        d_tr = np.load(train_file)
        X_tr = d_tr["X"]
        f_tr = d_tr["from_class"]
        t_tr = d_tr["to_class"]
    elif os.path.exists(part1_file) and os.path.exists(part2_file):
        print("Loading training cache from split parts (part1 + part2)...", flush=True)
        d1 = np.load(part1_file)
        d2 = np.load(part2_file)
        X_tr = np.concatenate([d1["X"], d2["X"]], axis=0)
        f_tr = np.concatenate([d1["from_class"], d2["from_class"]], axis=0)
        t_tr = np.concatenate([d1["to_class"], d2["to_class"]], axis=0)
    else:
        raise FileNotFoundError(f"Training cache not found in {args.cache_dir}")

    d_v = np.load(val_file)
    X_all = np.concatenate([X_tr, d_v["X"]], axis=0)
    f_all = np.concatenate([f_tr, d_v["from_class"]], axis=0)
    t_all = np.concatenate([t_tr, d_v["to_class"]], axis=0)


    y_all = np.array([get_semantic_transition(c1, c2) for c1, c2 in zip(f_all, t_all)], dtype=np.int64)

    # Filter to real changed patches (transition >= 0)
    change_mask = (y_all >= 0)
    X_ch = X_all[change_mask]
    y_ch = y_all[change_mask]

    print(f"Total Cached Patches:     {len(X_all):,}")
    print(f"Total Change Candidates:  {len(X_ch):,} patches\n", flush=True)

    classes = SEMANTIC_TRANSITION_CLASSES
    num_classes = len(classes)

    print(f"Semantic Classes ({num_classes}):", flush=True)
    for idx, c in enumerate(classes):
        cnt = int((y_ch == idx).sum())
        print(f"  [{idx}] {c:<45} : {cnt:>5} patches ({cnt/len(y_ch)*100:5.1f}%)", flush=True)
    print(flush=True)

    # 2. Expand Spectral Biophysical Features (2051-D -> 2059-D)
    print("Applying Biophysical Non-Linear Spectral Expansion (2051-D -> 2059-D)...", flush=True)
    X_exp = expand_spectral_features(X_ch)
    in_dim = X_exp.shape[1]

    # 3. Stratified 80/20 Train/Val Split
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_exp, y_ch, test_size=0.20, stratify=y_ch, random_state=42
    )

    print(f"Stratified Dataset Split:", flush=True)
    print(f"  Train Set: {len(X_tr_np):>5} samples (80.0%)", flush=True)
    print(f"  Val Set:   {len(X_val_np):>5} samples (20.0%)\n", flush=True)

    # 4. Balanced Sampler for Fair Gradient Updates Across Rare Classes
    class_counts = np.bincount(y_tr_np, minlength=num_classes)
    sample_weights = np.array([1.0 / np.sqrt(max(class_counts[c], 1)) for c in y_tr_np], dtype=np.float64)
    sample_weights = sample_weights / sample_weights.sum()
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(y_tr_np), replacement=True)

    train_ds = TensorDataset(torch.from_numpy(X_tr_np).float(), torch.from_numpy(y_tr_np).long())
    val_ds = TensorDataset(torch.from_numpy(X_val_np).float(), torch.from_numpy(y_val_np).long())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 5. Initialize SOTA Head
    model = TemporalChangeHead(in_dim=in_dim, num_classes=num_classes, dropout=0.25).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 75, flush=True)
    print(f"Architecture:   SOTA Gated Residual Head (LayerNorm + GELU + ResBlocks)", flush=True)
    print(f"Input Feature:  {in_dim} D (2048 Foundation Latent + 11 Biophysical Interaction)", flush=True)
    print(f"Model Params:   {total_params:,} total ({total_params * 4 / (1024*1024):.2f} MB float32)", flush=True)
    print("=" * 75 + "\n", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_f1 = 0.0
    best_epoch = 0
    best_acc = 0.0
    best_metrics = {}
    best_ckpt_path = os.path.join(args.output_dir, "temporal_change_head.pth")

    print("[TRAIN] Commencing training across epochs (Checkpointing on Peak Validation Macro-F1)...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Feature augmentation jitter
            bx_aug = bx + torch.randn_like(bx) * 0.008

            optimizer.zero_grad()
            logits = model(bx_aug)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += len(by)

        scheduler.step()
        train_loss /= max(1, train_total)
        train_acc = (train_correct / max(1, train_total)) * 100.0

        # Periodic Validation Check
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    logits = model(bx)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(by.numpy())

            preds_arr = np.array(all_preds)
            targets_arr = np.array(all_targets)
            metrics = compute_detailed_metrics(preds_arr, targets_arr, classes)

            val_acc = metrics["accuracy"]
            val_f1 = metrics["macro_f1"]
            val_prec = metrics["macro_precision"]
            val_rec = metrics["macro_recall"]

            is_best = val_f1 > best_f1
            mark = " [*BEST*]" if is_best else ""

            if is_best:
                best_f1 = val_f1
                best_epoch = epoch
                best_acc = val_acc
                best_metrics = metrics

                # Save best checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "in_dim": in_dim,
                    "num_classes": num_classes,
                    "classes": classes,
                    "metrics": metrics
                }, best_ckpt_path)

            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] - "
                f"Train Loss: {train_loss:.4f} (Acc: {train_acc:5.1f}%) | "
                f"Val Acc: {val_acc:5.1f}%, Prec: {val_prec:5.1f}%, Rec: {val_rec:5.1f}%, Macro-F1: {val_f1:5.1f}%"
                f"{mark}",
                flush=True
            )

    # 6. Save Classes Metadata JSON
    meta_json_path = os.path.join(args.output_dir, "classes.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "in_dim": in_dim,
            "best_epoch": best_epoch,
            "overall_accuracy": best_acc,
            "macro_f1": best_f1,
            "metrics": best_metrics
        }, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"[FINAL EVALUATION] PEAK VALIDATION PERFORMANCE AT EPOCH {best_epoch:03d}", flush=True)
    print("=" * 80, flush=True)
    print(f"Overall Accuracy:  {best_acc:.2f}%", flush=True)
    print(f"Macro Precision:   {best_metrics.get('macro_precision', 0):.2f}%", flush=True)
    print(f"Macro Recall:      {best_metrics.get('macro_recall', 0):.2f}%", flush=True)
    print(f"Macro F1-Score:    {best_f1:.2f}%\n", flush=True)

    header = f"{'Class Transition Category':<45} | {'Count':>6} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for c_name, c_dict in best_metrics.get("per_class", {}).items():
        print(
            f"{c_name:<45} | {c_dict['count']:>6} | {c_dict['precision']:>8.1f}% | {c_dict['recall']:>8.1f}% | {c_dict['f1']:>8.1f}%",
            flush=True
        )
    print("=" * 80, flush=True)
    print(f"[SUCCESS] Trained model saved to '{best_ckpt_path}'", flush=True)
    print(f"[SUCCESS] Metadata saved to '{meta_json_path}'", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
