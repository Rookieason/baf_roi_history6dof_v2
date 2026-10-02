from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from utility import ConfidenceDatasetBuilder, ConfidenceSample
from model import build_confidence_estimator


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConfidenceTorchDataset(Dataset):
    def __init__(self, samples: List[ConfidenceSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.tensor(s.x_window, dtype=torch.float32)   # [T, 6]
        y = torch.tensor(s.y_label, dtype=torch.float32)    # scalar, 0 or 1
        meta = {
            "file_name": s.file_name,
            "end_row_index": s.end_row_index,
            "time_t": s.time_t,
        }
        return x, y, meta


def create_dataloaders(
    samples: List[ConfidenceSample],
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset]:
    dataset = ConfidenceTorchDataset(samples)

    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def count_labels_in_subset(subset: Dataset) -> Dict[str, int]:
    explorer = 0
    focused = 0
    for i in range(len(subset)):
        _, y, _ = subset[i]
        if int(y.item()) == 1:
            explorer += 1
        else:
            focused += 1
    return {
        "num_total": explorer + focused,
        "num_explorer": explorer,
        "num_focused": focused,
    }


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focused_weight: float = 1.0,
    explorer_weight: float = 1.0,
) -> torch.Tensor:
    """
    Explorer = 1, Focused = 0

    To make the model care more about Focused samples,
    set focused_weight > explorer_weight.
    """
    targets = targets.float()
    base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets == 0.0,
        torch.full_like(targets, focused_weight),
        torch.full_like(targets, explorer_weight),
    )
    return (base_loss * weights).mean()


def confusion_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """
    Positive class = Explorer = 1
    Negative class = Focused = 0
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # Explorer -> Explorer
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # Focused  -> Focused
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # Focused  -> Explorer
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # Explorer -> Focused
    return {
        "tp_explorer": tp,
        "tn_focused": tn,
        "fp_focused_as_explorer": fp,
        "fn_explorer_as_focused": fn,
    }


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_from_predictions(y_true, y_pred)

    tp = cm["tp_explorer"]
    tn = cm["tn_focused"]
    fp = cm["fp_focused_as_explorer"]
    fn = cm["fn_explorer_as_focused"]

    total = tp + tn + fp + fn

    # Explorer class metrics (positive class)
    explorer_precision = safe_div(tp, tp + fp)
    explorer_recall = safe_div(tp, tp + fn)
    explorer_f1 = safe_div(
        2 * explorer_precision * explorer_recall,
        explorer_precision + explorer_recall,
    )

    # Focused class metrics (treat Focused as the class of interest)
    focused_precision = safe_div(tn, tn + fn)
    focused_recall = safe_div(tn, tn + fp)
    focused_f1 = safe_div(
        2 * focused_precision * focused_recall,
        focused_precision + focused_recall,
    )

    accuracy = safe_div(tp + tn, total)
    balanced_accuracy = 0.5 * (explorer_recall + focused_recall)
    macro_f1 = 0.5 * (explorer_f1 + focused_f1)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class": {
            "Explorer": {
                "precision": explorer_precision,
                "recall": explorer_recall,
                "f1": explorer_f1,
                "support": int(np.sum(y_true == 1)),
            },
            "Focused": {
                "precision": focused_precision,
                "recall": focused_recall,
                "f1": focused_f1,
                "support": int(np.sum(y_true == 0)),
            },
        },
    }


def format_cm(cm: Dict[str, int]) -> str:
    return (
        f"TP(Explorer->Explorer)={cm['tp_explorer']}, "
        f"TN(Focused->Focused)={cm['tn_focused']}, "
        f"FP(Focused->Explorer)={cm['fp_focused_as_explorer']}, "
        f"FN(Explorer->Focused)={cm['fn_explorer_as_focused']}"
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    focused_weight: float,
    explorer_weight: float,
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        logits = out["logits"]

        loss = weighted_bce_loss(
            logits=logits,
            targets=y,
            focused_weight=focused_weight,
            explorer_weight=explorer_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def collect_outputs(model, loader, device, focused_weight: float, explorer_weight: float):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        logits = out["logits"]
        probs = out["confidence"]

        loss = weighted_bce_loss(
            logits=logits,
            targets=y,
            focused_weight=focused_weight,
            explorer_weight=explorer_weight,
        )

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.float32)

    avg_loss = total_loss / max(total_samples, 1)

    return avg_loss, y_true.astype(np.int64), y_prob.astype(np.float32)


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> List[Dict[str, object]]:
    results = []
    for th in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=th)
        results.append(metrics)
    return results


def choose_best_threshold(
    sweep_results: List[Dict[str, object]],
    criterion: str = "macro_f1",
) -> Dict[str, object]:
    valid_criteria = {"macro_f1", "balanced_accuracy", "accuracy"}
    if criterion not in valid_criteria:
        raise ValueError(f"criterion must be one of {valid_criteria}")

    best = max(sweep_results, key=lambda x: x[criterion])
    return best


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_glob", type=str, required=True)
    parser.add_argument("--val_glob", type=str, required=True)
    parser.add_argument("--test_glob", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results/save_model")
    parser.add_argument("--json_dir", type=str, default="./results")
    parser.add_argument("--focused_weight", type=float, default=3.0)
    parser.add_argument("--explorer_weight", type=float, default=1.0)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated thresholds for sweep",
    )
    parser.add_argument(
        "--threshold_criterion",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy", "accuracy"],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    # Build dataset from utility.py
    # builder = ConfidenceDatasetBuilder(seq_len=args.seq_len)
    # samples = builder.build_from_glob(args.input_glob)

    # total_explorer = sum(1 for s in samples if s.y_label == 1)
    # total_focused = sum(1 for s in samples if s.y_label == 0)

    # print(f"[INFO] Total confidence samples: {len(samples)}")
    # print(f"[INFO] Explorer samples: {total_explorer}")
    # print(f"[INFO] Focused samples : {total_focused}")
    # print(f"[INFO] Using weighted BCE: focused_weight={args.focused_weight}, explorer_weight={args.explorer_weight}")

    # train_loader, val_loader, test_loader, train_set, val_set, test_set = create_dataloaders(
    #     samples=samples,
    #     batch_size=args.batch_size,
    #     seed=args.seed,
    # )
    builder = ConfidenceDatasetBuilder(seq_len=args.seq_len)

    train_samples = builder.build_from_glob(args.train_glob)
    val_samples = builder.build_from_glob(args.val_glob)
    test_samples = builder.build_from_glob(args.test_glob)

    train_set = ConfidenceTorchDataset(train_samples)
    val_set = ConfidenceTorchDataset(val_samples)
    test_set = ConfidenceTorchDataset(test_samples)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_stats = count_labels_in_subset(train_set)
    val_stats = count_labels_in_subset(val_set)
    test_stats = count_labels_in_subset(test_set)

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches  : {len(val_loader)}")
    print(f"[INFO] Test batches : {len(test_loader)}")

    print(f"[INFO] Train stats: {train_stats}")
    print(f"[INFO] Val stats  : {val_stats}")
    print(f"[INFO] Test stats : {test_stats}")

    model = build_confidence_estimator(
        input_dim=6,
        seq_len=args.seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.save_dir, "confidence_best.pt")

    history = []
    best_val_summary = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            focused_weight=args.focused_weight,
            explorer_weight=args.explorer_weight,
        )

        val_loss, y_true_val, y_prob_val = collect_outputs(
            model=model,
            loader=val_loader,
            device=device,
            focused_weight=args.focused_weight,
            explorer_weight=args.explorer_weight,
        )

        val_metrics_default = compute_binary_metrics(y_true_val, y_prob_val, threshold=0.5)
        val_sweep = threshold_sweep(y_true_val, y_prob_val, thresholds)
        val_best_threshold = choose_best_threshold(val_sweep, criterion=args.threshold_criterion)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_default_threshold_metrics": val_metrics_default,
            "val_threshold_sweep": val_sweep,
            "val_best_threshold_by_criterion": val_best_threshold,
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc@0.5={val_metrics_default['accuracy']:.4f} | "
            f"val_macro_f1@0.5={val_metrics_default['macro_f1']:.4f} | "
            f"best_th={val_best_threshold['threshold']:.2f} | "
            f"best_{args.threshold_criterion}={val_best_threshold[args.threshold_criterion]:.4f}"
        )
        print(f"           val CM @0.5: {format_cm(val_metrics_default['confusion_matrix'])}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_summary = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_default_threshold_metrics": val_metrics_default,
                "val_threshold_sweep": val_sweep,
                "val_best_threshold_by_criterion": val_best_threshold,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "seq_len": args.seq_len,
                    "focused_weight": args.focused_weight,
                    "explorer_weight": args.explorer_weight,
                },
                best_ckpt_path,
            )
            print(f"[INFO] Saved best checkpoint to: {best_ckpt_path}")

    print("[INFO] Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, y_true_test, y_prob_test = collect_outputs(
        model=model,
        loader=test_loader,
        device=device,
        focused_weight=args.focused_weight,
        explorer_weight=args.explorer_weight,
    )

    test_metrics_default = compute_binary_metrics(y_true_test, y_prob_test, threshold=0.5)
    test_sweep = threshold_sweep(y_true_test, y_prob_test, thresholds)
    test_best_threshold = choose_best_threshold(test_sweep, criterion=args.threshold_criterion)

    final_results = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
        },
        "config": {
            # "input_glob": args.input_glob,
            "train_glob": args.train_glob,
            "val_glob": args.val_glob,
            "test_glob": args.test_glob,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "focused_weight": args.focused_weight,
            "explorer_weight": args.explorer_weight,
            "thresholds": thresholds,
            "threshold_criterion": args.threshold_criterion,
        },
        "dataset_summary": {
            # "all_samples": {
            #     "num_total": len(samples),
            #     "num_explorer": total_explorer,
            #     "num_focused": total_focused,
            # },
            "train_split": train_stats,
            "val_split": val_stats,
            "test_split": test_stats,
        },
        "training_history": history,
        "best_validation_result": best_val_summary,
        "final_test_result": {
            "test_loss": test_loss,
            "test_metrics_default_threshold_0_5": test_metrics_default,
            "test_threshold_sweep": test_sweep,
            "test_best_threshold_by_criterion": test_best_threshold,
        },
    }

    results_json_path = os.path.join(args.json_dir, "confidence_training_results.json")
    save_json(final_results, results_json_path)

    print("\n=== Final Test Metrics @ threshold=0.5 ===")
    print(f"test_loss           : {test_loss:.6f}")
    print(f"test_accuracy       : {test_metrics_default['accuracy']:.4f}")
    print(f"test_balanced_acc   : {test_metrics_default['balanced_accuracy']:.4f}")
    print(f"test_macro_f1       : {test_metrics_default['macro_f1']:.4f}")
    print(f"test CM @0.5        : {format_cm(test_metrics_default['confusion_matrix'])}")
    print("test per-class:")
    for cls_name, cls_metrics in test_metrics_default["per_class"].items():
        print(
            f"  {cls_name}: "
            f"precision={cls_metrics['precision']:.4f}, "
            f"recall={cls_metrics['recall']:.4f}, "
            f"f1={cls_metrics['f1']:.4f}, "
            f"support={cls_metrics['support']}"
        )

    print(f"\n=== Best Test Threshold by {args.threshold_criterion} ===")
    print(f"threshold           : {test_best_threshold['threshold']:.2f}")
    print(f"accuracy            : {test_best_threshold['accuracy']:.4f}")
    print(f"balanced_accuracy   : {test_best_threshold['balanced_accuracy']:.4f}")
    print(f"macro_f1            : {test_best_threshold['macro_f1']:.4f}")
    print(f"test CM best_th     : {format_cm(test_best_threshold['confusion_matrix'])}")

    print(f"\n[INFO] JSON results written to: {results_json_path}")


if __name__ == "__main__":
    main()
