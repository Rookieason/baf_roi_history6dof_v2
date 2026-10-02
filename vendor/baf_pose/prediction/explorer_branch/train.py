from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utility import ExplorerDatasetBuilder, ExplorerSample
from model import build_explorer_branch, explorer_total_loss, quaternion_geodesic_distance


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExplorerTorchDataset(Dataset):
    def __init__(self, samples: List[ExplorerSample], use_aux: bool = False) -> None:
        self.samples = samples
        self.use_aux = use_aux

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        pose_x = torch.tensor(s.pose_x, dtype=torch.float32)                    # [T, 7]
        aux_x = torch.tensor(s.aux_x, dtype=torch.float32)                      # [4]
        target_translation = torch.tensor(s.target_translation, dtype=torch.float32)  # [3]
        target_quaternion = torch.tensor(s.target_quaternion, dtype=torch.float32)    # [4]
        current_quaternion = torch.tensor(s.current_quaternion, dtype=torch.float32)  # [4]
        prev_quaternion = torch.tensor(s.prev_quaternion, dtype=torch.float32)        # [4]

        meta = {
            "file_name": s.file_name,
            "end_row_index": s.end_row_index,
            "target_row_index": s.target_row_index,
            "time_t": s.time_t,
            "time_t1": s.time_t1,
        }

        return {
            "pose_x": pose_x,
            "aux_x": aux_x if self.use_aux else None,
            "target_translation": target_translation,
            "target_quaternion": target_quaternion,
            "current_quaternion": current_quaternion,
            "prev_quaternion": prev_quaternion,
            "meta": meta,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    pose_x = torch.stack([b["pose_x"] for b in batch], dim=0)

    use_aux = batch[0]["aux_x"] is not None
    aux_x = torch.stack([b["aux_x"] for b in batch], dim=0) if use_aux else None

    target_translation = torch.stack([b["target_translation"] for b in batch], dim=0)
    target_quaternion = torch.stack([b["target_quaternion"] for b in batch], dim=0)
    current_quaternion = torch.stack([b["current_quaternion"] for b in batch], dim=0)
    prev_quaternion = torch.stack([b["prev_quaternion"] for b in batch], dim=0)

    metas = [b["meta"] for b in batch]

    return {
        "pose_x": pose_x,
        "aux_x": aux_x,
        "target_translation": target_translation,
        "target_quaternion": target_quaternion,
        "current_quaternion": current_quaternion,
        "prev_quaternion": prev_quaternion,
        "meta": metas,
    }


def create_dataloaders(
    samples: List[ExplorerSample],
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_aux: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset]:
    dataset = ExplorerTorchDataset(samples, use_aux=use_aux)

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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def count_subset_samples(subset: Dataset) -> int:
    return len(subset)


def compute_metrics(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    gamma_rot: float,
    eta_tail: float,
) -> Dict[str, torch.Tensor]:
    losses = explorer_total_loss(
        delta_hat_tr=out["delta_hat_tr"],
        current_translation=out["current_translation"],
        target_translation=batch["target_translation"],
        q_hat_norm=out["q_hat_norm"],
        target_quaternion=batch["target_quaternion"],
        current_quaternion=batch["current_quaternion"],
        prev_quaternion=batch["prev_quaternion"],
        gamma_rot=gamma_rot,
        eta_tail=eta_tail,
        reduction="mean",
    )

    tr_mae = torch.mean(torch.abs(out["p_hat_tr"] - batch["target_translation"]))
    rot_geo = quaternion_geodesic_distance(out["q_hat_norm"], batch["target_quaternion"]).mean()

    return {
        "loss_total": losses["loss_total"],
        "loss_tr": losses["loss_tr"],
        "loss_rot": losses["loss_rot"],
        "tail_weight": losses["tail_weight"],
        "translation_mae": tr_mae,
        "rotation_geo": rot_geo,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    gamma_rot: float,
    eta_tail: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_tr_loss = 0.0
    total_rot_loss = 0.0
    total_tail = 0.0
    total_tr_mae = 0.0
    total_rot_geo = 0.0
    total_samples = 0

    for batch in loader:
        pose_x = batch["pose_x"].to(device)
        aux_x = batch["aux_x"].to(device) if batch["aux_x"] is not None else None
        target_translation = batch["target_translation"].to(device)
        target_quaternion = batch["target_quaternion"].to(device)
        current_quaternion = batch["current_quaternion"].to(device)
        prev_quaternion = batch["prev_quaternion"].to(device)

        batch_dev = {
            "target_translation": target_translation,
            "target_quaternion": target_quaternion,
            "current_quaternion": current_quaternion,
            "prev_quaternion": prev_quaternion,
        }

        optimizer.zero_grad()

        out = model(pose_x, aux_x)
        metrics = compute_metrics(out, batch_dev, gamma_rot=gamma_rot, eta_tail=eta_tail)

        loss = metrics["loss_total"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        bs = pose_x.size(0)
        total_loss += float(metrics["loss_total"].item()) * bs
        total_tr_loss += float(metrics["loss_tr"].item()) * bs
        total_rot_loss += float(metrics["loss_rot"].item()) * bs
        total_tail += float(metrics["tail_weight"].item()) * bs
        total_tr_mae += float(metrics["translation_mae"].item()) * bs
        total_rot_geo += float(metrics["rotation_geo"].item()) * bs
        total_samples += bs

    return {
        "loss_total": total_loss / max(total_samples, 1),
        "loss_tr": total_tr_loss / max(total_samples, 1),
        "loss_rot": total_rot_loss / max(total_samples, 1),
        "tail_weight": total_tail / max(total_samples, 1),
        "translation_mae": total_tr_mae / max(total_samples, 1),
        "rotation_geo": total_rot_geo / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    gamma_rot: float,
    eta_tail: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_tr_loss = 0.0
    total_rot_loss = 0.0
    total_tail = 0.0
    total_tr_mae = 0.0
    total_rot_geo = 0.0
    total_samples = 0

    for batch in loader:
        pose_x = batch["pose_x"].to(device)
        aux_x = batch["aux_x"].to(device) if batch["aux_x"] is not None else None
        target_translation = batch["target_translation"].to(device)
        target_quaternion = batch["target_quaternion"].to(device)
        current_quaternion = batch["current_quaternion"].to(device)
        prev_quaternion = batch["prev_quaternion"].to(device)

        batch_dev = {
            "target_translation": target_translation,
            "target_quaternion": target_quaternion,
            "current_quaternion": current_quaternion,
            "prev_quaternion": prev_quaternion,
        }

        out = model(pose_x, aux_x)
        metrics = compute_metrics(out, batch_dev, gamma_rot=gamma_rot, eta_tail=eta_tail)

        bs = pose_x.size(0)
        total_loss += float(metrics["loss_total"].item()) * bs
        total_tr_loss += float(metrics["loss_tr"].item()) * bs
        total_rot_loss += float(metrics["loss_rot"].item()) * bs
        total_tail += float(metrics["tail_weight"].item()) * bs
        total_tr_mae += float(metrics["translation_mae"].item()) * bs
        total_rot_geo += float(metrics["rotation_geo"].item()) * bs
        total_samples += bs

    return {
        "loss_total": total_loss / max(total_samples, 1),
        "loss_tr": total_tr_loss / max(total_samples, 1),
        "loss_rot": total_rot_loss / max(total_samples, 1),
        "tail_weight": total_tail / max(total_samples, 1),
        "translation_mae": total_tr_mae / max(total_samples, 1),
        "rotation_geo": total_rot_geo / max(total_samples, 1),
    }


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--train_glob", type=str, required=True)
    parser.add_argument("--val_glob", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_tcn_blocks", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--gamma_rot", type=float, default=0.3)
    parser.add_argument("--eta_tail", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./results/saved_model")
    parser.add_argument("--json_dir", type=str, default="./results")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # builder = ExplorerDatasetBuilder(seq_len=args.seq_len)
    # samples = builder.build_from_glob(args.input_glob)

    # if len(samples) == 0:
    #     raise RuntimeError("No Explorer samples found. Check input_glob or labels.")

    # print(f"[INFO] Total Explorer samples: {len(samples)}")
    # print(f"[INFO] use_aux: {args.use_aux}")

    # train_loader, val_loader, test_loader, train_set, val_set, test_set = create_dataloaders(
    #     samples=samples,
    #     batch_size=args.batch_size,
    #     seed=args.seed,
    #     use_aux=args.use_aux,
    # )

    # print(f"[INFO] Train samples: {count_subset_samples(train_set)}")
    # print(f"[INFO] Val samples  : {count_subset_samples(val_set)}")
    # print(f"[INFO] Test samples : {count_subset_samples(test_set)}")
    # print(f"[INFO] Train batches: {len(train_loader)}")
    # print(f"[INFO] Val batches  : {len(val_loader)}")
    # print(f"[INFO] Test batches : {len(test_loader)}")

    builder = ExplorerDatasetBuilder(seq_len=args.seq_len)

    train_samples = builder.build_from_glob(args.train_glob)
    val_samples = builder.build_from_glob(args.val_glob)

    if len(train_samples) == 0:
        raise RuntimeError("No Explorer training samples found. Check --train_glob or labels.")
    if len(val_samples) == 0:
        raise RuntimeError("No Explorer validation samples found. Check --val_glob or labels.")

    print(f"[INFO] Train Explorer samples: {len(train_samples)}")
    print(f"[INFO] Val Explorer samples  : {len(val_samples)}")
    print(f"[INFO] use_aux: {args.use_aux}")

    train_set = ExplorerTorchDataset(train_samples, use_aux=args.use_aux)
    val_set = ExplorerTorchDataset(val_samples, use_aux=args.use_aux)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches  : {len(val_loader)}")

    model = build_explorer_branch(
        pose_dim=7,
        aux_dim=4 if args.use_aux else 0,
        hidden_dim=args.hidden_dim,
        num_tcn_blocks=args.num_tcn_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        gamma_rot=args.gamma_rot,
        eta_tail=args.eta_tail,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    # best_ckpt_path = os.path.join(args.save_dir, "explorer_best.pt")
    best_ckpt_path = os.path.join(args.save_dir, "explorer_residual_rot_best.pt")
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            gamma_rot=args.gamma_rot,
            eta_tail=args.eta_tail,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            gamma_rot=args.gamma_rot,
            eta_tail=args.eta_tail,
        )

        scheduler.step(val_metrics["loss_total"])

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_total={train_metrics['loss_total']:.6f} | "
            f"train_tr={train_metrics['loss_tr']:.6f} | "
            f"train_rot={train_metrics['loss_rot']:.6f} | "
            f"val_total={val_metrics['loss_total']:.6f} | "
            f"val_tr_mae={val_metrics['translation_mae']:.6f} | "
            f"val_rot_geo={val_metrics['rotation_geo']:.6f} | "
            f"tail_w={val_metrics['tail_weight']:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                    "rotation_mode": "relative_quaternion_residual",
                },
                best_ckpt_path,
            )
            print(f"[INFO] Saved best checkpoint to: {best_ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # print("[INFO] Loading best checkpoint for final test evaluation...")
    # ckpt = torch.load(best_ckpt_path, map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"])

    # test_metrics = evaluate(
    #     model=model,
    #     loader=test_loader,
    #     device=device,
    #     gamma_rot=args.gamma_rot,
    #     eta_tail=args.eta_tail,
    # )

    # print("\n=== Final Test Metrics ===")
    # print(f"test_loss_total    : {test_metrics['loss_total']:.6f}")
    # print(f"test_loss_tr       : {test_metrics['loss_tr']:.6f}")
    # print(f"test_loss_rot      : {test_metrics['loss_rot']:.6f}")
    # print(f"test_tail_weight   : {test_metrics['tail_weight']:.6f}")
    # print(f"test_translation_mae: {test_metrics['translation_mae']:.6f}")
    # print(f"test_rotation_geo  : {test_metrics['rotation_geo']:.6f}")

    # result_obj = {
    #     "config": vars(args),
    #     "dataset_summary": {
    #         "num_total_explorer_samples": len(samples),
    #         "train_samples": count_subset_samples(train_set),
    #         "val_samples": count_subset_samples(val_set),
    #         "test_samples": count_subset_samples(test_set),
    #     },
    #     "best_val_loss": best_val_loss,
    #     "training_history": history,
    #     "final_test_metrics": test_metrics,
    # }

    result_obj = {
        "config": vars(args),
        "dataset_summary": {
            "train_samples": len(train_set),
            "val_samples": len(val_set),
        },
        "best_val_loss": best_val_loss,
        "training_history": history,
    }

    # result_path = os.path.join(args.json_dir, "explorer_training_results.json")
    result_path = os.path.join(args.json_dir, "explorer_residual_rot_training_results.json")
    save_json(result_obj, result_path)
    print(f"[INFO] JSON results written to: {result_path}")


if __name__ == "__main__":
    main()
