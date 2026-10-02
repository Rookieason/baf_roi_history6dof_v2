from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utility import FocusedDatasetBuilder, FocusedSample
from model import (
    FocusedBranchConfig,
    FocusedResidualLSTM,
    focused_total_loss,
    quaternion_geodesic_distance,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocusedTorchDataset(Dataset):
    def __init__(self, samples: List[FocusedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.tensor(s.x_window, dtype=torch.float32)   # [T, 6]
        # y = torch.tensor(s.y_next, dtype=torch.float32)     # [6]
        target_translation = torch.tensor(s.target_translation, dtype=torch.float32)
        target_quaternion = torch.tensor(s.target_quaternion, dtype=torch.float32)
        meta = {
            "file_name": s.file_name,
            "end_row_index": s.end_row_index,
            "target_row_index": s.target_row_index,
            "time_t": s.time_t,
            "time_t1": s.time_t1,
        }
        return x, target_translation, target_quaternion, meta


def create_dataloaders(
    samples: List[FocusedSample],
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset]:
    dataset = FocusedTorchDataset(samples)

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


def circular_abs_diff_deg(pred_angle: torch.Tensor, target_angle: torch.Tensor) -> torch.Tensor:
    diff = pred_angle - target_angle
    diff = (diff + 180.0) % 360.0 - 180.0
    return torch.abs(diff)


def mean_abs_error(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_hat - y_true))


def split_pose_mae(y_hat: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tr_mae = torch.mean(torch.abs(y_hat[:, :3] - y_true[:, :3]))
    rot_mae = torch.mean(circular_abs_diff_deg(y_hat[:, 3:], y_true[:, 3:]))
    return tr_mae, rot_mae


# def compute_losses(
#     out: Dict[str, torch.Tensor],
#     y: torch.Tensor,
#     lambda_smooth: float,
#     lambda_tr: float,
#     lambda_rot: float,
# ) -> Dict[str, torch.Tensor]:
#     return focused_total_loss(
#         delta_hat=out["delta_hat"],
#         current_pose=out["current_pose"],
#         prev_pose=out["prev_pose"],
#         target_pose=y,
#         lambda_smooth=lambda_smooth,
#         lambda_tr=lambda_tr,
#         lambda_rot=lambda_rot,
#         reduction="mean",
#     )
def compute_losses(
    out: Dict[str, torch.Tensor],
    target_translation: torch.Tensor,
    target_quaternion: torch.Tensor,
    lambda_smooth: float,
    lambda_tr: float,
    lambda_rot: float,
) -> Dict[str, torch.Tensor]:
    return focused_total_loss(
        delta_hat_tr=out["delta_hat_tr"],
        current_translation=out["current_translation"],
        prev_translation=out["prev_translation"],
        target_translation=target_translation,
        q_hat_norm=out["q_hat_norm"],
        current_quaternion=out["current_quaternion"],
        target_quaternion=target_quaternion,
        lambda_smooth=lambda_smooth,
        lambda_tr=lambda_tr,
        lambda_rot=lambda_rot,
        reduction="mean",
    )

def train_one_epoch(
    model: FocusedResidualLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_smooth: float,
    lambda_tr: float,
    lambda_rot: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_pred = 0.0
    total_smooth = 0.0
    total_mae = 0.0
    total_tr_mae = 0.0
    total_rot_mae = 0.0
    total_samples = 0

    # for x, y, _ in loader:
    #     x = x.to(device)
    #     y = y.to(device)
    for x, target_translation, target_quaternion, _ in loader:
        x = x.to(device)
        target_translation = target_translation.to(device)
        target_quaternion = target_quaternion.to(device)

        optimizer.zero_grad()

        out = model(x)
        # losses = compute_losses(
        #     out=out,
        #     y=y,
        #     lambda_smooth=lambda_smooth,
        #     lambda_tr=lambda_tr,
        #     lambda_rot=lambda_rot,
        # )
        losses = compute_losses(
            out=out,
            target_translation=target_translation,
            target_quaternion=target_quaternion,
            lambda_smooth=lambda_smooth,
            lambda_tr=lambda_tr,
            lambda_rot=lambda_rot,
        )

        loss = losses["loss_total"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # y_hat = out["y_hat"]
        # mae = mean_abs_error(y_hat, y)
        # tr_mae, rot_mae = split_pose_mae(y_hat, y)
        tr_mae, rot_mae = compute_batch_metrics(
            out=out,
            target_translation=target_translation,
            target_quaternion=target_quaternion,
        )
        mae = tr_mae + rot_mae

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        # total_pred += float(losses["loss_pred"].item()) * batch_size
        total_pred += float((losses["loss_tr"] + losses["loss_rot"]).item()) * batch_size
        total_smooth += float(losses["loss_smooth"].item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_tr_mae += float(tr_mae.item()) * batch_size
        total_rot_mae += float(rot_mae.item()) * batch_size
        total_samples += batch_size

    return {
        "loss_total": total_loss / max(total_samples, 1),
        "loss_pred": total_pred / max(total_samples, 1),
        "loss_smooth": total_smooth / max(total_samples, 1),
        "pose_mae": total_mae / max(total_samples, 1),
        "translation_mae": total_tr_mae / max(total_samples, 1),
        "rotation_mae": total_rot_mae / max(total_samples, 1),
    }

def compute_batch_metrics(
    out: Dict[str, torch.Tensor],
    target_translation: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tr_mae = torch.mean(torch.abs(out["p_hat_tr"] - target_translation))
    rot_geo = quaternion_geodesic_distance(
        out["q_hat_norm"],
        target_quaternion,
    ).mean()
    return tr_mae, rot_geo

@torch.no_grad()
def evaluate(
    model: FocusedResidualLSTM,
    loader: DataLoader,
    device: torch.device,
    lambda_smooth: float,
    lambda_tr: float,
    lambda_rot: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_pred = 0.0
    total_smooth = 0.0
    total_mae = 0.0
    total_tr_mae = 0.0
    total_rot_mae = 0.0
    total_samples = 0

    # for x, y, _ in loader:
    #     x = x.to(device)
    #     y = y.to(device)
    for x, target_translation, target_quaternion, _ in loader:
        x = x.to(device)
        target_translation = target_translation.to(device)
        target_quaternion = target_quaternion.to(device)

        out = model(x)
        # losses = compute_losses(
        #     out=out,
        #     y=y,
        #     lambda_smooth=lambda_smooth,
        #     lambda_tr=lambda_tr,
        #     lambda_rot=lambda_rot,
        # )
        losses = compute_losses(
            out=out,
            target_translation=target_translation,
            target_quaternion=target_quaternion,
            lambda_smooth=lambda_smooth,
            lambda_tr=lambda_tr,
            lambda_rot=lambda_rot,
        )

        # y_hat = out["y_hat"]
        # mae = mean_abs_error(y_hat, y)
        # tr_mae, rot_mae = split_pose_mae(y_hat, y)
        tr_mae, rot_mae = compute_batch_metrics(
            out=out,
            target_translation=target_translation,
            target_quaternion=target_quaternion,
        )
        mae = tr_mae + rot_mae

        batch_size = x.size(0)
        total_loss += float(losses["loss_total"].item()) * batch_size
        # total_pred += float(losses["loss_pred"].item()) * batch_size
        total_pred += float((losses["loss_tr"] + losses["loss_rot"]).item()) * batch_size
        total_smooth += float(losses["loss_smooth"].item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_tr_mae += float(tr_mae.item()) * batch_size
        total_rot_mae += float(rot_mae.item()) * batch_size
        total_samples += batch_size

    return {
        "loss_total": total_loss / max(total_samples, 1),
        "loss_pred": total_pred / max(total_samples, 1),
        "loss_smooth": total_smooth / max(total_samples, 1),
        "pose_mae": total_mae / max(total_samples, 1),
        "translation_mae": total_tr_mae / max(total_samples, 1),
        "rotation_mae": total_rot_mae / max(total_samples, 1),
    }


def count_subset_samples(subset: Dataset) -> int:
    return len(subset)


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
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_tr", type=float, default=1.0)
    parser.add_argument("--lambda_rot", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./results/saved_model")
    parser.add_argument("--json_dir", type=str, default="./results")
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="focused_clean",
        choices=["focused_clean", "all"],
        help="Sample selection mode: focused_clean for Focused->Focused, all for all valid transitions.",
    )

    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for initializing model weights before training.",
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="focused_quat_residual_best.pt",
        help="Checkpoint filename saved under save_dir.",
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default="focused_quat_residual_training_results.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # builder = FocusedDatasetBuilder(seq_len=args.seq_len)
    # samples = builder.build_from_glob(args.input_glob)

    # if len(samples) == 0:
    #     raise RuntimeError("No Focused samples found. Check your input_glob or labeling output.")

    # print(f"[INFO] Total Focused samples: {len(samples)}")

    # train_loader, val_loader, test_loader, train_set, val_set, test_set = create_dataloaders(
    #     samples=samples,
    #     batch_size=args.batch_size,
    #     seed=args.seed,
    # )

    # print(f"[INFO] Train samples: {count_subset_samples(train_set)}")
    # print(f"[INFO] Val samples  : {count_subset_samples(val_set)}")
    # print(f"[INFO] Test samples : {count_subset_samples(test_set)}")
    # print(f"[INFO] Train batches: {len(train_loader)}")
    # print(f"[INFO] Val batches  : {len(val_loader)}")
    # print(f"[INFO] Test batches : {len(test_loader)}")

    # builder = FocusedDatasetBuilder(seq_len=args.seq_len)
    builder = FocusedDatasetBuilder(
        seq_len=args.seq_len,
        sample_mode=args.sample_mode,
    )

    train_samples = builder.build_from_glob(args.train_glob)
    val_samples = builder.build_from_glob(args.val_glob)

    if len(train_samples) == 0:
        raise RuntimeError("No Focused training samples found. Check --train_glob or labeling output.")
    if len(val_samples) == 0:
        raise RuntimeError("No Focused validation samples found. Check --val_glob or labeling output.")

    train_set = FocusedTorchDataset(train_samples)
    val_set = FocusedTorchDataset(val_samples)

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

    print(f"[INFO] Train Focused samples: {len(train_set)}")
    print(f"[INFO] Val Focused samples  : {len(val_set)}")
    print(f"[INFO] Train batches        : {len(train_loader)}")
    print(f"[INFO] Val batches          : {len(val_loader)}")

    model = FocusedResidualLSTM(
        FocusedBranchConfig(
            input_dim=7,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_layer_norm=True,
        )
    ).to(device)

    if args.init_checkpoint is not None:
        print(f"[INFO] Loading initial checkpoint from: {args.init_checkpoint}")
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)

        missing, unexpected = model.load_state_dict(
            init_ckpt["model_state_dict"],
            strict=False,
        )

        print(f"[INFO] Loaded initial checkpoint.")
        print(f"[INFO] Missing keys   : {missing}")
        print(f"[INFO] Unexpected keys: {unexpected}")

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
    # best_ckpt_path = os.path.join(args.save_dir, "focused_best.pt")
    # best_ckpt_path = os.path.join(args.save_dir, "focused_quat_residual_best.pt")
    best_ckpt_path = os.path.join(args.save_dir, args.checkpoint_name)
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_smooth=args.lambda_smooth,
            lambda_tr=args.lambda_tr,
            lambda_rot=args.lambda_rot,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_smooth=args.lambda_smooth,
            lambda_tr=args.lambda_tr,
            lambda_rot=args.lambda_rot,
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
            f"train_pred={train_metrics['loss_pred']:.6f} | "
            f"train_smooth={train_metrics['loss_smooth']:.6f} | "
            f"val_total={val_metrics['loss_total']:.6f} | "
            f"val_tr_mae={val_metrics['translation_mae']:.6f} | "
            # f"val_rot_cmae={val_metrics['rotation_mae']:.6f} | "
            f"val_rot_geo_rad={val_metrics['rotation_mae']:.6f} | "
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
    #     lambda_smooth=args.lambda_smooth,
    #     lambda_tr=args.lambda_tr,
    #     lambda_rot=args.lambda_rot,
    # )

    # print("\n=== Final Test Metrics ===")
    # print(f"test_loss_total      : {test_metrics['loss_total']:.6f}")
    # print(f"test_loss_pred       : {test_metrics['loss_pred']:.6f}")
    # print(f"test_loss_smooth     : {test_metrics['loss_smooth']:.6f}")
    # print(f"test_pose_mae        : {test_metrics['pose_mae']:.6f}")
    # print(f"test_translation_mae : {test_metrics['translation_mae']:.6f}")
    # print(f"test_rotation_cmae   : {test_metrics['rotation_mae']:.6f}")

    # result_obj = {
    #     "config": vars(args),
    #     "dataset_summary": {
    #         "num_total_focused_samples": len(samples),
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

    # result_path = os.path.join(args.json_dir, "focused_training_results.json")
    # result_path = os.path.join(args.json_dir, "focused_quat_residual_training_results.json")
    result_path = os.path.join(args.json_dir, args.result_name)
    save_json(result_obj, result_path)
    print(f"[INFO] JSON results written to: {result_path}")


if __name__ == "__main__":
    main()
