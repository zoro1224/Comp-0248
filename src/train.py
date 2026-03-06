import os
import argparse
import time
import csv
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import HandGestureKeyframeDataset, collect_records
from model import TinyUNetMultiTask
from utils import (
    set_seed,
    DiceLoss,
    to_numpy_mask_from_logits,
    mean_iou_hand_bg,
    dice_coeff,
)


def split_records(records, val_ratio: float, seed: int):
    import random
    rng = random.Random(seed)
    recs = records[:]
    rng.shuffle(recs)
    n = len(recs)
    n_val = int(n * val_ratio)
    val = recs[:n_val]
    train = recs[n_val:]
    return train, val


def bbox_from_mask_np(mask: np.ndarray):

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def bbox_iou_np(b1, b2, eps: float = 1e-6):

    if b1 is None or b2 is None:
        return 0.0

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih

    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])

    union = a1 + a2 - inter + eps
    return float((inter + eps) / union)


def det_metrics_from_masks_np(pred_mask: np.ndarray, gt_mask: np.ndarray):
    b_pred = bbox_from_mask_np(pred_mask)
    b_gt = bbox_from_mask_np(gt_mask)
    iou = bbox_iou_np(b_pred, b_gt)
    acc05 = 1.0 if iou >= 0.5 else 0.0
    return {
        "bbox_iou": iou,
        "acc05": acc05,
    }


def cls_accuracy(cls_true, cls_pred):
    correct = sum(int(t == p) for t, p in zip(cls_true, cls_pred))
    return correct / max(1, len(cls_true))


def run_one_epoch(model, loader, device, optimizer=None, bce=None, dice=None, ce=None,
                  cls_w=4.0):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_seg = 0.0
    total_cls = 0.0
    n_batches = 0

    for x, y_mask, y_cls, _ in loader:
        x = x.to(device)
        y_mask = y_mask.to(device)
        y_cls = y_cls.to(device)


        seg_logits, cls_logits = model(x)

        loss_seg = bce(seg_logits, y_mask) + dice(seg_logits, y_mask)
        loss_cls = ce(cls_logits, y_cls)
        loss = loss_seg + cls_w * loss_cls

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_seg += float(loss_seg.item())
        total_cls += float(loss_cls.item())
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "loss_seg": total_seg / max(1, n_batches),
        "loss_cls": total_cls / max(1, n_batches),
        "loss_det": 0.0,   # no independent detection head
    }


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()

    seg_mious = []
    seg_dices = []
    det_ious = []
    det_acc05 = []
    cls_true = []
    cls_pred = []

    for x, y_mask, y_cls, _ in loader:
        x = x.to(device)
        y_mask = y_mask.to(device)


        seg_logits, cls_logits = model(x)


        pred_masks = to_numpy_mask_from_logits(seg_logits, thr=0.5)   # (B,H,W)
        gt_masks = (y_mask.cpu().numpy()[:, 0] > 0.5).astype("uint8")


        pred_c = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()
        gt_c = y_cls.cpu().numpy()

        for i in range(pred_masks.shape[0]):
            pm = pred_masks[i]
            gm = gt_masks[i]


            seg_mious.append(mean_iou_hand_bg(pm, gm))
            seg_dices.append(dice_coeff(pm, gm))


            dm = det_metrics_from_masks_np(pm, gm)
            det_ious.append(dm["bbox_iou"])
            det_acc05.append(dm["acc05"])

        cls_true.extend(gt_c.tolist())
        cls_pred.extend(pred_c.tolist())

    return {
        "seg_mean_iou_hand_bg": float(sum(seg_mious) / max(1, len(seg_mious))),
        "seg_dice": float(sum(seg_dices) / max(1, len(seg_dices))),
        "det_mean_bbox_iou": float(sum(det_ious) / max(1, len(det_ious))),
        "det_acc05": float(sum(det_acc05) / max(1, len(det_acc05))),
        "cls_acc": float(cls_accuracy(cls_true, cls_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to your gesture folder root")
    ap.add_argument("--use_depth", action="store_true",
                    help="Use RGB-D (4 channels). If not set, use RGB only.")
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cls_w", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--weights_dir", type=str, default="weights")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = collect_records(args.data_root)
    if len(records) == 0:
        raise RuntimeError(f"No annotated keyframes found under: {args.data_root}")

    train_recs, val_recs = split_records(records, args.val_ratio, args.seed)

    train_ds = HandGestureKeyframeDataset(
        args.data_root, train_recs,
        use_depth=args.use_depth,
        image_size=(args.image_w, args.image_h)
    )
    val_ds = HandGestureKeyframeDataset(
        args.data_root, val_recs,
        use_depth=args.use_depth,
        image_size=(args.image_w, args.image_h)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    in_ch = 4 if args.use_depth else 3
    model = TinyUNetMultiTask(in_channels=in_ch, num_classes=10, base=64).to(device)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_seg", "train_cls", "train_det",
            "val_mIoU", "val_dice", "val_det_acc05", "val_det_iou", "val_cls_acc"
        ])

    best_score = -1.0
    best_ckpt = os.path.join(args.weights_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = run_one_epoch(
            model, train_loader, device,
            optimizer=optimizer,
            bce=bce, dice=dice, ce=ce,
            cls_w=args.cls_w
        )
        val_stats = eval_metrics(model, val_loader, device)

        score = (
            1.0 * val_stats["seg_mean_iou_hand_bg"] +
            0.5 * val_stats["det_acc05"] +
            1.0 * val_stats["cls_acc"]
        )
        scheduler.step(score)

        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": in_ch,
                "use_depth": args.use_depth,
                "image_w": args.image_w,
                "image_h": args.image_h,
                "epoch": epoch,
                "best_score": best_score,
            }, best_ckpt)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_stats['loss']:.6f}",
                f"{train_stats['loss_seg']:.6f}",
                f"{train_stats['loss_cls']:.6f}",
                f"{train_stats['loss_det']:.6f}",
                f"{val_stats['seg_mean_iou_hand_bg']:.6f}",
                f"{val_stats['seg_dice']:.6f}",
                f"{val_stats['det_acc05']:.6f}",
                f"{val_stats['det_mean_bbox_iou']:.6f}",
                f"{val_stats['cls_acc']:.6f}",
            ])

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] "
            f"train loss={train_stats['loss']:.4f} "
            f"(seg={train_stats['loss_seg']:.4f}, cls={train_stats['loss_cls']:.4f}, det={train_stats['loss_det']:.4f}) | "
            f"val mIoU={val_stats['seg_mean_iou_hand_bg']:.3f}, "
            f"dice={val_stats['seg_dice']:.3f}, "
            f"det@0.5={val_stats['det_acc05']:.3f}, "
            f"detIoU={val_stats['det_mean_bbox_iou']:.3f}, "
            f"cls_acc={val_stats['cls_acc']:.3f} | "
            f"best={best_score:.3f} | {dt:.1f}s"
        )

    print(f"Training complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
