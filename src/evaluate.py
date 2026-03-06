import os
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataloader import HandGestureKeyframeDataset, GESTURE_NAMES, collect_records
from model import TinyUNetMultiTask
from utils import (
    to_numpy_mask_from_logits,
    mean_iou_hand_bg,
    dice_coeff,
    save_json,
)
from visualise import save_overlay, save_confusion_matrix


def split_records(records, val_ratio: float, seed: int):
    rng = random.Random(seed)
    recs = records[:]
    rng.shuffle(recs)
    n = len(recs)
    n_val = int(n * val_ratio)
    val = recs[:n_val]
    train = recs[n_val:]
    return train, val


def bbox_from_mask_np(mask: np.ndarray):
    """
    mask: (H,W) in {0,1}
    return: (x1,y1,x2,y2) or None
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def bbox_iou_np(b1, b2, eps: float = 1e-6):
    """
    b1, b2: (x1,y1,x2,y2) or None
    """
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


@torch.no_grad()
def evaluate(model, loader, device, out_dir: str, tag: str, max_overlays: int = 20):
    model.eval()

    seg_mious = []
    seg_dices = []
    bbox_ious = []
    det_acc05 = []

    y_true = []
    y_pred = []

    overlays_dir = os.path.join(out_dir, "overlays", tag)
    os.makedirs(overlays_dir, exist_ok=True)

    overlay_count = 0

    for x, y_mask, y_cls, meta in loader:
        x = x.to(device)

        seg_logits, cls_logits = model(x)

        pred_masks = to_numpy_mask_from_logits(seg_logits, thr=0.5)   # (B,H,W)
        gt_masks = (y_mask.numpy()[:, 0] > 0.5).astype("uint8")       # (B,H,W)

        pred_cls = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()
        gt_cls = y_cls.numpy()

        rgb = x.detach().cpu().numpy()  # (B,C,H,W)

        for i in range(pred_masks.shape[0]):
            pm = pred_masks[i]
            gm = gt_masks[i]
            
             # Segmentation metrics
            seg_mious.append(mean_iou_hand_bg(pm, gm))
            seg_dices.append(dice_coeff(pm, gm))

            # Detection metrics (mask -> bbox)
            dm = det_metrics_from_masks_np(pm, gm)
            bbox_ious.append(dm["bbox_iou"])
            det_acc05.append(dm["acc05"])

            # Classification metrics
            y_true.append(int(gt_cls[i]))
            y_pred.append(int(pred_cls[i]))


            if overlay_count < max_overlays:
                rgb_img = np.transpose(rgb[i, :3], (1, 2, 0))  # H,W,3
                title = f"{meta['gesture'][i]} | pred={GESTURE_NAMES[pred_cls[i]]}"
                out_path = os.path.join(overlays_dir, f"{overlay_count:03d}.png")

                gt_bbox = bbox_from_mask_np(gm)
                pred_bbox = bbox_from_mask_np(pm)

                save_overlay(
                    rgb=rgb_img,
                    gt_mask=gm,
                    pred_mask=pm,
                    out_path=out_path,
                    title=title,
                    draw_bbox=True,
                    gt_bbox=gt_bbox,
                    pred_bbox=pred_bbox,
                )
                overlay_count += 1

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GESTURE_NAMES))))

    metrics = {
        "det_accuracy_iou_0.5": float(np.mean(det_acc05)) if det_acc05 else 0.0,
        "det_mean_bbox_iou": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
        "seg_mean_iou_hand_bg": float(np.mean(seg_mious)) if seg_mious else 0.0,
        "seg_dice": float(np.mean(seg_dices)) if seg_dices else 0.0,
        "cls_top1_accuracy": acc,
        "cls_macro_f1": f1,
        "num_samples": len(y_true),
    }

    save_json(metrics, os.path.join(out_dir, f"metrics_{tag}.json"))
    save_confusion_matrix(
        cm,
        GESTURE_NAMES,
        os.path.join(out_dir, f"confusion_{tag}.png"),
        title=f"Confusion Matrix ({tag})"
    )
    np.save(os.path.join(out_dir, f"confusion_{tag}.npy"), cm)

    print(
        f"[{tag}] "
        f"det@0.5={metrics['det_accuracy_iou_0.5']:.3f}, "
        f"detIoU={metrics['det_mean_bbox_iou']:.3f}, "
        f"mIoU={metrics['seg_mean_iou_hand_bg']:.3f}, "
        f"dice={metrics['seg_dice']:.3f}, "
        f"acc={metrics['cls_top1_accuracy']:.3f}, "
        f"macroF1={metrics['cls_macro_f1']:.3f}"
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to gesture folder root.")
    ap.add_argument("--checkpoint", type=str, default="weights/best_model.pth")
    ap.add_argument("--use_depth", action="store_true")
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--tag", type=str, default="val", help="Output tag name, e.g. val or test")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "all"],
                    help="Evaluate a split reproduced by the same seed-based rule as training.")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Validation ratio used during training.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed used during training split.")
    ap.add_argument("--overlay_seed", type=int, default=None,
                    help="Random seed for sampling overlay examples (DataLoader shuffle). "
                         "If not set, falls back to --seed.")
    ap.add_argument("--max_overlays", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--base", type=int, default=64,
                    help="Base channel width of the model.")
    args = ap.parse_args()

    if args.overlay_seed is None:
        args.overlay_seed = args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyUNetMultiTask(
        in_channels=4 if args.use_depth else 3,
        num_classes=10,
        base=args.base
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)


    records = collect_records(args.data_root)
    if len(records) == 0:
        raise RuntimeError(f"No annotated keyframes found under: {args.data_root}")

    train_recs, val_recs = split_records(records, args.val_ratio, args.seed)

    if args.split == "train":
        eval_recs = train_recs
    elif args.split == "val":
        eval_recs = val_recs
    else:
        eval_recs = records

    print(f"Evaluating split='{args.split}' with seed={args.seed}, val_ratio={args.val_ratio}")
    print(f"Number of samples in this split: {len(eval_recs)}")
    print(f"Sampling overlays with overlay_seed={args.overlay_seed}, max_overlays={args.max_overlays}")

    ds = HandGestureKeyframeDataset(
        args.data_root,
        records=eval_recs,
        use_depth=args.use_depth,
        image_size=(args.image_w, args.image_h),
    )


    g = torch.Generator()
    g.manual_seed(args.overlay_seed)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    evaluate(model, loader, device, args.out_dir, args.tag, max_overlays=args.max_overlays)


if __name__ == "__main__":
    main()
