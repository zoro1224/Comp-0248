import os
import json
import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiceLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,1,H,W), target: (B,1,H,W) in {0,1}
        prob = torch.sigmoid(logits)
        prob = prob.view(prob.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


# =========================
# Segmentation metrics
# =========================
def mask_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return float((inter + eps) / (union + eps))


def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    s = (pred > 0).sum() + (gt > 0).sum()
    return float((2 * inter + eps) / (s + eps))


def mean_iou_hand_bg(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    iou_hand = mask_iou(pred, gt, eps)
    pred_bg = 1 - (pred > 0).astype(np.uint8)
    gt_bg = 1 - (gt > 0).astype(np.uint8)
    iou_bg = mask_iou(pred_bg, gt_bg, eps)
    return 0.5 * (iou_hand + iou_bg)


# =========================
# Mask -> bbox (still useful for GT box creation)
# =========================
def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    mask: H,W in {0,1}
    returns (x1, y1, x2, y2) inclusive-exclusive
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return int(x1), int(y1), int(x2), int(y2)


def masks_to_boxes_xyxy(mask_batch: torch.Tensor) -> torch.Tensor:
    """
    mask_batch: (B,1,H,W), values in {0,1}
    returns: (B,4) pixel coords [x1,y1,x2,y2]
    """
    B, _, _, _ = mask_batch.shape
    boxes = []
    for i in range(B):
        m = mask_batch[i, 0] > 0.5
        ys, xs = torch.where(m)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append(torch.tensor([0.0, 0.0, 0.0, 0.0], device=mask_batch.device))
        else:
            x1 = xs.min().float()
            y1 = ys.min().float()
            x2 = xs.max().float() + 1.0
            y2 = ys.max().float() + 1.0
            boxes.append(torch.stack([x1, y1, x2, y2]))
    return torch.stack(boxes, dim=0)


# =========================
# Box format conversion
# =========================
def xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    boxes_xyxy: (B,4) pixel coords [x1,y1,x2,y2]
    returns: (B,4) normalized [cx,cy,w,h] in [0,1]
    """
    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    out = torch.stack([cx, cy, bw, bh], dim=1)
    return torch.clamp(out, 0.0, 1.0)


def cxcywh_to_xyxy_norm(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """
    boxes_cxcywh: (B,4), normalized [cx,cy,w,h]
    returns: (B,4), normalized [x1,y1,x2,y2]
    """
    cx, cy, bw, bh = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    out = torch.stack([x1, y1, x2, y2], dim=1)
    return torch.clamp(out, 0.0, 1.0)


# =========================
# Detection metrics for independent detection head
# =========================
def pairwise_box_iou(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    """
    boxes1_xyxy, boxes2_xyxy: (B,4), normalized xyxy
    returns: (B,) IoU
    """
    x1 = torch.maximum(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
    y1 = torch.maximum(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
    x2 = torch.minimum(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
    y2 = torch.minimum(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    inter = inter_w * inter_h

    area1 = torch.clamp(boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0], min=0.0) * \
            torch.clamp(boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1], min=0.0)
    area2 = torch.clamp(boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0], min=0.0) * \
            torch.clamp(boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1], min=0.0)

    union = area1 + area2 - inter + 1e-6
    return inter / union


def det_metrics_from_boxes(pred_boxes_cxcywh: torch.Tensor, gt_boxes_cxcywh: torch.Tensor) -> Dict[str, float]:
    """
    pred_boxes_cxcywh, gt_boxes_cxcywh: (B,4), normalized [cx,cy,w,h]
    returns dict with mean IoU and acc@0.5
    """
    pred_xyxy = cxcywh_to_xyxy_norm(pred_boxes_cxcywh)
    gt_xyxy = cxcywh_to_xyxy_norm(gt_boxes_cxcywh)
    ious = pairwise_box_iou(pred_xyxy, gt_xyxy)
    acc05 = (ious >= 0.5).float().mean().item()
    mean_iou = ious.mean().item()
    return {
        "bbox_iou": mean_iou,
        "acc05": acc05,
    }


# =========================
# Misc
# =========================
def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def to_numpy_mask_from_logits(seg_logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    prob = torch.sigmoid(seg_logits).detach().cpu().numpy()
    pred = (prob[:, 0] >= thr).astype(np.uint8)  # B,H,W
    return pred