import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_overlay(
    rgb: np.ndarray,                  # H,W,3 in [0,1]
    gt_mask: np.ndarray,              # H,W {0,1}
    pred_mask: np.ndarray,            # H,W {0,1}
    out_path: str,
    title: str = "",
    draw_bbox: bool = True,
    gt_bbox: Optional[Tuple[int, int, int, int]] = None,   # (x1,y1,x2,y2)
    pred_bbox: Optional[Tuple[int, int, int, int]] = None, # (x1,y1,x2,y2)
):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.imshow(rgb)


    gt = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    gt[..., 1] = gt_mask * 0.6   
    gt[..., 3] = gt_mask * 0.6   
    ax.imshow(gt)

    pr = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    pr[..., 0] = pred_mask * 0.6 
    pr[..., 3] = pred_mask * 0.6 
    ax.imshow(pr)


    if draw_bbox:
        if gt_bbox is not None:
            x1, y1, x2, y2 = gt_bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)

        if pred_bbox is not None:
            x1, y1, x2, y2 = pred_bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names,
    out_path: str,
    title: str = "Confusion Matrix"
):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_cls = len(class_names)

    # Auto-scale figure size to avoid overcrowded labels for many classes
    fig_w = max(8, 0.9 * n_cls)
    fig_h = max(6, 0.8 * n_cls)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(title, fontsize=18, pad=22)

    ax.set_xticks(np.arange(n_cls))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

 
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Prediction", fontsize=14, labelpad=12)
    ax.set_ylabel("Actual", fontsize=14, labelpad=12)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")


    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j, i, f"{value}",
                ha="center", va="center",
                color="white" if value > thresh else "#1f4b99",
                fontsize=11
            )


    ax.set_ylim(n_cls - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
