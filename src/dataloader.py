import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

GESTURE_NAMES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]


@dataclass
class SampleRecord:
    rgb_path: str
    depth_path: Optional[str]
    depth_raw_path: Optional[str]
    mask_path: str
    label: int
    gesture: str
    clip: str
    frame_name: str


def _find_depth_paths(clip_dir: str, frame_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (depth_png_path, depth_raw_path) if they exist."""
    dp = os.path.join(clip_dir, "depth", frame_name)
    dr = os.path.join(clip_dir, "depth_raw", os.path.splitext(frame_name)[0] + ".npy")
    if not os.path.exists(dp):
        dp = None
    if not os.path.exists(dr):
        dr = None
    return dp, dr


def collect_records(root: str) -> List[SampleRecord]:
    from pathlib import Path
    import re

    IMG_EXTS = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]

    def list_images(folder: str):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(folder, f"*{ext}"))
        return sorted(files)

    def build_map(files):
        return {Path(p).stem: p for p in files}  # stem -> full path

    def match_rgb(rgb_map, mask_path: str):

        mstem = Path(mask_path).stem

        if mstem in rgb_map:
            return rgb_map[mstem]

        variants = [
            mstem.replace("_mask", ""),
            mstem.replace("mask_", ""),
            mstem.replace("-mask", ""),
            mstem.replace("mask-", ""),
            mstem.replace("_annotation", ""),
            mstem.replace("annotation_", ""),
        ]
        for v in variants:
            if v in rgb_map:
                return rgb_map[v]

        nums = re.findall(r"\d+", mstem)
        if nums:
            key = nums[-1]
            for cand in [key, key.zfill(3), f"frame_{key}", f"frame_{key.zfill(3)}"]:
                if cand in rgb_map:
                    return rgb_map[cand]
            for rs, rp in rgb_map.items():
                if rs.endswith(key) or (key in rs):
                    return rp

        return None

    records: List[SampleRecord] = []

    gesture_to_idx = {g: i for i, g in enumerate(GESTURE_NAMES)}
    for i, g in enumerate(GESTURE_NAMES):
        gesture_to_idx[g.split("_")[0]] = i  

    root = os.path.abspath(root)

    subject_dirs = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]

    for sdir in sorted(subject_dirs):
        gesture_dirs = [p for p in glob.glob(os.path.join(sdir, "G*")) if os.path.isdir(p)]

        for gdir in sorted(gesture_dirs):
            gname = os.path.basename(gdir)
            key = gname if gname in gesture_to_idx else gname.split("_")[0]
            if key not in gesture_to_idx:
                continue
            label = gesture_to_idx[key]

            clip_dirs = [p for p in glob.glob(os.path.join(gdir, "clip*")) if os.path.isdir(p)]
            clip_dirs += [p for p in glob.glob(os.path.join(gdir, "Clip*")) if os.path.isdir(p)]
            clip_dirs = sorted(set(clip_dirs))

            for cdir in clip_dirs:
                ann_dir = os.path.join(cdir, "annotation")
                rgb_dir = os.path.join(cdir, "rgb")
                if not os.path.isdir(ann_dir) or not os.path.isdir(rgb_dir):
                    continue

                mask_files = list_images(ann_dir)
                rgb_files = list_images(rgb_dir)
                if not mask_files or not rgb_files:
                    continue

                rgb_map = build_map(rgb_files)

                for mp in mask_files:
                    rgb_path = match_rgb(rgb_map, mp)
                    if rgb_path is None:
                        continue

                    depth_path = None
                    depth_raw_path = None
                    depth_dir = os.path.join(cdir, "depth")
                    depth_raw_dir = os.path.join(cdir, "depth_raw")

                    if os.path.isdir(depth_dir):
                        stem = Path(rgb_path).stem
                        for ext in IMG_EXTS:
                            cand = os.path.join(depth_dir, stem + ext)
                            if os.path.exists(cand):
                                depth_path = cand
                                break

                    if os.path.isdir(depth_raw_dir):
                        stem = Path(rgb_path).stem
                        cand = os.path.join(depth_raw_dir, stem + ".npy")
                        depth_raw_path = cand if os.path.exists(cand) else None

                    records.append(SampleRecord(
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        depth_raw_path=depth_raw_path,
                        mask_path=mp,
                        label=label,
                        gesture=gname,
                        clip=os.path.basename(cdir),
                        frame_name=os.path.basename(rgb_path),
                    ))

    return records


def _read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _read_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim != 2:
        m = m[..., 0]
    return (m > 127).astype(np.uint8)


def _read_depth(depth_png_path: Optional[str], depth_raw_path: Optional[str]) -> Optional[np.ndarray]:

    if depth_raw_path and os.path.exists(depth_raw_path):
        d = np.load(depth_raw_path).astype(np.float32)
        if np.nanmax(d) > 50.0:  # heuristic: mm -> meters
            d = d / 1000.0
        d = np.clip(d, 0.0, 4.0) / 4.0
        return d

    if depth_png_path and os.path.exists(depth_png_path):
        arr = np.array(Image.open(depth_png_path))
        if arr.ndim == 3:
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        arr = arr.astype(np.float32)
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr

    return None


class HandGestureKeyframeDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        records: Optional[List[SampleRecord]] = None,
        use_depth: bool = True,
        image_size: Tuple[int, int] = (320, 240),  # (W,H)
    ):
        super().__init__()
        self.data_root = data_root
        self.records = records if records is not None else collect_records(data_root)
        self.use_depth = use_depth
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]

        rgb = _read_rgb(r.rgb_path)
        mask = _read_mask(r.mask_path)
        depth = _read_depth(r.depth_path, r.depth_raw_path)

        W, H = self.image_size
        rgb_img = Image.fromarray(rgb).resize((W, H), resample=Image.BILINEAR)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)

        rgb = np.array(rgb_img).astype(np.float32) / 255.0
        mask = (np.array(mask_img) > 127).astype(np.float32)

        if self.use_depth:
            # Keep channel count consistent even if depth is missing
            if depth is None:
                depth = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
            else:
                depth_img = Image.fromarray((depth * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
                depth = np.array(depth_img).astype(np.float32) / 255.0
            x = np.concatenate([rgb, depth[..., None]], axis=2)  # H,W,4
        else:
            x = rgb  # H,W,3

        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y_mask = torch.from_numpy(mask[None, ...]).contiguous()
        y_cls = torch.tensor(r.label, dtype=torch.long)

        meta = {
            "rgb_path": r.rgb_path,
            "mask_path": r.mask_path,
            "gesture": r.gesture,
            "clip": r.clip,
            "frame": r.frame_name,
        }
        return x, y_mask, y_cls, meta
