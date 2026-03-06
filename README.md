


1.

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

2.

Dataset Format

The loader reads annotated keyframes only and expects the following structure:

<data_root>/
  <subject_1>/
    G01_call/
      clip01/
        rgb/
        annotation/
        depth/       (optional)
        depth_raw/   (optional .npy)
    ...
    G10_three/
  <subject_2>/
    ...


3.

Training
Train with RGB (default)

python train.py \
  --data_root <path_to_dataset_root> \
  --epochs 60 \
  --batch_size 8 \
  --lr 1e-4 \
  --cls_w 1.0 \
  --image_w 320 --image_h 240 \
  --seed 42 \
  --out_dir results \
  --weights_dir weights
  

Outputs:

Best checkpoint: weights/best_model.pth

Training log: results/train_log.csv

5) Evaluation (Metrics + Confusion Matrix + Overlays)

The evaluation script reproduces the same seed-based split rule used during training.

4.

Evaluate on validation split

python evaluate.py \
  --data_root <path_to_dataset_root> \
  --checkpoint weights/best_model.pth \
  --tag val \
  --split val \
  --val_ratio 0.2 \
  --seed 42 \
  --base 64 \
  --max_overlays 200 \
  --out_dir results



5.

Evaluate on train split 

python evaluate.py \
  --data_root <path_to_dataset_root> \
  --checkpoint weights/best_model.pth \
  --tag train \
  --split train \
  --val_ratio 0.2 \
  --seed 42 \
  --base 64 \
  --max_overlays 200 \
  --out_dir results
6.

Evaluate on all annotated samples

python evaluate.py \
  --data_root <path_to_dataset_root> \
  --checkpoint weights/choosed weights directory/best_model.pth \
  --tag all \
  --split all \
  --base 64 \
  --max_overlays 200 \
  --out_dir results

Evaluation outputs:

results/metrics_<tag>.json

results/confusion_<tag>.png and results/confusion_<tag>.npy

results/overlays/<tag>/*.png (qualitative overlays with masks + boxes)

Seg mIoU: mean IoU (hand vs background)

Seg Dice: Dice coefficient

Detection (derived from mask → bbox)

Det Acc@0.5: fraction of samples with bbox IoU ≥ 0.5

Det mean IoU: mean bbox IoU

Classification

Cls Acc: top-1 accuracy (10 classes)

Cls Macro-F1: macro-averaged F1


