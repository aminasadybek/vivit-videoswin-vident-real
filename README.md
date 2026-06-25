# ViViT-B vs Video Swin-B on Vident-real — Quadrant Classification

Dental quadrant (Q1–Q4) classification on the [Vident-real](https://mostwiedzy.pl/en/open-research-data/) intra-oral dental video dataset, comparing **ViViT-B** and **Video Swin-B** (Kinetics-400 pretrained) under **full fine-tuning** vs **LoRA**. Protocol: select on validation, report on test.

ESAI406, Nazarbayev University · ESAI Lab · supervised by Prof. Jurn Gyu Park.

## Setup
- Dataset: Vident-real (Gdansk Univ., CC BY-NC 4.0), 65/10/25 train/val/test videos; labels from the classification spreadsheet, matched to folders by frame count. Dataset not included.
- 4 classes (Q1–Q4); 32-frame clips at 224×224; class-weighted cross-entropy for imbalance.
- Clip-level training, **per-video** prediction (clips aggregated). Reported with balanced accuracy + confusion matrix (a trivial "always Q1" baseline ≈ 48%).

## Results

| Model | Regime | Val Acc | Test Acc | Test Balanced | Best Ep |
|---|---|---|---|---|---|
| Video Swin-B | Full FT | 0.700 | 0.520 | 0.458 | 1 |
| Video Swin-B | LoRA | 0.700 | 0.560 | 0.469 | 12 |
| ViViT-B | Full FT | 0.700 | 0.480 | 0.333 | 6 |
| ViViT-B | LoRA | 0.700 | **0.600** | **0.500** | 2 |

**Findings**
- **LoRA outperforms full fine-tuning** for both backbones; full FT overfits the small set almost immediately (train acc → ~100%, val loss diverges). Best: **ViViT-B + LoRA** (test 0.60 / balanced 0.50).
- Results are **modest and Q1-leaning** — balanced accuracy 0.33–0.50 (chance 0.25); quadrant-from-video is only weakly learnable on this small, imbalanced set.
- **Limitation:** with 10 validation videos, all four tie at 0.700 val, so validation cannot select between models — the test ranking is reported for transparency, not as a validated selection.

## Files
```
vident_classify.py        # both models × both regimes (set MODEL, REGIME)
plot_cls_results.py       # accuracy/loss curves + confusion matrices + summary
Vident_cls_results/       # JSONs, figures, summary.csv
```
