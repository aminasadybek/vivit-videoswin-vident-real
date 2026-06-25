# ViViT-B vs Video Swin-B on Vident-real

Comparison of two video transformer backbones — **ViViT-B** and **Video Swin-B** (both pretrained on Kinetics-400) — under **full fine-tuning** versus **LoRA** (parameter-efficient fine-tuning) on the [Vident-real](https://mostwiedzy.pl/en/open-research-data/) intra-oral dental video dataset. For every configuration we record both task quality and efficiency: training time, peak VRAM, average power, and energy (measured via NVML).

Conducted as part of **ESAI406** at Nazarbayev University, ESAI Lab, supervised by Prof. Jurn Gyu Park.

## Tasks

The repository covers two tasks on the same dataset, kept on separate branches:

| Branch | Task | Label source | Quality metric |
|---|---|---|---|
| `vident-real-segmentation` | Teeth segmentation (binary mask per frame) | `masks/` folder | IoU, Dice |
| `vident-real-classification` | Dental quadrant classification (Q1–Q4) | spreadsheet `Location` column | Accuracy, balanced accuracy |

The **2 models × 2 fine-tuning regimes** grid (4 cells) is run for each task.

## Dataset

Vident-real (Gdansk University of Technology, CC BY-NC 4.0): ~100 intra-oral surgical videos, 800×800, split into **65 train / 10 validation / 25 test** videos. Each video folder contains `input/` (RGB frames), `masks/` (teeth masks), `GT/`, and motion data. Quadrant labels (Q1–Q4) come from the accompanying classification spreadsheet and are matched to video folders by frame count.

> The dataset itself and trained checkpoints (`.pth`) are **not** included in this repository.

## Method

- **Backbones:** torchvision `swin3d_b` (Kinetics-400) and HuggingFace `vivit-b-16x2-kinetics400`.
- **Clips:** 32 frames at 224×224 (ViViT requires 32; Swin matched to 32 so efficiency metrics are comparable).
- **Regimes:** full fine-tuning vs LoRA (rank 8, applied to FFN/attention projections; task head trained fully).
- **Segmentation:** transformer used as encoder + a decoder; BCE + Dice loss; center-frame mask prediction.
- **Classification:** clip-level training, then clips aggregated to a **per-video** prediction; class-weighted cross-entropy to handle imbalance; model selected on validation, reported on test.
- **Efficiency:** NVML power sampling → average power and energy (Wh) for training and full-dataset inference; peak VRAM via CUDA stats.
- **Precision:** bfloat16 autocast on RTX 5090 (TF32 enabled, cuDNN benchmark on).

## Segmentation results

Center-frame teeth segmentation, 32-frame clips, full validation set.

| Model | Regime | Best Val IoU | Best Val Dice |
|---|---|---|---|
| Video Swin-B | Full FT | 0.8204 | 0.8944 |
| Video Swin-B | LoRA | 0.8204 | 0.8960 |
| ViViT-B | Full FT | ~0.78 | ~0.87 |
| ViViT-B | LoRA | ~0.76 | ~0.85 |

**Findings:**
- **LoRA matches full fine-tuning** while training only ~1% of the parameters (it ties rather than beats on this task).
- **Video Swin-B outperforms ViViT-B**, consistent with its multi-scale hierarchical features suiting dense prediction over ViViT's single coarse token grid.
- Adding a cosine LR scheduler + gradient clipping smooths the Video Swin validation fluctuation (caused by a constant LR with full FT) and shifts the best epoch later, with essentially unchanged final IoU (~+0.004) — i.e. a stability fix, not an accuracy gain.

Overlay panels (`input | ground truth | Video Swin | ViViT`) are in the segmentation branch and visually confirm both models localize teeth, with Swin producing slightly cleaner masks.

## Classification results

Dental quadrant (Q1–Q4) classification, per-video prediction (clips aggregated per video). The dataset is small (65/10/25 videos) and heavily imbalanced (Q1 ≈ half), so overall accuracy is reported alongside balanced accuracy and a confusion matrix. A trivial "always predict Q1" baseline reaches ~48% on the test set.

| Model | Regime | Val Acc | Test Acc | Test Balanced Acc | Best Epoch |
|---|---|---|---|---|---|
| Video Swin-B | Full FT | 0.700 | 0.520 | 0.458 | 1 |
| Video Swin-B | LoRA | 0.700 | 0.560 | 0.469 | 12 |
| ViViT-B | Full FT | 0.700 | 0.480 | 0.333 | 6 |
| ViViT-B | LoRA | 0.700 | **0.600** | **0.500** | 2 |

**Findings:**
- **LoRA outperforms full fine-tuning for both backbones** (Swin 0.56 vs 0.52, ViViT 0.60 vs 0.48). On a dataset this small, full fine-tuning overfits almost immediately — training accuracy reaches ~100% while validation/test collapse and validation loss diverges — whereas LoRA's limited capacity acts as a regularizer. Best configuration overall: **ViViT-B + LoRA** (test accuracy 0.60, balanced 0.50).
- Results are **modest and limited by the dataset**. Balanced accuracies (0.33–0.50) sit above the 0.25 random-chance level but show that predicting dental quadrant from video is only weakly learnable here; all confusion matrices show the models leaning toward the majority class Q1. ViViT-B Full FT (0.48) does not exceed the always-Q1 baseline.

**Limitation — model selection.** With only 10 validation videos, validation accuracy is quantized to 10% steps and **all four configurations tie at 0.700**, so validation cannot distinguish between them. The intended "select on validation, report on test" protocol is therefore unreliable for this split; the test ranking above is reported for transparency, not as a validated selection.

## Environment

RTX 5090 (Blackwell), Windows, Python 3.13, PyTorch 2.11 + CUDA 12.8, bfloat16.

## Acknowledgements

Vident-real dataset © Gdansk University of Technology (CC BY-NC 4.0). Work supervised by Prof. Jurn Gyu Park, ESAI Lab, Nazarbayev University.
