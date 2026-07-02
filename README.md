# ViViT-B vs Video Swin-B on Vident-real

Comparison of two video transformer backbones — **ViViT-B** and **Video Swin-B** (both pretrained on Kinetics-400) — on the [Vident-real](https://mostwiedzy.pl/en/open-research-data/) intra-oral dental video dataset, across a spectrum of **fine-tuning techniques** and two tasks. For every configuration we record both task quality and efficiency: training time, peak VRAM, average power, and energy (measured via NVML).

## Tasks

The repository covers two tasks on the same dataset, kept on separate branches:

| Branch | Task | Label source | Quality metric |
|---|---|---|---|
| `vident-real-segmentation` | Teeth segmentation (binary mask per frame) | `masks/` folder | IoU, Dice |
| `vident-real-classification` | Dental quadrant classification (Q1–Q4) | spreadsheet `Specified Frames` + `Location` columns | Accuracy, balanced accuracy |

## Dataset

Vident-real (Gdansk University of Technology, CC BY-NC 4.0): ~100 intra-oral surgical videos, 800×800, pre-split into **65 train / 10 validation / 25 test** videos. Each video folder contains `input/` (RGB frames), `masks/` (teeth masks), `GT/`, and motion data.

For classification, quadrant labels come from the accompanying spreadsheet. Each row's **Specified Frames** column defines the exact annotated frame range for a label, and following the supervisor's instruction, **each annotated segment is treated as an independent example** ("segment as video"). This expands the 65/10/25 *videos* into **81 / 13 / 27 *segments***. Video folders are matched to spreadsheet rows by frame count.

> The dataset itself and trained checkpoints (`.pth`) are **not** included in this repository.

## Method

- **Backbones:** torchvision `swin3d_b` (Kinetics-400) and HuggingFace `vivit-b-16x2-kinetics400`.
- **Clips:** 32 frames at 224×224. Within-clip temporal stride follows the papers' 32×2 sampling (every other frame) for segments long enough to fill it, falling back to stride 1 for short segments to avoid frame-repeat padding.
- **Fine-tuning techniques (classification):** a spectrum ordered by trainable-parameter count —
  - **Frozen** (linear probe, ~0.005%) — train only the classification head.
  - **BitFit** (~0.16%) — train only bias terms (of non-normalization layers) + head.
  - **LoRA** (~1%, rank 8) — low-rank adapters on FFN/attention projections + head.
  - **Partial** (~29%) — train the last *K* transformer blocks + head.
  - **Full** (100%) — train all parameters.
- **Loss:** class-weighted cross-entropy, and a class-weighted **focal loss** variant, to address the strong class imbalance.
- **Segmentation:** transformer used as encoder + a decoder; BCE + Dice loss; center-frame mask prediction; full FT vs LoRA.
- **Evaluation:** clip-level training, then clips aggregated to a **per-segment** prediction; model selected on validation, reported on test. Metrics reported at both segment and clip level, with overall and balanced accuracy plus confusion matrices.
- **Efficiency:** NVML power sampling produces average power and energy (Wh) for training and full-dataset inference; peak VRAM via CUDA stats.
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
- **Video Swin-B outperforms ViViT-B**, consistent with its multi-scale hierarchical features suiting dense prediction over ViViT's single coarse token grid.
- **LoRA matches full fine-tuning** while training only ~1% of the parameters (it ties rather than beats on this task).

Overlay panels (`input | ground truth | Video Swin | ViViT`) in the segmentation branch visually confirm both models localize teeth, with Swin producing slightly cleaner masks.

## Classification results

Dental quadrant (Q1–Q4) classification, per-segment prediction (clips aggregated per segment). The dataset is small (81/13/27 segments) and heavily imbalanced (Q1 dominant, Q3/Q4 rare), so overall accuracy is reported alongside balanced accuracy and confusion matrices. Selection is on validation; test is reported once.

Test accuracy (per-segment) across fine-tuning techniques, class-weighted focal loss:

| Model | Regime | Val Acc | Test Acc | Test Balanced Acc |
|---|---|---|---|---|
| Video Swin-B | Full | 0.692 | **0.630** | 0.503 |
| Video Swin-B | BitFit | 0.692 | 0.593 | 0.446 |
| Video Swin-B | LoRA | 0.615 | 0.556 | 0.467 |
| Video Swin-B | Partial | 0.615 | 0.444 | 0.348 |
| Video Swin-B | Frozen | 0.615 | 0.444 | 0.335 |
| ViViT-B | Full | 0.538 | 0.556 | 0.467 |
| ViViT-B | Frozen | 0.462 | 0.556 | 0.546 |
| ViViT-B | LoRA | 0.538 | 0.519 | 0.436 |
| ViViT-B | BitFit | 0.385 | 0.519 | 0.449 |
| ViViT-B | Partial | 0.615 | 0.407 | 0.461 |

**Loss comparison (cross-entropy vs focal).** For the full and LoRA regimes we ran both class-weighted cross-entropy and class-weighted focal loss. Test accuracy was **identical** between the two losses (Video Swin-B full 0.630, LoRA 0.556; ViViT-B full 0.556, LoRA 0.519 under both losses), with only minor differences in balanced accuracy. This is direct evidence that the loss function is not the limiting factor on this dataset.

**Findings:**
- **Across every fine-tuning technique (frozen → bitfit → lora → partial → full), both backbones, and both losses (cross-entropy and focal), test accuracy stayed in a narrow ~0.44–0.63 band.** This is the central result: on this dataset the **fine-tuning technique and loss are not the bottleneck**. The highest test accuracy (Video Swin-B, full FT, 0.63) sits within the noise of a 27-segment test set.
- The limiting factor is **dataset size and class imbalance** — 81 training segments with very few Q3/Q4 examples. Focal loss matched class-weighted cross-entropy exactly on test, consistent with imbalance already being partly handled by class weighting.
- Confusion matrices show the models leaning toward the majority class; balanced accuracies (~0.33–0.55) sit above the 0.25 random-chance level but confirm the task is only weakly learnable at this data scale.

**Limitation — model selection.** With only 13 validation segments, validation accuracy is quantized to ~7.7% steps and several configurations tie, so validation only weakly distinguishes techniques. The test numbers are reported for transparency; given the 27-segment test set, small differences between techniques are within noise and should not be over-interpreted.

## Environment

RTX 5090 (Blackwell), Windows, Python 3.13, PyTorch 2.11 + CUDA 12.8, bfloat16.
