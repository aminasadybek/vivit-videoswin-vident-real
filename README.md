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

Dental quadrant (Q1–Q4) classification. The dataset is small (65/10/25 videos) and heavily imbalanced (Q1 ≈ half), so overall accuracy is reported **alongside balanced accuracy, per-class accuracy, and a confusion matrix**; a trivial "always predict Q1" baseline reaches ~48%.

Validation accuracy is computed per video and moves in ~10% steps (only 10 validation videos). Under full fine-tuning the model overfits the small training set quickly (training accuracy rises while validation accuracy falls). Final per-model test results and the model selected by validation are summarized in `Vident_cls_results/plots/summary.csv`.

## Usage

```bash
# install
python -m pip install torch torchvision transformers peft pynvml openpyxl matplotlib

# segmentation (set REGIME = 'full' then 'lora' at the top of each file)
python vident_swin_seg_metrics.py
python vident_vivit_seg_metrics.py
python plot_seg_results.py

# classification (set MODEL and REGIME at the top, run all four cells)
python vident_classify.py
python plot_cls_results.py
```

Edit the `ROOT` / `LABELS_XLSX` / `SAVE_DIR` paths at the top of each script to point to a **local** copy of the data (reading clips off a shared network drive starves the GPU). Keep `SMOKE_TEST = True` for the first run to verify the folder→label mapping before launching the full run.

## Environment

RTX 5090 (Blackwell), Windows, Python 3.13, PyTorch 2.11 + CUDA 12.8, bfloat16.

## Acknowledgements

Vident-real dataset © Gdansk University of Technology (CC BY-NC 4.0). Work supervised by Prof. Jurn Gyu Park, ESAI Lab, Nazarbayev University.
