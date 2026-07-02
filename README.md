# ViViT-B vs Video Swin-B on Vident-real — Quadrant Classification

Dental quadrant (Q1–Q4) classification on the [Vident-real](https://mostwiedzy.pl/en/open-research-data/) intra-oral dental video dataset, comparing **ViViT-B** and **Video Swin-B** (Kinetics-400 pretrained) under **full fine-tuning** vs **LoRA**. Protocol: select on validation, report on test.

## Setup

- Dataset: Vident-real (Gdansk Univ., CC BY-NC 4.0), pre-split 65/10/25 train/val/test videos. Labels from the classification spreadsheet, matched to folders by frame count. Dataset not included.
- Each annotated **Specified-Frames** range is treated as an independent example ("segment as video"), expanding the videos into **81 / 13 / 27** train/val/test segments.
- 4 classes (Q1–Q4); 32-frame clips at 224×224 (within-clip temporal stride 2, papers' 32×2, with fallback to 1 for short segments); class-weighted cross-entropy for imbalance.
- Clip-level training, **per-segment** prediction (clips aggregated). Reported with balanced accuracy + confusion matrix (a trivial "always Q1" baseline ≈ 52% on the 27-segment test set).

## Labels

The quadrant labels are in `Vident-real classification label (Dental location).xlsx` (sheet `Modified`), included in this repo. Each row describes one annotated segment of a video:

| Column | Meaning |
|---|---|
| `Segmented Videos` | Running index of segments (a segmented clip within a video). Some videos are split into several rows. |
| `Vident-real Videos` | The video number (blank on continuation rows that belong to the video above). |
| `#Frames` | Total frame count of the video — used to match the spreadsheet row to its hex-named folder on disk. |
| `Specified Frames` | The annotated frame range for this label, e.g. `1-706`. **Only frames inside these ranges are used**; frames outside any range are excluded (some videos are only partially annotated). |
| `Location` | The quadrant label: `Q1` (maxillary right), `Q2` (maxillary left), `Q3` (mandibular left), `Q4` (mandibular right). |

Notes:
- All ranges within a single video share the same quadrant, so the label is unambiguous per video; the split into ranges only controls *which frames* are used.
- Following the supervisor's instruction, each `Specified Frames` range is treated as its own example, giving the 81 / 13 / 27 segment counts above.
- **Caveat:** cell `A1` of the spreadsheet links to *Vident-synth*, but the labels and frame counts correspond to *Vident-real* (the dataset used here, and the one provided by the supervisor). The link appears to be a stray reference; the annotations themselves are for Vident-real.

## Results

Per-segment metrics, class-weighted cross-entropy.

| Model | Regime | Val Acc | Test Acc | Test Balanced | Best Ep |
|---|---|---|---|---|---|
| Video Swin-B | Full FT | 0.5385 | **0.6296** | 0.5030 | 3 |
| Video Swin-B | LoRA | 0.6154 | 0.5556 | 0.4673 | 2 |
| ViViT-B | Full FT | 0.6154 | 0.5556 | 0.3735 | 8 |
| ViViT-B | LoRA | 0.5385 | 0.5185 | 0.4360 | 1 |

**Findings**

- **No fine-tuning regime clearly dominates.** Test accuracy spans 0.52–0.63, within the noise of a 27-segment test set. The highest test accuracy is **Video Swin-B + Full FT (0.63)**, and full FT is not outperformed by LoRA here.
- Full fine-tuning still saturates training accuracy quickly (an overfitting signature on 81 segments), but on this small test set that does not translate into worse test accuracy than LoRA.
- Results are **modest and Q1-leaning** — balanced accuracy 0.37–0.50 (chance 0.25); quadrant-from-video is only weakly learnable on this small, imbalanced set.
- **Limitation — model selection.** With only 13 validation segments, validation accuracy is quantized to ~7.7% steps and only weakly separates models. Notably, the highest-test model (Swin Full, 0.63) has among the *lowest* validation accuracy (0.5385), so it would **not** be the validation-selected model — a clear illustration that validation cannot reliably select on this split. The test ranking is reported for transparency, not as a validated selection.

## Files

```
vident_classify.py                                       # both models × both regimes (set MODEL, REGIME)
plot_cls_results.py                                      # accuracy/loss curves + confusion matrices + summary
Vident-real classification label (Dental location).xlsx  # quadrant labels (sheet: Modified)
Vident_cls_results/                                      # JSONs, figures, summary.csv
```
