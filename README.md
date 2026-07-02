# ViViT-B vs Video Swin-B on Vident-real — Quadrant Classification (Parameter-Efficient Fine-Tuning, Focal Loss)

Dental quadrant (Q1–Q4) classification on the [Vident-real](https://mostwiedzy.pl/en/open-research-data/) intra-oral dental video dataset, comparing **ViViT-B** and **Video Swin-B** (Kinetics-400 pretrained) under three additional fine-tuning techniques — **Frozen** (linear probe), **BitFit**, and **Partial** (last-K blocks) — using **class-weighted focal loss**. These extend the Full-FT and LoRA comparison with the low-capacity end of the fine-tuning spectrum. Protocol: select on validation, report on test.

## Setup

- Dataset: Vident-real (Gdansk Univ., CC BY-NC 4.0), pre-split 65/10/25 train/val/test videos. Labels from the classification spreadsheet, matched to folders by frame count. Dataset not included.
- Each annotated **Specified-Frames** range is treated as an independent example ("segment as video"), expanding the videos into **81 / 13 / 27** train/val/test segments.
- 4 classes (Q1–Q4); 32-frame clips at 224×224 (within-clip temporal stride 2, papers' 32×2, with fallback to 1 for short segments).
- **Loss:** class-weighted **focal loss** (γ = 2) — the same loss used for the Full/LoRA focal runs, so the whole fine-tuning spectrum is compared under a fixed loss.
- Clip-level training, **per-segment** prediction (clips aggregated). Reported with balanced accuracy + confusion matrix (a trivial "always Q1" baseline ≈ 52% on the 27-segment test set).

## Fine-tuning techniques

All techniques share the same Kinetics-400 pretrained backbone and a freshly-trained 4-class head; they differ in how many parameters are trainable. Ordered by trainable-parameter count:

| Regime | Trainable | What is trained |
|---|---|---|
| **Frozen** (linear probe) | ~0.005% | Only the classification head; the entire backbone is frozen. |
| **BitFit** | ~0.16% | Only the bias terms of non-normalization layers, plus the head. |
| **Partial** | ~29% | The last *K* transformer blocks (K = 2) plus the head; earlier layers frozen. |

For reference, the companion runs cover **LoRA** (~1%) and **Full FT** (100%), giving the complete spectrum **frozen → bitfit → lora → partial → full**.

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

Per-segment metrics, class-weighted focal loss (γ = 2).

| Model | Regime | Val Acc | Test Acc | Test Balanced | Best Ep |
|---|---|---|---|---|---|
| Video Swin-B | BitFit | 0.6923 | **0.5926** | 0.4464 | 2 |
| Video Swin-B | Frozen | 0.6154 | 0.4444 | 0.3348 | 12 |
| Video Swin-B | Partial | 0.6154 | 0.4444 | 0.3482 | 1 |
| ViViT-B | BitFit | 0.3846 | 0.5185 | 0.4494 | 1 |
| ViViT-B | Frozen | 0.4615 | 0.5556 | **0.5461** | 3 |
| ViViT-B | Partial | 0.6154 | 0.4074 | 0.4613 | 1 |

**Findings**

- **These low-capacity techniques land in the same ~0.41–0.59 test band** as Full FT and LoRA. Combined with the companion Full/LoRA runs, test accuracy across the entire spectrum (frozen → bitfit → lora → partial → full) stays within a narrow range — evidence that the **fine-tuning technique is not the bottleneck**; dataset size and class imbalance are.
- For Video Swin-B, **BitFit** is the strongest of the three (test 0.59), while Frozen and Partial drop to ~0.44. For ViViT-B, **Frozen** gives the best balanced accuracy of any run in this table (0.55), suggesting its Kinetics features transfer reasonably without adaptation.
- Results remain **modest and Q1-leaning**; balanced accuracies (0.33–0.55) sit above the 0.25 random-chance level but confirm the task is only weakly learnable at this data scale.
- **Limitation — model selection.** With only 13 validation segments, validation accuracy is quantized to ~7.7% steps and only weakly separates techniques. The test numbers are reported for transparency; given the 27-segment test set, small differences between techniques are within noise and should not be over-interpreted.

## Files

```
vident_classify_ft.py                                    # Frozen / Partial (set MODEL, REGIME, PARTIAL_K)
vident_classify_bitfit.py                                # BitFit (set MODEL)
plot_cls_results_focal.py                                # accuracy/loss curves + confusion matrices + summary
Vident-real classification label (Dental location).xlsx  # quadrant labels (sheet: Modified)
Vident_cls_results_ft_focal/                             # JSONs, figures, summary
```
