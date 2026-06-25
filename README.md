# ViViT-B vs Video Swin-B on Vident-real — Teeth Segmentation

Binary teeth segmentation on the [Vident-real](https://mostwiedzy.pl/en/open-research-data/) intra-oral dental video dataset, comparing **ViViT-B** and **Video Swin-B** (Kinetics-400 pretrained) under **full fine-tuning** vs **LoRA**, with efficiency metrics (time, VRAM, power, energy via NVML).


## Setup
- Dataset: Vident-real (Gdansk Univ., CC BY-NC 4.0), 65/10/25 train/val/test videos. Dataset and checkpoints not included.
- Backbones used as encoders + decoder; 32-frame clips at 224×224; BCE + Dice loss; center-frame mask.
- Metric: IoU / Dice on the full validation set.

## Results

| Model | Regime | Best Val IoU | Best Val Dice |
|---|---|---|---|
| Video Swin-B | Full FT | 0.8204 | 0.8944 |
| Video Swin-B | LoRA | 0.8204 | 0.8960 |
| ViViT-B | Full FT | ~0.78 | ~0.87 |
| ViViT-B | LoRA | ~0.76 | ~0.85 |

**Findings**
- **LoRA matches full fine-tuning** while training ~1% of parameters (ties, not beats).
- **Video Swin-B > ViViT-B** — multi-scale features suit dense prediction better than ViViT's single token grid.
- A cosine scheduler + gradient clipping smooths Video Swin's validation fluctuation (constant-LR full FT) with ~+0.004 IoU — a stability fix, not an accuracy gain.

Overlay panels (`input | GT | Swin | ViViT`) confirm both models localize teeth; Swin masks are slightly cleaner.

## Files
```
vident_swin_seg_metrics.py    # Video Swin training + metrics (set REGIME = 'full' / 'lora')
vident_vivit_seg_metrics.py   # ViViT training + metrics
plot_seg_results.py           # IoU/Dice + loss curves, summary table
make_overlays.py              # comparison panels (no retraining)
Vident_seg_results/           # JSONs, figures
```
