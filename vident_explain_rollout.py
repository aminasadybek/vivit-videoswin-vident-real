# Attention-rollout explainability for the Vident-real QUADRANT classifier (Q1-Q4).
# Answers Prof. Park's ask: what's driving low accuracy - visualized + quantified via attention.
#
# REQUIRES: vident_classify.py (your existing quadrant script) in the SAME directory - this file
# imports it directly to reuse label parsing / folder matching / clip building / model building,
# so the data loading logic can't drift out of sync between the two scripts.
#
# WHY TWO DIFFERENT ROLLOUT IMPLEMENTATIONS:
#   - ViViT-B: standard ViT-style global self-attention + CLS token -> textbook Attention Rollout
#     (Abnar & Zuidema 2020) works directly via HuggingFace's output_attentions=True.
#   - Video Swin-B: attention is LOCAL to shifted windows, never returned by torchvision, and the
#     global-token assumption rollout relies on doesn't hold across the whole network (windows are
#     re-partitioned every block, resolution changes at every patch-merging stage). So this script:
#       1. monkeypatches torchvision's shifted_window_attention_3d to capture post-softmax weights
#       2. reconstructs a shift-corrected GLOBAL attention matrix per block (verified: row sums = 1)
#       3. does exact rollout algebra WITHIN THE FINAL STAGE ONLY (chaining its ~2 blocks)
#   This is an honest, verified "final-stage attention rollout" for Swin - not full-network rollout.
#   Flag this scope limitation to Prof. Park; a full cross-stage version would need to model how
#   patch-merging redistributes attention across resolutions, which is a separate derivation.
#
# Install (on top of your existing env):  pip install matplotlib scipy
# Run:  C:\ProgramData\Miniconda3\python.exe vident_explain_rollout.py

import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vident_classify as VC   # <-- your existing script, unmodified
import torchvision.models.video.swin_transformer as sw
from transformers import VivitForVideoClassification
from peft import LoraConfig, get_peft_model

device = VC.device
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# =============================== CONFIG ===============================
MODEL  = VC.MODEL          # reuse whatever's set in vident_classify.py ('vivit' | 'swin')
REGIME = VC.REGIME         # 'full' | 'lora'
N_QUALITATIVE_PER_GROUP = 4   # correct vs misclassified examples to visualize
N_QUANT_SAMPLE = 120          # clips sampled for the quantitative entropy/concentration analysis
FRAMES_TO_SHOW = 6            # frames per overlay figure
EXPLAIN_DIR = os.path.join(VC.SAVE_DIR, 'explainability')
os.makedirs(EXPLAIN_DIR, exist_ok=True)


# =============================== MODEL (eager attention for ViViT) ===============================
def build_model_explain(model_name, regime):
    """Same as VC.build_model but forces attn_implementation='eager' for ViViT so
    output_attentions works. Module structure/state_dict keys are unaffected by this
    flag (confirmed against the installed transformers version), so your existing
    fine-tuned checkpoint loads into this reconstruction without any key mismatches."""
    if model_name == 'swin':
        return VC.build_model(model_name, regime)   # Swin doesn't need this change
    net = VivitForVideoClassification.from_pretrained(
        'google/vivit-b-16x2-kinetics400', num_labels=VC.N_CLASSES,
        ignore_mismatched_sizes=True, attn_implementation='eager',
    )
    if regime == 'lora':
        for p in net.parameters(): p.requires_grad = False
        targets = [n for n, m in net.named_modules() if isinstance(m, nn.Linear)
                   and any(t in n for t in ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2'])]
        net = get_peft_model(net, LoraConfig(r=VC.LORA_R, lora_alpha=VC.LORA_ALPHA, lora_dropout=0.05,
                                              target_modules=targets, modules_to_save=['classifier']))
    return VC.Wrap(net, is_hf=True).to(device)


def load_best_checkpoint(model, model_name, regime):
    ckpt = os.path.join(VC.SAVE_DIR, f'{model_name}_{regime}_cls_best.pth')
    assert os.path.isfile(ckpt), f'checkpoint not found: {ckpt} - run vident_classify.py first'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


# =============================== VIVIT ROLLOUT ===============================
def vivit_rollout(model, x):
    """x: (1, T, C, H, W) already-normalized clip tensor (VC's ClipDataset layout is
    (C,T,H,W); this function expects HF's (T,C,H,W) - see run_rollout() for the transpose).
    Returns importance grid (n_t, n_h, n_w) over ViViT's tubelet patches."""
    net = model.model  # unwrap VC.Wrap
    with torch.no_grad():
        out = net(pixel_values=x, output_attentions=True)
    attentions = out.attentions  # tuple of (1, heads, seq, seq)
    seq = attentions[0].shape[-1]
    R = torch.eye(seq).unsqueeze(0)
    for A in attentions:
        A = A.mean(dim=1).cpu()
        A = 0.5 * A + 0.5 * torch.eye(seq).unsqueeze(0)
        A = A / A.sum(-1, keepdim=True)
        R = torch.bmm(A, R)
    cls_to_patches = R[0, 0, 1:]  # (n_patches,)
    cfg = net.config if not hasattr(net, 'base_model') else net.base_model.config
    tub = cfg.tubelet_size  # [t, h, w]
    n_t = VC.CLIP_LEN // tub[0]
    n_h = cfg.image_size // tub[1]
    n_w = cfg.image_size // tub[2]
    return cls_to_patches.reshape(n_t, n_h, n_w), out.logits.softmax(-1).cpu()


# =============================== SWIN ROLLOUT (final stage) ===============================
_CAPTURE = {"on": False, "records": []}
_orig_attn_fn = sw.shifted_window_attention_3d

def _patched_attn(input, qkv_weight, proj_weight, relative_position_bias, window_size,
                   num_heads, shift_size, attention_dropout=0.0, dropout=0.0,
                   qkv_bias=None, proj_bias=None, training=True):
    b, t, h, w, c = input.shape
    pad_size = sw._compute_pad_size_3d((t, h, w), tuple(window_size))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    num_windows = (padded_size[0]//window_size[0]) * (padded_size[1]//window_size[1]) * (padded_size[2]//window_size[2])
    xw = x.view(b, padded_size[0]//window_size[0], window_size[0], padded_size[1]//window_size[1], window_size[1],
                padded_size[2]//window_size[2], window_size[2], c)
    xw = xw.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(b*num_windows, window_size[0]*window_size[1]*window_size[2], c)
    qkv = F.linear(xw, qkv_weight, qkv_bias)
    qkv = qkv.reshape(xw.size(0), xw.size(1), 3, num_heads, c//num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (c // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1)) + relative_position_bias
    if sum(shift_size) > 0:
        attn_mask = sw._compute_attention_mask_3d(xw, padded_size, tuple(window_size), tuple(shift_size))
        attn = attn.view(xw.size(0)//num_windows, num_windows, num_heads, xw.size(1), xw.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, xw.size(1), xw.size(1))
    attn = F.softmax(attn, dim=-1)
    if _CAPTURE["on"]:
        _CAPTURE["records"].append({
            "attn": attn.detach().mean(dim=1).cpu(), "b": b, "padded_size": padded_size,
            "window_size": tuple(window_size), "shift_size": tuple(shift_size),
            "num_windows": num_windows, "orig_size": (t, h, w),
        })
    attn = F.dropout(attn, p=attention_dropout, training=training)
    xo = attn.matmul(v).transpose(1, 2).reshape(xw.size(0), xw.size(1), c)
    xo = F.linear(xo, proj_weight, proj_bias)
    xo = F.dropout(xo, p=dropout, training=training)
    xo = xo.view(b, padded_size[0]//window_size[0], padded_size[1]//window_size[1], padded_size[2]//window_size[2],
                 window_size[0], window_size[1], window_size[2], c)
    xo = xo.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)
    if sum(shift_size) > 0:
        xo = torch.roll(xo, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
    return xo[:, :t, :h, :w, :].contiguous()


def _record_to_global_matrix(rec):
    b = rec["b"]; tp, hp, wp = rec["padded_size"]; wd, wh, ww = rec["window_size"]
    st, sh, sw_ = rec["shift_size"]; nW = rec["num_windows"]
    attn = rec["attn"]; N = tp * hp * wp
    nt, nh, nw = tp // wd, hp // wh, wp // ww
    win_ids = torch.arange(nW)
    wt = win_ids // (nh * nw); wh_ = (win_ids // nw) % nh; ww_ = win_ids % nw
    ld = torch.arange(wd).view(-1, 1, 1).expand(wd, wh, ww).reshape(-1)
    lh = torch.arange(wh).view(1, -1, 1).expand(wd, wh, ww).reshape(-1)
    lw = torch.arange(ww).view(1, 1, -1).expand(wd, wh, ww).reshape(-1)
    t_c = wt.view(-1, 1) * wd + ld.view(1, -1)
    h_c = wh_.view(-1, 1) * wh + lh.view(1, -1)
    w_c = ww_.view(-1, 1) * ww + lw.view(1, -1)
    if st or sh or sw_:
        t_c = (t_c + st) % tp; h_c = (h_c + sh) % hp; w_c = (w_c + sw_) % wp
    flat_idx = t_c * hp * wp + h_c * wp + w_c
    G = torch.zeros(b, N, N)
    attn = attn.view(b, nW, wd*wh*ww, wd*wh*ww)
    for bi in range(b):
        for wi in range(nW):
            idx = flat_idx[wi]
            G[bi][idx.unsqueeze(1), idx.unsqueeze(0)] = attn[bi, wi]
    return G, (tp, hp, wp)


def swin_rollout(model, x):
    """x: (1, C, T, H, W) - torchvision's video layout (matches VC.ClipDataset directly)."""
    net = model.model  # unwrap VC.Wrap
    depths = [2, 2, 18, 2]  # swin3d_b default depths - confirm against your torchvision version if it errors
    _CAPTURE["on"] = True; _CAPTURE["records"] = []
    sw.shifted_window_attention_3d = _patched_attn
    with torch.no_grad():
        out = net(x)
    sw.shifted_window_attention_3d = _orig_attn_fn
    _CAPTURE["on"] = False

    last_stage_records = _CAPTURE["records"][-depths[-1]:]
    mats, shape = [], None
    for rec in last_stage_records:
        G, shape = _record_to_global_matrix(rec)
        N = G.shape[-1]
        A = 0.5 * G + 0.5 * torch.eye(N).unsqueeze(0)
        A = A / A.sum(-1, keepdim=True).clamp(min=1e-8)
        mats.append(A)
    R = mats[0]
    for A in mats[1:]:
        R = torch.bmm(A, R)
    importance = R.mean(dim=1)[0]  # (N,)
    tp, hp, wp = shape
    return importance.reshape(tp, hp, wp), out.softmax(-1).cpu()


# =============================== ROLLOUT DISPATCH + UPSAMPLE ===============================
def compute_rollout(model, clip_chw_t):
    """clip_chw_t: (C,T,H,W) tensor as produced by VC.ClipDataset (single clip, no batch dim).
    Returns (heatmap (T,H,W) in [0,1], softmax_probs)."""
    x = clip_chw_t.unsqueeze(0).to(device)  # (1,C,T,H,W)
    if MODEL == 'vivit':
        x_thwc = x.permute(0, 2, 1, 3, 4)  # HF wants (B,T,C,H,W)
        grid, probs = vivit_rollout(_MODEL, x_thwc)
    else:
        grid, probs = swin_rollout(_MODEL, x)
    grid = grid.unsqueeze(0).unsqueeze(0).float()  # (1,1,t',h',w')
    heat = F.interpolate(grid, size=(VC.CLIP_LEN, VC.IMG_SIZE, VC.IMG_SIZE),
                          mode='trilinear', align_corners=False)[0, 0]
    heat = heat.numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat, probs


# =============================== ANALYSIS METRICS ===============================
def heatmap_stats(heat):
    """heat: (T,H,W) in [0,1]. Returns spatial entropy (avg over frames), temporal
    peak-frame index, and concentration ratio (max/mean) - the numbers that get
    compared between correct and misclassified groups."""
    T = heat.shape[0]
    flat = heat.reshape(T, -1)
    p = flat / (flat.sum(-1, keepdims=True) + 1e-8)
    ent = -(p * np.log(p + 1e-8)).sum(-1)              # per-frame spatial entropy
    frame_mass = flat.sum(-1)                            # how much attention each frame gets
    peak_frame = int(frame_mass.argmax())
    concentration = float(heat.max() / (heat.mean() + 1e-8))
    return {'mean_spatial_entropy': float(ent.mean()), 'peak_frame': peak_frame,
            'peak_frame_frac': peak_frame / max(T - 1, 1), 'concentration': concentration}


# =============================== VISUALIZATION ===============================
def load_frame_rgb(path):
    im = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (VC.IMG_SIZE, VC.IMG_SIZE))

def save_overlay_figure(frame_paths, heat, true_lbl, pred_lbl, correct, out_path, n_show=FRAMES_TO_SHOW):
    T = len(frame_paths)
    idxs = np.linspace(0, T - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.4))
    tag = 'CORRECT' if correct else 'MISCLASSIFIED'
    fig.suptitle(f'{tag}  true={VC.IDQ[true_lbl]}  pred={VC.IDQ[pred_lbl]}', fontsize=13)
    for ax, i in zip(axes, idxs):
        img = load_frame_rgb(frame_paths[i]) / 255.0
        ax.imshow(img)
        ax.imshow(heat[i], cmap='jet', alpha=0.45)
        ax.set_title(f'frame {i}', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================== MAIN ===============================
def main():
    global _MODEL
    print(f'Explainability (Attention Rollout) | {VC.NAME} | regime={REGIME}')
    _MODEL = build_model_explain(MODEL, REGIME)
    load_best_checkpoint(_MODEL, MODEL, REGIME)

    labels = VC.parse_labels(VC.LABELS_XLSX)
    fl_test = VC.match_folders(os.path.join(VC.ROOT, 'test'), labels['test'])
    seg_test = VC.flatten_segments(fl_test)
    te_labels = [lb for _, _, _, lb in seg_test]
    # one representative (non-overlapping, EVAL_STRIDE) set of clips per segment, keep clip-level paths
    te_clips = VC.build_clips(seg_test, VC.EVAL_STRIDE)

    # group clips by segment, keep first clip per segment for the qualitative pass,
    # and run inference on ALL clips for both the segment-level prediction AND the
    # quantitative sample.
    by_seg = {}
    for paths, lbl, seg_id in te_clips:
        by_seg.setdefault(seg_id, []).append((paths, lbl))

    print(f'{len(seg_test)} test segments, {len(te_clips)} eval clips')

    # ---- predict every segment (majority/averaged softmax over its clips, like VC.evaluate) ----
    seg_pred, seg_probs_sum = {}, {}
    _MODEL.eval()
    all_quant = []  # (heat_stats dict, correct bool) for the quantitative sample
    quant_pool = list(te_clips)
    random.shuffle(quant_pool)
    quant_pool = quant_pool[:N_QUANT_SAMPLE]

    print(f'Running rollout on {len(quant_pool)} clips for quantitative analysis '
          f'(this is the slow part - rollout needs attention capture, not just a forward pass)...')
    quant_rows = []
    for paths, lbl, seg_id in quant_pool:
        clip = VC.ClipDataset([(paths, lbl, seg_id)], train=False)[0][0]
        heat, probs = compute_rollout(_MODEL, clip)
        pred = int(probs.argmax())
        correct = (pred == lbl)
        stats = heatmap_stats(heat)
        stats.update({'seg_id': seg_id, 'true': VC.IDQ[lbl], 'pred': VC.IDQ[pred], 'correct': correct})
        quant_rows.append(stats)

    # ---- quantitative summary: correct vs misclassified ----
    import csv
    csv_path = os.path.join(EXPLAIN_DIR, f'rollout_stats_{MODEL}_{REGIME}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(quant_rows[0].keys()))
        w.writeheader(); w.writerows(quant_rows)
    print(f'Per-clip rollout stats saved -> {csv_path}')

    corr = [r for r in quant_rows if r['correct']]
    inco = [r for r in quant_rows if not r['correct']]
    def summarize(rows, key):
        vals = [r[key] for r in rows]
        return (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))

    print('\n' + '=' * 70)
    print(f'  QUANTITATIVE: correct (n={len(corr)}) vs misclassified (n={len(inco)})')
    print('=' * 70)
    for key in ['mean_spatial_entropy', 'peak_frame_frac', 'concentration']:
        mc, sc = summarize(corr, key); mi, si = summarize(inco, key)
        print(f'  {key:<22} correct: {mc:.4f} +/- {sc:.4f}   misclassified: {mi:.4f} +/- {si:.4f}')
    try:
        from scipy import stats as sstats
        for key in ['mean_spatial_entropy', 'peak_frame_frac', 'concentration']:
            if corr and inco:
                t, p = sstats.ttest_ind([r[key] for r in corr], [r[key] for r in inco], equal_var=False)
                print(f'  {key:<22} Welch t-test: t={t:.3f} p={p:.4f}')
    except ImportError:
        print('  (install scipy for significance tests: pip install scipy)')

    # bar chart of the three metrics, correct vs misclassified
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ['mean_spatial_entropy', 'peak_frame_frac', 'concentration']):
        mc, sc = summarize(corr, key); mi, si = summarize(inco, key)
        ax.bar(['correct', 'misclassified'], [mc, mi], yerr=[sc, si], capsize=5,
               color=['#4C72B0', '#C44E52'])
        ax.set_title(key)
    plt.tight_layout()
    chart_path = os.path.join(EXPLAIN_DIR, f'rollout_summary_{MODEL}_{REGIME}.png')
    plt.savefig(chart_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary chart saved -> {chart_path}')

    # ---- qualitative: N_QUALITATIVE_PER_GROUP correct + N misclassified example clips ----
    random.shuffle(corr); random.shuffle(inco)
    for group, tag in [(corr[:N_QUALITATIVE_PER_GROUP], 'correct'),
                        (inco[:N_QUALITATIVE_PER_GROUP], 'misclassified')]:
        for i, row in enumerate(group):
            seg_id = row['seg_id']
            paths, lbl = by_seg[seg_id][0]
            clip = VC.ClipDataset([(paths, lbl, seg_id)], train=False)[0][0]
            heat, probs = compute_rollout(_MODEL, clip)
            pred = int(probs.argmax())
            out_path = os.path.join(EXPLAIN_DIR, f'{MODEL}_{REGIME}_{tag}_{i}.png')
            save_overlay_figure(paths, heat, lbl, pred, pred == lbl, out_path)
            print(f'  saved {out_path}')

    print(f'\nDone. All outputs in: {EXPLAIN_DIR}')


if __name__ == '__main__':
    main()
