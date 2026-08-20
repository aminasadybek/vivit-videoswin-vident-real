# Vident-real DENTAL QUADRANT classification (Q1-Q4) - ViViT-B & Video Swin-B, Full & LoRA.
# Per-video labels from the spreadsheet. Protocol: select model by VAL accuracy, report TEST accuracy.
# Handles severe class imbalance (class-weighted loss + balanced/per-class accuracy).
#
#   MODEL = 'swin'|'vivit'   REGIME = 'full'|'lora'   -> run all four, pick best VAL, report its TEST.
#
# Install:  C:\ProgramData\Miniconda3\python.exe -m pip install peft transformers pynvml openpyxl
# Run:      C:\ProgramData\Miniconda3\python.exe vident_classify.py
# FIRST RUN: keep SMOKE_TEST = True.

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import time, json, random, threading, warnings
from collections import defaultdict, Counter
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import pynvml
import openpyxl
from scipy.optimize import linear_sum_assignment
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from transformers import VivitForVideoClassification
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
cv2.setNumThreads(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================== CONFIG ===============================
ROOT        = r'C:\Users\ESAI406_student2\Vident-real'
LABELS_XLSX = r'C:\Users\ESAI406_student2\Vident-real classification label (Dental location).xlsx'
SAVE_DIR    = r'C:\Users\ESAI406_student2\Vident_cls_results'

MODEL      = 'vivit'               # 'swin' | 'vivit'
REGIME     = 'full'               # 'full' | 'lora'
SMOKE_TEST = False                 # few videos / 1 epoch sanity run; flip to False for the real run

N_CLASSES  = 4                    # Q1..Q4
CLIP_LEN   = 32                   # ViViT requires 32; Swin matches
FRAME_STRIDE = 2                  # temporal stride WITHIN a clip (papers use 32x2); auto-falls to 1 for ranges <64 frames
IMG_SIZE   = 224
BATCH_SIZE = 4
NUM_EPOCHS = 15
LR         = 1e-4 if REGIME == 'full' else 5e-4
TRAIN_STRIDE = 16                 # clip every N frames per train video (overlapping windows -> more samples)
EVAL_STRIDE  = 32                 # non-overlapping clips per val/test video (aggregated to a video prediction)
NUM_WORKERS  = 4
USE_AMP    = True
AMP_DTYPE  = torch.bfloat16
LORA_R, LORA_ALPHA = 8, 16
# Stabilizers (OFF = same recipe; enable for ALL four if used). NOTE: with only 10 val videos,
# val accuracy moves in ~10% steps regardless - that coarseness is a separate cause of "fluctuation".
USE_SCHEDULER = False             # cosine LR decay
GRAD_CLIP     = None              # e.g. 1.0
if SMOKE_TEST:
    NUM_EPOCHS = 1

QMAP = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}
IDQ  = {v: k for k, v in QMAP.items()}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = ('.jpg', '.jpeg', '.png')
NAME = {'swin': 'Video Swin-B', 'vivit': 'ViViT-B'}[MODEL]

# =============================== POWER ===============================
class PowerMonitor:
    def __init__(self, gpu=0, interval=0.25):
        pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu); self.interval = interval
    def _loop(self):
        while not self._stop:
            try: self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.h)/1000.0)
            except Exception: pass
            time.sleep(self.interval)
    def start(self): self._stop, self.samples = False, []; self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()
    def stop(self): self._stop = True; self.t.join(); return float(np.mean(self.samples)) if self.samples else 0.0

# =============================== LABELS + FOLDER MATCHING ===============================
def _parse_range(rng, total):
    """Parse a 'start-end' Specified-Frames string into (start, end), 1-based inclusive.
    Excel turns ranges like '1-28' into dates, and those are always single full-video segments,
    so fall back to (1, total) when the cell is not a 'a-b' string."""
    if isinstance(rng, str) and '-' in rng:
        a, b = rng.split('-')[:2]
        try:
            return int(a), int(b)
        except ValueError:
            pass
    return 1, int(total)                                       # datetime / blank -> whole video

def parse_labels(xlsx):
    """Per split: list of videos, each {'total': #frames, 'ranges': [(start, end, label), ...]}.
    Uses the 'Specified Frames' column (col 5) so only annotated frames are used; a video can have
    several ranges (continuation rows where the video number is blank)."""
    ws = openpyxl.load_workbook(xlsx, data_only=True)['Modified (Location)']
    split = None; out = {'train': [], 'val': [], 'test': []}; cur = None
    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1).value
        if a:
            if 'Train' in str(a): split = 'train'
            elif 'Validation' in str(a): split = 'val'
            elif 'Test' in str(a): split = 'test'
        vid, nfr, rng, quad = (ws.cell(r, 3).value, ws.cell(r, 4).value,
                               ws.cell(r, 5).value, ws.cell(r, 6).value)
        if quad not in QMAP or not split:
            continue
        if vid is not None:                                   # new video row (has #frames)
            cur = {'total': int(nfr), 'ranges': []}
            out[split].append(cur)
        if cur is None:
            continue
        start, end = _parse_range(rng, cur['total'])
        cur['ranges'].append((start, end, QMAP[quad]))
    return out

def match_folders(split_dir, videos):
    """Map each hex folder -> its video (total frames + ranges) via OPTIMAL bipartite
    matching (Hungarian algorithm) on frame-count distance, not greedy first-fit.
    Greedy alphabetical-order matching let an earlier folder 'steal' another folder's
    exact frame-count match, cascading into wrong INEXACT pairings and, when there are
    more folders on disk than labeled rows, forcing leftover folders into a fake match
    against an arbitrary already-used row (silently wrong). This version finds the
    assignment that minimizes TOTAL |folder_frames - label_frames| across all pairs at
    once, and explicitly reports any folder that has no labeled row at all instead of
    forcing a bogus pairing."""
    folders = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    fcount = {}
    for f in folders:
        idir = os.path.join(split_dir, f, 'input')
        fcount[f] = len([x for x in os.listdir(idir) if x.lower().endswith(IMG_EXTS)]) if os.path.isdir(idir) else 0

    nF, nV = len(folders), len(videos)
    cost = np.array([[abs(fcount[f] - v['total']) for v in videos] for f in folders], dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost)  # globally optimal, handles nF != nV natively

    matched_rows = set(row_ind.tolist())
    mapping = []
    print(f'  matching {nF} folders to {nV} videos in {os.path.basename(split_dir)} (optimal assignment):')
    for i, j in zip(row_ind, col_ind):
        f = folders[i]; v = videos[j]; n = fcount[f]; d = int(cost[i, j])
        lbl = v['ranges'][0][2]                               # all ranges in a video share the quadrant
        covered = sum(e - s + 1 for s, e, _ in v['ranges'])
        mapping.append((os.path.join(split_dir, f), v['total'], v['ranges']))
        flag = '' if d == 0 else f'  <-- INEXACT (folder {n} vs label {v["total"]}, off {d})'
        note = '' if covered >= n else f'  [uses {covered}/{n} specified frames]'
        print(f'    {f[:8]}.. frames={n:<6} -> {IDQ[lbl]}{note}{flag}')

    unmatched = [folders[i] for i in range(nF) if i not in matched_rows]
    if unmatched:
        print(f'  *** {len(unmatched)} folder(s) have NO labeled row in this split at all '
              f'(more folders on disk than labeled videos) -- DROPPED, not guessed: {unmatched}')
    return mapping

# =============================== CLIP SAMPLING ===============================
def flatten_segments(videos):
    """Each Specified-Frames range becomes its OWN example (prof: treat segments as videos).
    Returns a flat list of (folder, start, end, label); its index is the example id."""
    segs = []
    for (folder, total, ranges) in videos:
        for (start, end, label) in ranges:
            segs.append((folder, start, end, label))
    return segs

def build_clips(segments, stride):
    """Return list of (clip_paths, label, seg_id). Each segment is an independent example;
    clips are sampled ONLY within its frame range, and seg_id = the segment's index."""
    samples = []
    for seg_id, (folder, start, end, label) in enumerate(segments):
        idir = os.path.join(folder, 'input')
        frames = sorted(f for f in os.listdir(idir) if f.lower().endswith(IMG_EXTS))
        n = len(frames)
        if n == 0: continue
        s0, e0 = max(0, start - 1), min(end, n)               # 1-based inclusive -> 0-based slice
        seg = frames[s0:e0]
        m = len(seg)
        if m == 0: continue
        # Temporal sampling stride WITHIN a clip (the "x2" in vivit-b-16x2 / Swin 32x2):
        # use 2 only when the segment is long enough to fill a 32x2 clip (>=64 frames),
        # else fall back to 1 so short segments aren't padded with a repeated frame.
        fs = FRAME_STRIDE if m >= CLIP_LEN * FRAME_STRIDE else 1
        span = CLIP_LEN * fs                                  # frames a clip reaches across
        starts = list(range(0, max(1, m - span + 1), stride)) or [0]
        for s in starts:
            idxs = [min(s + d * fs, m - 1) for d in range(CLIP_LEN)]
            samples.append(([os.path.join(idir, seg[k]) for k in idxs], label, seg_id))
    return samples

def _load(p):
    im = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    return (im.astype(np.float32) / 255.0 - MEAN) / STD

class ClipDataset(Dataset):
    def __init__(self, samples, train=False): self.samples, self.train = samples, train
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        paths, label, vid = self.samples[i]
        fr = [_load(p) for p in paths]
        if self.train and random.random() < 0.5:
            fr = [np.ascontiguousarray(f[:, ::-1]) for f in fr]
        x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float()   # (C,T,H,W)
        return x, label, vid

def collate(b):
    b = [z for z in b if z is not None]
    return (torch.stack([z[0] for z in b]), torch.tensor([z[1] for z in b]), torch.tensor([z[2] for z in b]))

# =============================== MODEL ===============================
class Wrap(nn.Module):
    def __init__(self, model, is_hf): super().__init__(); self.model, self.is_hf = model, is_hf
    def forward(self, x):
        if self.is_hf:
            return self.model(pixel_values=x.permute(0, 2, 1, 3, 4).contiguous()).logits
        return self.model(x)

def build_model(model_name, regime):
    if model_name == 'swin':
        net = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
        net.head = nn.Linear(net.head.in_features, N_CLASSES)
        inc, is_hf = ['mlp'], False
    else:
        net = VivitForVideoClassification.from_pretrained('google/vivit-b-16x2-kinetics400',
                                                          num_labels=N_CLASSES, ignore_mismatched_sizes=True)
        inc, is_hf = ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2'], True
    if regime == 'lora':
        for p in net.parameters(): p.requires_grad = False
        save = ['head'] if model_name == 'swin' else ['classifier']
        targets = [n for n, m in net.named_modules() if isinstance(m, nn.Linear) and any(t in n for t in inc)]
        assert targets, 'No LoRA targets matched'
        net = get_peft_model(net, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                                             target_modules=targets, modules_to_save=save))
        net.print_trainable_parameters()
    return Wrap(net, is_hf).to(device)

# =============================== EVAL (video-level) ===============================
@torch.no_grad()
def evaluate(model, loader, n_videos, video_labels, weight=None):
    """Aggregate clip logits per video -> video prediction. Also clip-level loss. Returns metrics."""
    model.eval()
    logit_sum = torch.zeros(n_videos, N_CLASSES)
    loss_tot = clip_n = clip_correct = 0.0
    clip_per_class = np.zeros(N_CLASSES); clip_per_total = np.zeros(N_CLASSES)
    for x, y, vid in loader:
        if x is None: continue
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            lo = model(x.to(device)).float().cpu()
        loss_tot += F.cross_entropy(lo, y, weight=weight.cpu() if weight is not None else None,
                                    reduction='sum').item()
        cp = lo.argmax(1)
        clip_correct += (cp == y).sum().item()
        for c in range(N_CLASSES):
            mc = (y == c)
            clip_per_total[c] += mc.sum().item()
            clip_per_class[c] += ((cp == y) & mc).sum().item()
        clip_n += x.size(0)
        for k in range(x.size(0)):
            logit_sum[vid[k]] += torch.softmax(lo[k], 0)
    pred = logit_sum.argmax(1).numpy()
    true = np.array(video_labels)
    overall = float((pred == true).mean())
    per_class, conf = {}, np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(true, pred): conf[t, p] += 1
    bal = []
    for c in range(N_CLASSES):
        m = true == c
        if m.sum(): acc = float((pred[m] == c).mean()); per_class[IDQ[c]] = round(acc, 3); bal.append(acc)
    cbal = [a for a in (clip_per_class[c] / clip_per_total[c]
            for c in range(N_CLASSES) if clip_per_total[c]) ]
    model.train()
    return {'overall_acc': overall, 'balanced_acc': float(np.mean(bal)) if bal else 0.0,
            'per_class': per_class, 'confusion': conf.tolist(),
            'loss': loss_tot / max(clip_n, 1), 'clip_acc': clip_correct / max(clip_n, 1),
            'clip_balanced_acc': float(np.mean(cbal)) if cbal else 0.0,
            'n_clips': int(clip_n)}

# =============================== MAIN ===============================
def main():
    assert os.path.isdir(ROOT) and os.path.isfile(LABELS_XLSX), 'check ROOT / LABELS_XLSX paths'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'{NAME} | regime={REGIME} | smoke={SMOKE_TEST} | {N_CLASSES} classes (Q1-Q4)')

    labels = parse_labels(LABELS_XLSX)
    fl = {sp: match_folders(os.path.join(ROOT, sp), labels[sp]) for sp in ['train', 'val', 'test']}
    if SMOKE_TEST:
        fl = {sp: v[:4] for sp, v in fl.items()}

    # each Specified-Frames range = one independent example ("segment treated as a video")
    seg = {sp: flatten_segments(fl[sp]) for sp in ['train', 'val', 'test']}
    tr = build_clips(seg['train'], TRAIN_STRIDE)
    va = build_clips(seg['val'],   EVAL_STRIDE)
    te = build_clips(seg['test'],  EVAL_STRIDE)
    va_labels = [lb for _, _, _, lb in seg['val']]             # one label per segment
    te_labels = [lb for _, _, _, lb in seg['test']]
    print(f'\nsegments -> train {len(seg["train"])} ({len(fl["train"])} videos) | '
          f'val {len(seg["val"])} ({len(fl["val"])} videos) | test {len(seg["test"])} ({len(fl["test"])} videos)')
    print(f'clips -> train {len(tr)} | val {len(va)} | test {len(te)}')
    print('train clip class balance:', dict(sorted(Counter(l for _, l, _ in tr).items())))
    # how many segments are long enough for stride-2 vs fall back to stride-1
    _rlens = [e - s + 1 for sp in ['train', 'val', 'test'] for _, s, e, _ in seg[sp]]
    _s2 = sum(1 for L in _rlens if L >= CLIP_LEN * FRAME_STRIDE)
    print(f'frame sampling: {_s2}/{len(_rlens)} ranges use stride {FRAME_STRIDE} (>={CLIP_LEN*FRAME_STRIDE} frames), '
          f'{len(_rlens)-_s2} fall back to stride 1')

    # class weights (inverse frequency) for the imbalance
    cc = Counter(l for _, l, _ in tr)
    w = torch.tensor([len(tr) / (N_CLASSES * cc.get(c, 1)) for c in range(N_CLASSES)], dtype=torch.float, device=device)
    ce = nn.CrossEntropyLoss(weight=w)

    lk = dict(batch_size=BATCH_SIZE, collate_fn=collate, pin_memory=True)
    if NUM_WORKERS > 0: lk.update(num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=4)
    tl = DataLoader(ClipDataset(tr, True),  shuffle=True,  drop_last=True, **lk)
    vl = DataLoader(ClipDataset(va, False), shuffle=False, **lk)
    el = DataLoader(ClipDataset(te, False), shuffle=False, **lk)

    model = build_model(MODEL, REGIME)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS) if USE_SCHEDULER else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    ckpt = os.path.join(SAVE_DIR, f'{MODEL}_{REGIME}_cls_best.pth')
    best = {'val': -1.0, 'epoch': 0, 'clip': 0.0, 'clipbal': 0.0}; log = []
    print('\n  [(clip) = averaged over individual 32-frame clips | (seg) = clips aggregated per segment;'
          ' loss is always clip-level, model is selected on val_acc(seg)]')
    torch.cuda.reset_peak_memory_stats(); pm = PowerMonitor(); pm.start(); t0 = time.time()
    for ep in range(NUM_EPOCHS):
        model.train(); run = seen = tr_correct = 0
        for x, y, _ in tqdm(tl, desc=f'epoch {ep+1}/{NUM_EPOCHS}'):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                out = model(x); loss = ce(out, y)
            opt.zero_grad(); loss.backward()
            if GRAD_CLIP: torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)
            tr_correct += (out.detach().argmax(1) == y).sum().item()
        if sched: sched.step()
        v = evaluate(model, vl, len(va_labels), va_labels, weight=w)
        tr_acc = tr_correct / max(seen, 1)
        log.append({'epoch': ep+1, 'train_loss': run/max(seen,1), 'train_acc': tr_acc,
                    'val_loss': v['loss'], 'val_clip_acc': v['clip_acc'],
                    'val_acc': v['overall_acc'], 'val_bal_acc': v['balanced_acc']})
        if v['overall_acc'] > best['val']:
            best.update(val=v['overall_acc'], epoch=ep+1, clip=v['clip_acc'],
                        clipbal=v['clip_balanced_acc']); torch.save(model.state_dict(), ckpt)
        print(f'  ep {ep+1:2d} | train_loss(clip) {log[-1]["train_loss"]:.4f}  train_acc(clip) {tr_acc:.4f}  '
              f'val_loss(clip) {v["loss"]:.4f}  val_acc(clip) {v["clip_acc"]:.4f}  '
              f'val_acc(seg) {v["overall_acc"]:.4f}  val_bal_acc(seg) {v["balanced_acc"]:.4f}  '
              f'-> best val_acc(seg) {best["val"]:.4f} @ ep{best["epoch"]}')
    ft_time = time.time() - t0; avg_pw = pm.stop(); peak = torch.cuda.max_memory_allocated()/1e9

    # ---- TEST with the best (val-selected) checkpoint ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    pm2 = PowerMonitor(); pm2.start(); ti = time.time()
    test = evaluate(model, el, len(te_labels), te_labels, weight=w)
    inf_t = time.time() - ti; inf_pw = pm2.stop()
    n_test_clips = test['n_clips']

    SEP = '=' * 70
    print('\n' + SEP)
    print(f'  RESULTS: vident-real quadrant cls | {NAME} | {REGIME} | {NUM_EPOCHS}ep')
    print(SEP)
    rows = [
        ('VAL accuracy (per-segment)',        f'{best["val"]:.4f}  (ep {best["epoch"]})'),
        ('VAL accuracy (per-clip)',         f'{best["clip"]:.4f}'),
        ('VAL balanced accuracy (per-clip)', f'{best["clipbal"]:.4f}'),
        ('TEST accuracy (per-segment)',       f'{test["overall_acc"]:.4f}'),
        ('TEST accuracy (per-clip)',        f'{test["clip_acc"]:.4f}'),
        ('TEST balanced accuracy (per-segment)', f'{test["balanced_acc"]:.4f}'),
        ('TEST balanced accuracy (per-clip)', f'{test["clip_balanced_acc"]:.4f}'),
        ('TEST per-class',                 str(test['per_class'])),
        ('FT Time (h)',                    f'{ft_time/3600:.2f}'),
        ('Test clips (N)',                 f'{n_test_clips}'),
        ('Inf Time, full test set (s)',    f'{inf_t:.1f}'),
        ('Inf Throughput (clips/s)',       f'{n_test_clips/max(inf_t,1e-9):.1f}'),
        ('Peak VRAM Train (GB)',           f'{peak:.2f}'),
        ('Avg Power Train (W)',            f'{avg_pw:.1f}'),
        ('Energy Train (Wh)',              f'{avg_pw*ft_time/3600:.2f}'),
        ('Avg Power Inf (W)',              f'{inf_pw:.1f}'),
        (f'Energy Inf, {n_test_clips} clips (Wh)', f'{inf_pw*inf_t/3600:.4f}'),
    ]
    for k, vv in rows: print(f'  {k:<32} {vv}')
    print('  TEST confusion [rows=true Q1..Q4]:'); [print('   ', r) for r in test['confusion']]
    print(SEP)

    res = {'model': MODEL, 'regime': REGIME, 'n_classes': N_CLASSES, 'epochs': NUM_EPOCHS,
           'clip_len': CLIP_LEN, 'batch_size': BATCH_SIZE, 'smoke_test': SMOKE_TEST,
           'val_acc': best['val'], 'val_clip_acc': best['clip'], 'val_clip_balanced_acc': best['clipbal'],
           'val_best_epoch': best['epoch'],
           'test_acc': test['overall_acc'], 'test_clip_acc': test['clip_acc'],
           'test_balanced_acc': test['balanced_acc'], 'test_clip_balanced_acc': test['clip_balanced_acc'],
           'test_per_class': test['per_class'], 'test_confusion': test['confusion'],
           'ft_time_h': ft_time/3600, 'n_test_clips': n_test_clips,
           'inf_time_sec': inf_t, 'inf_throughput_cps': n_test_clips/max(inf_t, 1e-9),
           'avg_power_train_w': avg_pw, 'energy_train_wh': avg_pw*ft_time/3600,
           'avg_power_inf_w': inf_pw, 'energy_inf_wh': inf_pw*inf_t/3600,
           'peak_vram_train_gb': peak, 'epoch_log': log}
    out = os.path.join(SAVE_DIR, f'results_cls_{MODEL}_{REGIME}.json')
    with open(out, 'w') as f: json.dump(res, f, indent=2)
    print(f'DONE - VAL {best["val"]:.4f} | TEST {test["overall_acc"]:.4f} (bal {test["balanced_acc"]:.4f}) | saved -> {out}')

if __name__ == '__main__':
    main()