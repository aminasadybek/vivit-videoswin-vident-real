# Vident-real teeth segmentation - VIDEO SWIN-B only, full metric suite.
#   REGIME = 'full' | 'lora'  -> run both cells.
# Captures: IoU/Dice, FT time, FULL-DATASET inference (total time + throughput + per-clip),
# peak VRAM, avg power (NVML) + energy (Wh) for train and inference, per-epoch train/val loss.
#
# Install:  C:\ProgramData\Miniconda3\python.exe -m pip install peft pynvml
# Run:      C:\ProgramData\Miniconda3\python.exe vident_swin_seg_metrics.py
# FIRST RUN: keep SMOKE_TEST = True.

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time, json, random, threading, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import pynvml

from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
cv2.setNumThreads(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================== CONFIG ===============================
ROOT      = r'C:\Users\ESAI406_student2\Vident-real'      # local copy (copy off the shared drive first)
SAVE_DIR  = r'C:\Users\ESAI406_student2\Vident_results'

MODEL      = 'swin'               # this file is Video Swin-B only
REGIME     = 'lora'               # 'full' | 'lora'    <- run both
SMOKE_TEST = False                 # 2 videos / 1 epoch sanity run; flip to False for the real run

CLIP_LEN   = 32                   # match the ViViT file (32) so the two models' efficiency metrics are comparable
IMG_SIZE   = 224
BATCH_SIZE = 4
NUM_EPOCHS = 15
LR         = 1e-4 if REGIME == 'full' else 5e-4
FRAME_STRIDE = 5
NUM_WORKERS  = 4                  # set 0 if the loader hangs on Windows
USE_AMP    = True
AMP_DTYPE  = torch.bfloat16
LORA_R, LORA_ALPHA = 8, 16

# Stabilizers for the Swin fluctuation (OFF = same recipe; enable for ALL four if you use them)
USE_SCHEDULER = False             # cosine LR decay
GRAD_CLIP     = None            # e.g. 1.0

if SMOKE_TEST:
    MAX_VIDEOS, FRAME_STRIDE, NUM_EPOCHS = 2, 20, 1

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = ('.jpg', '.jpeg', '.png')
NAME = 'Video Swin-B'

# =============================== POWER (NVML) ===============================
class PowerMonitor:
    def __init__(self, gpu=0, interval=0.25):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu); self.interval = interval
    def _loop(self):
        while not self._stop:
            try: self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
            except Exception: pass
            time.sleep(self.interval)
    def start(self):
        self._stop, self.samples = False, []
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()
    def stop(self):
        self._stop = True; self.t.join()
        return float(np.mean(self.samples)) if self.samples else 0.0

# =============================== DATA ===============================
def index_clips(split_root, stride, max_videos=None):
    vids = sorted(d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d)))
    if max_videos: vids = vids[:max_videos]
    samples = []
    for v in vids:
        idir = os.path.join(split_root, v, 'input'); mdir = os.path.join(split_root, v, 'masks')
        if not (os.path.isdir(idir) and os.path.isdir(mdir)): continue
        mask_by_stem = {os.path.splitext(m)[0]: m for m in os.listdir(mdir)}
        frames = sorted(f for f in os.listdir(idir) if f.lower().endswith(IMG_EXTS))
        n = len(frames)
        for ci in range(0, n, stride):
            stem = os.path.splitext(frames[ci])[0]
            if stem not in mask_by_stem: continue
            half = CLIP_LEN // 2
            idxs = [min(max(ci + d, 0), n - 1) for d in range(-half, CLIP_LEN - half)]
            samples.append(([os.path.join(idir, frames[k]) for k in idxs],
                            os.path.join(mdir, mask_by_stem[stem])))
    return samples

def _load_img(p):
    img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 255.0 - MEAN) / STD

class ClipSegDataset(Dataset):
    def __init__(self, samples, train=False): self.samples, self.train = samples, train
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        clip_paths, mp = self.samples[idx]
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None: return None
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        frames = [_load_img(p) for p in clip_paths]
        if self.train and random.random() < 0.5:
            frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames]
            mask   = np.ascontiguousarray(mask[:, ::-1])
        x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float()   # (C,T,H,W)
        y = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, y

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

# =============================== MODELS ===============================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None: x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SwinSeg(nn.Module):
    TAPS = [0, 2, 4, 6]
    def __init__(self):
        super().__init__()
        net = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
        self.patch_embed, self.pos_drop, self.features = net.patch_embed, net.pos_drop, net.features
        self.d3 = DecoderBlock(1024, 512, 512)
        self.d2 = DecoderBlock(512, 256, 256)
        self.d1 = DecoderBlock(256, 128, 128)
        self.up = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False),
                                nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        feats = {}
        h = self.pos_drop(self.patch_embed(x))
        for i, blk in enumerate(self.features):
            h = blk(h)
            if i in self.TAPS:
                feats[i] = h.permute(0, 4, 1, 2, 3).mean(dim=2)   # channels-last -> (B,C,H,W), temporal mean
        d = self.d3(feats[6], feats[4]); d = self.d2(d, feats[2]); d = self.d1(d, feats[0])
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d = self.up(d)
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(d)

def build_model(model_name, regime):
    model = SwinSeg()
    if regime == 'full':
        for p in model.parameters(): p.requires_grad = True
    elif regime == 'lora':
        for p in model.parameters(): p.requires_grad = False
        targets = [n for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and 'mlp' in n]      # Swin FFN layers
        assert targets, 'No LoRA targets matched'
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                         target_modules=targets,
                         modules_to_save=['d3', 'd2', 'd1', 'up', 'head'])  # decoder trains fully
        model = get_peft_model(model, cfg); model.print_trainable_parameters()
    return model.to(device)

# =============================== LOSS / METRICS ===============================
bce = nn.BCEWithLogitsLoss()
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    return (1 - (2 * (p * target).sum((2, 3)) + eps) / (p.sum((2, 3)) + target.sum((2, 3)) + eps)).mean()
def seg_loss(logits, target):
    return bce(logits, target) + dice_loss(logits, target)
@torch.no_grad()
def iou_dice(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum((2, 3)); psum = pred.sum((2, 3)) + target.sum((2, 3))
    return (((inter + 1e-6) / (psum - inter + 1e-6)).mean().item(),
            ((2 * inter + 1e-6) / (psum + 1e-6)).mean().item())

@torch.no_grad()
def quick_val(model, loader):
    """Per-epoch val: IoU, Dice, loss."""
    model.eval(); ious = dices = losses = n = 0
    for x, y in loader:
        if x is None: continue
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(x); loss = seg_loss(logits, y)
        i, d = iou_dice(logits, y); bs = x.size(0)
        ious += i * bs; dices += d * bs; losses += loss.item() * bs; n += bs
    model.train()
    return ious / max(n, 1), dices / max(n, 1), losses / max(n, 1)

@torch.no_grad()
def run_inference(model, loader):
    """Timed FULL-DATASET inference pass with power + peak VRAM."""
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    for x, y in loader:                       # warm-up
        if x is None: continue
        _ = model(x.to(device)); break
    torch.cuda.synchronize()
    ious = dices = n = 0
    pm = PowerMonitor(); pm.start(); t0 = time.time()
    for x, y in tqdm(loader, desc='inference', leave=False):
        if x is None: continue
        x = x.to(device); y = y.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = model(x)
        i, d = iou_dice(logits, y); bs = x.size(0)
        ious += i * bs; dices += d * bs; n += bs
    torch.cuda.synchronize()
    inf_time = time.time() - t0; avg_pw = pm.stop()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {'val_iou': ious / max(n, 1), 'val_dice': dices / max(n, 1), 'n_eval': n,
            'inf_time_sec': inf_time, 'inf_throughput_cps': n / max(inf_time, 1e-9),
            'inf_latency_ms': 1000.0 * inf_time / max(n, 1),
            'avg_inf_power_w': avg_pw, 'peak_vram_inf_gb': peak}

# =============================== MAIN ===============================
def main():
    assert os.path.isdir(ROOT), f'ROOT not found: {ROOT}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'{NAME} | regime={REGIME} | smoke={SMOKE_TEST} | clip={CLIP_LEN} | device={device}')

    mv = MAX_VIDEOS if SMOKE_TEST else None
    tr = index_clips(os.path.join(ROOT, 'train'), FRAME_STRIDE, mv)
    va = index_clips(os.path.join(ROOT, 'val'),   FRAME_STRIDE, mv)
    print(f'train clips: {len(tr)} | val clips: {len(va)}')
    assert tr and va, 'No clips found - check ROOT / layout.'

    lk = dict(batch_size=BATCH_SIZE, collate_fn=collate, pin_memory=True)
    if NUM_WORKERS > 0: lk.update(num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=4)
    tl = DataLoader(ClipSegDataset(tr, True),  shuffle=True,  drop_last=True, **lk)
    vl = DataLoader(ClipSegDataset(va, False), shuffle=False,                 **lk)

    model = build_model(MODEL, REGIME)
    with torch.no_grad():
        xb, _ = next(iter(tl)); out = model(xb.to(device))
        print('forward OK | input', tuple(xb.shape), '-> output', tuple(out.shape))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS) if USE_SCHEDULER else None
    trainable = [p for p in model.parameters() if p.requires_grad]

    torch.cuda.reset_peak_memory_stats()
    pm = PowerMonitor(); pm.start()
    best = {'iou': -1.0, 'epoch': 0, 'path': os.path.join(SAVE_DIR, f'{MODEL}seg_{REGIME}_best.pth')}
    log = []; t0 = time.time()
    for ep in range(NUM_EPOCHS):
        model.train(); running = seen = 0
        for x, y in tqdm(tl, desc=f'epoch {ep+1}/{NUM_EPOCHS}'):
            if x is None: continue
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                loss = seg_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            if GRAD_CLIP: torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()
            running += loss.item() * x.size(0); seen += x.size(0)
        if sched: sched.step()
        v_iou, v_dice, v_loss = quick_val(model, vl)
        log.append({'epoch': ep+1, 'train_loss': running/max(seen,1),
                    'val_loss': v_loss, 'val_iou': v_iou, 'val_dice': v_dice})
        if v_iou > best['iou']:
            best.update(iou=v_iou, epoch=ep+1)
            torch.save(model.state_dict(), best['path'])
        print(f'  epoch {ep+1}: loss {log[-1]["train_loss"]:.3f}  val_loss {v_loss:.3f}  '
              f'IoU {v_iou:.4f}  Dice {v_dice:.4f}  (best {best["iou"]:.4f} @ ep{best["epoch"]})')
    ft_time = time.time() - t0
    avg_train_pw = pm.stop()
    peak_train = torch.cuda.max_memory_allocated() / 1e9

    # reload best, then full-dataset inference with power/VRAM
    model.load_state_dict(torch.load(best['path'], map_location=device))
    ev = run_inference(model, vl)

    # ---- results table ----
    SEP = '=' * 70
    print('\n' + SEP)
    print(f'  RESULTS: vident-real seg | {NAME} | regime={REGIME} | {NUM_EPOCHS}ep '
          f'| clip{CLIP_LEN} batch{BATCH_SIZE}')
    print(SEP)
    rows = [
        ('Best Val IoU',            f'{best["iou"]:.4f}  (ep {best["epoch"]})'),
        ('Best Val Dice',           f'{ev["val_dice"]:.4f}'),
        ('FT Time (h)',             f'{ft_time/3600:.2f}'),
        ('Eval clips (N)',          f'{ev["n_eval"]}'),
        ('Inf Time, full set (s)',  f'{ev["inf_time_sec"]:.1f}'),
        ('Inf Throughput (clips/s)',f'{ev["inf_throughput_cps"]:.1f}'),
        ('Inf Latency (ms/clip)',   f'{ev["inf_latency_ms"]:.1f}'),
        ('Peak VRAM Train (GB)',    f'{peak_train:.2f}'),
        ('Peak VRAM Inf (GB)',      f'{ev["peak_vram_inf_gb"]:.2f}'),
        ('Avg Power Train (W)',     f'{avg_train_pw:.1f}'),
        ('Energy Train (Wh)',       f'{avg_train_pw*ft_time/3600:.2f}'),
        ('Avg Power Inf (W)',       f'{ev["avg_inf_power_w"]:.1f}'),
        (f'Energy Inf, {ev["n_eval"]} clips (Wh)',
                                    f'{ev["avg_inf_power_w"]*ev["inf_time_sec"]/3600:.4f}'),
    ]
    for k, v in rows: print(f'  {k:<28} {v:>22}')
    print(SEP)

    results = {'model': 'video_swin_b', 'regime': REGIME,
               'clip_len': CLIP_LEN, 'img_size': IMG_SIZE, 'frame_stride': FRAME_STRIDE,
               'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE, 'smoke_test': SMOKE_TEST,
               'best_iou': best['iou'], 'best_epoch': best['epoch'], 'best_dice': ev['val_dice'],
               'ft_time_h': ft_time/3600, 'ft_time_sec': ft_time,
               'avg_power_train_w': avg_train_pw, 'energy_train_wh': avg_train_pw*ft_time/3600,
               'peak_vram_train_gb': peak_train, 'epoch_log': log, **ev,
               'energy_inf_wh': ev['avg_inf_power_w']*ev['inf_time_sec']/3600}
    out = os.path.join(SAVE_DIR, f'results_{MODEL}seg_{REGIME}.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f'DONE - best IoU {best["iou"]:.4f} | saved -> {out}')

if __name__ == '__main__':
    main()
