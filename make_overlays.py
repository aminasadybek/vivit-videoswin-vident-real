# Mask-overlay generator for the presentation. NO TRAINING - loads your saved best
# checkpoints and runs them on a few val frames, writing comparison panels:
#     [ input | ground truth | Video Swin pred | ViViT pred ]
#
# Run:  C:\ProgramData\Miniconda3\python.exe make_overlays.py
# Output: <SAVE_DIR>\overlays\overlay_00.png ... (drop these into your slides)

import os, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from transformers import VivitModel

# =============================== CONFIG ===============================
ROOT       = r'C:\Users\ESAI406_student2\Vident-real'
SAVE_DIR   = r'C:\Users\ESAI406_student2\Vident_results'
SWIN_CKPT  = os.path.join(SAVE_DIR, 'swinseg_full_best.pth')    # Video Swin Full FT best
VIVIT_CKPT = os.path.join(SAVE_DIR, 'vivitseg_full_best.pth')   # ViViT Full FT best
OUT_DIR    = os.path.join(SAVE_DIR, 'overlays')

N_SAMPLES  = 6            # how many comparison panels to make (one per val video)
CLIP_LEN   = 32
IMG_SIZE   = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = ('.jpg', '.jpeg', '.png')

# =============================== MODELS (must match the training scripts) ===============================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
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
        self.d3 = DecoderBlock(1024, 512, 512); self.d2 = DecoderBlock(512, 256, 256); self.d1 = DecoderBlock(256, 128, 128)
        self.up = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        feats = {}; h = self.pos_drop(self.patch_embed(x))
        for i, blk in enumerate(self.features):
            h = blk(h)
            if i in self.TAPS: feats[i] = h.permute(0, 4, 1, 2, 3).mean(dim=2)
        d = self.d3(feats[6], feats[4]); d = self.d2(d, feats[2]); d = self.d1(d, feats[0])
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False); d = self.up(d)
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(d)

class VivitSeg(nn.Module):
    GRID = 14
    def __init__(self):
        super().__init__()
        self.backbone = VivitModel.from_pretrained('google/vivit-b-16x2-kinetics400')
        def up(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.dec = nn.ModuleList([up(768, 256), up(256, 128), up(128, 64), up(64, 32)]); self.head = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        tok = self.backbone(pixel_values=x).last_hidden_state[:, 1:, :]
        B, N, C = tok.shape; Tt = N // (self.GRID * self.GRID)
        d = tok.reshape(B, Tt, self.GRID, self.GRID, C).mean(1).permute(0, 3, 1, 2)
        for blk in self.dec:
            d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False); d = blk(d)
        return self.head(d)

# =============================== DATA ===============================
def pick_samples(split_root, n):
    """One center-frame sample from each of the first n val videos."""
    vids = sorted(d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d)))[:n]
    out = []
    for v in vids:
        idir = os.path.join(split_root, v, 'input'); mdir = os.path.join(split_root, v, 'masks')
        if not (os.path.isdir(idir) and os.path.isdir(mdir)): continue
        frames = sorted(f for f in os.listdir(idir) if f.lower().endswith(IMG_EXTS))
        mask_by_stem = {os.path.splitext(m)[0]: m for m in os.listdir(mdir)}
        ci = len(frames) // 2                                   # middle frame of the video
        stem = os.path.splitext(frames[ci])[0]
        if stem not in mask_by_stem: continue
        half = CLIP_LEN // 2
        idxs = [min(max(ci + d, 0), len(frames) - 1) for d in range(-half, CLIP_LEN - half)]
        out.append(([os.path.join(idir, frames[k]) for k in idxs],
                    os.path.join(idir, frames[ci]),
                    os.path.join(mdir, mask_by_stem[stem])))
    return out

def load_clip(clip_paths):
    fr = []
    for p in clip_paths:
        im = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
        fr.append((im.astype(np.float32) / 255.0 - MEAN) / STD)
    return torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float()[None]   # (1,C,T,H,W)

def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (IMG_SIZE, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

@torch.no_grad()
def predict(model, clip):
    with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.bfloat16):
        logits = model(clip.to(device))
    return (torch.sigmoid(logits)[0, 0] > 0.5).float().cpu().numpy()

def main():
    assert os.path.isfile(SWIN_CKPT),  f'missing {SWIN_CKPT}'
    assert os.path.isfile(VIVIT_CKPT), f'missing {VIVIT_CKPT}'
    os.makedirs(OUT_DIR, exist_ok=True)

    print('loading Video Swin ...');  swin  = SwinSeg().to(device).eval()
    swin.load_state_dict(torch.load(SWIN_CKPT, map_location=device))
    print('loading ViViT ...');       vivit = VivitSeg().to(device).eval()
    vivit.load_state_dict(torch.load(VIVIT_CKPT, map_location=device))

    samples = pick_samples(os.path.join(ROOT, 'val'), N_SAMPLES)
    print(f'making {len(samples)} overlay panels ...')
    for i, (clip_paths, center_path, mask_path) in enumerate(samples):
        clip = load_clip(clip_paths)
        inp  = cv2.resize(cv2.imread(center_path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))   # BGR for display
        gt   = cv2.cvtColor((cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        sw   = cv2.cvtColor((predict(swin,  clip) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vv   = cv2.cvtColor((predict(vivit, clip) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panel = np.concatenate([label(inp, 'Input'), label(gt, 'Ground truth'),
                                label(sw, 'Video Swin'), label(vv, 'ViViT')], axis=1)
        out = os.path.join(OUT_DIR, f'overlay_{i:02d}.png')
        cv2.imwrite(out, panel); print('  saved', out)
    print('DONE ->', OUT_DIR)

if __name__ == '__main__':
    main()
