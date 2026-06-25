# Reads every results_*seg_*.json from the Vident segmentation runs and makes,
# for EACH model x regime: an IoU/Dice curve and a loss curve.
# Plus a combined val-IoU comparison and a summary table. No re-training needed.
#
#   pip install matplotlib
#   python plot_seg_results.py

import os, json, glob, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = r'C:\Users\ESAI406_student2\Vident_results'   # same folder the runs wrote to
PLOT_DIR = os.path.join(SAVE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

LABEL = {'video_swin_b': 'Video Swin-B', 'vivit_b': 'ViViT-B',
         'swin': 'Video Swin-B', 'vivit': 'ViViT-B'}

def main():
    files = sorted(glob.glob(os.path.join(SAVE_DIR, 'results_*.json')))
    if not files:
        raise SystemExit(f'No results_*.json found in {SAVE_DIR}')

    summary_rows = []     # for the printed table + CSV
    iou_curves   = []     # (tag, epochs, val_iou) for the combined plot

    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        log = data.get('epoch_log') or []
        # segmentation logs have val_iou; skip anything that isn't this format
        if not log or 'val_iou' not in log[0]:
            print(f'skip (not a segmentation result): {os.path.basename(fp)}')
            continue

        model  = data.get('model', 'model')
        regime = data.get('regime', '?')
        epochs     = [e['epoch'] for e in log]
        val_iou    = [e['val_iou']  for e in log]
        val_dice   = [e['val_dice'] for e in log]
        train_loss = [e.get('train_loss') for e in log]
        val_loss   = [e.get('val_loss') for e in log]
        has_vloss  = all(v is not None for v in val_loss) and len(val_loss) > 0
        best_ep    = data.get('best_epoch')
        best_iou   = data.get('best_iou')

        name = LABEL.get(model, model)
        tag  = f'{name} - {regime.upper()} FT'
        base = f'{model}_{regime}'

        # ---------- IoU + Dice ("accuracy" equivalent for segmentation) ----------
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, val_iou,  'o-', label='Val IoU')
        plt.plot(epochs, val_dice, 's-', label='Val Dice')
        if best_ep:
            plt.axvline(best_ep, ls='--', color='gray', alpha=.7, label=f'Best epoch ({best_ep})')
        plt.title(f'Segmentation quality - {tag}')
        plt.xlabel('Epoch'); plt.ylabel('Score'); plt.ylim(0, 1)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{base}_iou_dice.png'), dpi=150); plt.close()

        # ---------- Error (loss) ----------
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, train_loss, 'o-', label='Train loss')
        if has_vloss:
            plt.plot(epochs, val_loss, 's-', label='Val loss')
        if best_ep:
            plt.axvline(best_ep, ls='--', color='gray', alpha=.7, label=f'Best epoch ({best_ep})')
        plt.title(f'Error (BCE + Dice loss) - {tag}')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{base}_loss.png'), dpi=150); plt.close()

        print(f'plotted {base}: iou_dice + loss')
        iou_curves.append((tag, epochs, val_iou))
        best_dice = max(e['val_dice'] for e in log)
        summary_rows.append([name, regime.upper(),
                             f'{best_iou:.4f}' if best_iou is not None else '',
                             f'{best_dice:.4f}', best_ep,
                             f'{data.get("ft_time_h", 0):.2f}'])

    # ---------- Combined val-IoU comparison ----------
    if len(iou_curves) > 1:
        plt.figure(figsize=(8, 5))
        for tag, ep, vi in iou_curves:
            plt.plot(ep, vi, 'o-', label=tag)
        plt.title('Validation IoU - all runs')
        plt.xlabel('Epoch'); plt.ylabel('Val IoU'); plt.ylim(0, 1)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'ALL_val_iou.png'), dpi=150); plt.close()
        print('plotted combined: ALL_val_iou.png')

    # ---------- Summary table (printed + CSV) ----------
    if summary_rows:
        hdr = ['Model', 'Regime', 'Best IoU', 'Best Dice', 'Best Epoch', 'Time (h)']
        print('\n' + '  '.join(f'{h:<12}' for h in hdr))
        print('-' * 74)
        for r in summary_rows:
            print('  '.join(f'{str(c):<12}' for c in r))
        with open(os.path.join(PLOT_DIR, 'summary.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(summary_rows)
        print('\nsaved summary.csv')

    print('done ->', PLOT_DIR)

if __name__ == '__main__':
    main()
