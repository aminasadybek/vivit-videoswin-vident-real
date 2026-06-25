# Reads every results_cls_*.json from the quadrant-classification runs and makes,
# for EACH model x regime: an accuracy curve, a loss curve, and a TEST confusion matrix.
# Plus a combined val-accuracy comparison and a summary table (val/test/balanced acc).
#
#   pip install matplotlib
#   python plot_cls_results.py

import os, json, glob, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = r'C:\Users\ESAI406_student2\Vident_cls_results'
PLOT_DIR = os.path.join(SAVE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

LABEL   = {'swin': 'Video Swin-B', 'vivit': 'ViViT-B'}
CLASSES = ['Q1', 'Q2', 'Q3', 'Q4']

def main():
    files = sorted(glob.glob(os.path.join(SAVE_DIR, 'results_cls_*.json')))
    if not files:
        raise SystemExit(f'No results_cls_*.json in {SAVE_DIR}')

    summary, acc_curves = [], []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        log = d.get('epoch_log') or []
        if not log:
            print(f'skip (no epoch_log): {os.path.basename(fp)}'); continue
        model, regime = d.get('model', '?'), d.get('regime', '?')
        tag  = f'{LABEL.get(model, model)} - {regime.upper()} FT'
        base = f'{model}_{regime}'
        epochs    = [e['epoch'] for e in log]
        val_acc   = [e.get('val_acc') for e in log]
        val_bal   = [e.get('val_bal_acc') for e in log]
        train_acc = [e.get('train_acc') for e in log]
        val_clip  = [e.get('val_clip_acc') for e in log]
        tr_loss   = [e.get('train_loss') for e in log]
        vl_loss   = [e.get('val_loss') for e in log]
        has_vloss = all(v is not None for v in vl_loss) and len(vl_loss) > 0
        best_ep   = d.get('val_best_epoch')

        # ---------- Accuracy ----------
        plt.figure(figsize=(7, 4.5))
        if any(v is not None for v in train_acc):
            plt.plot(epochs, train_acc, '^-', label='Train accuracy (clip)')
        if any(v is not None for v in val_clip):
            plt.plot(epochs, val_clip, 'o-', label='Val accuracy (clip)')
        plt.plot(epochs, val_acc, 's-', label='Val accuracy (video, reported)')
        if best_ep:
            plt.axvline(best_ep, ls='--', color='gray', alpha=.7, label=f'Best epoch ({best_ep})')
        plt.title(f'Accuracy - {tag}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.ylim(0, 1)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{base}_accuracy.png'), dpi=150); plt.close()

        # ---------- Loss ----------
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, tr_loss, 'o-', label='Train loss')
        if has_vloss:
            plt.plot(epochs, vl_loss, 's-', label='Val loss')
        if best_ep:
            plt.axvline(best_ep, ls='--', color='gray', alpha=.7, label=f'Best epoch ({best_ep})')
        plt.title(f'Error (cross-entropy loss) - {tag}'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{base}_loss.png'), dpi=150); plt.close()

        # ---------- Test confusion matrix ----------
        conf = d.get('test_confusion')
        if conf:
            cm = np.array(conf)
            plt.figure(figsize=(5.2, 4.6))
            plt.imshow(cm, cmap='Blues')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                             color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)
            plt.xticks(range(len(CLASSES)), CLASSES); plt.yticks(range(len(CLASSES)), CLASSES)
            plt.xlabel('Predicted'); plt.ylabel('True')
            ta = d.get('test_acc'); tb = d.get('test_balanced_acc')
            plt.title(f'Test confusion - {tag}\nacc {ta:.3f} | balanced {tb:.3f}')
            plt.colorbar(fraction=0.046); plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f'{base}_confusion.png'), dpi=150); plt.close()

        acc_curves.append((tag, epochs, val_acc))
        summary.append([LABEL.get(model, model), regime.upper(),
                        f'{d.get("val_acc",0):.4f}', f'{d.get("test_acc",0):.4f}',
                        f'{d.get("test_balanced_acc",0):.4f}', best_ep, f'{d.get("ft_time_h",0):.2f}'])
        print(f'plotted {base}: accuracy + loss + confusion')

    # ---------- Combined val accuracy ----------
    if len(acc_curves) > 1:
        plt.figure(figsize=(8, 5))
        for tag, ep, va in acc_curves:
            plt.plot(ep, va, 'o-', label=tag)
        plt.title('Validation accuracy - all runs'); plt.xlabel('Epoch'); plt.ylabel('Val accuracy'); plt.ylim(0, 1)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'ALL_val_accuracy.png'), dpi=150); plt.close()
        print('plotted combined: ALL_val_accuracy.png')

    # ---------- Summary table ----------
    if summary:
        hdr = ['Model', 'Regime', 'Val Acc', 'Test Acc', 'Test Bal Acc', 'Best Ep', 'Time (h)']
        print('\n' + '  '.join(f'{h:<13}' for h in hdr)); print('-' * 92)
        # the model with the best VAL acc is the selected one
        best_i = max(range(len(summary)), key=lambda i: float(summary[i][2]))
        for i, r in enumerate(summary):
            star = '  <- selected (best val)' if i == best_i else ''
            print('  '.join(f'{str(c):<13}' for c in r) + star)
        with open(os.path.join(PLOT_DIR, 'summary.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(summary)
        print('\nThe SELECTED model (highest val acc) -> report its TEST accuracy.')
        print('saved summary.csv')
    print('done ->', PLOT_DIR)

if __name__ == '__main__':
    main()