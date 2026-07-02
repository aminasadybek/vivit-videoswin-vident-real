# Plots the FOCAL-loss quadrant-classification runs (results_cls_focal_*.json).
# For EACH model x regime: accuracy curve, loss curve, TEST confusion matrix - every point
# annotated with its value. Plus a combined val comparison and a summary table/CSV.
# Titles are tagged "(focal)" so these never get confused with the CE-baseline plots.
#
#   pip install matplotlib
#   C:\ProgramData\Miniconda3\python.exe C:\ESAI406_Shared\Downloads\plot_cls_results_focal.py

import os, json, glob, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# point this at the FOCAL results folder
SAVE_DIR = r'C:\Users\ESAI406_student2\Vident_cls_results_ft_focal'
PLOT_DIR = os.path.join(SAVE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

LABEL   = {'swin': 'Video Swin-B', 'vivit': 'ViViT-B'}
CLASSES = ['Q1', 'Q2', 'Q3', 'Q4']

def _annotate(xs, ys, color, dy, fmt='{:.3f}', fs=6.5):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        plt.annotate(fmt.format(y), (x, y), textcoords='offset points', xytext=(0, dy),
                     ha='center', fontsize=fs, color=color)

def main():
    # focal JSONs are named results_cls_focal_*.json
    files = sorted(glob.glob(os.path.join(SAVE_DIR, 'results_cls_focal_*.json')))
    if not files:                                              # fall back to any results file in this folder
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
        gamma = d.get('focal_gamma')
        loss_tag = f'focal, gamma={gamma}' if gamma is not None else 'focal'
        tag  = f'{LABEL.get(model, model)} - {regime.upper()} FT ({loss_tag})'
        base = f'{model}_{regime}_focal'                       # distinct from CE plot filenames
        epochs    = [e['epoch'] for e in log]
        train_acc = [e.get('train_acc') for e in log]     # clip-level
        val_clip  = [e.get('val_clip_acc') for e in log]  # clip-level
        val_acc   = [e.get('val_acc') for e in log]       # segment-level (reported)
        val_bal   = [e.get('val_bal_acc') for e in log]   # segment-level balanced
        tr_loss   = [e.get('train_loss') for e in log]    # clip-level (focal)
        vl_loss   = [e.get('val_loss') for e in log]      # clip-level (CE reference)
        has_vloss = all(v is not None for v in vl_loss) and len(vl_loss) > 0
        best_ep   = d.get('val_best_epoch')

        # ============ Accuracy ============
        plt.figure(figsize=(10.5, 5.5))
        if any(v is not None for v in train_acc):
            ln, = plt.plot(epochs, train_acc, '^-', label='Train accuracy (clip)')
            _annotate(epochs, train_acc, ln.get_color(), dy=7)
        if any(v is not None for v in val_clip):
            ln, = plt.plot(epochs, val_clip, 'o-', label='Val accuracy (clip)')
            _annotate(epochs, val_clip, ln.get_color(), dy=-12)
        ln, = plt.plot(epochs, val_acc, 's-', linewidth=2, label='Val accuracy (segment, reported)')
        _annotate(epochs, val_acc, ln.get_color(), dy=9, fs=7)
        if any(v is not None for v in val_bal):
            ln, = plt.plot(epochs, val_bal, 'd-', label='Val balanced accuracy (segment)')
            _annotate(epochs, val_bal, ln.get_color(), dy=-13)
        if best_ep:
            plt.axvline(best_ep, ls='--', color='gray', alpha=.7)
            bi = epochs.index(best_ep) if best_ep in epochs else None
            if bi is not None:
                plt.scatter([best_ep], [val_acc[bi]], s=160, facecolors='none',
                            edgecolors='red', linewidths=2, zorder=5, label=f'Best epoch ({best_ep})')
                plt.annotate(f'{val_acc[bi]:.3f}', (best_ep, val_acc[bi]),
                             textcoords='offset points', xytext=(0, 20), ha='center',
                             fontsize=10, fontweight='bold', color='red')
        if best_ep and best_ep in epochs:
            bi = epochs.index(best_ep)
            box = (f'best ep {best_ep}  (focal)\n'
                   f'val_acc(seg)    {val_acc[bi]:.4f}\n'
                   f'val_bal(seg)    {val_bal[bi]:.4f}\n'
                   f'val_acc(clip)   {val_clip[bi]:.4f}\n'
                   f'train_acc(clip) {train_acc[bi]:.4f}')
            plt.gca().text(0.015, 0.015, box, transform=plt.gca().transAxes, fontsize=8,
                           va='bottom', ha='left', family='monospace',
                           bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=.85))
        plt.title(f'Accuracy - {tag}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.ylim(0, 1.05)
        plt.grid(alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, borderaxespad=0)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{base}_accuracy.png'), dpi=150); plt.close()

        # ============ Loss ============
        plt.figure(figsize=(9, 5))
        ln, = plt.plot(epochs, tr_loss, 'o-', label='Train loss (clip, focal)')
        _annotate(epochs, tr_loss, ln.get_color(), dy=8, fmt='{:.2f}')
        if has_vloss:
            ln, = plt.plot(epochs, vl_loss, 's-', label='Val loss (clip, CE ref)')
            _annotate(epochs, vl_loss, ln.get_color(), dy=-13, fmt='{:.2f}')
        if best_ep:
            plt.axvline(best_ep, ls='--', color='gray', alpha=.7, label=f'Best epoch ({best_ep})')
        plt.title(f'Loss (clip-level) - {tag}')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{base}_loss.png'), dpi=150); plt.close()

        # ============ Test confusion ============
        conf = d.get('test_confusion')
        if conf:
            cm = np.array(conf)
            plt.figure(figsize=(5.4, 4.8))
            plt.imshow(cm, cmap='Blues')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                             color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
            plt.xticks(range(len(CLASSES)), CLASSES); plt.yticks(range(len(CLASSES)), CLASSES)
            plt.xlabel('Predicted'); plt.ylabel('True')
            ta = d.get('test_acc'); tb = d.get('test_balanced_acc')
            tac = d.get('test_clip_acc'); tbc = d.get('test_clip_balanced_acc')
            sub = f'acc(seg) {ta:.3f} | bal(seg) {tb:.3f}'
            if tac is not None:
                sub += f'\nacc(clip) {tac:.3f} | bal(clip) {tbc:.3f}'
            plt.title(f'Test confusion - {tag}\n{sub}', fontsize=9)
            plt.colorbar(fraction=0.046); plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f'{base}_confusion.png'), dpi=150); plt.close()

        acc_curves.append((tag, epochs, val_acc, best_ep))
        summary.append([LABEL.get(model, model), regime.upper(),
                        f'{d.get("val_acc",0):.4f}', f'{d.get("val_clip_acc",0):.4f}',
                        f'{d.get("test_acc",0):.4f}', f'{d.get("test_clip_acc",0):.4f}',
                        f'{d.get("test_balanced_acc",0):.4f}', f'{d.get("test_clip_balanced_acc",0):.4f}',
                        best_ep, f'{d.get("ft_time_h",0):.2f}'])
        print(f'plotted {base}: accuracy + loss + confusion')

    # ============ Combined val accuracy ============
    if len(acc_curves) > 1:
        plt.figure(figsize=(9, 5.5))
        for tag, ep, va, bep in acc_curves:
            ln, = plt.plot(ep, va, 'o-', label=tag)
            if bep and bep in ep:
                bi = ep.index(bep)
                plt.annotate(f'{va[bi]:.3f}', (bep, va[bi]), textcoords='offset points',
                             xytext=(0, 8), ha='center', fontsize=8,
                             fontweight='bold', color=ln.get_color())
        plt.title('Validation accuracy (segment, reported) - all FOCAL runs')
        plt.xlabel('Epoch'); plt.ylabel('Val accuracy (segment)'); plt.ylim(0, 1.05)
        plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'ALL_val_accuracy_focal.png'), dpi=150); plt.close()
        print('plotted combined: ALL_val_accuracy_focal.png')

    # ============ Summary table + CSV ============
    if summary:
        hdr = ['Model', 'Regime', 'Val(seg)', 'Val(clip)', 'Test(seg)', 'Test(clip)',
               'TestBal(seg)', 'TestBal(clip)', 'BestEp', 'Time(h)']
        print('\n  FOCAL loss results')
        print('  ' + '  '.join(f'{h:<12}' for h in hdr)); print('-' * 132)
        best_i = max(range(len(summary)), key=lambda i: float(summary[i][2]))
        for i, r in enumerate(summary):
            star = '  <- selected (best val seg)' if i == best_i else ''
            print('  ' + '  '.join(f'{str(c):<12}' for c in r) + star)
        with open(os.path.join(PLOT_DIR, 'summary_focal.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(summary)
        print('\nThe SELECTED model (highest val seg accuracy) -> report its TEST accuracy.')
        print('saved summary_focal.csv')
    print('done ->', PLOT_DIR)

if __name__ == '__main__':
    main()
