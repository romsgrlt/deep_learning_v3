import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main(log_dir):
    train_df = pd.read_csv(f'{log_dir}/train.csv')
    val_df = pd.read_csv(f'{log_dir}/val.csv')
    test_df = pd.read_csv(f'{log_dir}/test.csv')

    GROUP_LABELS = ['landbird/land', 'landbird/water', 'waterbird/land', 'waterbird/water']
    GROUP_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    epochs = val_df['epoch']

    fig = plt.figure(figsize=(20, 30))
    fig.suptitle('Group DRO — Waterbirds (300 epochs)', fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)

    best_epoch = val_df.loc[val_df['worst_acc'].idxmax(), 'epoch']

    # ── 1. Worst-group accuracy Train / Val / Test ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(train_df['epoch'], train_df['worst_acc'], color='#4CAF50', linewidth=2, label='Train worst-group acc')
    ax0.plot(epochs, val_df['worst_acc'], color='#2196F3', linewidth=2, label='Val worst-group acc')
    ax0.plot(epochs, test_df['worst_acc'], color='#E91E63', linewidth=2, label='Test worst-group acc')
    ax0.set_title('Worst-group accuracy — Train / Val / Test', fontweight='bold')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Accuracy')
    ax0.legend()
    ax0.grid(alpha=0.3)

    # ── 2. Accuracy par groupe — Val ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    for g, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ax1.plot(epochs, val_df[f'acc_g{g}'], color=color, linewidth=2, label=label)
    ax1.plot(epochs, val_df['avg_acc'], color='#000000', linewidth=2, label='Val avg acc', linestyle='--', alpha=0.4)
    ax1.set_title('Accuracy par groupe — Validation', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── 3. Worst-group accuracy & Average accuracy (val vs test) ───────────────
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(epochs, val_df['worst_acc'], color='#2196F3', linewidth=1.5, label='Val worst-group acc')
    ax2.plot(epochs, test_df['worst_acc'], color='#E91E63', linewidth=1.5, label='Test worst-group acc', linestyle='--')
    ax2.plot(epochs, val_df['avg_acc'], color='#2196F3', linewidth=1.5, alpha=0.4, label='Val avg acc')
    ax2.plot(epochs, test_df['avg_acc'], color='#E91E63', linewidth=1.5, alpha=0.4, label='Test avg acc', linestyle='--')
    ax2.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=1.5, label=f'Best val epoch ({best_epoch})')
    ax2.set_title('Worst-group accuracy & Average accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── 4. Accuracy par groupe — Test ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    for g, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ax3.plot(epochs, test_df[f'acc_g{g}'], color=color, linewidth=1.5, label=label)
    ax3.set_title('Accuracy par groupe — Test', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── 5. Loss par groupe — Train ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    for g, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ax4.plot(train_df['epoch'], train_df[f'loss_g{g}'], color=color, linewidth=1.5, label=label)
    ax4.set_title('Loss par groupe — Train', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── 6. Loss par groupe — Val ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    for g, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ax5.plot(epochs, val_df[f'loss_g{g}'], color=color, linewidth=1.5, label=label)
    ax5.set_title('Loss par groupe — Validation', fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # ── 7. Adversarial probabilities — Train ───────────────────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    for g, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ax6.plot(train_df['epoch'], train_df[f'adv_prob_g{g}'], color=color, linewidth=1.5, label=label)
    ax6.axhline(y=0.25, color='gray', linestyle=':', linewidth=1, label='Uniforme (0.25)')
    ax6.set_title('Probabilités adversariales (adv_probs) — Train', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('adv_prob')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    plt.savefig(f'{log_dir}/results.png', dpi=150, bbox_inches='tight')
    print("Plot sauvegardé")


if __name__ == '__main__':
    for i in range(5):
        main(f'logs/adjustments/{i}')
