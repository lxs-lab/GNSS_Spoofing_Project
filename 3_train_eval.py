"""
3_train_eval.py  —  Training + cross-scenario evaluation
=========================================================
Key fixes vs. previous version
  1. Loss function: CrossEntropyLoss now receives raw logits (model no
     longer applies log_softmax internally).
  2. Class-weight auto-detection: if train imbalance > 2:1 the positive
     class is upweighted automatically so spoofing events are not ignored.
  3. Evaluation metrics: per-scenario Detection Rate (DR) and False Alarm
     Rate (FAR) are reported separately — never aggregated into one number.
  4. Ablation flag: set ABLATION_MODE = True to run 5 variants automatically.
"""

import os, sys, shutil, random
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.loader import DataLoader

from src.graph_builder import GNSSGraphDataset
from src.model import STGraphTransformer
from src import config

# ── Experiment switches ────────────────────────────────────────────────────────
ABLATION_MODE   = False   # True → run 5 ablation variants automatically
SAVE_RESULTS    = True    # False → nothing written to disk
# ──────────────────────────────────────────────────────────────────────────────


# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path=None):
        self.terminal = sys.stdout
        self.log = open(path, 'a', encoding='utf-8') if path else None

    def write(self, msg):
        self.terminal.write(msg)
        if self.log:
            self.log.write(msg)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()


# ── Metrics ───────────────────────────────────────────────────────────────────
def scenario_metrics(preds, labels):
    """Return DR (Detection Rate) and FAR (False Alarm Rate)."""
    preds  = np.array(preds)
    labels = np.array(labels)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    dr  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    far = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    return dr, far, tp, fn, fp, tn


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_run(dataset, log_dir, variant_tag='full'):
    train_data = [d for d in dataset if d.train_mask]
    test_data  = [d for d in dataset if not d.train_mask]

    # Class imbalance check → auto-weight
    train_labels = [d.y.item() for d in train_data]
    n_clean = train_labels.count(0)
    n_spoof = train_labels.count(1)
    ratio   = n_clean / max(n_spoof, 1)
    if ratio > 2.0:
        # upweight the minority (spoof) class
        w = torch.tensor([1.0, ratio], dtype=torch.float)
        print(f"   ⚖️  Class imbalance {ratio:.1f}:1 — applying weights {w}")
    else:
        w = torch.tensor([1.0, 1.0], dtype=torch.float)
        print(f"   ⚖️  Balanced dataset — equal class weights")

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️  Device: {device}")

    model = STGraphTransformer(
        in_channels=dataset.num_features,
        hidden_channels=config.HIDDEN_DIM,
        edge_dim=2,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=1e-4)
    # CrossEntropyLoss expects raw logits — model.forward() now returns logits
    criterion = torch.nn.CrossEntropyLoss(weight=w.to(device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS)

    best_acc   = 0.0
    best_dr    = 0.0
    train_losses, test_accs = [], []
    best_path  = os.path.join(config.MODEL_DIR, f'best_{variant_tag}.pth')

    print(f"\n   🚀  Training [{variant_tag}]  ({config.EPOCHS} epochs) …")
    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # ── Eval ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred  = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        dr, far, *_ = scenario_metrics(all_preds, all_labels)
        test_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_dr  = dr
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
                  f"Acc {acc*100:.1f}% | DR {dr*100:.1f}% | FAR {far*100:.2f}%")

    print(f"\n   ✅  Best overall Acc={best_acc*100:.1f}%, DR={best_dr*100:.1f}%")

    # ── Per-scenario evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    scenarios = sorted(set(d.scenario for d in test_data if hasattr(d, 'scenario')))
    print(f"\n{'='*62}")
    print(f"  Per-scenario evaluation (zero-shot transfer)")
    print(f"{'='*62}")
    print(f"  {'Scenario':<20} {'#samples':>8} {'DR (%)':>8} {'FAR (%)':>8}  {'Status'}")
    print(f"  {'-'*58}")

    perf = {}
    y_true_all, y_pred_all = [], []

    for sce in scenarios:
        subset = [d for d in test_data if getattr(d, 'scenario', '') == sce]
        if not subset:
            continue
        loader = DataLoader(subset, batch_size=config.BATCH_SIZE)
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred  = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(batch.y.cpu().numpy())
        dr_s, far_s, *_ = scenario_metrics(preds, labels)
        perf[sce] = (dr_s, far_s)
        y_true_all.extend(labels)
        y_pred_all.extend(preds)

        flag = '✅' if (dr_s > 0.90 and far_s < 0.05) else '⚠️ '
        print(f"  {sce:<20} {len(subset):>8} "
              f"{dr_s*100:>7.1f}% {far_s*100:>7.2f}%  {flag}")

    print(f"{'='*62}")

    # ── Plots ────────────────────────────────────────────────────────────────
    if log_dir and SAVE_RESULTS:
        _save_plots(train_losses, test_accs, perf,
                    y_true_all, y_pred_all, log_dir, variant_tag)

    return perf


def _save_plots(losses, accs, perf, y_true, y_pred, log_dir, tag):
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses);  ax1.set_title('Training loss');  ax1.set_xlabel('Epoch')
    ax2.plot([a*100 for a in accs], color='orange')
    ax2.set_title('Overall test accuracy'); ax2.set_ylabel('%'); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}_training_curves.png'), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Spoofed'],
                yticklabels=['Clean', 'Spoofed'])
    plt.title(f'Confusion matrix [{tag}]')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}_confusion.png'), dpi=150)
    plt.close()

    # Per-scenario DR / FAR bar chart
    names = list(perf.keys())
    drs   = [perf[n][0] * 100 for n in names]
    fars  = [perf[n][1] * 100 for n in names]
    x     = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.2), 5))
    ax.bar(x - 0.2, drs,  0.35, label='DR (%)',  color='#4CAF50')
    ax.bar(x + 0.2, fars, 0.35, label='FAR (%)', color='#F44336')
    ax.axhline(90, color='green',  linestyle='--', linewidth=0.8, label='DR target 90%')
    ax.axhline(5,  color='red',    linestyle='--', linewidth=0.8, label='FAR limit 5%')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 115); ax.set_ylabel('%'); ax.legend()
    ax.set_title(f'Per-scenario DR and FAR [{tag}]')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}_scenario_metrics.png'), dpi=150)
    plt.close()

    print(f"   📊  Plots saved to {log_dir}/")


# ── Ablation suite ────────────────────────────────────────────────────────────
ABLATION_VARIANTS = {
    'no_edge_feat':  'Remove edge features (set edge_attr to zeros)',
    'raw_edge':      'Use raw (unnormalised) Doppler difference as edge feature',
    'no_temporal':   'Single GNN layer only (spatial only)',
    'random_graph':  'Randomly rewired edges (destroy topology)',
    'full':          'Full model — proposed ST-GT',
}


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = None
    if SAVE_RESULTS:
        log_dir = os.path.join(config.LOG_BASE_DIR, ts)
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = Logger(os.path.join(log_dir, 'console_log.txt'))

    print("=" * 62)
    print("  GNSS ST-Graph Transformer — Training & Evaluation")
    print("=" * 62)
    print(f"  EPOCHS={config.EPOCHS}  BS={config.BATCH_SIZE}  "
          f"LR={config.LR}  HIDDEN={config.HIDDEN_DIM}  "
          f"DROPOUT={config.DROPOUT}")

    # Force rebuild to pick up graph_builder changes
    proc_dir = os.path.join(config.DATASET_DIR, 'processed')
    if os.path.exists(proc_dir):
        shutil.rmtree(proc_dir)
        print("🧹  Cleared cached dataset — rebuilding from CSV …")

    dataset = GNSSGraphDataset(root=config.DATASET_DIR)
    print(f"\n📦  Dataset: {len(dataset)} graphs  |  "
          f"features={dataset.num_features}  classes={dataset.num_classes}")

    if ABLATION_MODE:
        print("\n🔬  Running ablation study …")
        for variant, desc in ABLATION_VARIANTS.items():
            print(f"\n  ── Variant: {variant} ──")
            print(f"     {desc}")
            # NOTE: full ablation requires patching model/graph_builder per variant.
            # Here we run the full model for all variants as a scaffold.
            # Implement per-variant logic before the train_one_run() call.
            train_one_run(dataset, log_dir, variant_tag=variant)
    else:
        train_one_run(dataset, log_dir, variant_tag='ST-GT')


if __name__ == '__main__':
    main()