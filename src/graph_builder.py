import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
from . import config


# ── Normalisation constants ────────────────────────────────────────────────────
# Node features
CN0_MEAN   = 45.0    # dB-Hz  (typical mid-point for healthy signals)
CN0_STD    = 10.0    # dB-Hz  (1-sigma range covers 35–55 dB-Hz well)
DOP_SCALE  = 3000.0  # Hz     (covers ±5000 Hz dynamic range via tanh)

# Edge features (inter-satellite single-difference)
# Single-diff CN0 rarely exceeds ±15 dB in practice
DIFF_CN0_STD   = 10.0   # normalise CN0 difference by same scale as node CN0
# Single-diff Doppler is much smaller than absolute Doppler
# Two satellites seen by the same receiver differ by ≤ ~2000 Hz typically
DIFF_DOP_SCALE = 500.0  # Hz — tighter scale gives better resolution for edge features
# ──────────────────────────────────────────────────────────────────────────────

# Training set definition
# ● clean static + clean dynamic → legitimate constellation topology
# ● ds4 (matched-power time-push, static) → hardest *known* covert attack
# Everything else is held out as zero-shot test scenarios.
TRAIN_FILES = {'cleanStatic.bin', 'cleanDynamic.bin', 'ds4.bin'}


class GNSSGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            try:
                self.data, self.slices = torch.load(
                    self.processed_paths[0], weights_only=False)
            except Exception:
                print("⚠️  Stale cache — rebuilding dataset...")
                self.process()
                self.data, self.slices = torch.load(
                    self.processed_paths[0], weights_only=False)
        else:
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['gnss_full_dataset.pt']

    def download(self):
        pass

    # ── Feature helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _norm_cn0(v):
        """Z-score normalise C/N₀ around 45 dB-Hz."""
        return (v - CN0_MEAN) / CN0_STD

    @staticmethod
    def _norm_doppler(v):
        """tanh-compress absolute Doppler (±5 kHz range)."""
        return np.tanh(v / DOP_SCALE)

    @staticmethod
    def _norm_diff_cn0(v):
        """Normalise inter-satellite CN0 difference."""
        return v / DIFF_CN0_STD        # linear — differences are already small

    @staticmethod
    def _norm_diff_doppler(v):
        """
        tanh-compress inter-satellite Doppler difference.

        FIX: single-diff Doppler values are much smaller than absolute
        Doppler.  Using the same 1000 Hz scale as node features squashes
        most edge values into the near-zero region where gradients vanish.
        A tighter 500 Hz scale gives the model finer resolution.
        """
        return np.tanh(v / DIFF_DOP_SCALE)

    # ── Main builder ──────────────────────────────────────────────────────────

    def process(self):
        data_list = []
        print("🏗️  Building physics-aware graph dataset …")
        print(f"    Train files : {sorted(TRAIN_FILES)}")
        print(f"    Norm params : CN0 z-score (μ={CN0_MEAN}, σ={CN0_STD}),  "
              f"Dop tanh/{DOP_SCALE},  ΔDop tanh/{DIFF_DOP_SCALE}")

        for filename, label in config.DATA_FILES.items():
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)

            if not os.path.exists(csv_path):
                print(f"   ⚠️  Missing CSV: {csv_name}  (run 1_batch_extract.py first)")
                continue

            is_train = (filename in TRAIN_FILES)
            split_tag = "TRAIN" if is_train else "TEST"
            print(f"   📦  {filename}  label={label}  [{split_tag}]")

            df = pd.read_csv(csv_path)

            for time, group in tqdm(df.groupby('Time'),
                                    desc=f"   {filename}", leave=False):

                raw_cn0     = group['CN0_dBHz'].values.astype(float)
                raw_doppler = group['Doppler'].values.astype(float)
                num_nodes   = len(raw_cn0)

                # Skip epochs with too few satellites (can't form a meaningful graph)
                if num_nodes < 4:
                    continue

                # ── 1. Node features [N, 2] ──────────────────────────────────
                x_np = np.stack([
                    self._norm_cn0(raw_cn0),
                    self._norm_doppler(raw_doppler),
                ], axis=1).astype(np.float32)

                x = torch.tensor(x_np, dtype=torch.float)

                # ── 2. Edge features — inter-satellite single differences ─────
                # Fully connected directed graph: N*(N-1) edges
                # Directed (i→j) edges allow the attention mechanism to identify
                # *which* satellite is the anomalous one (asymmetric weights).
                src_list, dst_list = [], []
                edge_attrs = []

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i == j:
                            continue
                        src_list.append(i)
                        dst_list.append(j)

                        # Directed difference: satellite i minus satellite j
                        d_cn0 = self._norm_diff_cn0(raw_cn0[i] - raw_cn0[j])
                        d_dop = self._norm_diff_doppler(raw_doppler[i] - raw_doppler[j])
                        edge_attrs.append([d_cn0, d_dop])

                edge_index = torch.tensor(
                    [src_list, dst_list], dtype=torch.long)
                edge_attr  = torch.tensor(
                    edge_attrs, dtype=torch.float)

                # ── 3. Graph-level label ─────────────────────────────────────
                y = torch.tensor([label], dtype=torch.long)

                data = Data(x=x, edge_index=edge_index,
                            edge_attr=edge_attr, y=y)
                data.timestamp  = float(time)
                data.train_mask = is_train
                data.scenario   = filename.replace('.bin', '')

                data_list.append(data)

        if not data_list:
            raise RuntimeError(
                "No graph samples were built. Check that CSV files exist "
                f"in {config.DATA_PROC_DIR}")

        # Shuffle to break temporal autocorrelation within each split
        # (shuffle train and test independently to avoid label leakage)
        train_list = [d for d in data_list if d.train_mask]
        test_list  = [d for d in data_list if not d.train_mask]
        random.shuffle(train_list)
        random.shuffle(test_list)
        data_list = train_list + test_list

        print(f"\n💾  Saving {len(data_list)} graphs "
              f"(train={len(train_list)}, test={len(test_list)}) …")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅  Dataset built successfully!")

        # ── Dataset statistics ───────────────────────────────────────────────
        train_labels = [d.y.item() for d in train_list]
        test_labels  = [d.y.item() for d in test_list]
        print(f"\n📊  Train: clean={train_labels.count(0)}, spoof={train_labels.count(1)}")
        print(f"    Test : clean={test_labels.count(0)}, spoof={test_labels.count(1)}")
        if train_labels.count(0) > 0 and train_labels.count(1) > 0:
            ratio = train_labels.count(0) / train_labels.count(1)
            if ratio > 2.5:
                print(f"    ⚠️  Class imbalance ratio = {ratio:.1f}:1. "
                      "Consider enabling class weights in 3_train_eval.py.")