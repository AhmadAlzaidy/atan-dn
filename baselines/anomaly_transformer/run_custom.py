import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import argparse
from solver import Solver
from model.AnomalyTransformer import AnomalyTransformer
from utils.utils import *
from solver import my_kl_loss

# Define Configuration Class or Dict
class Config:
    # Model Config
    win_size = 100
    enc_in = 0  # Will be set based on data
    c_out = 0   # Will be set based on data
    batch_size = 32
    num_epochs = 10 # Adjusted for quick trial, user can increase
    lr = 1e-4
    k = 3
    anormly_ratio = 4.0 # Default from main.py
    
    # Paths
    # Default relative path or allow overwrite
    data_path = 'all_telemetry_41_46.parquet' 
    model_save_path = 'checkpoints_custom'
    dataset = 'Custom'
    mode = 'train'

class TelemetryDataset(Dataset):
    def __init__(self, data_path, win_size=100, step=1, mode='train'):
        self.mode = mode
        self.win_size = win_size
        self.step = step
        
        # Load Data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}. Please provide correct path via --data_path argument.")
            
        df = pd.read_parquet(data_path)
        
        # Select Features
        features = [c for c in df.columns if c.startswith("chchannel_")]
        label_col = "label"
        
        data = df[features].values.astype(np.float32)
        labels = df[label_col].values.astype(np.int32)
        
        # 50/50 Chronological Split
        n = len(data)
        split_idx = int(n * 0.5)
        
        train_data = data[:split_idx]
        train_labels = labels[:split_idx]
        
        test_data = data[split_idx:]
        test_labels = labels[split_idx:]
        
        # Normalization (Fit on Train, Transform Both)
        self.scaler = StandardScaler()
        self.train_data = self.scaler.fit_transform(train_data)
        self.test_data = self.scaler.transform(test_data)
        
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        print(f"Data Loaded. Split: 50/50 Chronological.")
        print(f"Train Size: {len(self.train_data)}, Test Size: {len(self.test_data)}")
        print(f"Features: {len(features)}")
        
        # Assign based on mode
        if self.mode == 'train':
            self.data = self.train_data
            self.labels = self.train_labels  # Not typically used in unsupervised training loop of AnomalyTransformer but kept for consistency
        elif self.mode == 'val' or self.mode == 'test' or self.mode == 'thre':
            self.data = self.test_data
            self.labels = self.test_labels
            
    def __len__(self):
        return (len(self.data) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        seq = self.data[index : index + self.win_size]
        label = self.labels[index : index + self.win_size]
        return torch.FloatTensor(seq), torch.FloatTensor(label)

class CustomSolver(Solver):
    def __init__(self, config, train_loader, val_loader, test_loader):
        # Initialize parent class manually or skip it if it does too much. 
        # Solver.__init__ calls build_model and sets up loaders. We want to override loader setup.
        
        # Copying relevant parts from Solver.__init__ to avoid duplicate/wrong loader init
        self.__dict__.update(Solver.DEFAULTS, **config)
        
        self.train_loader = train_loader
        self.vali_loader = val_loader
        self.test_loader = test_loader
        self.thre_loader = test_loader # Use test/val for threshold calculation
        
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

# ============================================================================
# HONEST PROTOCOL MODE (--split-json / --export-scores), see baselines/README.md
# ============================================================================
# Trains on the FIT slice of the split file only (early stopping on the unlabelled loss of the VALIDATION slice,
# which belongs to the training split; the legacy code used the test loader there) and exports per-window raw scores
# of the fit / validation / test slices. No threshold, no metric and no label is used here.

def _honest_runner():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import honest_runner
    return honest_runner


class ArrayWindows(Dataset):
    """Windows of an in-memory ``(N, C)`` array with ZERO labels (honest mode never passes a label on)."""

    def __init__(self, data, win_size, step):
        self.data, self.win_size, self.step = data, win_size, step

    def __len__(self):
        return max(0, (len(self.data) - self.win_size) // self.step + 1)

    def __getitem__(self, index):
        s = index * self.step
        return torch.FloatTensor(self.data[s:s + self.win_size]), torch.zeros(self.win_size)


def window_energy(model, x, win_size, device, temperature=50):
    """Anomaly Transformer energy of solver.test() (softmax(-assoc. discrepancy) * reconstruction error), averaged
    over the time steps of each window -> one score per window (aggregation of ours: the upstream code scores
    time steps)."""
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        output, series, prior, _ = model(x)
        loss = torch.mean(criterion(x, output), dim=-1)
        series_loss, prior_loss = 0.0, 0.0
        for u in range(len(prior)):
            p_norm = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)
            series_loss = series_loss + my_kl_loss(series[u], p_norm.detach()) * temperature
            prior_loss = prior_loss + my_kl_loss(p_norm, series[u].detach()) * temperature
        metric = torch.softmax(-series_loss - prior_loss, dim=-1)
        return (metric * loss).mean(dim=1).cpu().numpy()


def honest_main(args, kind, mission):
    hr = _honest_runner()
    conf = Config()
    conf.model_save_path = args.out
    os.makedirs(conf.model_save_path, exist_ok=True)
    values, split, stride = hr.load_honest_inputs(args, kind)
    scaled = hr.scale_with_fit(values, split)          # scaler fitted on the FIT slice only
    fa, fb = hr.split_io.part_range(split, "fit")
    va, vb = hr.split_io.part_range(split, "val")
    fit_data, val_data = scaled[fa:fb], scaled[va:vb]
    conf.input_c = conf.output_c = scaled.shape[1]
    train_ds = ArrayWindows(fit_data, conf.win_size, 1)            # published runner setting: step 1
    val_ds = ArrayWindows(val_data, conf.win_size, conf.win_size)  # non-overlapping, unlabelled validation loss
    if len(val_ds) == 0:
        raise ValueError("the validation slice is shorter than one window; cannot early-stop without it")
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=conf.batch_size, shuffle=False, num_workers=0)
    config_dict = {k: getattr(conf, k) for k in dir(conf) if not k.startswith('__') and not callable(getattr(conf, k))}
    # Solver.train() early-stops on `test_loader`: in honest mode that is the validation slice, never the test split
    solver = CustomSolver(config_dict, train_loader, val_loader, val_loader)
    solver.train()
    ckpt = os.path.join(str(conf.model_save_path), str(conf.dataset) + '_checkpoint.pth')
    solver.model.load_state_dict(torch.load(ckpt, map_location=solver.device))
    solver.model.eval()

    def score_fn(X):
        x = torch.from_numpy(X).float().to(solver.device)
        return window_energy(solver.model, x, conf.win_size, solver.device)

    parts = hr.score_parts(score_fn, scaled, split, conf.win_size, stride, chunk=256)
    hr.export_scores(args, split, parts, conf.win_size, stride, "anomaly_transformer", split.get("mission", mission))


def main():
    parser = argparse.ArgumentParser(description='Run Anomaly Transformer on Custom Data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to parquet data file')
    parser.add_argument('--out', type=str, default='baseline_out', help='Output directory for checkpoints')
    _honest_runner().add_honest_args(parser)
    args = parser.parse_args()
    mode = _honest_runner().resolve_mode(args, parser)
    if mode == "honest":
        args.data = args.data_path
        honest_main(args, "esa", None)
        return

    # 1. Config
    conf = Config()
    conf.data_path = args.data_path # Override with arg
    conf.model_save_path = args.out
    
    if not os.path.exists(conf.model_save_path):
        os.makedirs(conf.model_save_path)
        
    # 2. Data Loading
    # We need to know input_c (number of features) before initializing model?
    # data_loader will help us find out.
    
    # Create Datasets
    # Step can be 1 for fine-grained sliding window, or more to speed up
    step = 1 
    
    train_dataset = TelemetryDataset(conf.data_path, win_size=conf.win_size, step=step, mode='train')
    test_dataset = TelemetryDataset(conf.data_path, win_size=conf.win_size, step=step, mode='test')
    
    # Update Config with Data Dimensions
    conf.input_c = train_dataset.data.shape[1]
    conf.output_c = train_dataset.data.shape[1] 
    
    print(f"Input Channels: {conf.input_c}")
    
    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
    # Using the same test dataset for val, test, and threshold finding as per 50/50 split request
    val_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0)
    
    # 3. Initialize Solver
    # vars(conf) might not work if Config is a class with class attributes
    config_dict = {k: getattr(conf, k) for k in dir(conf) if not k.startswith('__') and not callable(getattr(conf, k))}
    
    solver = CustomSolver(config_dict, train_loader, val_loader, test_loader)
    
    # 4. Train
    print("\nStarting Training...")
    solver.train()
    
    # 5. Test
    print("\nStarting Testing...")
    solver.test()

if __name__ == '__main__':
    main()
