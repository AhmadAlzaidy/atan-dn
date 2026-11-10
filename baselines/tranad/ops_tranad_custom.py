import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import time
import math
import copy

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_PATH = "segments.csv"
    WINDOW_SIZE = 10
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Early stopping and checkpointing
    PATIENCE = 10
    CHECKPOINT_PATH = "best_tranad_ops.pt"
    # Split ratio
    TRAIN_RATIO = 0.75  # 75% train, 25% test

# ============================================================================
# UTILS
# ============================================================================

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

# ============================================================================
# TRANAD MODEL (adapted for single feature)
# ============================================================================

class TranAD(nn.Module):
    def __init__(self, feats, lr=0.0001):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        
        # For single feature, we need nhead to divide 2*feats
        # With feats=1, 2*feats=2, so nhead must be 1 or 2
        nhead = 1 if feats == 1 else min(feats, 2)
        
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=nhead, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=nhead, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=nhead, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

# ============================================================================
# DATA LOADING (75/25 Chronological Split using Timestamp)
# ============================================================================

def convert_to_windows(data, w_size):
    windows = []
    for i in range(len(data)):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]]) if i > 0 else data[0].repeat(w_size, 1)
        windows.append(w)
    return torch.stack(windows)

def load_data(config):
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {config.DATA_PATH}")

    # Load CSV
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Parse timestamp and sort chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Sorted by timestamp: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Extract value as single feature
    data = df['value'].values.astype(np.float32).reshape(-1, 1)
    
    # Convert anomaly label
    if df['anomaly'].dtype == 'object':
        labels = (df['anomaly'].str.lower() == 'anomaly').astype(np.int32).values
    else:
        labels = df['anomaly'].astype(np.int32).values
    
    # 75/25 Chronological Split
    n = len(data)
    split_idx = int(n * config.TRAIN_RATIO)
    
    train_data_raw = data[:split_idx]
    train_labels = labels[:split_idx]
    test_data_raw = data[split_idx:]
    test_labels = labels[split_idx:]
    
    # Print label distribution
    train_anom = train_labels.sum()
    test_anom = test_labels.sum()
    print(f"\n[Data Split] {int(config.TRAIN_RATIO*100)}/{int((1-config.TRAIN_RATIO)*100)} Chronological")
    print(f"Train samples: {len(train_data_raw)} (anomalies: {train_anom}, {100*train_anom/len(train_data_raw):.2f}%)")
    print(f"Test samples:  {len(test_data_raw)} (anomalies: {test_anom}, {100*test_anom/len(test_data_raw):.2f}%)")
    print(f"Features: 1 (value)")
    
    # Normalization
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)
    
    # Convert to tensors
    train_tensor = torch.DoubleTensor(train_data)
    test_tensor = torch.DoubleTensor(test_data)
    
    # Convert to windows
    train_windows = convert_to_windows(train_tensor, config.WINDOW_SIZE)
    test_windows = convert_to_windows(test_tensor, config.WINDOW_SIZE)
    
    # Subsample for speed (step=WINDOW_SIZE for non-overlapping)
    step = config.WINDOW_SIZE
    train_windows = train_windows[::step]
    train_labels_sub = train_labels[::step]
    test_windows = test_windows[::step]
    test_labels_sub = test_labels[::step]
    
    print(f"Train windows (subsampled): {len(train_windows)}")
    print(f"Test windows (subsampled):  {len(test_windows)}")
    
    return train_windows, test_windows, train_labels_sub, test_labels_sub, 1  # 1 feature

# ============================================================================
# TRAINING & TESTING
# ============================================================================

def train_model(model, train_data, config):
    model = to_device(model, config.DEVICE)
    model = model.double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    l = nn.MSELoss(reduction='none')
    
    scaler = torch.amp.GradScaler('cuda', enabled=(config.DEVICE.type == 'cuda'))
    
    # Best model tracking
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\nStarting training on {config.DEVICE}...")
    print(f"Early stopping patience: {config.PATIENCE} epochs")
    start_time = time.time()
    
    feats = model.n_feats
    data_x = to_device(train_data, config.DEVICE)
    dataset = TensorDataset(data_x, data_x)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    for epoch in range(config.EPOCHS):
        model.train()
        l1s = []
        
        for d, _ in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(config.DEVICE.type == 'cuda'), dtype=torch.float16):
                z = model(window, elem)
                n = epoch + 1
                l1 = (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                loss = torch.mean(l1)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            l1s.append(loss.item())
        
        scheduler.step()
        mean_loss = np.mean(l1s)
        
        # Check for NaN
        if np.isnan(mean_loss):
            print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: NaN - STOPPING (NaN detected)")
            break
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {mean_loss:.6f}")
        
        # Track best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model saved (Loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {config.PATIENCE} epochs)")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, config.CHECKPOINT_PATH)
    
    print(f"Training complete in {(time.time()-start_time)/60:.2f} mins")
    return model

def evaluate_model(model, test_data, test_labels, config):
    model = to_device(model, config.DEVICE)
    model.eval()
    l = nn.MSELoss(reduction='none')
    
    feats = model.n_feats
    data_x = to_device(test_data, config.DEVICE)
    dataset = TensorDataset(data_x, data_x)
    dataloader = DataLoader(dataset, batch_size=len(data_x))
    
    print("Evaluating...")
    with torch.no_grad():
        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, d.shape[0], feats)
            z = model(window, elem)
            z = z[1]
            loss = l(z, elem)[0]
    
    scores = loss.mean(dim=1).cpu().numpy()
    labels = test_labels
    
    return scores, labels

def find_best_threshold(scores, labels):
    best_f1 = 0
    best_thresh = 0
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1

# ============================================================================
# MAIN
# ============================================================================

def main():
    conf = Config()
    
    train_data, test_data, train_labels, test_labels, num_feats = load_data(conf)
    model = TranAD(num_feats, conf.LR)
    
    model = train_model(model, train_data, conf)
    scores, labels = evaluate_model(model, test_data, test_labels, conf)
    
    best_thresh, best_f1 = find_best_threshold(scores, labels)
    preds = (scores > best_thresh).astype(int)
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.5
    
    print("\n" + "="*50)
    print("FINAL RESULTS (OPS-SAT TranAD)")
    print("="*50)
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"AUC:           {auc:.4f}")
    print(f"Threshold:     {best_thresh:.6f}")
    print("="*50)
    
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)

# ============================================================================
# HONEST PROTOCOL MODE (--split-json / --export-scores), see baselines/README.md
# ============================================================================
# Trains on the FIT slice of the split file only and exports per-window raw scores of the fit / validation / test
# slices (window ends are absolute sample indices). No threshold, no metric and no label is used here; the
# threshold, the known-event mask and the metrics come from baselines/protocol_adapter.py.

def _honest_runner():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import honest_runner
    return honest_runner

def honest_main(args, kind, mission):
    hr = _honest_runner()
    conf = Config()
    values, split, stride = hr.load_honest_inputs(args, kind)
    scaled = hr.scale_with_fit(values, split)          # scaler fitted on the FIT slice only
    a, b = hr.split_io.part_range(split, "fit")
    # published runner setting: non-overlapping training windows (step = window); a window ends at its last sample
    starts = hr.split_io.window_starts_in(a, b, conf.WINDOW_SIZE, conf.WINDOW_SIZE)
    train_windows = torch.DoubleTensor(hr.make_windows(scaled, starts, conf.WINDOW_SIZE))
    model = TranAD(scaled.shape[1], conf.LR)
    model = train_model(model, train_windows, conf)

    def score_fn(X):
        return evaluate_model(model, torch.DoubleTensor(X), None, conf)[0]

    parts = hr.score_parts(score_fn, scaled, split, conf.WINDOW_SIZE, stride)
    hr.export_scores(args, split, parts, conf.WINDOW_SIZE, stride, "tranad", mission)

if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description="Custom baseline runner")
    _p.add_argument("--data", required=True, help="Path to the mission data file (parquet or csv)")
    _p.add_argument("--out", default="baseline_out", help="Output directory for checkpoints")
    _honest_runner().add_honest_args(_p)
    _a = _p.parse_args()
    _mode = _honest_runner().resolve_mode(_a, _p)
    os.makedirs(_a.out, exist_ok=True)
    Config.DATA_PATH = _a.data
    Config.CHECKPOINT_PATH = os.path.join(_a.out, os.path.basename(Config.CHECKPOINT_PATH))
    if _mode == "honest":
        honest_main(_a, "opssat", "ops_sat")
    else:
        main()
