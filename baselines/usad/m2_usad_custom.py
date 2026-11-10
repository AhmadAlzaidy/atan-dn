import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import time
import copy

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_PATH = "all_telemetry_18_28.parquet"
    WINDOW_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 100
    LATENT_SIZE = 40
    ALPHA = 0.5
    BETA = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Early stopping and checkpointing
    PATIENCE = 10  # Stop if no improvement for this many epochs
    CHECKPOINT_PATH = "best_usad_m2.pt"

# ============================================================================
# UTILS
# ============================================================================

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# ============================================================================
# DATA LOADING (50/50 Chronological Split)
# ============================================================================

class TelemetryDataset(Dataset):
    def __init__(self, data, labels, window_size=100, step=100):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.step = step
        
    def __len__(self):
        return (len(self.data) - self.window_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        seq = self.data[index : index + self.window_size]
        label_seq = self.labels[index : index + self.window_size]
        label = np.max(label_seq) 
        return torch.FloatTensor(seq), torch.FloatTensor([label])

def load_data(config):
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {config.DATA_PATH}")

    df = pd.read_parquet(config.DATA_PATH)
    features = [c for c in df.columns if c.startswith("chchannel_")]
    label_col = "label"
    
    data = df[features].values.astype(np.float32)
    labels = df[label_col].values.astype(np.int32)
    
    n = len(data)
    split_idx = int(n * 0.5)
    
    train_data_raw = data[:split_idx]
    train_labels = labels[:split_idx]
    test_data_raw = data[split_idx:]
    test_labels = labels[split_idx:]
    
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)
    
    print(f"[Data Split] 50/50 Chronological")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")
    print(f"Features: {len(features)}")
    
    train_dataset = TelemetryDataset(train_data, train_labels, config.WINDOW_SIZE)
    test_dataset = TelemetryDataset(test_data, test_labels, config.WINDOW_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, len(features)

# ============================================================================
# MODEL ARCHITECTURE (USAD)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        
    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    
class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
  
    def training_step(self, batch, epoch):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/(epoch+1)*torch.mean((batch-w1)**2) + (1-1/(epoch+1))*torch.mean((batch-w3)**2)
        loss2 = 1/(epoch+1)*torch.mean((batch-w2)**2) - (1-1/(epoch+1))*torch.mean((batch-w3)**2)
        return loss1, loss2

# ============================================================================
# TRAINING & TESTING
# ============================================================================

def train_model(model, train_loader, val_loader, config):
    model = to_device(model, config.DEVICE)
    optimizer1 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    
    scaler = torch.amp.GradScaler('cuda', enabled=(config.DEVICE.type == 'cuda'))
    
    # Best model tracking
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"Starting training on {config.DEVICE} (Mixed Precision: {'On' if config.DEVICE.type == 'cuda' else 'Off'})...")
    print(f"Early stopping patience: {config.PATIENCE} epochs")
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        model.train()
        l1_vals = []
        l2_vals = []
        
        for batch in train_loader:
            batch_data = to_device(batch[0], config.DEVICE)
            batch_data = batch_data.view(batch_data.size(0), -1)
            
            # --- Train AE1 ---
            optimizer1.zero_grad()
            with torch.amp.autocast('cuda', enabled=(config.DEVICE.type == 'cuda')):
                loss1, loss2 = model.training_step(batch_data, epoch+1)
            
            scaler.scale(loss1).backward()
            scaler.step(optimizer1)
            
            # --- Train AE2 ---
            optimizer2.zero_grad()
            with torch.amp.autocast('cuda', enabled=(config.DEVICE.type == 'cuda')):
                loss1, loss2 = model.training_step(batch_data, epoch+1)
            
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            
            l1_vals.append(loss1.item())
            l2_vals.append(loss2.item())
        
        mean_l1 = np.mean(l1_vals)
        mean_l2 = np.mean(l2_vals)
        
        # Check for NaN
        if np.isnan(mean_l1) or np.isnan(mean_l2):
            print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss1: NaN, Loss2: NaN - STOPPING (NaN detected)")
            break
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss1: {mean_l1:.4f}, Loss2: {mean_l2:.4f}")
        
        # Track best model (using Loss1 as primary metric)
        if mean_l1 < best_loss:
            best_loss = mean_l1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model saved (Loss1: {best_loss:.4f})")
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

def evaluate_model(model, test_loader, config):
    model = to_device(model, config.DEVICE)
    model.eval()
    
    results = []
    labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in test_loader:
            batch_data = to_device(batch[0], config.DEVICE)
            batch_labels = batch[1].cpu().numpy()
            batch_data = batch_data.view(batch_data.size(0), -1)
            
            w1 = model.decoder1(model.encoder(batch_data))
            w2 = model.decoder2(model.encoder(w1))
            
            rec1 = torch.mean((batch_data - w1)**2, dim=1)
            rec2 = torch.mean((batch_data - w2)**2, dim=1)
            score = config.ALPHA * rec1 + config.BETA * rec2
            
            results.extend(score.cpu().numpy())
            labels.extend(batch_labels)
            
    return np.array(results), np.array(labels).flatten()

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
    
    train_loader, test_loader, num_feats = load_data(conf)
    input_size = conf.WINDOW_SIZE * num_feats
    model = UsadModel(input_size, conf.LATENT_SIZE)
    
    model = train_model(model, train_loader, test_loader, conf)
    scores, labels = evaluate_model(model, test_loader, conf)
    
    best_thresh, best_f1 = find_best_threshold(scores, labels)
    preds = (scores > best_thresh).astype(int)
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.5
        
    print("\n" + "="*50)
    print("FINAL RESULTS")
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
    from torch.utils.data import TensorDataset
    hr = _honest_runner()
    conf = Config()
    values, split, stride = hr.load_honest_inputs(args, kind)
    scaled = hr.scale_with_fit(values, split)          # scaler fitted on the FIT slice only
    a, b = hr.split_io.part_range(split, "fit")
    fit_data = scaled[a:b]
    # published runner setting: non-overlapping training windows (step = window); zero labels (unsupervised)
    train_ds = TelemetryDataset(fit_data, np.zeros(len(fit_data), dtype=np.int32), conf.WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=conf.BATCH_SIZE, shuffle=True)
    model = UsadModel(conf.WINDOW_SIZE * fit_data.shape[1], conf.LATENT_SIZE)
    model = train_model(model, train_loader, train_loader, conf)   # train_model does not read the validation loader

    def score_fn(X):
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.zeros(len(X), 1)), batch_size=conf.BATCH_SIZE)
        return evaluate_model(model, loader, conf)[0]

    parts = hr.score_parts(score_fn, scaled, split, conf.WINDOW_SIZE, stride)
    hr.export_scores(args, split, parts, conf.WINDOW_SIZE, stride, "usad", mission)

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
        honest_main(_a, "esa", "m2")
    else:
        main()
