"""
OPS-SAT Anomaly-Transformer Custom Script
Standalone version with 75/25 chronological split using timestamp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import time
import copy
import math
from math import sqrt

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_PATH = "segments.csv"
    WIN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-4
    D_MODEL = 64
    N_HEADS = 4
    E_LAYERS = 2
    D_FF = 64
    DROPOUT = 0.1
    K = 3  # For anomaly ratio in loss
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Early stopping and checkpointing
    PATIENCE = 10
    CHECKPOINT_PATH = "best_anomaly_transformer_ops.pt"
    # Split ratio
    TRAIN_RATIO = 0.75

# ============================================================================
# EMBEDDINGS
# ============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

# ============================================================================
# ATTENTION
# ============================================================================

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.win_size = win_size
        # Pre-compute distances on CPU, will move to device during forward
        self.register_buffer('distances', torch.zeros((win_size, win_size)))
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        
        sigma = sigma.transpose(1, 2)
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)
        
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)
        
        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        
        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), series, prior, sigma

# ============================================================================
# ENCODER
# ============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list

# ============================================================================
# ANOMALY TRANSFORMER MODEL
# ============================================================================

class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out

# ============================================================================
# DATA LOADING
# ============================================================================

class TelemetryDataset(Dataset):
    def __init__(self, data, labels, win_size=100, step=100):
        self.data = data
        self.labels = labels
        self.win_size = win_size
        self.step = step
        
    def __len__(self):
        return max(0, (len(self.data) - self.win_size) // self.step + 1)

    def __getitem__(self, index):
        index = index * self.step
        seq = self.data[index : index + self.win_size]
        label_seq = self.labels[index : index + self.win_size]
        return torch.FloatTensor(seq), torch.FloatTensor(label_seq)

def load_data(config):
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {config.DATA_PATH}")

    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} rows")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Sorted by timestamp: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    data = df['value'].values.astype(np.float32).reshape(-1, 1)
    
    if df['anomaly'].dtype == 'object':
        labels = (df['anomaly'].str.lower() == 'anomaly').astype(np.int32).values
    else:
        labels = df['anomaly'].astype(np.int32).values
    
    n = len(data)
    split_idx = int(n * config.TRAIN_RATIO)
    
    train_data_raw = data[:split_idx]
    train_labels = labels[:split_idx]
    test_data_raw = data[split_idx:]
    test_labels = labels[split_idx:]
    
    train_anom = train_labels.sum()
    test_anom = test_labels.sum()
    print(f"\n[Data Split] {int(config.TRAIN_RATIO*100)}/{int((1-config.TRAIN_RATIO)*100)} Chronological")
    print(f"Train samples: {len(train_data_raw)} (anomalies: {train_anom}, {100*train_anom/len(train_data_raw):.2f}%)")
    print(f"Test samples:  {len(test_data_raw)} (anomalies: {test_anom}, {100*test_anom/len(test_data_raw):.2f}%)")
    print(f"Features: 1 (value)")
    
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)
    
    train_dataset = TelemetryDataset(train_data, train_labels, config.WIN_SIZE, step=config.WIN_SIZE)
    test_dataset = TelemetryDataset(test_data, test_labels, config.WIN_SIZE, step=config.WIN_SIZE)
    
    print(f"Train windows: {len(train_dataset)}")
    print(f"Test windows:  {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, 1

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def train_model(model, train_loader, config):
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\nStarting training on {config.DEVICE}...")
    print(f"Early stopping patience: {config.PATIENCE} epochs")
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        model.train()
        losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            output, series, prior, _ = model(batch_x)
            
            # Reconstruction loss
            rec_loss = criterion(output, batch_x)
            
            # Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, config.WIN_SIZE)).detach())) 
                    + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, config.WIN_SIZE)).detach(), series[u])))
                prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, config.WIN_SIZE)), 
                    series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), 
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, config.WIN_SIZE)))))
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            
            # Total loss (minimize reconstruction + association discrepancy)
            loss = rec_loss - config.K * series_loss + config.K * prior_loss
            
            loss.backward()
            optimizer.step()
            
            losses.append(rec_loss.item())
        
        mean_loss = np.mean(losses)
        
        if np.isnan(mean_loss):
            print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: NaN - STOPPING")
            break
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {mean_loss:.6f}")
        
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model saved (Loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, config.CHECKPOINT_PATH)
    
    print(f"Training complete in {(time.time()-start_time)/60:.2f} mins")
    return model

def evaluate_model(model, test_loader, config):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    scores = []
    labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            
            output, series, prior, _ = model(batch_x)
            
            # Anomaly score = reconstruction error (per time step, averaged over features)
            rec_error = torch.mean(criterion(output, batch_x), dim=-1)  # [B, L]
            
            # Association discrepancy (per batch sample)
            assoc_discrepancy = torch.zeros(batch_x.shape[0], device=config.DEVICE)
            for u in range(len(prior)):
                if prior[u].shape[2] != config.WIN_SIZE or prior[u].shape[3] != config.WIN_SIZE:
                    continue
                # Normalize prior
                prior_norm = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, config.WIN_SIZE)
                # KL divergence between series and prior
                series_loss = my_kl_loss(series[u], prior_norm.detach())  # [B, H]
                prior_loss = my_kl_loss(prior_norm, series[u].detach())    # [B, H]
                # Sum over heads
                assoc_discrepancy += (series_loss + prior_loss).mean(dim=-1)
            
            # Combine: mean reconstruction error per window + association discrepancy
            score = rec_error.mean(dim=-1) + assoc_discrepancy
            
            scores.extend(score.cpu().numpy())
            
            # Labels: max label in window
            label_per_window = batch_y.max(dim=-1)[0]
            labels.extend(label_per_window.numpy())
    
    return np.array(scores), np.array(labels)

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
    config = Config()
    
    train_loader, test_loader, num_feats = load_data(config)
    
    model = AnomalyTransformer(
        win_size=config.WIN_SIZE,
        enc_in=num_feats,
        c_out=num_feats,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        e_layers=config.E_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        activation='gelu',
        output_attention=True
    )
    
    model = train_model(model, train_loader, config)
    scores, labels = evaluate_model(model, test_loader, config)
    
    best_thresh, best_f1 = find_best_threshold(scores, labels)
    preds = (scores > best_thresh).astype(int)
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.5
    
    print("\n" + "="*50)
    print("FINAL RESULTS (OPS-SAT Anomaly-Transformer)")
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
    config = Config()
    values, split, stride = hr.load_honest_inputs(args, kind)
    scaled = hr.scale_with_fit(values, split)          # scaler fitted on the FIT slice only
    a, b = hr.split_io.part_range(split, "fit")
    fit_data = scaled[a:b]
    # published runner setting: non-overlapping training windows (step = window); zero labels (unsupervised)
    train_ds = TelemetryDataset(fit_data, np.zeros(len(fit_data), dtype=np.int32), config.WIN_SIZE, step=config.WIN_SIZE)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    model = AnomalyTransformer(win_size=config.WIN_SIZE, enc_in=fit_data.shape[1], c_out=fit_data.shape[1],
                               d_model=config.D_MODEL, n_heads=config.N_HEADS, e_layers=config.E_LAYERS,
                               d_ff=config.D_FF, dropout=config.DROPOUT, activation='gelu', output_attention=True)
    model = train_model(model, train_loader, config)

    def score_fn(X):
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.zeros(len(X), X.shape[1])),
                            batch_size=config.BATCH_SIZE)
        return evaluate_model(model, loader, config)[0]

    parts = hr.score_parts(score_fn, scaled, split, config.WIN_SIZE, stride)
    hr.export_scores(args, split, parts, config.WIN_SIZE, stride, "anomaly_transformer", mission)

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
