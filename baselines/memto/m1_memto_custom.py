"""
MEMTO baseline for Mission 1 (M1), in the same self-contained-script style as
Final_code/TranAD/m1_tranad_custom.py.

This script re-implements MEMTO's real two-phase training procedure (Song et al.,
NeurIPS 2023, https://github.com/gunny97/MEMTO) using the UPSTREAM model classes
imported from ../memto/model (TransformerVar, EntropyLoss, GatheringLoss) and the
UPSTREAM k-means initializer (utils.utils.k_means_clustering), instead of the
upstream solver.py/data_factory pipeline (which is hardcoded to SMD/SMAP/PSM/MSL/
SWaT .npy/.csv loaders and argparse). Deviations from solver.py are documented
inline and in memto/README.md.

Requires a CUDA GPU: the upstream model code (model/embedding.py's
InputEmbedding.forward and model/ours_memory_module.py's MemoryModule.read/update)
calls .cuda() unconditionally, and utils/utils.py's k_means_clustering uses
kmeans_pytorch with device='cuda:0' hardcoded. These are upstream files we do not
modify, so this baseline cannot run on CPU-only machines.
"""

import os
import sys
import copy
import time
import gc
import random

# ---------------------------------------------------------------------------
# GPU memory safety (single-card OOM protection).
# These env vars MUST be set before torch is imported:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> far less fragmentation,
#       the most common cause of a late-run OOM on a nearly-full card.
#   MEMTO_GPU=<id> -> pin this run to one card (e.g. MEMTO_GPU=1 to use the
#       second, idle GPU instead of the busy one).
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if os.environ.get("MEMTO_GPU", "").strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["MEMTO_GPU"].strip()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# ----------------------------------------------------------------------------
# Make the local MEMTO `model` / `utils` packages importable regardless of CWD.
# ----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model.Transformer import TransformerVar          # noqa: E402
from model.loss_functions import EntropyLoss, GatheringLoss  # noqa: E402
from utils.utils import k_means_clustering             # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    DATA_PATH = os.path.join(_THIS_DIR, "..", "m1", "m1_data", "all_telemetry_41_46.parquet")

    # --- windowing ---
    WIN_SIZE = 100          # upstream main.py default win_size=100
    STEP = WIN_SIZE         # non-overlapping windows (matches data_factory's default step=win_size)

    # --- model hyperparameters (upstream defaults / test.sh values) ---
    D_MODEL = 512            # upstream main.py default d_model=512
    N_MEMORY = 10             # upstream test.sh uses n_memory=10 for real datasets (main.py argparse default is 128)
    E_LAYERS = 3              # upstream default e_layers=3 (hardcoded in solver.build_model)
    N_HEADS = 8               # upstream TransformerVar default n_heads=8
    LAMBD = 0.01              # upstream test.sh --lambd 0.01 (entropy loss weight)
    TEMPERATURE = 0.1         # upstream main.py default --temperature 0.1
    LR_PHASE1 = 1e-4          # upstream test.sh phase-1 (mode=train) --lr 1e-4
    LR_PHASE2 = 5e-5          # upstream test.sh phase-2 (mode=memory_initial) --lr 5e-5
    BATCH_SIZE = 256          # upstream test.sh --batch_size 256

    # --- training schedule ---
    # Upstream main.py default num_epochs=10 (and test.sh uses num_epochs=100 for the
    # real training runs). We use 100 epochs with early stopping (patience 10), matching
    # the protocol used by the TranAD/Anomaly-Transformer/usad baselines in this repo.
    EPOCHS = 100
    PATIENCE = 10

    # --- validation / k-means split (mirrors data_factory.get_loader_segment) ---
    VAL_RATIO = 0.2           # upstream default val_ratio=0.2
    K_RATIO = 0.1             # upstream: k_loader uses first 10% of the train-subset windows

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_NAME = "M1_MEMTO"
    CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints_m1")
    MEMORY_ITEM_DIR = os.path.join(_THIS_DIR, "memory_item")
    CHECKPOINT_PATH = os.path.join(_THIS_DIR, "best_memto_m1.pt")  # final (phase-2 best) weights, saved flat too

    SEED = 42

    # --- OOM safety ---
    # Cap on how many query VECTORS (windows x win_size) are fed to k-means.
    # Upstream uses every k_loader query; on a 16 GB card the full set can be
    # several GB, so we deterministically subsample above this cap (see
    # run_kmeans_safely). 500k x 512 floats is about 1.0 GB.
    KMEANS_MAX_POINTS = 500_000


# ============================================================================
# UTILS
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    """Simple (window, window_label) dataset. Labels are unused during MEMTO
    training (the loss is purely reconstruction + entropy, unsupervised), but are
    kept alongside for evaluation-time convenience."""

    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def make_windows(data, labels, win_size, step):
    """Non-overlapping (stride == win_size by default) windowing.
    Label per window = max over the window (documented in README.md)."""
    n = len(data)
    if n < win_size:
        # Degenerate/tiny split (can happen in smoke-test mode): pad by repeating
        # the last row so we still get exactly one window.
        pad = win_size - n
        pad_block = np.repeat(data[-1:, :], pad, axis=0) if n > 0 else np.zeros((win_size, data.shape[1]), dtype=data.dtype)
        data = np.concatenate([data, pad_block], axis=0) if n > 0 else pad_block
        pad_labels = np.repeat(labels[-1:], pad, axis=0) if n > 0 else np.zeros(win_size, dtype=labels.dtype)
        labels = np.concatenate([labels, pad_labels], axis=0) if n > 0 else pad_labels
        n = win_size

    n_windows = (n - win_size) // step + 1
    n_feats = data.shape[1]
    X = np.zeros((n_windows, win_size, n_feats), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.int64)
    for i in range(n_windows):
        s = i * step
        X[i] = data[s:s + win_size]
        y[i] = int(labels[s:s + win_size].max())
    return X, y


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
# DATA LOADING
# ============================================================================

def load_data(config, is_smoke_test=False):
    data_path = config.DATA_PATH
    if not os.path.exists(data_path):
        alt = "all_telemetry_41_46.parquet"
        if os.path.exists(alt):
            data_path = alt
        else:
            raise FileNotFoundError(f"Data file not found at: {config.DATA_PATH}")

    df = pd.read_parquet(data_path)
    features = [c for c in df.columns if c.startswith("chchannel_")]
    label_col = "label"

    if is_smoke_test:
        print("\n!!! SMOKE TEST MODE !!! Subsampling data aggressively.\n")
        df = df.iloc[:10000].copy()
        # Inject a small synthetic anomaly block so metrics are computable even if
        # the first 10k rows have no positives (mirrors multi_seed_eval/m1_multiseed.py).
        label_pos = df.columns.get_loc(label_col)
        df.iloc[500:601, label_pos] = 1
        df.iloc[9000:9101, label_pos] = 1

    data = df[features].values.astype(np.float32)
    labels = df[label_col].values.astype(np.int64)

    n = len(data)
    split_idx = int(n * 0.5)

    train_data_raw = data[:split_idx]
    train_labels = labels[:split_idx]
    test_data_raw = data[split_idx:]
    test_labels = labels[split_idx:]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data_raw).astype(np.float32)
    test_data = scaler.transform(test_data_raw).astype(np.float32)

    print("[Data Split] 50/50 Chronological")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")
    print(f"Features: {len(features)}")

    win_size = config.WIN_SIZE
    if is_smoke_test:
        win_size = min(config.WIN_SIZE, max(10, len(train_data) // 4))
        config.WIN_SIZE = win_size
        config.STEP = win_size

    train_windows, train_win_labels = make_windows(train_data, train_labels, win_size, config.STEP)
    test_windows, test_win_labels = make_windows(test_data, test_labels, win_size, config.STEP)

    print(f"Train windows: {len(train_windows)} (win_size={win_size})")
    print(f"Test windows:  {len(test_windows)}")
    if is_smoke_test:
        print(f"Test windows with positive label: {int(test_win_labels.sum())}/{len(test_win_labels)}")

    return train_windows, train_win_labels, test_windows, test_win_labels, len(features)


# ============================================================================
# TWO-PHASE MEMTO TRAINING (faithful re-implementation of solver.py)
# ============================================================================

# ============================================================================
# GPU MEMORY / OOM SAFETY HELPERS
# ============================================================================

def _is_oom(exc):
    """True if this exception is a CUDA out-of-memory error."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_mem(stage):
    if torch.cuda.is_available():
        print(f"[GPU] {stage}: peak allocated {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
              f"reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def _train_step(model, x, entropy_loss_fn, criterion, config, weight=1.0):
    """Forward+backward for one (sub)batch -> (weighted loss, memory embedding).

    On CUDA OOM the batch is split in half and the halves are gradient-accumulated
    instead, recursively down to a single sample. Loss contributions are weighted by
    sub-batch fraction, so the accumulated gradient matches the full-batch one.
    Nothing is skipped: if even a single sample cannot fit, the error is re-raised.
    """
    try:
        out = model(x)
        loss = criterion(out["out"], x) + config.LAMBD * entropy_loss_fn(out["attn"])
        (loss * weight).backward()
        return loss.item() * weight, out["memory_item_embedding"]
    except RuntimeError as exc:
        if (not _is_oom(exc)) or x.shape[0] < 2:
            raise
        _free_cuda()
        n = x.shape[0]
        half = n // 2
        print(f"      [OOM-SAFE] train sub-batch of {n} did not fit -> splitting in half")
        loss_a, mem_a = _train_step(model, x[:half], entropy_loss_fn, criterion, config, weight * half / n)
        loss_b, mem_b = _train_step(model, x[half:], entropy_loss_fn, criterion, config, weight * (n - half) / n)
        return loss_a + loss_b, (mem_b if mem_b is not None else mem_a)


def _eval_map(model, x, fn):
    """Run fn(model(x), x) under no_grad with OOM protection.

    fn must return a tensor whose first dimension is the batch, so split results
    can be concatenated. On OOM the batch is halved recursively.
    """
    try:
        with torch.no_grad():
            return fn(model(x), x)
    except RuntimeError as exc:
        if (not _is_oom(exc)) or x.shape[0] < 2:
            raise
        _free_cuda()
        n = x.shape[0]
        half = n // 2
        print(f"      [OOM-SAFE] eval batch of {n} did not fit -> splitting in half")
        a = _eval_map(model, x[:half], fn)
        b = _eval_map(model, x[half:], fn)
        return torch.cat([a, b], dim=0)


def run_kmeans_safely(queries_cpu, config):
    """k-means over encoder queries, degrading instead of crashing.

    Fallback order: GPU k-means -> GPU k-means on progressively halved subsamples
    -> CPU k-means (slower, but cannot OOM the card). utils/utils.py hardcodes
    cuda:0, so the CPU fallback calls kmeans_pytorch directly with the same
    settings (euclidean distance, same cluster count).
    """
    points = queries_cpu
    while True:
        try:
            _free_cuda()
            centers = k_means_clustering(x=points.to(config.DEVICE), n_mem=config.N_MEMORY,
                                         d_model=config.D_MODEL)
            return centers.detach()
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            _free_cuda()
            if points.shape[0] > max(1000, 4 * config.N_MEMORY):
                points = points[::2].contiguous()
                print(f"      [OOM-SAFE] k-means hit OOM -> retrying with {points.shape[0]} points")
                continue
            print("      [OOM-SAFE] k-means still OOM -> falling back to CPU k-means (slower)")
            from kmeans_pytorch import kmeans
            _, centers = kmeans(X=points.view(-1, config.D_MODEL), num_clusters=config.N_MEMORY,
                                distance="euclidean", device=torch.device("cpu"))
            return centers.detach()


def build_model(config, n_feats, memory_initial, memory_init_embedding, phase_type):
    model = TransformerVar(
        win_size=config.WIN_SIZE,
        enc_in=n_feats,
        c_out=n_feats,
        e_layers=config.E_LAYERS,
        d_model=config.D_MODEL,
        n_memory=config.N_MEMORY,
        n_heads=config.N_HEADS,
        device=config.DEVICE,
        memory_initial=memory_initial,
        memory_init_embedding=memory_init_embedding,
        phase_type=phase_type,
        dataset_name=config.DATASET_NAME,
    ).to(config.DEVICE)
    return model


def split_train_val(windows, config):
    """Mirrors data_factory.get_loader_segment's mode='train' split: a contiguous
    val_ratio-sized block held out for validation/early stopping, plus a k_loader
    subset (first k_ratio of the train-subset) used for k-means input.

    Deviation from upstream: upstream picks a *random* contiguous validation block
    (random.randrange); we deterministically hold out the *last* val_ratio windows
    (chronological) for reproducibility across runs/seeds, matching how the other
    baseline scripts in this repo build their validation splits.
    """
    n = len(windows)
    val_len = max(1, int(n * config.VAL_RATIO))
    train_len = n - val_len

    train_windows = windows[:train_len]
    val_windows = windows[train_len:]

    k_len = max(1, int(train_len * config.K_RATIO))
    k_windows = train_windows[:k_len]

    return train_windows, val_windows, k_windows


def run_train_phase(model, train_loader, val_loader, lr, config, phase_name, checkpoint_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    entropy_loss_fn = EntropyLoss()
    criterion = nn.MSELoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    last_memory_item_embedding = None

    print(f"\n===================== {phase_name} =====================")
    start_time = time.time()
    _reset_peak()

    for epoch in range(config.EPOCHS):
        model.train()
        train_losses = []
        for x, _ in train_loader:
            x = x.to(config.DEVICE).float()
            optimizer.zero_grad()
            # OOM-safe: identical math to a plain forward/backward when memory is
            # sufficient; splits the batch and accumulates gradients if it is not.
            loss_value, mem_embedding = _train_step(model, x, entropy_loss_fn, criterion, config)
            if mem_embedding is not None:
                last_memory_item_embedding = mem_embedding
            optimizer.step()
            train_losses.append(loss_value)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(config.DEVICE).float()
                output_dict = model(x)
                output = output_dict["out"]
                attn = output_dict["attn"]
                rec_loss = criterion(output, x)
                ent_loss = entropy_loss_fn(attn)
                loss = rec_loss + config.LAMBD * ent_loss
                val_losses.append(loss.item())
        mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss

        print(f"[{phase_name}] Epoch [{epoch + 1}/{config.EPOCHS}] "
              f"Train Loss: {mean_train_loss:.6f}  Val Loss: {mean_val_loss:.6f}")

        if np.isnan(mean_train_loss):
            print(f"[{phase_name}] NaN loss detected - stopping.")
            break

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"[{phase_name}] Early stopping triggered after {epoch + 1} epochs "
                      f"(no val-loss improvement for {config.PATIENCE} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[{phase_name}] Training complete in {(time.time() - start_time) / 60:.2f} mins")
    _gpu_mem(phase_name)
    _free_cuda()
    return model, last_memory_item_embedding, ckpt_path


def train_memto(train_windows, train_win_labels, n_feats, config):
    """Faithful two-phase MEMTO training:
      Phase 1 (first_train): random memory init, full model trained (lr=LR_PHASE1).
      K-means step: encoder queries from a k-loader subset -> k-means -> memory init.
      Phase 2 (second_train): memory re-initialized from k-means centers, model
        fine-tuned further (lr=LR_PHASE2), memory continues to adapt via the
        gating-update rule (model/ours_memory_module.py MemoryModule.update).
    Returns the phase-2 checkpoint path and the final memory item tensor (saved to
    disk, exactly like upstream solver.get_memory_initial_embedding).
    """
    train_sub, val_sub, k_sub = split_train_val(train_windows, config)
    train_labels_sub = train_win_labels[:len(train_sub)]
    val_labels_sub = train_win_labels[len(train_sub):len(train_sub) + len(val_sub)]

    train_ds = WindowDataset(train_sub, train_labels_sub)
    val_ds = WindowDataset(val_sub, val_labels_sub)
    k_ds = WindowDataset(k_sub, train_win_labels[:len(k_sub)])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    k_loader = DataLoader(k_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---------------- Phase 1: first_train (random memory init) ----------------
    model1 = build_model(config, n_feats, memory_initial=False, memory_init_embedding=None, phase_type="None")
    model1, _, phase1_ckpt = run_train_phase(
        model1, train_loader, val_loader, config.LR_PHASE1, config,
        phase_name="PHASE 1 (first_train, random memory init)",
        checkpoint_name=f"{config.DATASET_NAME}_checkpoint_first_train.pth",
    )

    # ---------------- K-means memory initialization ----------------
    print("\n===================== K-MEANS MEMORY INITIALIZATION =====================")
    _reset_peak()
    model1.eval()
    # Collect encoder queries on the CPU (upstream keeps every batch on the GPU,
    # which is the single largest allocation of the whole run).
    query_chunks = []
    for x, _ in k_loader:
        x = x.to(config.DEVICE).float()
        q = _eval_map(model1, x, lambda out, _x: out["queries"].detach())
        query_chunks.append(q.reshape(-1, config.D_MODEL).cpu())
        del x, q
    queries_cpu = torch.cat(query_chunks, dim=0)
    del query_chunks

    # Free the phase-1 model before k-means: it is not needed again, and this is
    # where peak memory would otherwise overlap.
    model1.to("cpu")
    del model1
    _free_cuda()

    collected = queries_cpu.shape[0]
    if collected > config.KMEANS_MAX_POINTS:
        # Deterministic (strided) subsample -> identical result on every rerun.
        stride = (collected + config.KMEANS_MAX_POINTS - 1) // config.KMEANS_MAX_POINTS
        queries_cpu = queries_cpu[::stride].contiguous()
        print(f"      -> k-means input capped: {queries_cpu.shape[0]} of {collected} query vectors "
              f"(stride {stride}, cap KMEANS_MAX_POINTS={config.KMEANS_MAX_POINTS})")
    else:
        print(f"      -> k-means input: {collected} query vectors (under cap, no subsampling)")

    memory_init_embedding = run_kmeans_safely(queries_cpu, config)
    del queries_cpu
    _gpu_mem("k-means")
    _free_cuda()

    # ---------------- Phase 2: second_train (k-means memory init, fine-tune) ----------------
    model2 = build_model(config, n_feats, memory_initial=False,
                          memory_init_embedding=memory_init_embedding, phase_type="second_train")
    model2, last_mem_embedding, phase2_ckpt = run_train_phase(
        model2, train_loader, val_loader, config.LR_PHASE2, config,
        phase_name="PHASE 2 (second_train, k-means memory init)",
        checkpoint_name=f"{config.DATASET_NAME}_checkpoint_second_train.pth",
    )

    # Save the final memory item snapshot to disk, exactly like upstream
    # solver.get_memory_initial_embedding (mem is a plain tensor attribute, NOT part
    # of model.state_dict(), so it must be persisted separately). Note this is the
    # *last training-step* snapshot of memory (not the best-val-loss epoch's memory);
    # this quirk is preserved from upstream, not "fixed", per the task instructions.
    os.makedirs(config.MEMORY_ITEM_DIR, exist_ok=True)
    if last_mem_embedding is None:
        last_mem_embedding = model2.mem_module.mem.detach()
    mem_to_save = last_mem_embedding[: config.N_MEMORY, :].detach().cpu()
    mem_item_path = os.path.join(config.MEMORY_ITEM_DIR, f"{config.DATASET_NAME}_memory_item.pth")
    torch.save(mem_to_save, mem_item_path)

    # Also save a flat "final" checkpoint for convenience/consistency with the other
    # baseline scripts in this repo (which all save a single best_*.pt file).
    torch.save(model2.state_dict(), config.CHECKPOINT_PATH)

    # Release phase-2 weights before the test-phase model is built.
    model2.to("cpu")
    del model2
    _free_cuda()

    return phase2_ckpt, mem_item_path


# ============================================================================
# TESTING (GatheringLoss-based anomaly score, as in solver.py's test())
# ============================================================================

def evaluate_memto(test_windows, test_win_labels, n_feats, phase2_ckpt_path, mem_item_path, config):
    # Test-time model: phase_type='test' makes MemoryModule load memory items from
    # disk (mem_item_path, via the dataset_name convention baked into
    # model/ours_memory_module.py) and skips the memory-update step
    # (`if self.phase_type != 'test': self.update(query)`), so the memory is frozen
    # at test time exactly as in upstream.
    assert os.path.exists(mem_item_path), f"Expected memory item file at {mem_item_path}"
    _free_cuda()
    _reset_peak()
    model = build_model(config, n_feats, memory_initial=False, memory_init_embedding=None, phase_type="test")

    state = torch.load(phase2_ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    test_ds = WindowDataset(test_windows, test_win_labels)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss(reduce=False)
    gathering_loss = GatheringLoss(reduce=False)
    temperature = config.TEMPERATURE

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.DEVICE).float()
            output_dict = model(x)
            output, queries, mem_items = output_dict["out"], output_dict["queries"], output_dict["mem"]

            rec_loss = torch.mean(criterion(x, output), dim=-1)                     # N x L
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)  # N x L
            loss = latent_score * rec_loss                                          # N x L

            # Aggregate the per-timestep MEMTO score to one score per window (mean
            # over time), to match the window-level (max-aggregated) labels used by
            # this evaluation protocol. Upstream instead flattens per-timestep
            # scores against per-timestep labels (see solver.py test()); we deviate
            # here so MEMTO's "FINAL RESULTS" block reports one score+label per
            # window, consistent with the other baselines in this repo.
            window_score = loss.mean(dim=1).detach().cpu().numpy()
            all_scores.append(window_score)
            all_labels.append(y.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return scores, labels


# ============================================================================
# MAIN
# ============================================================================

def main(epochs_override=None, is_smoke_test=False):
    conf = Config()
    set_seed(conf.SEED)

    if epochs_override is not None:
        conf.EPOCHS = epochs_override
        conf.PATIENCE = max(1, epochs_override)

    if is_smoke_test:
        conf.DATASET_NAME = "M1_MEMTO_SMOKE"
        conf.N_MEMORY = min(conf.N_MEMORY, 4)
        conf.D_MODEL = 64
        conf.BATCH_SIZE = min(conf.BATCH_SIZE, 32)

    train_windows, train_win_labels, test_windows, test_win_labels, n_feats = load_data(conf, is_smoke_test=is_smoke_test)

    phase2_ckpt, mem_item_path = train_memto(train_windows, train_win_labels, n_feats, conf)
    scores, labels = evaluate_memto(test_windows, test_win_labels, n_feats, phase2_ckpt, mem_item_path, conf)

    best_thresh, best_f1 = find_best_threshold(scores, labels)
    preds = (scores > best_thresh).astype(int)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    has_positives = bool(labels.sum() > 0) and bool(labels.sum() < len(labels))
    if has_positives:
        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.5
    else:
        auc = float("nan")
        print("WARNING: test labels have no positive (or no negative) class; AUC skipped.")

    print("\n" + "=" * 50)
    print("FINAL RESULTS (M1 MEMTO)")
    print("=" * 50)
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"AUC:           {auc:.4f}" if has_positives else "AUC:           N/A")
    print(f"Threshold:     {best_thresh:.6f}")
    print("=" * 50)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print("\nConfusion Matrix:")
    print(cm)

    if is_smoke_test:
        print("\n!!! SMOKE TEST MODE COMPLETE !!!")


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
    set_seed(conf.SEED)
    values, split, stride = hr.load_honest_inputs(args, kind)
    scaled = hr.scale_with_fit(values, split)          # scaler fitted on the FIT slice only
    a, b = hr.split_io.part_range(split, "fit")
    fit_data = scaled[a:b]
    # published runner setting: non-overlapping windows (step = WIN_SIZE); zero labels (unsupervised)
    train_windows, train_win_labels = make_windows(fit_data, np.zeros(len(fit_data), dtype=np.int64),
                                                   conf.WIN_SIZE, conf.STEP)
    n_feats = fit_data.shape[1]
    # train_memto holds out the last VAL_RATIO of the FIT windows for early stopping (unsupervised loss)
    phase2_ckpt, mem_item_path = train_memto(train_windows, train_win_labels, n_feats, conf)

    def score_fn(X):
        return evaluate_memto(X, np.zeros(len(X), dtype=np.int64), n_feats, phase2_ckpt, mem_item_path, conf)[0]

    parts = hr.score_parts(score_fn, scaled, split, conf.WIN_SIZE, stride, chunk=None)
    hr.export_scores(args, split, parts, conf.WIN_SIZE, stride, "memto", mission)

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
    Config.CHECKPOINT_DIR = os.path.join(_a.out, os.path.basename(Config.CHECKPOINT_DIR))
    Config.MEMORY_ITEM_DIR = os.path.join(_a.out, "memory_item")
    if _mode == "honest":
        honest_main(_a, "esa", "m1")
    else:
        main()
