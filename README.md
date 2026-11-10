# ATAN-DN: Mission-Agnostic Attention Network with Divisive Normalization for Operational Satellite Anomaly Detection

Official PyTorch implementation of **ATAN-DN** (Attention-Augmented Temporal Anomaly Network with Divisive Normalization) for operational spacecraft telemetry monitoring.

---

## Abstract

Robust telemetry anomaly detection is a cornerstone of autonomous satellite mission operations and Spacecraft Constellation State-of-Health monitoring. Spacecraft telemetry streams present severe operational challenges: complex cross-channel coupling, non-Gaussian sensor noise, extreme class imbalance (with anomalies often comprising less than 0.01% of flight time), and planned spacecraft operations (such as orbital maneuvers and thruster firings) that trigger extensive false alarms. 

This repository provides the complete implementation of **ATAN-DN**, a unified, mission-agnostic deep learning architecture combining:
1. **Bounded-Influence Gain Normalization (BIGN):** A bio-inspired divisive normalization layer providing provably bounded activation norms, finite kurtosis, and asymptotic common-mode noise rejection to prevent attention softmax saturation.
2. **Dual-Head Architecture:** Joint supervision combining a temporal classification head with an autoencoder reconstruction head governed by an exact algebraic linear-system inverse BIGN layer.
3. **Reconstruction-Regularized Weighted Focal Loss (RR-WFL):** An imbalance-aware objective with label smoothing ($\epsilon_{\text{ls}} = 0.05$) and hard-sample mining.
4. **Three-Stage Operational Pipeline:** Strict chronological splitting (50/50 for ESA-AD Missions 1 & 2, 75/25 for OPS-SAT), automated Stage 2 settling buffer derivation strictly from training history, and Stage 3 false alarm elimination with zero test-split leakage.

Evaluated on multi-year flight streams from **ESA-AD Mission 1** (14 years), **ESA-AD Mission 2** (3.5 years), and **OPS-SAT** (4.5 years), ATAN-DN establishes state-of-the-art detection performance with zero false alarms ($\mathrm{FPR} = 0.0000$) on ESA-AD and millisecond-level inference latency.

---

## Key Architectural Components

```
Telemetry Window X in R^(B x C x T)
      │
      ▼
Anti-Drift Window Normalization (Causal local stats)
      │
      ▼
Encoder Blocks [x2]:
  Conv1D (k=5, s=2) ──► BatchNorm ──► BIGN ──► Multi-Head Attention ──► Squeeze-and-Excitation ──► ReLU
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
Latent Bottleneck (Attention + LayerNorm)   Decoder Blocks [x2]:
      │                                       ConvTranspose1D ──► BatchNorm ──► Exact Inverse BIGN ──► ReLU
      ▼                                         │
Classifier Head:                                ▼
  Linear ──► LayerNorm ──► GELU [x3]         Reconstructed Window X_hat
      │                                         │
      ▼                                         ▼
Anomaly Probability p(t)                    Reconstruction Error e_rec(t)
      └─────────────────────┬───────────────────┘
                            ▼
              Dual-Evidence Score s(t)
                            │
                            ▼
           Stage 3 Operational Filtering
        (Maneuver Buffers + Spike Filtering)
                            │
                            ▼
              Calibrated Decision y_hat(t)
```

---

## Repository Structure

```
repo/
├── atandn/                       # Core ATAN-DN library
│   ├── layers/
│   │   ├── bign.py               # BIGN forward and exact algebraic inverse layers
│   │   ├── attention.py          # 1D Multi-Head Attention, Squeeze-and-Excitation
│   │   └── norms.py              # Normalization layers for ablation benchmarks
│   ├── models/
│   │   └── atandn.py             # Complete ATAN-DN dual-head model
│   ├── data/
│   │   ├── windows.py            # Streaming causal windowing and dataset utilities
│   │   ├── splits.py             # Strict chronological and event-aware splitters
│   │   ├── esa.py                # ESA-ADB telemetry loader
│   │   └── opssat.py             # OPS-SAT benchmark telemetry loader
│   ├── losses.py                 # RR-WFL loss formulation and focal loss
│   ├── training/
│   │   ├── trainer.py            # AdamW trainer with cosine restarts and EMA
│   │   └── search.py             # Automated Bayesian TPE hyperparameter search
│   └── inference/
│       ├── scoring.py            # Dual-evidence score computation
│       ├── threshold.py          # Validation-grid F1-optimal threshold calibration
│       └── postprocess.py        # Operational maneuver buffers and spike filters
├── configs/                      # Mission configurations and ablation specifications
│   ├── m1.yaml                   # ESA Mission 1 configuration
│   ├── m2.yaml                   # ESA Mission 2 configuration
│   ├── ops_sat.yaml              # OPS-SAT configuration
│   ├── ablation_table6.yaml      # BIGN vs Alternative Normalizations
│   └── obc_hardware.yaml         # On-board computing platforms specification
├── experiments/                  # Reusable research experiment suites
│   ├── multi_seed/               # Multi-seed evaluation runner (seeds: 100, 777, 2026)
│   └── ablation/                 # Comprehensive architectural and component ablations
├── baselines/                    # Standard spacecraft anomaly detection baselines
│   ├── tranad/                   # TranAD implementation & adapter
│   ├── usad/                     # USAD implementation & adapter
│   ├── anomaly_transformer/      # Anomaly Transformer implementation & adapter
│   └── memto/                    # MEMTO implementation & adapter
├── scripts/                      # Command-line entry points
│   ├── prepare_data.py           # Ingest raw ESA-ADB / OPS-SAT files to Parquet
│   ├── train.py                  # Train ATAN-DN models
│   ├── evaluate.py               # Run frozen evaluation protocol
│   ├── tune.py                   # Run Bayesian hyperparameter optimization
│   ├── complexity.py             # Parameters, FLOPs, and OBC latency profiling
│   └── run_pipeline.py           # End-to-end mission pipeline orchestrator
├── tests/                        # Comprehensive unit and integration test suite
├── requirements.txt              # Production runtime dependencies
└── pyproject.toml                # Package configuration
```

---

## Installation

### Prerequisites
- Python $\ge 3.9$
- PyTorch $\ge 2.2$ (tested on PyTorch 2.2–2.8, CPU and CUDA)

```bash
# Clone the repository
git clone https://github.com/<your-username>/ATAN-DN.git
cd ATAN-DN

# Install dependencies
pip install -r requirements.txt

# (Optional) Install the package in editable mode
pip install -e .
```

Verify installation:
```bash
pytest tests/
```

---

## Data Ingestion & Preprocessing

The benchmarks use telemetry data from **ESA-ADB** (European Space Agency Anomaly Detection Benchmark) and **OPS-SAT**.

1. **ESA-ADB (Mission 1 and Mission 2):**
   Place raw mission folders (containing `channels.csv`, `labels.csv`, `anomaly_types.csv`, and `channels/`) in your data directory:
   ```bash
   # Process ESA Mission 1 (Subsystem 5: Channels 41-46, 0.033 Hz)
   python scripts/prepare_data.py --config configs/m1.yaml --raw-dir data/m1/raw --data-root data

   # Process ESA Mission 2 (Subsystem 1: Channels 18-28, 0.056 Hz)
   python scripts/prepare_data.py --config configs/m2.yaml --raw-dir data/m2/raw --data-root data
   ```

2. **OPS-SAT (OPSSAT-AD):**
   Place `segments.csv` under `data/ops_sat/segments.csv`:
   ```bash
   python scripts/prepare_data.py --config configs/ops_sat.yaml --data-root data
   ```

---

## How to Run

### 1. Model Training
Train ATAN-DN under the strict chronological split using the optimized mission configuration:
```bash
python scripts/train.py --config configs/m1.yaml --data-root data --out-dir runs/m1
```
This performs:
- Chronological partitioning with an event-aware validation partition carved strictly from training data.
- AdamW optimization with linear warmup, cosine restarts, and Exponential Moving Average (EMA).
- Automated derivation of Stage 2 operational filters (P95 settling buffers and noise cluster distributions).
- Protocol freezing in `runs/m1/model.pt`.

### 2. Frozen Protocol Evaluation
Evaluate the trained model on the blind test split:
```bash
python scripts/evaluate.py --config configs/m1.yaml --data-root data --out-dir runs/m1
```
Reports both **Point-Adjusted (PA)** event scoring (matching ESA-ADB Supplementary Table 9) and unadjusted window-level / point-wise metrics.

### 3. End-to-End Automated Pipeline
To execute data preparation, training, evaluation, and latency profiling sequentially:
```bash
python scripts/run_pipeline.py --config configs/m1.yaml --data-root data --out-dir runs/m1
```

### 4. Multi-Seed Robustness Evaluation (Table 2 / Section 6)
To evaluate statistical stability across seeds $\{100, 777, 2026\}$ with frozen parameters:
```bash
python experiments/multi_seed/run_multi_seed.py \
    --config configs/m1.yaml \
    --data-root data/m1 \
    --out-dir results/multi_seed_m1 \
    --seeds 100 777 2026
```

### 5. Ablation Studies (Table 6)
To reproduce the normalization layer isolation benchmark (BIGN vs. BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm, Identity):
```bash
python experiments/ablation/run_ablation.py \
    --config configs/m1.yaml \
    --study configs/ablation_table6.yaml \
    --data-root data/m1 \
    --out-dir results/ablation_norm_m1

# Summarize LaTeX Table 6
python experiments/ablation/summarize.py results/ablation_norm_m1/ablation_table6_results.json
```

### 6. Baseline Comparisons (Table 3)
Run standardized baselines (TranAD, USAD, Anomaly Transformer, MEMTO) under identical chronological splits and operational filters:
```bash
# Example: TranAD on Mission 1
python baselines/run_baseline.py --model tranad --mission m1 --data-root data --out-dir runs/baselines/m1_tranad
```

### 7. Computational Complexity & OBC Hardware Latency (Tables 6 & 7)
Calculate parameters, FLOPs, and latency across physical hardware and flight-grade onboard computers:
```bash
python scripts/complexity.py \
    --config configs/m1.yaml \
    --checkpoint runs/m1/model.pt \
    --project configs/obc_hardware.yaml \
    --out runs/m1/complexity.json
```

---

## Evaluation Methodology & Zero-Leakage Guarantee

- **Strict Chronological Splitting:** Telemetry is split chronologically into 50% training and 50% testing (75/25 for OPS-SAT) *prior* to sliding window construction. No future samples contaminate the past.
- **Event-Aware Validation Block:** Model selection, early stopping, and $F_1$-optimal decision thresholds ($\tau^\ast$) are derived strictly from a held-out validation block inside the training split.
- **Zero Test-Split Leakage in Stage 2/3:** Maneuver buffers and noise cluster thresholds are computed entirely on training history. At test time, these frozen parameters are applied directly without test label access.
- **Fair Baseline Protocol:** All baselines are re-trained and evaluated on identical chronological splits with identical Stage 3 operational filtering.

---

## Citation

If you find this code or our paper useful in your research, please cite:

```bibtex
@article{alzaidy2026atandn,
  title={ATAN-DN: Mission-Agnostic Attention Network with Divisive Normalization for Operational Satellite Anomaly Detection},
  author={Al-Zaidy, Ahmad M. and Gaber, Mohamed Medhat and Awad, Mohamed Ibrahim and Elhady, Nancy E.},
  journal={Springer Nature},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
Upstream baseline codes belong to their respective original authors.
