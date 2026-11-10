"""Tests of the command-line scripts: --help, and a tiny synthetic end-to-end run (tune/train/evaluate)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT_NAMES = ["prepare_data", "tune", "train", "evaluate", "complexity", "run_pipeline"]

CONFIG = """
mission: m1
name: m1
data:
  kind: esa
  path: data/m1/all_telemetry_41_46.parquet
  train_ratio: 0.5
model:
  in_channels: 2
  window_size: 32
  conv_filters: [8, 16]
  num_heads: 2
train:
  max_epochs: 2
  warmup_epochs: 1
  patience: 5
  seed: 7
eval:
  use_known_event_mask: true
  mask_source: rare_events
  min_consecutive: 1
search:
  n_trials: 2
  seed: 1
  fixed:
    window_size: 32
    overlap: 0.5
    batch_size: 16
    num_heads: 2
    conv_f1: 8
    conv_f2: 16
    use_weighted_sampling: false
    use_augmentation: false
"""


def _make_data(root: Path, zero_test_labels: bool,
               events: tuple = ((100, 130), (520, 545), (800, 830), (1000, 1040))) -> None:
    """Synthetic 2-channel stream (1200 samples). Anomalies in both halves; optionally none in the test half."""
    rng = np.random.default_rng(0)
    n = 1200
    t = np.arange(n)
    values = np.stack([np.sin(t / 7.0), np.cos(t / 11.0)], axis=1) + 0.05 * rng.standard_normal((n, 2))
    labels = np.zeros(n)
    for a, b in events:
        values[a:b] += 4.0
        labels[a:b] = 1.0
    if zero_test_labels:
        labels[n // 2:] = 0.0
    idx = pd.date_range("2020-01-01", periods=n, freq="30s")
    df = pd.DataFrame({"chchannel_41": values[:, 0], "chchannel_42": values[:, 1], "label": labels}, index=idx)
    d = root / "data" / "m1"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "all_telemetry_41_46.parquet")
    pd.DataFrame({"ID": ["e1"], "Channel": ["channel_41"], "StartTime": [str(idx[300])],
                  "EndTime": [str(idx[320])]}).to_csv(d / "labels.csv", index=False)
    pd.DataFrame({"ID": ["e1"], "Category": ["Rare Event"]}).to_csv(d / "anomaly_types.csv", index=False)


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_help(name: str) -> None:
    res = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert "--config" in res.stdout


def test_missing_config_exit_code(tmp_path: Path) -> None:
    import tune

    assert tune.main(["--config", str(tmp_path / "nope.yaml")]) == 2


def test_end_to_end_smoke(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    import complexity
    import evaluate
    import train
    import tune

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(CONFIG, encoding="utf-8")
    real, zero = tmp_path / "real", tmp_path / "zero"
    _make_data(real, zero_test_labels=False)
    _make_data(zero, zero_test_labels=True)
    out = tmp_path / "out"
    base = ["--config", str(cfg_path), "--data-root", str(real), "--out-dir", str(out), "--device", "cpu"]

    assert tune.main(base + ["--n-trials", "2"]) == 0
    best = json.loads((out / "best_params.json").read_text())
    assert best["n_trials"] == 2 and best["params"]["window_size"] == 32
    assert train.main(base) == 0
    assert (out / "model.pt").is_file() and (out / "mask_params.json").is_file()
    assert evaluate.main(base) == 0

    rep = json.loads((out / "report.json").read_text())
    assert rep["threshold_source"] == "validation" and rep["protocol"]["threshold_source"] == "validation"
    assert rep["protocol"]["normalizer_source"].startswith("training")
    assert rep["used_legacy_defaults"] is False
    assert rep["n_test_events"] == 2 and rep["n_test_pos_samples"] == 70 and rep["n_test_pos_windows"] > 0
    assert rep["validation"]["val_has_positives"] is True and rep["n_val_pos_windows"] > 0
    assert rep["protocol"]["test_labels_used_for_fitting"] is False
    assert rep["n_test_windows"] > 0 and rep["n_train_windows"] > 0
    assert rep["postprocessing"]["known_event_mask"] is True
    assert {"f1", "f0_5", "tp", "fp", "tn", "fn"} <= set(rep["window_metrics"])
    assert "f1" in rep["pointwise_metrics_rule_A"]
    assert rep["esa_adb_event_wise"]["computed"] is False   # flag not given

    # Same checkpoint, test labels replaced by all zeros: threshold/normaliser/buffer must not change.
    rep2_path = tmp_path / "report_zero.json"
    zbase = ["--config", str(cfg_path), "--data-root", str(zero), "--out-dir", str(out), "--device", "cpu"]
    assert evaluate.main(zbase + ["--report", str(rep2_path)]) == 0
    rep2 = json.loads(rep2_path.read_text())
    assert rep2["threshold"] == rep["threshold"]
    assert rep2["normalizer"] == rep["normalizer"]
    assert rep2["postprocessing"]["mask_buffer_after_minutes"] == rep["postprocessing"]["mask_buffer_after_minutes"]
    assert rep2["postprocessing"]["mask_buffer_before_minutes"] == 0.0
    assert rep2["n_test_windows"] == rep["n_test_windows"]
    assert rep2["n_test_events"] == 0 and rep2["n_test_pos_samples"] == 0 and rep2["n_test_pos_windows"] == 0

    # Checkpoint is the source of truth: the experiment path (frozen=None -> refit from the training split) gives the
    # same metrics as evaluate.py, which uses the values frozen in the checkpoint (same function, same data).
    from atandn.data.esa import load_esa_mission
    from atandn.evaluation.pipeline import evaluate_mission, prepare_protocol_data
    from atandn.training import load_checkpoint
    model, ckcfg, extra = load_checkpoint(out / "model.pt", "cpu")
    assert extra["window_size"] == 32 and extra["train_ratio"] == 0.5 and extra["test_stride"] == 10
    assert extra["threshold"] == rep["threshold"] and extra["normalizer"] == rep["normalizer"]
    assert extra["mask_params"]["mask_buffer_after_minutes"] == rep["postprocessing"]["mask_buffer_after_minutes"]
    tr, te = load_esa_mission(real / "data" / "m1", "m1").split(0.5)
    again = evaluate_mission(model, prepare_protocol_data(tr, te, ckcfg), ckcfg, device="cpu")
    for k in ("threshold", "alpha_rec", "normalizer", "window_metrics", "pointwise_metrics_rule_A", "postprocessing"):
        assert again[k] == rep[k], k

    # YAML that disagrees with the checkpoint is refused, unless --override (then the protocol is re-fitted)
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(CONFIG.replace("train_ratio: 0.5", "train_ratio: 0.75"), encoding="utf-8")
    bbase = ["--config", str(bad_cfg), "--data-root", str(real), "--out-dir", str(out), "--device", "cpu"]
    assert evaluate.main(bbase + ["--report", str(tmp_path / "x.json")]) == 2
    assert evaluate.main(bbase + ["--override", "--report", str(tmp_path / "ov.json")]) == 0
    assert json.loads((tmp_path / "ov.json").read_text())["protocol"]["train_ratio"] == 0.75

    # --esa-adb without the official package: logged as not computed, run still succeeds.
    rep3_path = tmp_path / "report_esa.json"
    assert evaluate.main(base + ["--report", str(rep3_path), "--esa-adb"]) == 0
    assert json.loads(rep3_path.read_text())["esa_adb_event_wise"]["computed"] is False

    out_json = tmp_path / "complexity.json"
    assert complexity.main(["--config", str(cfg_path), "--device", "cpu", "--repeats", "3", "--warmup", "1",
                            "--out", str(out_json)]) == 0
    comp = json.loads(out_json.read_text())
    assert comp["parameters"]["with_decoder"] > comp["parameters"]["without_decoder"]
    assert comp["flops_per_window"]["without_decoder"]["total"] < comp["flops_per_window"]["decoder_inverse_solve"]["total"]


def test_pipeline_resumable(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    import run_pipeline

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(CONFIG, encoding="utf-8")
    _make_data(tmp_path / "d", zero_test_labels=False)
    out = tmp_path / "o"
    argv = ["--config", str(cfg_path), "--data-root", str(tmp_path / "d"), "--out-dir", str(out), "--device", "cpu",
            "--n-trials", "1"]
    assert run_pipeline.main(argv) == 0
    stamp = (out / "model.pt").stat().st_mtime_ns
    assert run_pipeline.main(argv) == 0            # everything exists: all steps skipped
    assert (out / "model.pt").stat().st_mtime_ns == stamp


def test_train_refuses_legacy_defaults(tmp_path: Path) -> None:
    import evaluate
    import train

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(CONFIG, encoding="utf-8")
    _make_data(tmp_path / "d", zero_test_labels=False)
    out = tmp_path / "o"
    base = ["--config", str(cfg_path), "--data-root", str(tmp_path / "d"), "--out-dir", str(out), "--device", "cpu",
            "--max-epochs", "1"]
    assert train.main(base) == 2                                  # no best-params, no flag: refused
    assert not (out / "model.pt").exists()
    assert train.main(base + ["--allow-legacy-defaults"]) == 0
    assert evaluate.main(base[:8]) == 0
    assert json.loads((out / "report.json").read_text())["used_legacy_defaults"] is True
    assert json.loads((out / "train_summary.json").read_text())["used_legacy_defaults"] is True


def test_evaluate_honours_train_threshold_fallback(tmp_path: Path) -> None:
    """M3: train.py cannot freeze the protocol (no positive in the validation block); evaluate.py then refits and
    must honour the fallback flag (CLI or YAML) instead of repeating an error whose advice does not work."""
    import evaluate
    import train

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(CONFIG, encoding="utf-8")
    _make_data(tmp_path / "d", zero_test_labels=False, events=((100, 130), (800, 830), (1000, 1040)))
    out = tmp_path / "o"
    base = ["--config", str(cfg_path), "--data-root", str(tmp_path / "d"), "--out-dir", str(out), "--device", "cpu"]
    assert train.main(base + ["--max-epochs", "1", "--allow-legacy-defaults"]) == 2     # model saved, protocol not frozen
    assert (out / "model.pt").is_file()
    assert evaluate.main(base) != 0                                                    # refused: no fallback allowed
    rep_path = tmp_path / "r.json"
    assert evaluate.main(base + ["--allow-train-threshold-fallback", "--report", str(rep_path)]) == 0
    rep = json.loads(rep_path.read_text())
    assert rep["threshold_source"] == "train_fallback" and rep["val_has_positives"] is False
    # the YAML key works as well
    yaml_cfg = tmp_path / "fb.yaml"
    extra_line = "  allow_train_threshold_fallback: true"
    yaml_cfg.write_text(CONFIG.replace("  min_consecutive: 1", "  min_consecutive: 1" + chr(10) + extra_line),
                        encoding="utf-8")
    rep2_path = tmp_path / "r2.json"
    assert evaluate.main(["--config", str(yaml_cfg), "--data-root", str(tmp_path / "d"), "--out-dir", str(out),
                          "--device", "cpu", "--report", str(rep2_path)]) == 0
    assert json.loads(rep2_path.read_text())["threshold_source"] == "train_fallback"
