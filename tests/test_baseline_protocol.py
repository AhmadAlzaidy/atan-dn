"""Honest baseline protocol: split export, score adapter (synthetic scores) and a CPU smoke run of three runners."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import protocol_adapter as pa  # noqa: E402
import split_io  # noqa: E402
from atandn.config import MissionConfig  # noqa: E402

N, T0 = 3000, 1800


def _stream(seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(N, dtype=np.int64)
    for a, b in ((400, 450), (1300, 1360), (1550, 1600), (2000, 2100), (2500, 2560)):
        labels[a:b] = 1
    ts = pd.date_range("2020-01-01", periods=N, freq="1min")
    return rng, labels, ts


def _split(ts) -> dict:
    return {"schema": split_io.SPLIT_SCHEMA, "mission": "m1", "n_samples": N, "train_stop": T0,
            "fit": {"start": 0, "stop": 1000}, "val": {"start": 1100, "stop": T0},
            "test": {"start": T0, "stop": N}, "window_size": 60, "test_stride": 10,
            "timestamps": split_io.timestamps_identity(ts)}


def _scores(path: Path, labels, split, rng, window: int = 20, stride: int = 10, test_value: float | None = None):
    parts = {}
    for name in split_io.PARTS:
        a, b = split_io.part_range(split, name)
        starts = split_io.window_starts_in(a, b, window, stride)
        ends = starts + window - 1
        y = pa.window_labels_from_ends(labels, ends, window)
        s = 0.2 * rng.random(len(ends)) + 0.6 * y
        if name == "test" and test_value is not None:
            s = np.full(len(ends), test_value)
        parts[name] = (s, ends)
    split_io.save_scores(path, parts, window_size=window, stride=stride, model="toy", mission="m1",
                         ts_sha1=split["timestamps"]["sha1"])


def _cfg(mask: bool = False, min_consec=1) -> MissionConfig:
    cfg = MissionConfig()
    cfg.eval.use_known_event_mask = mask
    cfg.eval.min_consecutive = min_consec
    cfg.eval.mask_source = "rare_events"      # these tests pass rare-event intervals; parity with telecommands: test_adapter_mask_parity.py
    return cfg


def test_test_labels_never_enter_threshold_or_predictions(tmp_path: Path) -> None:
    rng, labels, ts = _stream()
    split = _split(ts)
    _scores(tmp_path / "s.npz", labels, split, rng)
    intervals = [(ts[2200], ts[2230])]
    kw = dict(timestamps=ts, split=split, intervals=intervals, mask_params={"mask_buffer_after_minutes": 15.0,
              "min_consecutive_windows": 2}, return_arrays=True)
    a = pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(True, "derived"), labels=labels, **kw)
    perm = labels.copy()
    perm[T0:] = np.random.default_rng(5).permutation(perm[T0:])
    b = pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(True, "derived"), labels=perm, **kw)
    assert a["threshold"] == b["threshold"] and a["normalizer"] == b["normalizer"]
    assert np.array_equal(a["_arrays"]["test_pred"], b["_arrays"]["test_pred"])
    assert np.array_equal(a["_arrays"]["test_scores"], b["_arrays"]["test_scores"])
    assert a["window_metrics"] != b["window_metrics"]          # the metrics do depend on the test labels
    assert a["threshold_source"] == "validation" and a["postprocessing"]["known_event_mask"] is True
    assert a["postprocessing"]["mask_buffer_after_minutes"] == 15.0 and a["postprocessing"]["min_consecutive"] == 2
    assert {"window_metrics", "pointwise_metrics_rule_A", "n_test_events", "n_test_pos_samples"} <= set(a)
    assert a["n_test_events"] == 2


def test_all_positive_stream_precision_equals_positive_rate(tmp_path: Path) -> None:
    rng, labels, ts = _stream(1)
    split = _split(ts)
    _scores(tmp_path / "s.npz", labels, split, rng, test_value=50.0)
    rep = pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(), labels=labels, timestamps=ts, split=split)
    wm = rep["window_metrics"]
    assert wm["fp"] + wm["tp"] == rep["n_test_windows"]            # everything predicted positive
    assert wm["precision"] == pytest.approx(rep["n_test_pos_windows"] / rep["n_test_windows"])
    assert wm["recall"] == 1.0
    pm = rep["pointwise_metrics_rule_A"]
    assert pm["precision"] == pytest.approx(pm["tp"] / (pm["tp"] + pm["fp"]))


def test_mapping_of_windows_of_different_lengths_is_causal() -> None:
    ends = np.arange(19, 400, 10)                                   # baseline windows of length 20, stride 10
    s = np.arange(len(ends), dtype=float)
    grid = np.arange(59, 400, 5)                                    # a different (ATAN-DN-like) window grid
    base, ok = pa.map_scores_to_grid(s, ends, grid, max_age=9)
    assert ok.all()
    for g, v in zip(grid, base):
        assert ends[int(v)] <= g < ends[int(v)] + 10               # latest window that ended at or before g
    s2 = s.copy()
    cut = 200
    s2[ends >= cut] += 1000.0                                       # change the future
    base2, _ = pa.map_scores_to_grid(s2, ends, grid, max_age=9)
    assert np.array_equal(base[grid < cut], base2[grid < cut])
    _, ok_early = pa.map_scores_to_grid(s, ends, np.array([5, 18, 19]), max_age=9)
    assert ok_early.tolist() == [False, False, True]               # nothing is available before the first window ended
    _, ok_old = pa.map_scores_to_grid(s, ends[:2], np.array([100]), max_age=9)
    assert not ok_old[0]                                            # stale scores are not carried forward


def test_atandn_grid_and_no_stage3(tmp_path: Path) -> None:
    rng, labels, ts = _stream(2)
    split = _split(ts)
    _scores(tmp_path / "s.npz", labels, split, rng, window=20, stride=10)
    intervals = [(ts[2000], ts[2100])]                              # covers a true event: masking must cost recall
    kw = dict(labels=labels, timestamps=ts, split=split, intervals=intervals,
              mask_params={"mask_buffer_after_minutes": 0.0, "min_consecutive_windows": 2})
    on = pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(True, "derived"), grid="atandn", **kw)
    off = pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(True, "derived"), grid="atandn", stage3=False, **kw)
    assert on["protocol"]["window_size"] == 60 and on["protocol"]["grid"] == "atandn"
    assert off["postprocessing"]["known_event_mask"] is False and off["postprocessing"]["min_consecutive"] == 1
    assert on["window_metrics"]["recall"] < off["window_metrics"]["recall"]
    native = pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(), labels=labels, timestamps=ts, split=split)
    assert native["protocol"]["window_size"] == 20


def test_split_mismatch_and_missing_positives(tmp_path: Path) -> None:
    rng, labels, ts = _stream(3)
    split = _split(ts)
    _scores(tmp_path / "s.npz", labels, split, rng)
    other = copy.deepcopy(split)
    other["timestamps"]["sha1"] = "0" * 40
    with pytest.raises(ValueError, match="different split"):
        pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(), labels=labels, timestamps=ts, split=other)
    nolab = labels.copy()
    nolab[1100:T0] = 0
    with pytest.raises(ValueError, match="no positive window"):
        pa.evaluate_baseline_scores(tmp_path / "s.npz", _cfg(), labels=nolab, timestamps=ts, split=split)
    cfg = _cfg()
    cfg.eval.allow_train_threshold_fallback = True
    rep = pa.evaluate_baseline_scores(tmp_path / "s.npz", cfg, labels=nolab, timestamps=ts, split=split)
    assert rep["threshold_source"] == "train_fallback"


# ------------------------------------------------------------------------------------------------------------------
# export_split + runner smoke runs on CPU
# ------------------------------------------------------------------------------------------------------------------
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
eval:
  use_known_event_mask: true
  mask_source: rare_events
  min_consecutive: 1
"""


def _make_mission(root: Path, n: int = 2400) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    values = np.stack([np.sin(t / 7.0), np.cos(t / 11.0)], axis=1) + 0.05 * rng.standard_normal((n, 2))
    labels = np.zeros(n)
    for a, b in ((100, 130), (520, 545), (800, 830), (1000, 1040), (1300, 1340), (1700, 1750), (2100, 2140)):
        values[a:b] += 4.0
        labels[a:b] = 1.0
    idx = pd.date_range("2020-01-01", periods=n, freq="30s")
    df = pd.DataFrame({"chchannel_41": values[:, 0], "chchannel_42": values[:, 1], "label": labels}, index=idx)
    d = root / "data" / "m1"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "all_telemetry_41_46.parquet")
    pd.DataFrame({"ID": ["e1"], "Channel": ["channel_41"], "StartTime": [str(idx[300])],
                  "EndTime": [str(idx[320])]}).to_csv(d / "labels.csv", index=False)
    pd.DataFrame({"ID": ["e1"], "Category": ["Rare Event"]}).to_csv(d / "anomaly_types.csv", index=False)


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> dict:
    import export_split

    tmp = tmp_path_factory.mktemp("honest")
    cfg = tmp / "cfg.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    _make_mission(tmp)
    out = tmp / "split.json"
    assert export_split.main(["--config", str(cfg), "--data-root", str(tmp), "--out", str(out)]) == 0
    return {"tmp": tmp, "cfg": cfg, "split": out, "data": tmp / "data" / "m1" / "all_telemetry_41_46.parquet"}


def test_export_split_matches_pipeline(exported: dict) -> None:
    from atandn.data import load_esa_mission
    from atandn.evaluation.pipeline import prepare_protocol_data
    from atandn.config import load_config

    sp = split_io.read_split(exported["split"])
    cfg = load_config(exported["cfg"])
    md = load_esa_mission(exported["data"].parent, "m1")
    train, test = md.split(cfg.data.train_ratio)
    cfg.model.in_channels = 2
    ref = prepare_protocol_data(train, test, cfg).split
    assert (sp["fit"]["start"], sp["fit"]["stop"]) == (ref.fit.start, ref.fit.stop)
    assert (sp["val"]["start"], sp["val"]["stop"]) == (ref.val.start, ref.val.stop)
    assert sp["test"] == {"start": len(train), "stop": len(md)} and sp["window_size"] == 32 and sp["test_stride"] == 10
    split_io.check_stream_matches_split(sp, md.timestamps)
    with pytest.raises(ValueError):
        split_io.check_stream_matches_split(sp, md.timestamps[::-1])


def _run_runner(exported: dict, rel: str, extra: list[str], name: str) -> subprocess.CompletedProcess:
    npz = exported["tmp"] / f"{name}.npz"
    cmd = [sys.executable, str(ROOT / "baselines" / rel), "--data", str(exported["data"]),
           "--out", str(exported["tmp"] / f"out_{name}")] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT / "baselines" / Path(rel).parent))


def test_runner_refuses_without_protocol_flags(exported: dict) -> None:
    res = _run_runner(exported, "usad/m1_usad_custom.py", [], "none")
    assert res.returncode == 2 and "--split-json" in res.stderr
    res = _run_runner(exported, "usad/m1_usad_custom.py",
                      ["--legacy-test-label-threshold", "--export-scores", "x.npz"], "mix")
    assert res.returncode == 2


@pytest.mark.parametrize("rel,model,window", [("usad/m1_usad_custom.py", "usad", 100),
                                              ("tranad/m1_tranad_custom.py", "tranad", 10)])
def test_runner_honest_smoke_and_adapter(exported: dict, rel: str, model: str, window: int) -> None:
    npz = exported["tmp"] / f"{model}.npz"
    res = _run_runner(exported, rel, ["--split-json", str(exported["split"]), "--export-scores", str(npz)], model)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "oracle" not in res.stdout and "Best F1" not in res.stdout       # no threshold sweep in honest mode
    sc = split_io.load_scores(npz)
    sp = split_io.read_split(exported["split"])
    assert sc["window_size"] == window and sc["stride"] == 10 and sc["model"] == model
    for part in split_io.PARTS:
        s, e = sc[part]
        a, b = split_io.part_range(sp, part)
        assert len(s) == len(e) > 0 and np.isfinite(s).all()
        assert e.min() - window + 1 >= a and e.max() < b           # windows lie entirely inside their slice
    # the adapter evaluates it end to end through the CLI
    import evaluate_baseline_scores as cli

    rep = exported["tmp"] / f"{model}_report.json"
    code = cli.main(["--config", str(exported["cfg"]), "--data-root", str(exported["tmp"]), "--scores", str(npz),
                     "--split-json", str(exported["split"]), "--report", str(rep), "--allow-train-threshold-fallback"])
    assert code == 0
    r = json.loads(rep.read_text())
    assert r["baseline"]["model"] == model and r["protocol"]["test_labels_used_for_fitting"] is False
    assert "f1" in r["window_metrics"]
