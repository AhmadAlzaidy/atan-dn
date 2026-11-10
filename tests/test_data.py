"""Data tests: windows, labels, splits (no leakage), config, prepare_esa_mission and the OPS-SAT loader."""
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from atandn.config import MissionConfig, load_config, mission_defaults
from atandn.data import (WindowDataset, chronological_split, load_esa_mission, load_opssat, make_test_windows,
                         prepare_esa_mission, train_val_tail_split, window_zscore)


def test_window_labels_are_max_of_samples():
    labels = np.zeros(100, dtype=int)
    labels[45] = 1
    ds = WindowDataset(np.random.randn(100, 2), labels, window_size=20, stride=10)
    starts = ds.starts
    expected = [int(labels[s:s + 20].max()) for s in starts]
    assert ds.window_labels.tolist() == expected
    x, y = ds[0]
    assert x.shape == (2, 20) and x.dtype == torch.float32 and y.dtype == torch.float32
    assert abs(float(x.mean())) < 1e-5
    assert len(ds.sampler_weights()) == len(ds)


def test_make_test_windows_causal_timestamp():
    n, w, s = 120, 30, 10
    ts = pd.date_range("2020-01-01", periods=n, freq="min")
    d = make_test_windows(np.random.randn(n, 3), np.zeros(n), ts, w, s)
    assert d["X"].shape == ((n - w) // s + 1, 3, w)
    assert (d["end_index"] == d["end_index"][0] + s * np.arange(len(d["end_index"]))).all()
    assert d["end_index"][0] == w - 1
    assert (d["ts"] == ts.values[d["end_index"]]).all()


def test_window_zscore_constant_channel_is_finite():
    z = window_zscore(np.ones((10, 2)))
    assert np.isfinite(z).all()


def test_chronological_split_no_overlap():
    tr, te = chronological_split(1000, 0.5)
    assert tr.stop == te.start == 500 and tr.start == 0 and te.stop == 1000
    tr, te = chronological_split(1000, 0.75)
    assert tr.stop == 750


def test_train_val_tail_split_has_gap_and_stays_in_train():
    fit, val = train_val_tail_split(1000, 0.2, 50)
    assert val.stop == 1000 and val.start == 800
    assert fit.stop <= val.start - 49          # no window of the fit part reaches the validation tail


def test_no_leakage_between_train_and_test_windows():
    n = 500
    vals = np.arange(n, dtype=float)[:, None]
    tr, te = chronological_split(n, 0.5)
    train_ds = WindowDataset(vals[tr], np.zeros(n)[tr], 20, 5)
    test = make_test_windows(vals[te], np.zeros(n)[te], np.arange(n)[te], 20, 10)
    assert train_ds.starts.max() + 20 <= tr.stop
    assert test["ts"].min() >= te.start


def test_config_defaults_and_yaml(tmp_path):
    for m, c, w in (("m1", 6, 300), ("m2", 11, 300), ("ops_sat", 9, 400)):
        cfg = mission_defaults(m)
        cfg.validate()
        assert (cfg.model.in_channels, cfg.model.window_size) == (c, w)
    assert mission_defaults("ops_sat").data.train_ratio == 0.75
    p = tmp_path / "c.yaml"
    p.write_text("mission: m2\ntrain:\n  lr: 0.001\nmodel:\n  conv_filters: [64, 128]\n")
    cfg = load_config(p)
    assert cfg.train.lr == 0.001 and cfg.model.in_channels == 11 and cfg.model.conv_filters == (64, 128)
    assert MissionConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    p.write_text("train:\n  bogus: 1\n")
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text("mission: m1\nmodel:\n  num_heads: 5\n")
    with pytest.raises(ValueError):
        load_config(p)


def _make_raw_mission(base, n=2000):
    (base / "channels").mkdir(parents=True)
    idx = pd.date_range("2000-01-01", periods=n, freq="30s")
    names = [f"channel_{i}" for i in range(1, 6)]
    pd.DataFrame({"Channel": names}).to_csv(base / "channels.csv", index=False)
    rng = np.random.default_rng(0)
    for nm in names:
        with open(base / "channels" / nm, "wb") as f:
            pickle.dump(pd.DataFrame({nm: rng.standard_normal(n)}, index=idx), f)
    pd.DataFrame({"ID": ["a", "b"], "Channel": ["channel_2", "channel_3"],
                  "StartTime": [str(idx[500]) + "+00:00", str(idx[1200]) + "+00:00"],
                  "EndTime": [str(idx[600]) + "+00:00", str(idx[1300]) + "+00:00"]}).to_csv(base / "labels.csv", index=False)
    pd.DataFrame({"ID": ["a", "b"], "Category": ["Anomaly", "Rare Event"]}).to_csv(base / "anomaly_types.csv", index=False)
    return idx


def test_prepare_esa_mission_labels_and_load(tmp_path):
    base = tmp_path / "raw"
    idx = _make_raw_mission(base)
    out = prepare_esa_mission(base, tmp_path / "out" / "all_telemetry_18_28.parquet", 2, 4, 1 / 30)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["chchannel_2", "chchannel_3", "chchannel_4", "label"]
    assert all(df[c].dtype == np.float64 for c in df.columns)
    assert df.isnull().sum().sum() == 0
    lab = df["label"]
    assert lab.loc[idx[500]:idx[600]].eq(1).all()          # Anomaly interval on a selected channel
    assert lab.loc[idx[1200]:idx[1300]].eq(0).all()        # Rare Event stays 0
    assert lab.sum() == 101
    # load via load_esa_mission (file name of mission m2 is used here only as a container)
    m = load_esa_mission(out.parent, "m2")
    assert m.values.shape == (len(df), 3) and m.labels.sum() == 101
    tr, te = m.split(0.5)
    assert len(tr) + len(te) == len(m)


def test_load_opssat_merges_by_timestamp(tmp_path):
    rows = []
    for ch in ("A", "B"):
        for i in range(5):
            rows.append({"channel": ch, "timestamp": f"2022-06-01T00:00:0{i}.000Z", "value": i + (10 if ch == "B" else 0),
                         "label": "anomaly" if i == 3 else "normal", "anomaly": int(i == 3), "segment": 1, "train": 1})
    pd.DataFrame(rows[::-1]).to_csv(tmp_path / "segments.csv", index=False)
    m = load_opssat(tmp_path / "segments.csv")
    assert m.values.shape == (10, 1) and m.labels.sum() == 2
    assert m.timestamps.is_monotonic_increasing
    assert m.values[:2, 0].tolist() == [0.0, 10.0]
