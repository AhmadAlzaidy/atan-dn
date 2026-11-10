"""Telecommand mask source, threshold floor, unmasked report, no_stage3 variant and OPS-SAT multichannel loader."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from atandn.config import MissionConfig, mission_defaults
from atandn.data import MissionData, chronological_split, load_mission_from_config, load_opssat_multichannel
from atandn.data.telecommands import load_telecommand_times, load_telecommands
from atandn.evaluation.pipeline import ProtocolData, evaluate_mission, fit_frozen_protocol, prepare_protocol_data
from atandn.inference.threshold import apply_threshold_floor, nominal_score_quantile
from experiments.ablation.summarize import DEFAULT_METRICS
from experiments.ablation.variants import STUDIES, apply_variant
from test_protocol import Toy

EPOCH = np.datetime64("2020-01-01T00:00")


# ---------------------------------------------------------------------------------------------------------
# telecommand mask source in the shared pipeline
# ---------------------------------------------------------------------------------------------------------
def _tc_stream(n: int = 6000, seed: int = 0) -> tuple[MissionData, pd.DataFrame]:
    """Minute telemetry; type A (every 200 min) is followed by a 25-minute pulse, type B never."""
    rng = np.random.default_rng(seed)
    x = 0.05 * rng.standard_normal((n, 2))
    a = np.arange(60, n - 100, 200)
    b = a + 100
    for t in a:
        x[t + 1:t + 26, 0] += 8.0
    y = np.zeros(n, dtype=np.int64)
    for s in (700, 1500, 2300, 3100, 4000, 4800):                          # labelled anomalies away from the pulses
        x[s:s + 20, 1] += 5.0
        y[s:s + 20] = 1
    ts = EPOCH + np.arange(n).astype("timedelta64[m]")
    tc = pd.DataFrame({"time": pd.DatetimeIndex(ts[np.r_[a, b]]), "type": ["A"] * len(a) + ["B"] * len(b),
                       "priority": 3}).sort_values("time", kind="stable").reset_index(drop=True)
    return MissionData(x, y, ts, ["a", "b"], {"telecommands": tc}), tc


def _cfg(source: str = "telecommands") -> MissionConfig:
    cfg = mission_defaults("m1")
    cfg.model.in_channels, cfg.model.window_size = 2, 32
    cfg.train.val_fraction = 0.25
    cfg.eval.use_known_event_mask, cfg.eval.mask_source = True, source
    cfg.eval.min_consecutive = 1
    cfg.eval.mask_buffer_after_minutes = None
    return cfg


def _split(full: MissionData, ratio: float = 0.6):
    tr, te = full.split(ratio)
    return tr, te


def test_telecommand_mask_buffers_come_from_training_and_test_intervals_from_schedule():
    torch.manual_seed(0)
    full, tc = _tc_stream()
    train, test = _split(full)
    cfg, model = _cfg(), Toy()
    data = prepare_protocol_data(train, test, cfg)
    fz = fit_frozen_protocol(model, data, cfg, device="cpu")
    mp = fz["mask_params"]
    assert fz["mask_source"] == "telecommands" and mp["mask_source"] == "telecommands"
    assert mp["telecommand"]["selected_types"] == ["A"]
    assert mp["mask_buffer_after_minutes"] == mp["telecommand"]["buffer_minutes"] == 25.0   # the pulse lasts 25 samples
    assert mp["telecommand"]["n_telecommands_train"] > mp["telecommand"]["n_excursion"] > 0
    assert fz["mask_buffer_before_minutes"] == 0.0 and fz["known_event_mask"]
    rep = evaluate_mission(model, data, cfg, frozen=fz, device="cpu")
    n_test_a = int(((tc["type"] == "A") & (tc["time"] > train.timestamps[-1])).sum())
    assert rep["postprocessing"]["mask_source"] == "telecommands" and rep["postprocessing"]["n_known_events"] == n_test_a
    assert rep["postprocessing"]["telecommand_selected_types"] == ["A"]
    assert 0.0 < rep["postprocessing"]["mask_share_of_test_windows"] < 1.0
    assert rep["threshold_link"]["diagnostic"] is True and rep["threshold_floor"]["mode"] == "none"


def test_telecommand_buffers_and_threshold_ignore_the_test_split():
    torch.manual_seed(0)
    full, tc = _tc_stream()
    train, test = _split(full)
    cfg, model = _cfg(), Toy()
    base = fit_frozen_protocol(model, prepare_protocol_data(train, test, cfg), cfg, device="cpu")
    # extra telecommands in the test period and a different test stream: the frozen protocol is identical
    extra = pd.DataFrame({"time": pd.DatetimeIndex([test.timestamps[10], test.timestamps[50]]), "type": ["A", "A"],
                          "priority": 3})
    full2 = MissionData(full.values, full.labels, full.timestamps, full.channel_names,
                        {"telecommands": pd.concat([tc, extra], ignore_index=True)})
    train2, _ = _split(full2)
    other = fit_frozen_protocol(model, prepare_protocol_data(train2, None, cfg), cfg, device="cpu")
    for k in ("threshold", "mask_buffer_after_minutes", "min_consecutive", "normalizer", "mask_source"):
        assert other[k] == base[k], k
    assert other["mask_params"]["telecommand"] == base["mask_params"]["telecommand"]


def test_rare_events_mode_keeps_the_legacy_behaviour_and_telecommands_are_optional():
    torch.manual_seed(0)
    full, _ = _tc_stream()
    train, test = _split(full)
    train.meta = {"known_event_intervals": [(train.timestamps[300], train.timestamps[320])]}
    test.meta = train.meta
    cfg, model = _cfg("rare_events"), Toy()
    data = prepare_protocol_data(train, test, cfg)
    fz = fit_frozen_protocol(model, data, cfg, device="cpu")
    assert fz["mask_source"] == "rare_events" and "telecommand" not in fz["mask_params"]
    assert fz["n_known_events"] == 1


def test_telecommand_mode_without_telecommands_is_an_error():
    full, _ = _tc_stream()
    train, test = _split(full)
    bare = MissionData(train.values, train.labels, train.timestamps, train.channel_names, {})
    cfg = _cfg()
    with pytest.raises(ValueError, match="telecommands"):
        fit_frozen_protocol(Toy(), prepare_protocol_data(bare, None, cfg), cfg, device="cpu")
    cfg.eval.mask_source = "bogus"
    with pytest.raises(ValueError):
        cfg.validate()


def test_report_unmasked_uses_the_same_threshold_and_never_has_fewer_alarms():
    torch.manual_seed(0)
    full, _ = _tc_stream()
    train, test = _split(full)
    cfg, model = _cfg(), Toy()
    cfg.eval.min_consecutive = 2
    data = prepare_protocol_data(train, test, cfg)
    rep = evaluate_mission(model, data, cfg, device="cpu", report_unmasked=True)
    c = rep["stage3_comparison"]
    w, u = c["with_stage3"]["window_metrics"], c["without_stage3"]["window_metrics"]
    assert w == rep["window_metrics"]
    assert u["tp"] + u["fp"] >= w["tp"] + w["fp"] and u["tp"] + u["tn"] + u["fp"] + u["fn"] == w["tp"] + w["tn"] + w["fp"] + w["fn"]
    assert "not re-tuned" in c["note"] and "stage3_comparison" not in evaluate_mission(model, data, cfg, device="cpu")


# ---------------------------------------------------------------------------------------------------------
# threshold floor
# ---------------------------------------------------------------------------------------------------------
def test_threshold_floor_is_max_of_grid_threshold_and_nominal_p99():
    rng = np.random.default_rng(0)
    s = rng.random(2000) * 0.5
    y = np.zeros(2000)
    s[:50], y[:50] = 0.99, 1                                   # positives must not enter the floor
    q = float(np.quantile(s[y == 0], 0.99))
    assert nominal_score_quantile(s, y) == q
    thr, info = apply_threshold_floor(0.2, "nominal_p99", s, y)
    assert thr == q and info["applied"] and info["floor"] == q and info["threshold_before_floor"] == 0.2
    thr2, info2 = apply_threshold_floor(0.9, "nominal_p99", s, y)
    assert thr2 == 0.9 and not info2["applied"]
    assert apply_threshold_floor(0.2, "none", s, y) == (0.2, {"mode": "none", "applied": False, "floor": None})
    s2 = s.copy()
    s2[y == 1] = 0.0                                           # changing positive scores leaves the floor unchanged
    assert apply_threshold_floor(0.2, "nominal_p99", s2, y)[0] == q
    with pytest.raises(ValueError):
        apply_threshold_floor(0.2, "bad", s, y)


def test_threshold_floor_in_pipeline_never_uses_test_labels():
    torch.manual_seed(0)
    full, _ = _tc_stream()
    train, test = _split(full)
    cfg, model = _cfg(), Toy()
    cfg.eval.threshold_floor = "nominal_p99"
    fz = fit_frozen_protocol(model, prepare_protocol_data(train, None, cfg), cfg, device="cpu")
    fl = fz["threshold_floor"]
    assert fl["mode"] == "nominal_p99" and fz["threshold"] == max(fz["threshold_before_floor"], fl["floor"])
    assert "fitting part" in fl["source"]
    t0 = MissionData(test.values, np.zeros_like(test.labels), test.timestamps, test.channel_names, test.meta)
    reps = [evaluate_mission(model, prepare_protocol_data(train, t, cfg), cfg, device="cpu") for t in (test, t0)]
    assert reps[0]["threshold"] == reps[1]["threshold"] == fz["threshold"]
    assert reps[0]["threshold_floor"] == fl
    cfg.eval.threshold_floor = "x"
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------------------------------------
# ablation variant no_stage3
# ---------------------------------------------------------------------------------------------------------
def test_no_stage3_variant():
    cfg = _cfg()
    cfg.eval.min_consecutive = 3
    out = apply_variant(cfg, "no_stage3")
    assert out.eval.use_known_event_mask is False and out.eval.min_consecutive == 1
    assert cfg.eval.use_known_event_mask is True                # the base config is untouched
    assert "full" in STUDIES["operational"] and "no_stage3" in STUDIES["operational"] and "window_fpr" in DEFAULT_METRICS
    torch.manual_seed(0)
    full, _ = _tc_stream()
    train, test = _split(full)
    rep = evaluate_mission(Toy(), prepare_protocol_data(train, test, out), out, device="cpu")
    assert rep["postprocessing"]["known_event_mask"] is False and rep["postprocessing"]["min_consecutive"] == 1
    assert rep["postprocessing"]["mask_source"] == "none"


# ---------------------------------------------------------------------------------------------------------
# telecommand files and OPS-SAT multichannel
# ---------------------------------------------------------------------------------------------------------
def test_load_telecommands_both_layouts(tmp_path):
    for mission, sub in (("m1", "preprocessed_telecommand_{n}.pkl"), ("m2", "telecommand_{n}/telecommand_{n}")):
        root = tmp_path / mission
        (root / "telecommands").mkdir(parents=True)
        pd.DataFrame({"Telecommand": ["telecommand_1", "telecommand_2"], "Priority": [3, 2]}).to_csv(
            root / "telecommands.csv", index=False)
        for n, idx in ((1, ["2001-01-01 00:10:00", "2001-01-01 00:05:00"]), (2, ["2001-01-01 00:07:00"])):
            ix = pd.DatetimeIndex(idx, tz="UTC") if mission == "m1" else pd.DatetimeIndex(idx)
            df = pd.DataFrame({f"telecommand_{n}": [1.0] * len(idx)}, index=ix)
            path = root / "telecommands" / sub.format(n=n)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(path)
        times = load_telecommand_times(root, mission)
        assert times.tz is None and times.is_monotonic_increasing and len(times) == 3
        assert str(times[0]) == "2001-01-01 00:05:00"
        tab = load_telecommands(root, mission, priorities=[3])
        assert set(tab["type"]) == {"telecommand_1"} and len(tab) == 2


def _ops_csv(path):
    rows = []
    spec = {"A": range(0, 5), "B": range(2, 7), "C": (0, 6)}      # observation seconds per channel
    for ch, secs in spec.items():
        for i in secs:
            rows.append({"channel": ch, "timestamp": f"2022-06-01T00:00:0{i}.000Z",
                         "value": float(i + {"A": 0, "B": 100, "C": 200}[ch]), "label": "x",
                         "anomaly": int(ch == "B" and i == 4), "segment": 1, "train": 1})
    pd.DataFrame(rows[::-1]).to_csv(path, index=False)


def test_opssat_multichannel_pivot_zoh_labels_and_split(tmp_path):
    _ops_csv(tmp_path / "segments.csv")
    m = load_opssat_multichannel(tmp_path / "segments.csv")
    assert m.values.shape == (7, 3) and m.channel_names == ["A", "B", "C"] and len(m.timestamps) == 7
    assert m.timestamps.is_monotonic_increasing and m.timestamps.is_unique
    # ZOH: A ends at second 4 -> 4 afterwards; B starts at 2 -> back-filled with its first value (102) before
    assert m.values[:, 0].tolist() == [0, 1, 2, 3, 4, 4, 4]
    assert m.values[:, 1].tolist() == [102, 102, 102, 103, 104, 105, 106]
    assert m.values[:, 2].tolist() == [200, 200, 200, 200, 200, 200, 206]
    # label: any OBSERVED channel anomalous (B at second 4), never forward-filled
    assert m.labels.tolist() == [0, 0, 0, 0, 1, 0, 0]
    maj = load_opssat_multichannel(tmp_path / "segments.csv", label_mode="majority")
    assert maj.labels.tolist() == [0, 0, 0, 0, 1, 0, 0]          # second 4: B anomalous, A and B observed = half
    tr, te = m.split(0.75)
    assert len(tr) == 5 and len(te) == 2 and chronological_split(7, 0.75)[0] == slice(0, 5)
    shuffled = pd.read_csv(tmp_path / "segments.csv").sample(frac=1.0, random_state=1)
    shuffled.to_csv(tmp_path / "s2.csv", index=False)
    m2 = load_opssat_multichannel(tmp_path / "s2.csv")
    assert np.array_equal(m2.values, m.values) and np.array_equal(m2.labels, m.labels)


def test_opssat_mode_selection_through_the_config(tmp_path):
    _ops_csv(tmp_path / "segments.csv")
    cfg = mission_defaults("ops_sat")
    cfg.data.path = "segments.csv"
    cfg.data.opssat_mode = "interleaved"
    assert load_mission_from_config(cfg, tmp_path).values.shape == (12, 1)      # 5 + 5 + 2 rows
    cfg.data.opssat_mode = "multichannel"
    assert load_mission_from_config(cfg, tmp_path).values.shape == (7, 3)
    bad = copy.deepcopy(cfg)
    bad.data.opssat_mode = "other"
    with pytest.raises(ValueError):
        bad.validate()


def test_esa_loader_attaches_telecommands_from_the_raw_folder(tmp_path):
    ts = pd.date_range("2001-01-01", periods=50, freq="min")
    (tmp_path / "data" / "m1").mkdir(parents=True)
    pd.DataFrame({"chchannel_41": np.arange(50.0), "label": 0}, index=ts).to_parquet(
        tmp_path / "data" / "m1" / "all_telemetry_41_46.parquet")
    raw = tmp_path / "data" / "m1" / "raw"
    (raw / "telecommands").mkdir(parents=True)
    pd.DataFrame({"Telecommand": ["telecommand_1"], "Priority": [2]}).to_csv(raw / "telecommands.csv", index=False)
    pd.DataFrame({"telecommand_1": [1]}, index=pd.DatetimeIndex(["2001-01-01 00:10:00"], tz="UTC")).to_pickle(
        raw / "telecommands" / "preprocessed_telecommand_1.pkl")
    cfg = mission_defaults("m1")
    cfg.data.path, cfg.data.telecommand_dir = "data/m1/all_telemetry_41_46.parquet", "data/m1/raw"
    cfg.eval.use_known_event_mask = True
    m = load_mission_from_config(cfg, tmp_path)
    assert list(m.meta["telecommands"]["type"]) == ["telecommand_1"]
    m2 = load_mission_from_config(cfg, tmp_path / "data" / "m1", parquet_in_root=True)     # experiment runners
    assert len(m2.meta["telecommands"]) == 1
    cfg.data.telecommand_dir = ""
    with pytest.raises(FileNotFoundError, match="telecommand_dir"):
        load_mission_from_config(cfg, tmp_path)
    cfg.eval.mask_source = "rare_events"
    assert "telecommands" not in load_mission_from_config(cfg, tmp_path).meta
