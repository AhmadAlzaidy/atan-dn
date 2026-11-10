"""make_tables, sensitivity, latency projection, dataset_stats and the tune.py provenance (synthetic fixtures only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import make_tables  # noqa: E402
import sensitivity  # noqa: E402
from atandn.evaluation.projection import classify_viability, latency_ms, load_hardware, project_latency  # noqa: E402

METRIC_VALUES = {"precision": 0.9, "recall": 0.8, "fpr": 0.01, "fnr": 0.2, "f1": 0.847, "f0_5": 0.88}


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------------------------
# projection
# ------------------------------------------------------------------------------------------------------------------
def test_projection_formula_orin_example() -> None:
    assert latency_ms(9.28e7, 5000.0, 0.5) == pytest.approx(0.0371, abs=5e-5)
    rows = project_latency(9.28e7, {"Orin": {"peak_gflops": 5000.0, "efficiency": 0.5}})
    assert rows[0]["latency_ms"] == pytest.approx(0.03712)
    assert rows[0]["fps"] == pytest.approx(1000.0 / rows[0]["latency_ms"])
    assert rows[0]["viability"] == "Real-Time"
    with pytest.raises(ValueError):
        latency_ms(1e7, 0.0, 0.5)
    with pytest.raises(ValueError):
        latency_ms(1e7, 10.0, 1.5)


def test_viability_rule_reproduces_the_papers_classes() -> None:
    # paper latencies in ms and their printed class, decision period 1 s
    for lat, cls in ((0.037, "Real-Time"), (47.2, "Real-Time"), (103, "Real-Time"), (646, "Real-Time*"),
                     (2217, "Snapshot"), (6748, "Snapshot")):
        assert classify_viability(lat) == cls
    assert classify_viability(1000.0) == "Real-Time*" and classify_viability(1000.1) == "Snapshot"
    assert classify_viability(300.0, decision_period_s=0.5) == "Real-Time*"


def test_hardware_yaml_loads_with_sources() -> None:
    hw, period, frac = load_hardware(ROOT / "configs" / "obc_hardware.yaml")
    assert len(hw) == 10 and period == 1.0 and frac == 0.5
    assert hw["NVIDIA Jetson AGX Orin"]["peak_gflops"] == 5300.0
    assert all("from legacy script, vendor value not re-verified" in v["source"] for v in hw.values())
    rows = project_latency(8.5792256e7, hw, period, frac)
    assert {r["viability"] for r in rows} <= {"Real-Time", "Real-Time*", "Snapshot"}
    assert rows[0]["latency_ms"] == pytest.approx(8.5792256e7 / (0.7 * 5300e9) * 1e3)


# ------------------------------------------------------------------------------------------------------------------
# make_tables
# ------------------------------------------------------------------------------------------------------------------
def test_make_tables_from_fixtures(tmp_path: Path) -> None:
    rep = _write(tmp_path / "m1" / "report.json", {"window_metrics": METRIC_VALUES,
                                                    "pointwise_metrics_rule_A": {**METRIC_VALUES, "f1": 0.5}})
    base = _write(tmp_path / "m1" / "usad_report.json", {"window_metrics": {**METRIC_VALUES, "f1": 0.1}})
    seeds = _write(tmp_path / "m1" / "multi_seed_summary.json", {"summary": {
        f"window_{k}": {"mean": v, "std": 0.0123, "n": 3} for k, v in METRIC_VALUES.items()}})
    abl = _write(tmp_path / "abl.json", {"meta": {"variants": ["bign", "identity"]}, "summary": {
        "bign": {"kurtosis_norm": {"mean": 1.5, "std": 0.2, "n": 3}, "fisher_latent": {"mean": 2.0, "std": 0.1, "n": 3},
                 "window_f1": {"mean": 0.9, "std": 0.01, "n": 3}},
        "identity": {"kurtosis_norm": {"mean": 3.5, "std": 0.2, "n": 3}, "window_f1": {"mean": 0.7, "std": 0.02, "n": 3}}}})
    mask = _write(tmp_path / "mask.json", {"mask_buffer_after_minutes": 15.5, "min_consecutive_windows": 2,
                                           "noise_cluster_median": 3.0, "transition_zones": {"n": 4, "x": "text"}})
    hw = {"Orin": {"peak_gflops": 5000.0, "efficiency": 0.5, "source": "test"}}
    cx = _write(tmp_path / "cx.json", {"parameters": {"with_decoder": 1000, "without_decoder": 900},
                                       "flops_per_window": {"decoder_inverse_solve": {"total": 9.28e7},
                                                            "without_decoder": {"total": 8.0e7}},
                                       "latency_with_decoder": {"mean_ms": 3.78}, "latency_without_decoder": {"mean_ms": 2.7},
                                       "projection": {"flops": 9.28e7, "flops_key": "decoder_inverse_solve",
                                                      "hardware_file": "hw.yaml", "platforms": project_latency(9.28e7, hw)}})
    res = make_tables.make_tables(tmp_path / "out", [("m1", "ATAN-DN", rep), ("m1", "USAD", base),
                                                     ("m2", "ATAN-DN", tmp_path / "missing.json")],
                                  {"m1": seeds}, {"norm": abl}, {"m1": mask}, {"m1": cx})
    out = tmp_path / "out"
    t1 = (out / "table1_results.tex").read_text(encoding="utf-8")
    assert "0.9000" in t1 and r"0.9000 $\pm$ 0.0123" in t1 and "USAD" in t1 and "0.1000" in t1
    assert "Prec." in t1 and "F0.5" in t1 and "FNR" in t1
    assert "0.5000" not in t1                                      # window level, not the point-wise F1
    assert "0.9000 ± 0.0123" in (out / "table1_results.md").read_text(encoding="utf-8")
    ab = (out / "table_ablation_norm.tex").read_text(encoding="utf-8")
    assert "1.500 $\\pm$ 0.200" in ab and "--" in ab               # identity has no Fisher value: shown as '--'
    dp = (out / "table_derived_params.tex").read_text(encoding="utf-8")
    assert "15.5" in dp and "transition\\_zones.n" in dp and "transition\\_zones.x" not in dp
    assert "9.280e+07" in (out / "table_complexity.tex").read_text(encoding="utf-8")
    assert "0.03712" in (out / "table_latency_projection.tex").read_text(encoding="utf-8")
    assert (out / "tables.md").is_file()
    assert any("missing.json" in s for s in res["skipped"])        # the missing input is reported, nothing invented


def test_make_tables_skips_tables_without_inputs(tmp_path: Path, caplog) -> None:
    with caplog.at_level("WARNING"):
        res = make_tables.make_tables(tmp_path / "o", [("m1", "ATAN-DN", tmp_path / "nope.json")])
    assert res["written"] == [] and not list((tmp_path / "o").glob("*.tex"))
    assert "nope.json" in caplog.text and "table skipped" in caplog.text
    assert make_tables.main(["--out-dir", str(tmp_path / "o2")]) == 0
    assert make_tables.main(["--out-dir", str(tmp_path / "o2"), "--report", "bad"]) == 2
    code = make_tables.main(["--out-dir", str(tmp_path / "o3"), "--run-dir", f"m1={tmp_path}"])
    assert code == 0 and not list((tmp_path / "o3").glob("*.tex"))


# ------------------------------------------------------------------------------------------------------------------
# sensitivity
# ------------------------------------------------------------------------------------------------------------------
def _synthetic_study(tmp_path: Path) -> Path:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna.trial import TrialState, create_trial

    dist = {"window_size": IntDistribution(200, 400, step=50), "pos_weight": FloatDistribution(3.0, 25.0),
            "recon_weight": FloatDistribution(0.1, 0.4)}
    study = optuna.create_study(direction="maximize")
    rng = np.random.default_rng(0)
    for i in range(12):
        params = {"window_size": int(200 + 50 * (i % 5)), "pos_weight": float(3.0 + 1.7 * i),
                  "recon_weight": float(0.1 + 0.02 * ((i * 7) % 12))}
        study.add_trial(create_trial(params=params, distributions=dist, value=float(rng.random()), state=TrialState.COMPLETE))
    # a pruned trial with extreme parameters must never be reported
    study.add_trial(create_trial(params={"window_size": 400, "pos_weight": 99.0, "recon_weight": 0.4},
                                 distributions={"window_size": dist["window_size"],
                                                "pos_weight": FloatDistribution(3.0, 100.0), "recon_weight": dist["recon_weight"]},
                                 state=TrialState.PRUNED, value=None, intermediate_values={0: 0.1}))
    import tune

    return tune.write_trials(study, tmp_path / "m1" / "trials.csv")


def test_sensitivity_rows_are_the_extreme_valued_completed_trials(tmp_path: Path) -> None:
    csv = _synthetic_study(tmp_path)
    df = sensitivity.load_trials(csv)
    assert {"number", "value", "state", "duration"} <= set(df.columns) and "params_pos_weight" in df.columns
    done = df[df["state"] == "COMPLETE"]
    for param in ("pos_weight", "window_size", "recon_weight"):
        low, best, high = sensitivity.sensitivity_rows(df, param)
        col = f"params_{param}"
        assert low["param_value"] == done[col].min() and high["param_value"] == done[col].max()
        assert low["value"] == done[done[col] == done[col].min()]["value"].max()
        assert best["value"] == done["value"].max()
        assert best["param_value"] == done.loc[done["value"].idxmax(), col]
        assert high["trial"] != int(df[df["state"] == "PRUNED"]["number"].iloc[0])   # the pruned trial is ignored
        assert low["n_completed"] == len(done) == 12


def test_sensitivity_cli_writes_tables_and_validates(tmp_path: Path) -> None:
    csv = _synthetic_study(tmp_path)
    pre = tmp_path / "res" / "sens"
    assert sensitivity.main(["--trials", str(csv), "--names", "M1", "--out-prefix", str(pre)]) == 0
    data = json.loads(pre.with_suffix(".json").read_text())
    assert set(data["table"]["M1"]) == {"pos_weight", "window_size", "recon_weight"}
    assert "validation average precision" in data["metric"]
    tex = pre.with_suffix(".tex").read_text()
    assert r"\textbf" in tex and "Pos. Weight" in tex and "Window Size" in tex
    md = pre.with_suffix(".md").read_text()
    assert "| L |" in md and "| H |" in md
    assert sensitivity.main(["--trials", str(csv), "--params", "nonexistent", "--out-prefix", str(pre)]) == 2
    assert sensitivity.main(["--trials", str(tmp_path / "none.csv"), "--out-prefix", str(pre)]) == 2
    assert sensitivity.main(["--out-prefix", str(pre)]) == 2


def test_search_ranges_come_from_the_search_space() -> None:
    r = sensitivity.search_ranges()
    assert r["window_size"] == (200, 400) and r["pos_weight"] == (3.0, 25.0) and r["recon_weight"] == (0.1, 0.4)


# ------------------------------------------------------------------------------------------------------------------
# tune.py provenance, complexity --project, dataset_stats (synthetic mission)
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
EVENTS = ((100, 130), (520, 545), (800, 830), (1000, 1040), (1300, 1340), (1700, 1750), (2100, 2140))


def _mission(root: Path, n: int = 2400) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    values = np.stack([np.sin(t / 7.0), np.cos(t / 11.0)], axis=1) + 0.05 * rng.standard_normal((n, 2))
    labels = np.zeros(n)
    for a, b in EVENTS:
        values[a:b] += 4.0
        labels[a:b] = 1.0
    idx = pd.date_range("2020-01-01", periods=n, freq="30s")
    df = pd.DataFrame({"chchannel_41": values[:, 0], "chchannel_42": values[:, 1], "label": labels}, index=idx)
    d = root / "data" / "m1"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "all_telemetry_41_46.parquet")
    pd.DataFrame({"ID": ["e1"], "Channel": ["channel_41"], "StartTime": [str(idx[300])], "EndTime": [str(idx[320])]}
                 ).to_csv(d / "labels.csv", index=False)
    pd.DataFrame({"ID": ["e1"], "Category": ["Rare Event"]}).to_csv(d / "anomaly_types.csv", index=False)


def test_dataset_stats_counts(tmp_path: Path) -> None:
    import dataset_stats

    (tmp_path / "cfg.yaml").write_text(CONFIG, encoding="utf-8")
    _mission(tmp_path)
    pre = tmp_path / "stats"
    assert dataset_stats.main(["--config", str(tmp_path / "cfg.yaml"), "--data-root", str(tmp_path),
                               "--out-prefix", str(pre)]) == 0
    st = json.loads(pre.with_suffix(".json").read_text())
    assert st["channels"] == 2 and st["first_timestamp"].startswith("2020-01-01 00:00:00")
    assert st["sampling_step_seconds"]["median"] == 30.0
    p = st["parts"]
    assert p["full"]["samples"] == 2400 and p["train"]["samples"] == 1200 and p["test"]["samples"] == 1200
    assert p["train"]["anomalous_samples"] == 125 and p["train"]["events"] == 4
    assert p["test"]["anomalous_samples"] == 130 and p["test"]["events"] == 3
    assert p["full"]["events"] == 7 and p["test"]["anomaly_percent"] == pytest.approx(100 * 130 / 1200)
    assert p["fit"]["samples"] + p["val"]["samples"] <= 1200
    assert p["test"]["windows"] == (1200 - 32) // 10 + 1
    assert "| test |" in pre.with_suffix(".md").read_text()


def test_tune_writes_trials_and_provenance(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    import tune

    (tmp_path / "cfg.yaml").write_text(CONFIG, encoding="utf-8")
    _mission(tmp_path)
    out = tmp_path / "out"
    assert tune.main(["--config", str(tmp_path / "cfg.yaml"), "--data-root", str(tmp_path), "--out-dir", str(out),
                      "--device", "cpu", "--n-trials", "2"]) == 0
    best = json.loads((out / "best_params.json").read_text())
    prov = best["provenance"]
    assert prov["duration_s"] >= 0 and prov["n_pruned"] >= 0 and prov["versions"]["optuna"] and prov["versions"]["torch"]
    df = pd.read_csv(out / "trials.csv")
    assert len(df) == 2 and {"number", "value", "state", "duration"} <= set(df.columns)
    # the sensitivity script reads what tune.py wrote
    assert sensitivity.main(["--trials", str(out / "trials.csv"), "--params", "pos_weight",
                             "--out-prefix", str(tmp_path / "s")]) == 0


def test_complexity_project_flag(tmp_path: Path) -> None:
    import complexity

    (tmp_path / "cfg.yaml").write_text(CONFIG, encoding="utf-8")
    out = tmp_path / "cx.json"
    code = complexity.main(["--config", str(tmp_path / "cfg.yaml"), "--device", "cpu", "--repeats", "2", "--warmup", "1",
                            "--project", str(ROOT / "configs" / "obc_hardware.yaml"), "--out", str(out)])
    assert code == 0
    proj = json.loads(out.read_text())["projection"]
    assert len(proj["platforms"]) == 10 and proj["flops"] > 0
    first = proj["platforms"][0]
    assert first["latency_ms"] == pytest.approx(proj["flops"] / (first["efficiency"] * first["peak_gflops"] * 1e9) * 1e3)
