"""Tests of the ablation and multi-seed runners on tiny synthetic data (CPU, 1 epoch)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atandn.config import DataConfig, MissionConfig, ModelConfig, TrainConfig
from atandn.data import MissionData
from experiments.ablation.run_ablation import load_study_file, run_ablation
from experiments.ablation.summarize import load_summary, to_latex, to_markdown
from experiments.ablation.variants import STUDIES, VARIANTS, apply_variant
from experiments.multi_seed.run_multi_seed import resolve_seeds, run_multi_seed
from experiments.protocol import build_model, excess_kurtosis, fisher_criterion, read_rows_csv, train_and_evaluate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _stream(n: int, seed: int) -> MissionData:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.stack([np.sin(t / 7.0), np.cos(t / 11.0)], 1) + 0.1 * rng.standard_normal((n, 2))
    y = np.zeros(n, dtype=np.int64)
    for s in range(150, n - 60, 300):
        x[s:s + 40] += rng.normal(3.0, 0.5, size=(40, 2))
        y[s:s + 40] = 1
    return MissionData(x, y, np.arange(n).astype("datetime64[s]"), ["a", "b"], {})


@pytest.fixture(scope="module")
def setup():
    cfg = MissionConfig(
        name="m1",
        model=ModelConfig(in_channels=2, window_size=32, conv_filters=(8, 16), num_heads=2, dropout=0.1),
        train=TrainConfig(max_epochs=1, batch_size=16, overlap=0.5, lr=1e-3, warmup_epochs=1, val_fraction=0.25,
                          label_smoothing=0.0, recon_score_weight=0.2),
        data=DataConfig(kind="esa", test_stride=10),
    )
    cfg.validate()
    return cfg, _stream(1200, 0), _stream(900, 1)


def test_registry_and_study_files():
    for names in STUDIES.values():
        assert all(n in VARIANTS for n in names)
    for f in ("ablation_norm.yaml", "ablation_components.yaml"):
        variants, seeds = load_study_file(CONFIGS / f)
        assert variants and seeds
    ident = apply_variant(MissionConfig(), "norm_identity")
    assert ident.model.norm == "identity"
    assert apply_variant(MissionConfig(), "classifier_only").model.use_reconstruction is False
    assert apply_variant(MissionConfig(), "no_recon_score").eval.alpha_rec == 0.0
    assert apply_variant(MissionConfig(), "loss_bce").train.classification == "bce"
    with pytest.raises(KeyError):
        apply_variant(MissionConfig(), "nope")


def test_variants_differ_in_structure(setup):
    cfg, _, _ = setup
    full = build_model(apply_variant(cfg, "full"))
    ident = build_model(apply_variant(cfg, "norm_identity"))
    ln = build_model(apply_variant(cfg, "norm_layernorm"))
    clf = build_model(apply_variant(cfg, "classifier_only"))
    assert len(full.bign_layers()) > 0
    assert len(ident.bign_layers()) == 0 and len(ln.bign_layers()) == 0
    n_full, n_ident = (sum(p.numel() for p in m.parameters()) for m in (full, ident))
    assert n_ident < n_full
    assert clf.decoder is None
    import torch
    x = torch.randn(3, 2, 32)
    for m in (full, ident, ln, clf):
        m.eval()
        recon, logits = m(x)
        assert logits.shape == (3,)
        assert (recon is None) == (m is clf) and (recon is None or recon.shape == x.shape)


def test_diagnostic_definitions():
    rng = np.random.default_rng(0)
    assert abs(excess_kurtosis(rng.standard_normal(200000))) < 0.1
    assert excess_kurtosis(np.array([1.0, 1.0, 1.0])) != excess_kurtosis(np.array([1.0, 1.0, 1.0]))  # NaN
    f = np.r_[rng.normal(0, 1, (500, 3)), rng.normal(2, 1, (500, 3))]
    y = np.r_[np.zeros(500), np.ones(500)]
    assert fisher_criterion(f, y) == pytest.approx(2.0, abs=0.3)  # (2-0)^2 / (1+1)
    assert np.isnan(fisher_criterion(f, np.zeros(1000)))


def test_ablation_run_outputs_resume_and_no_overwrite(setup, tmp_path):
    cfg, train, test = setup
    rows = run_ablation(cfg, train, test, ["full", "norm_identity"], [0, 1], tmp_path, name="abl")
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    csv_rows = read_rows_csv(tmp_path / "abl_results.csv")
    assert len(csv_rows) == 4
    data = json.loads((tmp_path / "abl_results.json").read_text(encoding="utf-8"))
    assert set(data["summary"]) == {"full", "norm_identity"}
    assert data["summary"]["full"]["window_f1"]["n"] == 2
    assert (tmp_path / "runs" / "full__seed0.json").exists()
    by = {(r["variant"], r["seed"]): r for r in rows}
    assert by[("norm_identity", 0)]["n_params"] < by[("full", 0)]["n_params"]
    for r in rows:
        assert r["threshold_source"] == "validation" and r["used_legacy_defaults"] is False
        assert {"n_test_events", "n_test_pos_windows", "n_test_pos_samples", "n_test_windows"} <= set(r)
        assert "kurtosis_norm" in r and "fisher_latent" in r and "point_f1" in r and "window_f0_5" in r
    with pytest.raises(FileExistsError):
        run_ablation(cfg, train, test, ["full"], [0], tmp_path, name="abl")
    again = run_ablation(cfg, train, test, ["full", "norm_identity"], [0, 1], tmp_path, name="abl", force=True, resume=True)
    assert [r["window_f1"] for r in again] == [r["window_f1"] for r in rows]  # reused, not retrained
    # summarize only formats
    order, summary, _ = load_summary(tmp_path / "abl_results.json")
    md = to_markdown(order, summary, ["window_f1", "kurtosis_norm"])
    tex = to_latex(order, summary, ["window_f1", "kurtosis_norm"])
    assert "norm_identity" in md and r"\toprule" in tex and r"norm\_identity" in tex
    order_c, summary_c, _ = load_summary(tmp_path / "abl_results.csv")
    assert order_c == order and summary_c["full"]["window_f1"]["n"] == 2


def test_threshold_and_normalizer_do_not_depend_on_test_labels(setup):
    cfg, train, test = setup
    a = train_and_evaluate(cfg, train, test, 3, keep_artifacts=True)
    rep = a["_artifacts"]["report"]
    grid = np.linspace(0.05, 0.95, 181)
    assert a["threshold_source"] == "validation" and np.isclose(grid, a["threshold"]).any()
    assert rep["val_has_positives"] and rep["n_val_pos_windows"] > 0
    flipped = MissionData(test.values, 1 - test.labels, test.timestamps, test.channel_names, {})
    zeros = MissionData(test.values, np.zeros_like(test.labels), test.timestamps, test.channel_names, {})
    for other in (flipped, zeros):
        b = train_and_evaluate(cfg, train, other, 3)
        assert b["threshold"] == a["threshold"] and b["norm_lo"] == a["norm_lo"] and b["norm_hi"] == a["norm_hi"]
        assert b["mask_buffer_after_minutes"] == a["mask_buffer_after_minutes"]
    assert b["n_test_events"] == 0 and b["n_test_pos_samples"] == 0
    c = train_and_evaluate(cfg, train, flipped, 3)
    assert c["window_tp"] != a["window_tp"] or c["window_fp"] != a["window_fp"]   # labels only change the metrics


def test_experiments_and_scripts_share_one_function():
    import atandn.evaluation.pipeline as pipe
    import experiments.protocol as proto
    assert proto.evaluate_mission is pipe.evaluate_mission
    src = (Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "evaluate_mission" in src and "apply_known_event_mask" not in src and "f1_optimal_threshold" not in src


def test_variants_get_mask_and_spike_filter(setup):
    cfg, train, test = setup
    import copy
    iv = [(np.datetime64(160, "s"), np.datetime64(190, "s"))]
    tr = MissionData(train.values, train.labels, train.timestamps, train.channel_names, {"known_event_intervals": iv})
    c = copy.deepcopy(cfg)
    c.eval.use_known_event_mask, c.eval.min_consecutive = True, 2
    c.eval.mask_source = "rare_events"
    for variant in ("full", "classifier_only"):
        row = train_and_evaluate(apply_variant(c, variant), tr, test, 1)
        assert row["known_event_mask"] is True and row["min_consecutive"] == 2
        assert row["mask_buffer_before_minutes"] == 0.0 and row["mask_buffer_after_minutes"] >= 0.0
    assert row["alpha_rec"] == 0.0                      # no reconstruction head -> alpha forced to 0 (L11)


def test_runners_refuse_legacy_defaults(tmp_path):
    import sys
    from experiments.ablation import run_ablation
    from experiments.multi_seed import run_multi_seed
    cfgp = CONFIGS / "m1.yaml"
    for mod, extra in ((run_multi_seed, []), (run_ablation, ["--variants", "full"])):
        argv = ["--config", str(cfgp), "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "o")] + extra
        assert mod.main(argv) == 2                       # refused before any data is read
    from atandn.training import LegacyDefaultsError, check_legacy_defaults
    assert check_legacy_defaults(True, False) is False
    assert check_legacy_defaults(False, True) is True
    with pytest.raises(LegacyDefaultsError):
        check_legacy_defaults(False, False)


def test_legacy_flag_is_written_into_rows(setup, tmp_path):
    cfg, train, test = setup
    rows = run_multi_seed(cfg, train, test, [0], tmp_path, used_legacy_defaults=True)
    assert rows[0]["used_legacy_defaults"] is True
    meta = json.loads((tmp_path / "multi_seed_summary.json").read_text(encoding="utf-8"))["meta"]
    assert meta["used_legacy_defaults"] is True


def test_reproducible_same_seed(setup):
    cfg, train, test = setup
    a = train_and_evaluate(cfg, train, test, 5)
    b = train_and_evaluate(cfg, train, test, 5)
    for k in ("window_f1", "window_precision", "window_recall", "point_f1", "threshold", "kurtosis_norm", "fisher_latent"):
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k


def test_multi_seed_run_and_resume(setup, tmp_path):
    cfg, train, test = setup
    assert resolve_seeds(None, None) == [100, 777, 2026]
    assert resolve_seeds(None, 2) == [100, 777]
    assert resolve_seeds(None, 5) == [100, 777, 2026, 2027, 2028]
    rows = run_multi_seed(cfg, train, test, [0, 1], tmp_path)
    for f in ("seed_0.json", "seed_1.json", "multi_seed_runs.csv", "multi_seed_summary.csv", "multi_seed_summary.json"):
        assert (tmp_path / f).exists(), f
    summ = json.loads((tmp_path / "multi_seed_summary.json").read_text(encoding="utf-8"))["summary"]
    for scope in ("window", "point"):
        for k in ("precision", "recall", "fpr", "fnr", "f1", "f0_5"):
            assert summ[f"{scope}_{k}"]["n"] == 2
    vals = [r["window_f1"] for r in rows]
    assert summ["window_f1"]["mean"] == pytest.approx(np.mean(vals))
    assert summ["window_f1"]["std"] == pytest.approx(np.std(vals, ddof=1))
    mtime = (tmp_path / "seed_0.json").stat().st_mtime_ns
    rows2 = run_multi_seed(cfg, train, test, [0, 1], tmp_path)
    assert (tmp_path / "seed_0.json").stat().st_mtime_ns == mtime  # resumed, not retrained
    assert [r["window_f1"] for r in rows2] == vals
