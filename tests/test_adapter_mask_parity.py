"""The baseline adapter builds the known-event mask exactly like ``evaluate_mission`` (both ``mask_source`` values)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import protocol_adapter as pa  # noqa: E402
import split_io  # noqa: E402
from atandn.data import MissionData  # noqa: E402
from atandn.evaluation.pipeline import evaluate_mission, fit_frozen_protocol, prepare_protocol_data  # noqa: E402
from atandn.inference.postprocess import min_consecutive_filter  # noqa: E402
from atandn.inference.scoring import ScoreNormalizer, dual_evidence, score_stream  # noqa: E402
from test_protocol import Toy  # noqa: E402
from test_stage3_options import _cfg, _tc_stream  # noqa: E402


def _adapter_inputs(tmp: Path, full: MissionData, train: MissionData, cfg, fz, te_raw, te_end):
    """Score file of a 'baseline' whose test scores are the model's raw scores and whose fit scores make the
    percentile normalizer the identity on [0, 1]."""
    n, t0, w, s = len(full), len(train), cfg.model.window_size, cfg.data.test_stride
    split = {"schema": split_io.SPLIT_SCHEMA, "mission": "m1", "n_samples": n, "train_stop": t0,
             "fit": {"start": 0, "stop": int(0.6 * t0)}, "val": {"start": int(0.7 * t0), "stop": t0},
             "test": {"start": t0, "stop": n}, "window_size": w, "test_stride": s,
             "timestamps": split_io.timestamps_identity(full.timestamps)}
    rng = np.random.default_rng(0)
    parts = {}
    for name in ("fit", "val"):
        a, b = split_io.part_range(split, name)
        ends = split_io.window_starts_in(a, b, w, s) + w - 1
        sc = rng.random(len(ends))
        sc[::4], sc[1::4] = 0.0, 1.0                       # 1st / 99th percentile = 0 / 1
        parts[name] = (sc, ends)
    parts["test"] = (te_raw, te_end + t0)
    split_io.save_scores(tmp / "s.npz", parts, window_size=w, stride=s, model="toy", mission="m1",
                         ts_sha1=split["timestamps"]["sha1"])
    return split


@pytest.mark.parametrize("source", ["telecommands", "rare_events"])
@pytest.mark.parametrize("give_mask_params", [True, False])
def test_adapter_masked_predictions_equal_evaluate_mission(tmp_path: Path, source: str, give_mask_params: bool) -> None:
    torch.manual_seed(0)
    full, _ = _tc_stream()
    train, test = full.split(0.6)
    if source == "rare_events":
        meta = {"known_event_intervals": [(train.timestamps[300], train.timestamps[320]),
                                          (test.timestamps[400], test.timestamps[430])]}
        train.meta, test.meta, full.meta = meta, meta, meta
    cfg, model = _cfg(source), Toy()
    data = prepare_protocol_data(train, test, cfg)
    fz = fit_frozen_protocol(model, data, cfg, device="cpu")
    rep = evaluate_mission(model, data, cfg, frozen=fz, device="cpu", return_arrays=True)
    w, s = cfg.model.window_size, cfg.data.test_stride
    te = score_stream(model, test.values, w, s, cfg.eval.batch_size, "cpu")
    raw = dual_evidence(te["prob"], te["recon_err"], fz["alpha_rec"], ScoreNormalizer.from_state_dict(fz["normalizer"]))
    assert np.all((raw >= 0) & (raw <= 1))
    split = _adapter_inputs(tmp_path, full, train, cfg, fz, raw, te["end_index"])
    cfg.eval.allow_train_threshold_fallback = True          # only the threshold differs from the pipeline's
    mp = fz["mask_params"] if give_mask_params else None
    out = pa.evaluate_baseline_scores(
        tmp_path / "s.npz", cfg, labels=full.labels, timestamps=full.timestamps, split=split,
        intervals=full.meta.get("known_event_intervals"), telecommands=full.meta.get("telecommands"),
        values=full.values, mask_params=mp, return_arrays=True)
    a, p = out["_arrays"], rep["_arrays"]
    assert np.allclose(a["test_scores"], p["test_scores"], atol=1e-12)           # identical masked score stream
    pred = min_consecutive_filter((p["test_scores"] > out["threshold"]).astype(int), fz["min_consecutive"])
    assert np.array_equal(a["test_pred"], pred)
    pp, rp = out["postprocessing"], rep["postprocessing"]
    for k in ("mask_source", "n_known_events", "mask_buffer_before_minutes", "mask_buffer_after_minutes"):
        assert pp[k] == rp[k], k
    assert pp["known_event_mask"] and (a["test_scores"] == 0).any()
    if source == "telecommands":
        assert pp["telecommand_selected_types"] == rp["telecommand_selected_types"] == ["A"]


def test_adapter_without_telecommands_raises_the_pipeline_error(tmp_path: Path) -> None:
    torch.manual_seed(0)
    full, _ = _tc_stream()
    train, test = full.split(0.6)
    cfg = _cfg("telecommands")
    w, s = cfg.model.window_size, cfg.data.test_stride
    te = score_stream(Toy(), test.values, w, s, cfg.eval.batch_size, "cpu")
    split = _adapter_inputs(tmp_path, full, train, cfg, None, np.random.default_rng(0).random(len(te["end_index"])),
                            te["end_index"])
    with pytest.raises(ValueError, match="telecommands") as e:
        pa.evaluate_baseline_scores(tmp_path / "s.npz", cfg, labels=full.labels, timestamps=full.timestamps, split=split,
                                    mask_params={"min_consecutive_windows": 1})
    from atandn.evaluation.pipeline import require_telecommands
    with pytest.raises(ValueError) as e2:
        require_telecommands(None)
    assert str(e.value) == str(e2.value)
    cfg.eval.use_known_event_mask = False                    # mask off: no telecommands needed (same as the pipeline)
    cfg.eval.allow_train_threshold_fallback = True
    pa.evaluate_baseline_scores(tmp_path / "s.npz", cfg, labels=full.labels, timestamps=full.timestamps, split=split)
