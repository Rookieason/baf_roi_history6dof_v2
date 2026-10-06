"""Source fidelity, causal rollout, and shared-origin end-to-end regression checks."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from baf_roi.data import prepare, Cache, OriginDataset
from baf_roi.models import build_models
from baf_roi.evaluate import evaluate as evaluate_baf
from baf_roi.pose import ypr_to_quat, quaternion_error_deg
from motion_transformer_roi.models import build_model, source_hashes, utility
from motion_transformer_roi.train import train, compute_losses
from motion_transformer_roi.runtime import Predictor
from motion_transformer_roi.evaluate import evaluate
from scripts.smoke_test import make_fixture


def test_vendored_source_fidelity():
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'vendor/motion_transformer/provenance.json').read_text())
    assert source_hashes() == provenance['sha256']
    for name, source in provenance['sources'].items():
        if (root / source).exists():
            assert (root / source).read_bytes() == (root / 'vendor/motion_transformer' / name).read_bytes()
    cfg = asdict(build_model().config)
    assert (cfg['d_model'], cfg['num_heads'], cfg['num_layers'], cfg['seq_len']) == (128, 4, 2, 30)


def test_original_losses_and_quaternion_sign():
    p = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
    q = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
    batch = {'target_tr': p + 2, 'target_q': -q}
    losses = compute_losses({'p_hat_tr': p, 'q_hat_norm': q}, batch)
    assert torch.equal(losses['loss_total'], utility.translation_l1_loss(p, p + 2)
                       + .3 * utility.quaternion_cosine_loss(q, -q))
    assert losses['loss_total'].item() == pytest.approx(2., abs=1e-6)


class ConstantVelocity(torch.nn.Module):
    """Makes recursive history updates observable; predicts from last two poses."""
    def forward(self, x):
        return {'p_hat_tr': 2 * x[:, -1, :3] - x[:, -2, :3], 'q_hat_norm': x[:, -1, 3:]}


def test_recursive_rollout_exact_horizons_and_online_units():
    stats = {'mean': [2., -1., 3.], 'std': [.5, 2., 3.]}
    predictor = Predictor(ConstantVelocity(), stats, data_config={'position_unit': 'cm', 'max_gap_ms': 100})
    ts = np.arange(61) / 30
    pos = np.column_stack((2 + 3 * ts, ts * 0 - 1, ts * 0 + 3))
    rot = np.tile([170., 0., 0.], (len(ts), 1))
    result = predictor.predict_observations(ts, pos * 100, rot)
    assert np.allclose(result['positions_m'][0, :, 0], pos[-1, 0] + 3 * np.arange(50, 251, 50) / 1000, atol=2e-5)
    assert result['rollout_positions_m'].shape == (1, 9, 3)
    assert np.max(quaternion_error_deg(result['quaternions_wxyz'], ypr_to_quat(rot[-1]))) < 1e-5
    h = np.concatenate((pos[-30:], rot[-30:]), 1)[None]
    assert np.allclose(result['positions_m'], predictor.forecast_batch(h)['positions_m'])
    assert np.array_equal(h[0, :, :3], pos[-30:])
    for bad in ([], [49], [251], [50, 50], [float('nan')]):
        with pytest.raises(ValueError):
            predictor.forecast_batch(h, bad)


def test_real_model_is_history_only_and_batch_independent(monkeypatch):
    torch.manual_seed(13)
    predictor = Predictor(build_model(), {'mean': [0., 0., 0.], 'std': [1., 1., 1.]})
    h = np.zeros((2, 30, 6)); h[:, :, 2] = 1.5
    h[1, :, 3] = np.linspace(170, 190, 30)
    before = h.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not call behavior supervision.')
    monkeypatch.setattr('baf_roi.data.centered_pair', forbidden)
    result = predictor.forecast_batch(h)
    alone = predictor.forecast_batch(h[1:])
    assert np.array_equal(h, before)
    assert np.allclose(result['positions_m'][1:], alone['positions_m'], atol=3e-5)
    assert np.max(quaternion_error_deg(result['quaternions_wxyz'][1:], alone['quaternions_wxyz'])) < .005
    assert np.isfinite(result['positions_m']).all()
    assert np.allclose(np.linalg.norm(result['quaternions_wxyz'], axis=-1), 1.)


def read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def test_train_evaluate_identical_baf_origins_and_metrics(tmp_path):
    cfg = make_fixture(tmp_path)
    prepare(cfg, tmp_path / 'cache')
    cache = Cache(tmp_path / 'cache')
    # The baseline accepts all valid targets, independent of centered behavior labels.
    dataset = OriginDataset(tmp_path / 'cache', 'train', 'all')
    assert len(dataset) == sum(r['counts']['all_next'] for r in cache.records if r['split'] == 'train')
    checkpoint = train(tmp_path / 'cache', tmp_path / 'mt', epochs=1)
    predictor = Predictor.load(checkpoint)
    assert predictor.bundle['normalization'] == cache.stats
    report = evaluate(tmp_path / 'cache', checkpoint, tmp_path / 'mt_eval', batch_size=32)
    # Untrained BAF weights suffice to verify evaluator alignment, not model accuracy.
    baf_checkpoint = tmp_path / 'baf.pt'
    torch.save({'format_version': 2, 'models': {k: m.state_dict() for k, m in build_models().items()},
                'normalization': cache.stats, 'config': cfg, 'cache_id': cache.manifest['cache_id']}, baf_checkpoint)
    baf = evaluate_baf(tmp_path / 'cache', baf_checkpoint, tmp_path / 'baf_eval', batch_size=32)
    rows = read_rows(tmp_path / 'mt_eval/per_origin_metrics.csv')
    other = read_rows(tmp_path / 'baf_eval/per_origin_metrics.csv')
    columns = ('trajectory', 'session', 'origin_epoch_s', 'horizon_ms', 'target_epoch_s')
    assert [tuple(r[k] for k in columns) for r in rows] == [tuple(r[k] for k in columns) for r in other]
    assert all(r['alpha_t'] == '' for r in rows)
    assert report['scene_id'] == baf['scene_id']
    for h in report['horizons_ms']:
        assert report['metrics'][str(h)]['n_origins'] == baf['metrics'][str(h)]['n_origins'] > 0
    files = sorted((tmp_path / 'mt_eval/predictions').glob('*.npz'))
    for f in files:
        with np.load(f) as a, np.load(tmp_path / 'baf_eval/predictions' / f.name) as b:
            assert np.array_equal(a['roi_true_packed'], b['roi_true_packed'])
            n = int(a['num_partitions'])
            pred = np.unpackbits(a['roi_pred_packed'], axis=-1, count=n, bitorder='little').astype(bool)
            truth = np.unpackbits(a['roi_true_packed'], axis=-1, count=n, bitorder='little').astype(bool)
            assert pred.shape == truth.shape
    valid = evaluate(tmp_path / 'cache', checkpoint, tmp_path / 'valid_eval', split='valid', save_masks=False)
    assert valid['split'] == 'valid'
    assert not (tmp_path / 'valid_eval/predictions').exists()
    raw = json.loads((tmp_path / 'history.json').read_text())
    assert predictor.predict_observations(**raw)['positions_m'].shape == (1, 5, 3)
    with pytest.raises(ValueError, match='new run'):
        train(tmp_path / 'cache', tmp_path / 'mt', epochs=1)
    with pytest.raises(ValueError, match='Expected adapted'):
        Predictor.load(baf_checkpoint)
    broken = dict(predictor.bundle, cache_id='different-cache')
    torch.save(broken, tmp_path / 'mismatch.pt')
    with pytest.raises(ValueError, match='mismatch'):
        evaluate(tmp_path / 'cache', tmp_path / 'mismatch.pt', tmp_path / 'rejected')
