"""Train, reload, evaluate, and deploy the internal residual on synthetic data.

This checks executable software paths. Synthetic accuracy, throughput, or a
chosen checkpoint here cannot establish a real CSI benefit or full-system FPS.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from baf_roi.data import prepare as prepare_hmd, Series
from baf_roi.train import train as train_baf
from baf_roi_heatmap.contracts import InputConfig, write_json
from baf_roi_heatmap.data import ForecastDataset, encode_window
from baf_roi_heatmap.prepare import prepare
from baf_roi_heatmap_residual.runtime import (
    HeatmapResidualPredictor, load_residual, predict_dataset_batch,
)
from roi_experiments.evaluation import evaluate, read_rows
from roi_experiments.runner import parser as experiment_parser, run as run_experiment
from scripts.smoke_heatmap import make_fixture, make_radio


def main(output, prepared=None):
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory for this software smoke test.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if prepared is None:
        config = make_fixture(out)
        cache = prepare_hmd(config, out/'cache')
        baseline = train_baf(out/'cache', out/'baf', 'cpu', epochs_override=1)
        make_radio(out, cache)
        prepared = out/'prepared'
        prepare(out/'cache', baseline, out/'rf_manifest.json', prepared,
                horizons=[50, 250, 1000], input_config=InputConfig(size=8), batch_size=64)
    else:
        prepared = Path(prepared).resolve()
        manifest = json.loads((prepared/'manifest.json').read_text())
        if manifest.get('rf_manifest', {}).get('evidence_type') != 'synthetic':
            raise ValueError('--prepared may only reuse an explicitly synthetic RF fixture.')

    run = out/'run'
    run_experiment(experiment_parser().parse_args([
        '--prepared', str(prepared), '--method', 'baf_roi', 'baf_roi_heatmap_residual',
        '--smoke-epochs', '1', '--batch-size', '32', '--device', 'cpu', '--threads', '2',
        '--residual-hidden', '8', '--residual-bottleneck', '8', '--margins', '0',
        '--output', str(run),
    ]))
    experiment = json.loads((run/'experiment.json').read_text())
    assert experiment['complete']
    assert experiment['methods'] == ['baf_roi', 'baf_roi_heatmap_residual']
    selected = Path(experiment['checkpoints']['baf_roi_heatmap_residual'])
    trained = selected.with_name('best_trained.pt')
    assert trained.is_file() and selected.with_name('last.pt').is_file()
    trained_bundle = torch.load(trained, map_location='cpu', weights_only=True)
    assert trained_bundle['selected_epoch'] == 1

    # Even if validation retains epoch zero, exercise a trained assisted path.
    dataset = ForecastDataset(prepared, 'test')
    model, bundle = load_residual(trained, dataset)
    assert not model.is_zero_residual()
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    with torch.inference_mode():
        prediction = predict_dataset_batch(model, dataset, batch)
    assert torch.isfinite(prediction['p']).all() and torch.isfinite(prediction['q']).all()
    evaluate(prepared, out/'trained_evaluation', residual_checkpoint=trained, margins=[0.])
    evaluate(prepared, out/'delayed_evaluation', residual_checkpoint=trained,
             delivery_delay_s=2., margins=[0.])
    original_rows = read_rows(run/'evaluations/test/baf_roi')
    delayed_rows = read_rows(out/'delayed_evaluation')
    assert len(original_rows) == len(delayed_rows)
    for original, delayed in zip(original_rows, delayed_rows):
        assert not delayed['rf_available'] and not delayed['correction_applied']
        for key in ('tp', 'fp', 'fn', 'tn', 'iou', 'translation_error_m', 'rotation_error_deg'):
            assert original[key] == delayed[key], key

    # Compare actual online execution with the same resampled inputs, observed
    # alpha, and causal RF window submitted directly to the loaded model.
    index = next(i for i in range(len(dataset)) if dataset[i]['inputs']['rf_available'])
    ri, oi = map(int, dataset.index[index])
    record = dataset.records[ri]
    frozen = dataset.arrays(ri)
    raw = dataset.cache.arrays(record['cache_index'])
    history = raw['series'].history(int(frozen['anchors'][oi]))
    origin = float(frozen['origin_s'][oi])
    times = origin-np.arange(29, -1, -1)/30
    online = HeatmapResidualPredictor(dataset.manifest['baseline_checkpoint'], trained)
    unit_scale = 100. if online.base.data_config['position_unit'] == 'cm' else 1.
    radio = dataset.rf_series(record['key'])
    baseline_calls = []
    original_forecast = online.base.forecast_batch
    def counted(*args, **kwargs):
        baseline_calls.append(True)
        return original_forecast(*args, **kwargs)
    online.base.forecast_batch = counted
    actual = online.predict_observations(times, history[:, :3]*unit_scale, history[:, 3:],
                                         radio, stream_id='synthetic')
    assert actual['execution_path'] == 'internal_residual_single_rollout'
    assert actual['adapted'] and not baseline_calls
    resampled = Series(times-times[0], history, online.base.data_config['max_gap_ms']/1000).history(29)
    window = encode_window(radio, origin, bundle['normalizer'], online.cfg,
                           age_basis=bundle.get('rf_age_basis', 'support_end'))
    inputs = {key: torch.as_tensor(np.asarray(value)[None]) for key, value in window.items()}
    inputs.update(alpha=torch.as_tensor(actual['alpha_t']),
                  horizon_s=torch.tensor([dataset.horizons], dtype=torch.float32)/1000)
    with torch.inference_mode():
        direct = model(inputs, torch.tensor(resampled[None]))
    np.testing.assert_allclose(actual['positions_m'], direct['p'].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual['quaternions_wxyz'], direct['q'].numpy(), rtol=0, atol=1e-6)
    saved = online.base.observed_state.copy()
    bad_radio = type('BadGrid', (), {'grid_id': 'incorrect', 'metadata': {'physical_axes_verified': True}})()
    try:
        online.predict_observations(times+1/30, history[:, :3]*unit_scale, history[:, 3:],
                                    bad_radio, stream_id='synthetic')
    except ValueError:
        pass
    else:
        raise AssertionError('Runtime accepted RF from an incompatible physical grid.')
    assert online.base.observed_state == saved

    # Re-evaluation uses recorded checkpoint identities without any retraining.
    run_experiment(experiment_parser().parse_args([
        'evaluate', '--run', str(run), '--output', str(out/'reevaluated'),
        '--device', 'cpu', '--margins', '0',
    ]))
    result = {
        'status': 'PASS', 'evidence_type': 'SYNTHETIC SOFTWARE INTEGRATION ONLY',
        'method': 'baf_roi_heatmap_residual', 'train_epochs': 1,
        'horizons_ms': dataset.horizons, 'test_origins': len(dataset),
        'selected_epoch': torch.load(selected, map_location='cpu', weights_only=True)['selected_epoch'],
        'trained_checkpoint_reloaded': True, 'single_assisted_online_rollout': True,
        'online_matches_direct_model': True, 'exact_delayed_rf_fallback': True,
        'online_error_preserves_observed_state': True, 'unified_reevaluation': True,
        'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu',
    }
    write_json(out/'smoke_result.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepared', help='Optional existing synthetic prepared fixture; never real research data.')
    arguments = parser.parse_args()
    main(arguments.output, arguments.prepared)
