"""Accounting conventions and real-runtime analysis integration."""
import argparse
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from baf_roi.complexity import (analyze, count_forecast, dense_macs, memory_probe,
                                model_inventory, stage_probe, summarize_times, synthetic_history)
from baf_roi.models import build_models
from baf_roi.runtime import Predictor


@pytest.mark.parametrize('batch_first,bidirectional,projection', [
    (True, False, 0), (False, True, 0), (True, True, 2)])
def test_lstm_accounts_all_layers_timesteps_and_directions(batch_first, bidirectional, projection):
    model = nn.LSTM(3, 5, 2, batch_first=batch_first, bidirectional=bidirectional, proj_size=projection)
    x = torch.zeros((2, 7, 3) if batch_first else (7, 2, 3))
    d, r = (2 if bidirectional else 1), (projection or 5)
    expected = 2 * 7 * d * (4 * 5 * (3 + r) + 4 * 5 * (d * r + r) + 2 * 5 * projection)
    assert dense_macs(model, (x,), None) == expected


def test_conv_counts_padding_before_chomp_and_groups():
    conv = nn.Conv1d(4, 6, 3, padding=4, dilation=2, groups=2)
    x = torch.zeros(2, 4, 30)
    out = conv(x)
    assert out.shape == (2, 6, 34)
    assert dense_macs(conv, (x,), out) == 2 * 6 * 34 * 2 * 3
    linear = nn.Linear(4, 3)
    assert dense_macs(linear, (torch.zeros(2, 7, 4),), torch.zeros(2, 7, 3)) == 2 * 7 * 3 * 4


def test_counts_whole_shared_rollout_without_changing_prediction():
    torch.set_num_threads(2)
    predictor = Predictor(build_models(), {'mean': [0, 0, 0], 'std': [1, 1, 1]})
    history = synthetic_history(1)
    original = history.copy()
    short, _ = count_forecast(predictor, history, [50])  # 2 steps
    full, result = count_forecast(predictor, history, [50, 100, 150, 200, 250])  # 8
    single, _ = count_forecast(predictor, history, [250])
    long, _ = count_forecast(predictor, history, [1000])  # 30
    batched, _ = count_forecast(predictor, np.repeat(history, 2, axis=0), [250])
    assert full['dense_macs_per_call'] == 4 * short['dense_macs_per_call']
    assert single['dense_macs_per_call'] == full['dense_macs_per_call']
    assert long['dense_macs_per_call'] * 8 == full['dense_macs_per_call'] * 30
    assert batched['dense_macs_per_call'] == 2 * full['dense_macs_per_call']
    assert np.array_equal(original, history)
    assert np.array_equal(result['positions_m'], predictor.forecast_batch(history)['positions_m'])
    assert all(not module._forward_hooks for m in predictor.models.values() for module in m.modules())
    assert all(row['calls'] == 8 for row in full['layers'])
    diagnostic = stage_probe(predictor, history, [250])
    assert diagnostic['stages']['branch/focused']['calls'] == 8
    assert diagnostic['remainder_ms'] >= 0
    assert model_inventory(predictor.models)['total']['parameter_bytes'] == sum(
        p.numel() * p.element_size() for m in predictor.models.values() for p in m.parameters())


def test_latency_does_not_confuse_amortized_time_with_response_time():
    stats = summarize_times([10, 20, 30, 40], 4, 25)
    assert stats['mean_ms'] == 25
    assert stats['amortized_ms_per_history'] == 6.25
    assert stats['histories_per_second'] == 160
    assert stats['deadline_miss_fraction'] == .5
    probe = memory_probe(lambda: {'a': np.ones(10)}, 'cpu')
    assert probe['returned_numpy_array_bytes'] == 80
    assert probe['cpu_native_peak_bytes'] is None


@pytest.mark.parametrize('checkpoint_mode', [False, True])
def test_full_report_with_observation_adapter_roi_and_checkpoint(tmp_path, checkpoint_mode):
    stats = {'mean': [0, 0, 0], 'std': [1, 1, 1]}
    checkpoint = tmp_path / 'test.pt'
    if checkpoint_mode:
        torch.save({'format_version': 2, 'models': {k: m.state_dict() for k, m in build_models().items()},
                    'normalization': stats, 'config': {'model': {'temporal_mu': .6},
                    'data': {'position_unit': 'cm', 'max_gap_ms': 100}}}, checkpoint)
    h = synthetic_history(1)[0]
    history = tmp_path / 'history.json'
    history.write_text(json.dumps({'timestamps_s': (np.arange(30) / 30).tolist(),
                       'positions': (h[:, :3] * (100 if checkpoint_mode else 1)).tolist(),
                       'rotations_deg': h[:, 3:].tolist()}))
    centers = tmp_path / 'centers.json'
    centers.write_text(json.dumps({'coordinate_convention': 'legacy_ypr_xforward', 'unit': 'm',
                                  'centers_m': [[3, 0, 1.6]], 'partition_ids': ['p0']}))
    scene = tmp_path / 'scene.json'
    scene.write_text(json.dumps({'geometry_confirmed': True, 'partition_centers_file': 'centers.json',
                                'visibility_mode': 'legacy_center_cone', 'fov_deg': 90}))
    args = argparse.Namespace(threads=2, iterations=2, warmup=1, batch_sizes=[1, 2], deadline_ms=1000 / 30,
                              horizon_sets=[[50]], seed=57, device='cpu', output=str(tmp_path / 'report'),
                              checkpoint=str(checkpoint) if checkpoint_mode else None,
                              scene_config=str(scene), history=str(history), profile=not checkpoint_mode)
    report = analyze(args)
    loaded = json.loads((tmp_path / 'report/report.json').read_text())
    assert loaded['model']['total']['parameters'] == report['model']['total']['parameters']
    assert loaded['cases'][0]['roi_output_shape'] == [1, 1, 1]
    assert 'observations_to_roi' in loaded['cases'][0]['latency']
    assert 'observations_to_pose' not in loaded['cases'][1]['latency']
    assert (tmp_path / 'report/summary.md').is_file()
    with (tmp_path / 'report/measurements.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert any(row['endpoint'] == 'observations_to_roi' for row in rows)
    assert len(rows) == sum(len(case['latency']) for case in report['cases'])
    if args.profile:
        trace = json.loads((tmp_path / 'report/profile_000/trace.json').read_text())
        labels = {event.get('name') for event in trace['traceEvents']}
        assert {'pose_to_roi', 'observation_resampling', 'branch/focused'} <= labels
    assert report['weights']['kind'] == ('trained_checkpoint' if checkpoint_mode else 'UNTRAINED_ARCHITECTURE_ONLY')
    with pytest.raises(ValueError, match='empty output'):
        analyze(args)
