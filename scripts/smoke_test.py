"""Synthetic integration test only. No result from this script is research evidence."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi.config import dump_json, load_config
from baf_roi.data import prepare
from baf_roi.train import train
from baf_roi.evaluate import evaluate
from baf_roi.runtime import Predictor
from baf_roi.audit import verify


def make_fixture(output):
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs/default.json').read_text())
    dataset = output / 'recordings'
    for s in range(3):
        directory = dataset / 'db' / f'session_{s}' / 'agent6_head'
        directory.mkdir(parents=True, exist_ok=True)
        t = np.arange(0, 8, 1 / 90)
        explore = np.clip(t - 2, 0, 3)
        pos = np.stack((2 + .008 * np.sin(t) + .35 * explore,
                        .005 * np.sin(t * 1.7) + .07 * explore,
                        1.5 + .003 * np.cos(t * .8) + .025 * explore), 1)
        pos += s * np.array([.01, .005, 0])
        rot = np.stack((10 + 2 * np.sin(1.5 * t) + 180 * explore, 3 * np.sin(t), np.sin(t * .8)), 1)
        rows = []
        with (directory / 'head.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['recv_unix', 'payload_json']); writer.writeheader()
            for i in range(len(t)):
                timestamp = float(1700000000 + s * 100 + t[i])
                payload = {'ts_utc': timestamp, 'pos': (pos[i] * 100).tolist(), 'rot_deg': rot[i].tolist()}
                writer.writerow({'recv_unix': timestamp, 'payload_json': json.dumps(payload)})
                rows.append((timestamp, payload['pos'], payload['rot_deg']))
        if s == 2:
            prefix = rows[:360]
            dump_json(output / 'history.json', {'timestamps_s': [v[0] for v in prefix],
                                               'positions': [v[1] for v in prefix],
                                               'rotations_deg': [v[2] for v in prefix]})
    split = output / 'split.json'
    dump_json(split, {'train': ['session_0'], 'valid': ['session_1'], 'test': ['session_2']})
    scene = output / 'synthetic_scene.json'
    dump_json(scene, {'defaults': {'config': {}}, 'streams': [
        {'config': {'obj_transform_pos_x_cm': 500, 'obj_transform_pos_y_cm': -100, 'obj_transform_pos_z_cm': 150}},
        {'config': {'obj_transform_pos_x_cm': 600, 'obj_transform_pos_y_cm': 150, 'obj_transform_pos_z_cm': 150}}]})
    cfg['data'].update(dataset_root=str(dataset), coordinate_contract_confirmed=True)
    cfg['split']['manifest'] = str(split)
    cfg['training']['batch_size'] = 32
    cfg['scene'].update(adaptive_config_path=str(scene), geometry_confirmed=True,
                        allow_uniform_grid_approximation=True, grid_dim=4)
    dump_json(output / 'config.json', cfg)
    dump_json(output / 'scene_config.json', cfg['scene'])
    return load_config(output / 'config.json')


def main(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Smoke output must be a new directory.')
    torch.set_num_threads(2)
    cfg = make_fixture(output)
    manifest = prepare(cfg, output / 'cache')
    checkpoint = train(output / 'cache', output / 'run', 'cpu', epochs_override=1)
    report = evaluate(output / 'cache', checkpoint, output / 'evaluation', 'cpu', batch_size=16)
    raw = json.loads((output / 'history.json').read_text())
    forecast = Predictor.load(checkpoint).predict_observations(raw['timestamps_s'], raw['positions'], raw['rotations_deg'])
    assert forecast['positions_m'].shape == (1, 5, 3)
    assert forecast['quaternions_wxyz'].shape == (1, 5, 4)
    assert forecast['rollout_positions_m'].shape[1] == 9
    assert all(report['metrics'][str(h)]['n_origins'] > 0 for h in (50, 100, 150, 200, 250))
    final = {'status': 'PASS', 'evidence_type': 'SYNTHETIC INTEGRATION TEST, NOT REAL DATA ACCURACY',
             'epochs_per_stage': 1, 'trajectories': len(manifest['trajectories']),
             'test_origins_per_horizon': report['metrics']['50']['n_origins'],
             'all_five_horizons': report['horizons_ms'], 'audit': verify(),
             'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu'}
    dump_json(output / 'smoke_result.json', final)
    print(json.dumps(final, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.output)
