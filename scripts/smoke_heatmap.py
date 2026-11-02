"""Real executable integration exercise on SYNTHETIC data, never research evidence."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
from baf_roi.config import dump_json, load_config
from baf_roi.data import prepare as prepare_hmd
from baf_roi.train import train as train_baf
from baf_roi_heatmap.contracts import InputConfig, ModelConfig, write_json
from baf_roi_heatmap.music import generate
from baf_roi_heatmap.prepare import prepare
from baf_roi_heatmap.train import train
from baf_roi_heatmap.data import ForecastDataset
from baf_roi_heatmap.evaluation import evaluate, compare, read_rows, select_policy
from baf_roi_heatmap.diagnostics import oracle, plot
from baf_roi_heatmap.runtime import benchmark


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

def make_radio(output, cache_manifest):
    """Known steering simulation with independent noise; NOT a human CSI measurement."""
    records = []
    config = {'window_packets': 10, 'stride_packets': 5, 'rx_window': 3,
              'frequency_points': 4, 'frequency_hop': 1, 'signal_dim': 1,
              'theta_deg': list(range(-60, 61, 15)), 'tau_s': np.linspace(0, 20e-9, 8).tolist(),
              'carrier_hz': 5.57e9, 'antenna_spacing_m': .02, 'max_raw_gap_s': .02,
              'projection': 'sin', 'preprocess': 'window_mean', 'delivery_delay_s': .005,
              'phase_calibrated': True, 'frequency_mapping_verified': True, 'array_convention_verified': True,
              'clock_domain': 'unix_seconds', 'evidence_type': 'synthetic known steering only'}
    write_json(output/'radio_config.json', config)
    for index, record in enumerate(cache_manifest['trajectories']):
        t = np.arange(0, 8, .01)
        ts = record['epoch_start_s'] + t
        rng = np.random.default_rng(117 + index)
        freq = 5.57e9 + np.arange(8)*2.5e6
        theta = .45*np.sin(t*1.2)
        tau = (10+2*np.cos(t))*1e-9
        a = np.exp(-2j*np.pi*5.57e9*.02/3e8*np.sin(theta)[:, None]*np.arange(4))
        b = np.exp(-2j*np.pi*tau[:, None]*freq[None])
        csi = a[:, None, :, None]*b[:, None, None, :]*np.exp(2j*np.pi*3*t)[:, None, None, None]
        csi += .01*(rng.normal(size=csi.shape)+1j*rng.normal(size=csi.shape))
        raw = output/(record['key']+'_raw.npz'); maps = output/(record['key']+'_maps.npz')
        np.savez_compressed(raw, csi=csi, timestamps_s=ts, available_at_s=ts+.001, frequencies_hz=freq)
        generate(raw, output/'radio_config.json', maps)
        records.append({'key': record['key'], 'file': str(maps), 'source_clock_domain': 'unix_seconds',
                        'clock_offset_s': 0., 'clock_uncertainty_s': 0., 'group_id': record['session'], 'scenario': 'synthetic'})
    manifest = {'format_version': 1, 'hmd_cache_id': cache_manifest['cache_id'],
                'hmd_clock_domain': 'unix_seconds', 'synchronization_confirmed': True,
                'allow_exploratory': False, 'evidence_type': 'synthetic', 'records': records}
    write_json(output/'rf_manifest.json', manifest)


def main(output, skip_plots=False):
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new output directory.')
    out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    config = make_fixture(out)
    cache = prepare_hmd(config, out/'cache')
    base = train_baf(out/'cache', out/'baf', 'cpu', epochs_override=1)
    from motion_transformer_roi.train import train as train_motion_transformer
    mt = train_motion_transformer(out/'cache', out/'motion_transformer', epochs=1, batch_size=32)
    make_radio(out, cache)
    frozen = prepare(out/'cache', base, out/'rf_manifest.json', out/'prepared',
                     horizons=[50, 250, 1000], input_config=InputConfig(size=8), batch_size=64)
    methods = {}
    for method in ('residual', 'expert_gate'):
        for modality in ('hmd', 'rf'):
            name = method+'_'+modality
            methods[name] = train(out/'prepared', out/name, ModelConfig(method=method, modality=modality, hidden=8), epochs=1)
    for baseline in ('baf', 'focused', 'explorer', 'fixed'):
        evaluate(out/'prepared', out/('eval_'+baseline), baseline=baseline)
    evaluate(out/'prepared', out/'eval_motion_transformer', motion_transformer_checkpoint=mt)
    compare(out/'eval_baf', out/'eval_motion_transformer', out/'mt_paired.json', bootstrap_samples=100)
    assert all(r['gate'] is None for r in read_rows(out/'eval_motion_transformer'))
    for name, checkpoint in methods.items():
        evaluate(out/'prepared', out/('eval_'+name), checkpoint=checkpoint)
    for method in ('residual', 'expert_gate'):
        compare(out/('eval_'+method+'_hmd'), out/('eval_'+method+'_rf'), out/(method+'_paired.json'), bootstrap_samples=100)
        paired = json.loads((out/(method+'_paired.json')).read_text())
        assert all(r['cluster_percentile_ci95'] is None for r in paired['results'])
    evaluate(out/'prepared', out/'eval_delayed', checkpoint=methods['residual_rf'], delivery_delay_s=2.)
    a, b = read_rows(out/'eval_baf'), read_rows(out/'eval_delayed')
    assert len(a) == len(b)
    for original, delayed in zip(a, b):
        assert not delayed['rf_available']
        for key in ('tp', 'fp', 'fn', 'tn', 'iou', 'translation_error_m', 'rotation_error_deg'):
            assert original[key] == delayed[key], key
    evaluate(out/'prepared', out/'valid_margin', checkpoint=methods['residual_rf'], split='valid', margins=[0., 5., 20.])
    # This low synthetic-only threshold tests policy plumbing, NOT a thesis quality target.
    policy = select_policy(out/'valid_margin', out/'policy.json', coverage=.01)
    if all(v is not None for v in policy['margins_by_horizon'].values()):
        evaluate(out/'prepared', out/'test_policy', checkpoint=methods['residual_rf'], policy_path=out/'policy.json')
    oracle(out/'prepared', out/'oracle.json', grid_points=5)
    benchmark(out/'prepared', methods['residual_rf'], out/'benchmark.json', count=2)
    if not skip_plots:
        plot([out/'eval_baf', out/'eval_residual_hmd', out/'eval_residual_rf'], out/'plots')
    dataset = ForecastDataset(out/'prepared', 'test')
    result = {'status': 'PASS', 'evidence_type': 'SYNTHETIC SOFTWARE INTEGRATION ONLY',
              'base_epochs_each_stage': 1, 'adapter_epochs': 1, 'adapter_methods': list(methods),
              'matched_origins_per_horizon': len(dataset), 'horizons_ms': frozen['horizons_ms'],
              'frozen_baselines': 4, 'matched_motion_transformer': True, 'exact_missing_rf_fallback': True,
              'single_cluster_ci_correctly_undefined': True, 'plots': 0 if skip_plots else 8,
              'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu'}
    write_json(out/'smoke_result.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--skip-plots', action='store_true', help='Useful in headless CI; the plot CLI can run separately.')
    args = parser.parse_args(); main(args.output, args.skip_plots)
