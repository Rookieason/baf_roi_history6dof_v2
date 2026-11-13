"""Import existing valder AoA-ToF maps and train on one explicitly selected date.

No CSI/heatmap generation and no old-day checkpoints are used. Input timestamps
are collector reception times; they are not asserted to be radio/HMD capture times.
"""
import argparse
import csv
from functools import lru_cache
from pathlib import Path
import json
import os
import sys
try:
    import resource
except ImportError:  # Import helpers in Windows tests; execution targets Linux.
    resource = None
import threading
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from baf_roi.config import load_config, dump_json
from baf_roi.data import Cache, OriginDataset, prepare as prepare_pose
from .contracts import InputConfig, ModelConfig, file_hash, read_json, write_json
from .data import HeatmapSeries

GIB = 1024 ** 3


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def check_resources(path, initial=False):
    import shutil
    path = Path(path)
    while not path.exists():
        path = path.parent
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    free = shutil.disk_usage(path).free
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if free < (8 if initial else 2) * GIB or memory['MemAvailable'] < 2 * GIB or rss > 6 * GIB:
        raise RuntimeError(f'Resource reserve reached: disk={free/GIB:.2f} GiB, '
                           f'available RAM={memory["MemAvailable"]/GIB:.2f} GiB, peak RSS={rss/GIB:.2f} GiB')
    return {'free_disk_gib': round(free/GIB, 3), 'available_ram_gib': round(memory['MemAvailable']/GIB, 3),
            'peak_rss_mib': round(rss/1024**2, 2)}


def dependency_times(matched, nic_receivers, half_window=12, past_only=False):
    """Include future interpolation endpoints and every centered-background input.

    A merged value interpolated between two packets depends on BOTH endpoints,
    even when its own CSV row is a phantom. Never timestamp it as available earlier.
    """
    n = len(matched)
    lower, upper, observed = [], [], np.zeros(n, bool)
    for nic in range(1, 5):
        receiver = nic_receivers[str(nic)]
        stamps = np.array([receiver.get(row[f'nic{nic}_array_saved'], np.nan) for row in matched])
        known = np.flatnonzero(np.isfinite(stamps))
        if not len(known):
            raise ValueError(f'NIC {nic} has no recorded reception timestamps.')
        observed |= np.isfinite(stamps)
        right = np.searchsorted(known, np.arange(n)).clip(0, len(known)-1)
        left = np.searchsorted(known, np.arange(n), side='right')-1
        left = left.clip(0, len(known)-1)
        a, b = stamps[known[left]], stamps[known[right]]
        lower.append(np.minimum(a, b)); upper.append(np.maximum(a, b))
    lower, upper = np.min(lower, axis=0), np.max(upper, axis=0)
    starts, ends = np.empty(n), np.empty(n)
    for i in range(n):
        if past_only:
            end = max(i+1, min(n, 2*half_window+1))
            start = max(0, end-(2*half_window+1))
        else:
            start, end = max(0, i-half_window), min(n, i+half_window+1)
        starts[i], ends[i] = np.min(lower[start:end]), np.max(upper[start:end])
    return starts, ends, observed


def selected_map_indices(starts, ends, observed, max_rate_hz=30.):
    # Prefix-greedy selection by actual completed support time, never index/fs.
    selected, previous = [], -float('inf')
    for i in np.argsort(ends, kind='stable'):
        if not observed[i] or ends[i]-starts[i] > 29/30:
            continue
        if ends[i]-previous >= 1/max_rate_hz:
            selected.append(int(i)); previous = ends[i]
    return np.array(selected, np.int64)


@lru_cache(maxsize=1)
def valder_steering(source, subcarriers, center_mhz, past_only):
    sys.path.insert(0, str(Path(source)/'rt_data_processing'))
    import heatmap_generator as hg
    if Path(hg.__file__).resolve() != Path(source)/'rt_data_processing/heatmap_generator.py':
        raise ValueError('Imported the wrong valder heatmap_generator module.')
    config = hg.make_config(subcarriers, center_mhz*1e6, past_only)
    setting = hg.heatmap_setup(config)
    steering, shape = hg.create_steering_matrix_AoA_ToF(setting)
    return hg, config, setting, steering, shape


def verify_generation_sample(dataset, session, settings):
    """Recompute ONE existing frame, streaming 25 CSI frames instead of loading GBs.

    This tests the inferred generation contract against the recorded numeric map;
    it does not replace or regenerate any recording or original heatmap file.
    """
    generation = settings['generation']
    hg, config, setting, steering, shape = valder_steering(
        str(dataset.parent/'valder'), generation['subcarriers'], generation['center_mhz'], generation['past_only'])
    folder = dataset/'intermediates'/session
    with zipfile.ZipFile(folder/'merged_csi'/f'{session}_merged.npz') as archive:
        with archive.open('csi.npy') as stream:
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
            dims, fortran, dtype = reader(stream)
            if fortran or dims[1:] != (2,8,generation['subcarriers']):
                raise ValueError(f'{session}: unexpected merged CSI layout.')
            i = min(100, dims[0]-13)
            half = generation['background_half_frames']
            lo, hi = max(0,i-half), min(dims[0],i+half+1)
            if generation['past_only']:
                hi = max(i+1,min(dims[0],2*half+1)); lo=max(0,hi-(2*half+1))
            frame_bytes = int(np.prod(dims[1:]))*dtype.itemsize
            stream.seek(lo*frame_bytes, 1)
            csi = np.frombuffer(stream.read((hi-lo)*frame_bytes), dtype=dtype).reshape(hi-lo,*dims[1:])
    power = np.abs(csi); np.square(power,out=power)
    residual = hg.background_subtraction(power.astype(np.float64), half, generation['past_only'])[[i-lo]]
    covariance = hg.calculate_correlation_matrix_wo_weight(hg.smoothed_CSI(setting,residual))
    covariance, _ = hg.weighted_correlation_matrix(config,0,covariance)
    actual = hg.run_music_algorithm(covariance,steering).reshape(-1,*shape)[0]
    with np.load(folder/'heatmaps'/f'{session}_heatmaps.npz',allow_pickle=False) as archive:
        expected = archive[str(i)]
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5,
                               err_msg=f'{session}: existing heatmap does not match the pinned valder settings')
    return {'frame':i,'status':'PASS','max_abs_db_error':float(np.max(abs(actual-expected))),
            'rtol':2e-5,'atol':2e-5,'scope':'One interior frame per session, not a proof of all historical generation settings'}


def profile(path):
    cfg = load_config(path)
    settings = cfg['recorded_heatmaps']
    if cfg['data'].get('session_prefix') != '20261031' or cfg['data'].get('payload_format') != 'sensor_full_v2':
        raise ValueError('This profile exclusively trains on 20261031 sensor.full v2 recordings.')
    cfg['scene']['approximation_note'] = 'Existing training proxy: 8^3 bounding-cube centers per object; not the server 48^3 visibility/occlusion renderer.'
    InputConfig(**settings['input_config']).validate()
    return cfg


def inspect_session(dataset, session, settings):
    folder = dataset/'intermediates'/session
    params = read_json(folder/'run_params.json')
    expected = settings['generation']
    for key in ('subcarriers', 'center_mhz', 'antenna_order'):
        if params.get(key) != expected[key]:
            raise ValueError(f'{session}: {key} disagrees with the training profile: {params.get(key)}')
    if bool(params.get('past_only', False)) != expected['past_only']:
        raise ValueError(f'{session}: inconsistent background-window mode.')
    matched_path = folder/'matched_csi'/f'{session}_matched.csv'
    matched = csv_rows(matched_path)
    receivers = {}
    for nic in range(1, 5):
        receiver = {}
        for path in sorted((dataset/'db'/session/f'csi.rx.{nic}').glob('*.csv')):
            for row in csv_rows(path):
                name = Path(row['array_saved']).name
                timestamp = float(row['recv_unix'])
                receiver[name] = min(timestamp, receiver.get(name, timestamp))
        receivers[str(nic)] = receiver
    starts, ends, observed = dependency_times(matched, receivers, expected['background_half_frames'], expected['past_only'])
    selected = selected_map_indices(starts, ends, observed, settings['map_rate_hz'])
    if not len(selected):
        raise ValueError(f'{session}: no maps with bounded, observed input support.')
    path = folder/'heatmaps'/f'{session}_heatmaps.npz'
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {f'{i}.npy' for i in range(len(matched))}:
            raise ValueError(f'{session}: map keys do not match matched_csi rows.')
        with archive.open('0.npy') as stream:
            shape = np.lib.format.read_array(stream, allow_pickle=False).shape
    if list(shape) != expected['source_shape']:
        raise ValueError(f'{session}: expected AoA-ToF shape {expected["source_shape"]}, got {shape}')
    return path, matched_path, starts, ends, selected, {
        'session': session, 'parameters': params, 'source_frames': len(matched), 'source_shape': list(shape),
        'observed_frames': int(observed.sum()), 'selected_maps': len(selected),
        'selected_support_span_p50_p99_s': np.percentile((ends-starts)[selected], [50,99]).tolist(),
        'map_start_s': float(ends[selected[0]]), 'map_end_s': float(ends[selected[-1]])}


def prepare_recorded(config_path, output):
    cfg = profile(config_path)
    out = Path(output).resolve()
    print('RESOURCES', check_resources(out, initial=True), flush=True)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory; previous runs are not overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(cfg['data']['dataset_root'])
    settings = cfg['recorded_heatmaps']
    source = dataset.parent/'valder'
    expected_hashes = settings.get('valder_source_sha256', {})
    if not expected_hashes:
        raise ValueError('Pin the inspected valder source files in this training profile.')
    for name, expected in expected_hashes.items():
        if file_hash(source/name) != expected:
            raise ValueError(f'Valder source changed: {name}; review settings before importing its maps.')
    pose_manifest = prepare_pose(cfg, out/'pose_cache')
    cache = Cache(out/'pose_cache')
    eligible = {stage: {split: len(OriginDataset(out/'pose_cache', split, mode))
                        for split in ('train', 'valid')}
                for stage, mode in [('confidence','confidence'), ('focused_pretrain','all'),
                                    ('focused_finetune','focused_clean'), ('explorer','explorer_clean')]}
    if any(not count for splits in eligible.values() for count in splits.values()):
        raise ValueError(f'Insufficient training/validation regimes: {eligible}')
    split_config = {session: split for split, sessions in pose_manifest['split'].items() for session in sessions}
    files, audits = {}, []
    maps_dir = out/'maps'; maps_dir.mkdir()
    input_cfg = InputConfig(**settings['input_config']).validate()
    theta = np.linspace(-90, 88, input_cfg.size)
    tau = np.linspace(0, 6.45/299792458., input_cfg.size)
    for session in sorted(split_config):
        check_resources(out)
        path, matched_path, starts, ends, selected, audit = inspect_session(dataset, session, settings)
        audit['generation_sample_check'] = verify_generation_sample(dataset, session, settings)
        maps = np.empty((len(selected), input_cfg.size, input_cfg.size), np.float32)
        with zipfile.ZipFile(path) as archive:
            for j, i in enumerate(selected):
                with archive.open(f'{i}.npy') as stream:
                    spectrum = np.lib.format.read_array(stream, allow_pickle=False)
                if not np.isfinite(spectrum).all() or list(spectrum.shape) != settings['generation']['source_shape']:
                    raise ValueError(f'{session}, frame {i}: invalid spectrum.')
                maps[j] = F.interpolate(torch.from_numpy(spectrum.astype(np.float32))[None,None],
                                        (input_cfg.size, input_cfg.size), mode='bilinear', align_corners=True)[0,0].numpy()
        metadata = {'format_version': 1, 'representation': 'aoa_tof', 'unit': 'db',
                    'clock_domain': 'unix_seconds', 'physical_axes_verified': False,
                    'availability_source': 'offline replay: maximum collector receive time of all interpolation and background inputs; zero computation delay lower bound',
                    'source_npz_sha256': file_hash(path), 'matched_csv_sha256': file_hash(matched_path),
                    'raw_sha256': file_hash(dataset/'intermediates'/session/'merged_csi'/f'{session}_merged.npz'),
                    'raw_source_kind': 'valder_merged_csi',
                    'session': session, 'generation_parameters': settings['generation'],
                    'historical_settings_evidence': 'run_params.json, pinned generator, and one recomputed numeric frame per session; no historical full generator snapshot exists',
                    'regridding': 'dB bilinear align_corners=True to training size; no heatmap regeneration',
                    'map_rate_hz_max': settings['map_rate_hz'], 'reception_clock_only': True}
        destination = maps_dir/(session+'.npz')
        np.savez_compressed(destination, maps=maps, valid=np.ones(len(maps), bool),
                            support_start_s=starts[selected], support_end_s=ends[selected],
                            available_at_s=ends[selected], theta_deg=theta, tau_s=tau,
                            source_map_index=selected, metadata_json=json.dumps(metadata))
        HeatmapSeries(destination, allow_exploratory=True)
        files[session] = destination
        audit.update(split=split_config[session], source_sha256=metadata['source_npz_sha256'],
                     converted_sha256=file_hash(destination))
        audits.append(audit)
        print('IMPORTED', session, len(maps), 'maps', check_resources(out), flush=True)
    rf = {'format_version': 1, 'hmd_cache_id': cache.manifest['cache_id'], 'hmd_clock_domain': 'unix_seconds',
          'synchronization_confirmed': True, 'allow_exploratory': True,
          'synchronization_evidence': 'Both HMD and all CSI CSV recv_unix are collector reception timestamps; radio rx_system_ns is deliberately not used. Capture-time synchronization is not claimed.',
          'records': [{'key': r['key'], 'file': str(files[r['session']]), 'group_id': r['session'],
                       'scenario': r['session'].split('_',1)[1], 'source_clock_domain': 'unix_seconds',
                       'clock_offset_s': 0., 'clock_uncertainty_s': .001} for r in cache.records]}
    write_json(out/'rf_manifest.json', rf)
    coverage = {s: {'available': 0, 'origins': 0} for s in ('train','valid','test')}
    series_cache = {}
    max_horizon = max(cfg['evaluation']['horizons_ms'])/1000
    for ri, record in enumerate(cache.records):
        series = series_cache.setdefault(record['session'], None)
        if series is None:
            series = HeatmapSeries(files[record['session']], uncertainty_s=.001, allow_exploratory=True)
            series_cache[record['session']] = series
        a = cache.arrays(ri)
        for anchor in a['anchors']:
            if a['t'][anchor]+max_horizon > a['t'][-1]:
                continue
            _, ids = series.indices(record['epoch_start_s']+a['t'][anchor], input_cfg)
            usable = ids >= 0
            counts = coverage[record['split']]; counts['origins'] += 1
            counts['available'] += int(usable[-1] and usable.mean() >= input_cfg.min_valid_fraction)
    for name, count in coverage.items():
        count['fraction'] = count['available']/max(count['origins'],1)
        if count['fraction'] < settings['minimum_rf_coverage']:
            raise ValueError(f'{name} RF coverage is insufficient: {count}')
    report = {'dataset_prefix': '20261031', 'sessions': audits, 'split': pose_manifest['split'],
              'excluded_sessions': settings.get('excluded_session_reasons', {}),
              'eligible_expert_samples': eligible, 'rf_coverage': coverage, 'resources': check_resources(out),
              'interpretation': 'Session holdout within this recording day; subject identity unknown, no unseen-subject claim. ROI uses the explicitly retained uniform-grid proxy.',
              'profile_sha256': file_hash(config_path), 'final_model': 'BAF + HMD/CSI AoA-ToF residual adapter'}
    write_json(out/'data_audit.json', report)
    dump_json(out/'resolved_config.json', cfg)
    print('PREPARED', json.dumps({'output': str(out), 'eligible_samples': eligible, 'rf_coverage': coverage}), flush=True)
    return cfg, report


def train_recorded(config_path, output, device):
    cfg, _ = prepare_recorded(config_path, output)
    out = Path(output).resolve()
    from baf_roi.train import train as train_experts
    from .prepare import prepare as prepare_adapter
    from .train import train as train_adapter
    from .evaluation import evaluate
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device = f'cuda:{device_index}'
        torch.cuda.set_device(device_index)
        torch.cuda.set_per_process_memory_fraction(.5, device_index)
        free, total = torch.cuda.mem_get_info(device_index)
        if free < 2*GIB:
            raise RuntimeError('Less than 2 GiB free GPU memory; use --device cpu or free the GPU.')
        print('GPU', torch.cuda.get_device_name(device_index), free/GIB, 'GiB free', flush=True)
    print('STAGE 1/4: train required pose experts on the NEW cache only', flush=True)
    baseline = train_experts(out/'pose_cache', out/'experts', device)
    print('STAGE 2/4: freeze paired predictions and CSI training targets', flush=True)
    prepare_adapter(out/'pose_cache', baseline, out/'rf_manifest.json', out/'prepared',
                    cfg['evaluation']['horizons_ms'], InputConfig(**cfg['recorded_heatmaps']['input_config']), device)
    print('STAGE 3/4: train HMD + CSI residual adapter (modality=rf)', flush=True)
    checkpoint = train_adapter(out/'prepared', out/'csi_residual', ModelConfig(method='residual', modality='rf'),
                               epochs=cfg['recorded_heatmaps']['epochs'], batch_size=cfg['training']['batch_size'],
                               seed=cfg['training']['seed'], device=device)
    print('STAGE 4/4: evaluate frozen selected checkpoint on held-out test sessions', flush=True)
    evaluate(out/'prepared', out/'test_csi', checkpoint=checkpoint, device=device)
    write_json(out/'complete.json', {'adapter': checkpoint, 'experts': str(baseline), 'device': device,
                                    'dataset_prefix': '20261031', 'model': 'HMD+CSI residual adapter'})
    print('TRAINING COMPLETE', checkpoint, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','train'])
    parser.add_argument('--config', default='configs/recorded_20261031.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    stop = threading.Event()
    def monitor():
        while not stop.wait(5):
            try:
                check_resources(args.output)
            except Exception as exc:
                print('STOPPING:', exc, flush=True)
                os._exit(75)
    guard = threading.Thread(target=monitor, daemon=True); guard.start()
    try:
        if args.command == 'prepare':
            prepare_recorded(args.config, args.output)
        else:
            train_recorded(args.config, args.output, args.device)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
