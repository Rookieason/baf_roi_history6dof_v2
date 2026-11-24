"""One method registry, one data protocol, and one comparison report.

This module orchestrates existing trainers. Neural models, losses, and inference
stay in baf_roi/, motion_transformer_roi/, baf_roi_heatmap/, and
baf_roi_heatmap_alpha/. Importing the CLI does not require loading PyTorch.
"""
import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import threading

from .methods import METHODS, resolve_methods


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/recorded_20261031.json'
CHECKPOINT_ARGUMENTS = {
    'baf_roi': 'baseline_checkpoint',
    'motion_transformer_roi': 'motion_transformer_checkpoint',
    'baf_roi_heatmap': 'heatmap_checkpoint',
    'baf_roi_heatmap_alpha': 'alpha_checkpoint',
}
RESOURCE_CHECK_SECONDS = 5


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', nargs='?', choices=['run', 'evaluate'], default='run',
                   help='run trains/reuses selected methods and evaluates them; evaluate reuses a completed run.')
    p.add_argument('--method', nargs='+', choices=['all', *METHODS],
                   help='One or more model packages; omitted/all runs all four. For evaluate, omitted means all methods in that run.')
    p.add_argument('--output', required=True, help='New or empty output directory; existing experiments are never overwritten.')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--config', help='Pose configuration, or the recorded_20261031 profile. Default: %(default)s')
    source.add_argument('--cache', help='Reuse an existing BAF pose cache without rebuilding its split.')
    source.add_argument('--prepared', help='Reuse one existing paired heatmap prepared experiment and its frozen BAF.')
    p.add_argument('--run', help='For evaluate: directory containing experiment.json from the unified runner.')
    p.add_argument('--rf-manifest', help='Reviewed RF mapping for the exact cache; required for generic heatmap runs.')
    p.add_argument('--input-config', help='Heatmap InputConfig JSON when preparing a new paired experiment.')
    p.add_argument('--scene-config', help='One shared scene JSON for every evaluated method.')
    p.add_argument('--horizons-ms', nargs='+', type=float, help='Shared prediction horizons; default from the cache/config.')
    p.add_argument('--baseline-checkpoint', help='Reuse BAF experts trained on the exact shared cache.')
    p.add_argument('--motion-transformer-checkpoint', help='Reuse Motion Transformer trained on the exact shared cache.')
    p.add_argument('--heatmap-checkpoint', help='Reuse an adapter trained on the exact prepared experiment.')
    p.add_argument('--alpha-checkpoint', help='Reuse an alpha adapter trained on the exact prepared experiment.')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, help='Default from the pose training config.')
    p.add_argument('--seed', type=int, help='Training seed; default from the shared cache/config.')
    p.add_argument('--mt-epochs', type=int, default=30)
    p.add_argument('--heatmap-epochs', type=int, help='Default from recorded_heatmaps.epochs, otherwise 20.')
    p.add_argument('--alpha-epochs', type=int, help='Default equal to heatmap-epochs.')
    p.add_argument('--smoke-epochs', type=int,
                   help='Explicit software-test override for ALL stages/models; outputs are marked as smoke tests.')
    p.add_argument('--lr', type=float, default=.001, help='Learning rate of the two heatmap adapters.')
    p.add_argument('--heatmap-objective', choices=['roi_iou', 'pose_roi'],
                   help='Default: pose_roi for internal motion experts; roi_iou for legacy adapters and alpha-only.')
    p.add_argument('--heatmap-aggregation', choices=['recording', 'origin'],
                   help='Default: recording for internal motion experts; origin for legacy adapters.')
    p.add_argument('--alpha-aggregation', choices=['recording', 'origin'], default='recording',
                   help='Alpha-only gate: default recording-macro weighting.')
    p.add_argument('--heatmap-adapter', choices=['motion_moe', 'residual', 'expert_gate'], default='motion_moe',
                   help='Default: CSI inside BAF motion experts and per-step routing. Other choices reproduce legacy adapters.')
    p.add_argument('--motion-experts', type=int, choices=[2, 3, 4], default=3)
    p.add_argument('--motion-conditioning', choices=['joint', 'router_only'], default='joint',
                   help='joint conditions expert latent features and trains pose heads; router_only freezes original experts.')
    p.add_argument('--heatmap-ablation', choices=['full', 'spatial_mean', 'last_frame', 'angle_only', 'delay_only', 'abs_change', 'metadata_only'], default='full')
    p.add_argument('--validation-coverage-drop', type=float, default=.005,
                   help='Motion model selection: maximum per-horizon validation micro-coverage drop from BAF (fraction).')
    p.add_argument('--validation-translation-increase-m', type=float, default=.005)
    p.add_argument('--validation-rotation-increase-deg', type=float, default=.5)
    p.add_argument('--modality', choices=['rf', 'hmd'], default='rf',
                   help='baf_roi_heatmap only; hmd ignores RF pixels, timing and availability for a matched architecture control.')
    p.add_argument('--hidden', type=int, default=32, help='baf_roi_heatmap context hidden width.')
    p.add_argument('--split', choices=['valid', 'test'], default='test')
    p.add_argument('--margins', type=float, nargs='+', default=[0., 5., 10., 20.],
                   help='Shared prediction-side angular margin sweep in degrees. Ground-truth FoV is fixed.')
    p.add_argument('--delivery-delay-ms', type=float, default=0.)
    p.add_argument('--dry-run', action='store_true', help='Print selected methods and dependencies without training or writing files.')
    return p


def _validate_options(a):
    for name in ('threads', 'batch_size', 'mt_epochs', 'heatmap_epochs', 'alpha_epochs', 'smoke_epochs'):
        value = getattr(a, name)
        if value is not None and value < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if a.hidden < 4 or not math.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('--hidden must be >= 4 and --lr must be finite and positive.')
    if a.heatmap_adapter == 'motion_moe' and a.motion_conditioning == 'router_only' and a.motion_experts != 2:
        raise ValueError('--motion-conditioning router_only requires --motion-experts 2.')
    for name in ('validation_coverage_drop', 'validation_translation_increase_m', 'validation_rotation_increase_deg'):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and nonnegative.')
    if not math.isfinite(a.delivery_delay_ms) or a.delivery_delay_ms < 0:
        raise ValueError('--delivery-delay-ms must be finite and nonnegative.')
    if not a.margins or len(set(a.margins)) != len(a.margins) or any(not math.isfinite(x) or x < 0 for x in a.margins):
        raise ValueError('--margins must contain distinct finite nonnegative angles.')
    if a.horizons_ms is not None:
        from baf_roi.config import validate_horizons
        validate_horizons(a.horizons_ms)
    if a.command == 'run' and a.run:
        raise ValueError('--run is only used with the evaluate command.')
    if a.command == 'evaluate':
        if not a.run:
            raise ValueError('evaluate requires --run with a unified experiment directory.')
        if any((a.config, a.cache, a.prepared, a.rf_manifest, a.input_config)):
            raise ValueError('evaluate obtains its cache and prepared data from --run.')
        if any(getattr(a, name) for name in CHECKPOINT_ARGUMENTS.values()):
            raise ValueError('evaluate uses the recorded checkpoints; use run to create a different comparison.')


def resolve_device(requested):
    import torch
    device = 'cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested)
    resolved = torch.device(device)
    if resolved.type == 'cuda':
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        torch.cuda.set_device(index)
        # Retain the recorded-data launcher's memory reserve on shared machines.
        torch.cuda.set_per_process_memory_fraction(.5, index)
        free, _ = torch.cuda.mem_get_info(index)
        if free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB free GPU memory; use --device cpu or free the GPU.')
        device = f'cuda:{index}'
    return device


def _monitor_resources(stop, output):
    """Retain the recorded launcher's process-wide emergency stop semantics."""
    from baf_roi_heatmap.recorded import check_resources
    while not stop.wait(RESOURCE_CHECK_SECONDS):
        try:
            check_resources(output)
        except Exception as exc:
            print('STOPPING:', exc, file=sys.stderr, flush=True)
            os._exit(75)
            return  # Also terminate cleanly when the exit function is replaced in tests.


@contextmanager
def resource_guard(config, output):
    """Monitor every stage of recorded-profile runs, including reused caches.

    The existing check_resources function owns the unchanged thresholds. Generic
    datasets do not opt into this machine-oriented recording profile's limits.
    """
    if 'recorded_heatmaps' not in config:
        yield
        return
    from baf_roi_heatmap.recorded import check_resources
    check_resources(output, initial=True)
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor_resources, args=(stop, output),
                               name='roi-recorded-resource-monitor', daemon=True)
    monitor.start()
    try:
        yield
    finally:
        stop.set()
        monitor.join(timeout=1.)


def _source_plan(a, methods):
    from baf_roi.config import load_config
    uses_heatmap = any(METHODS[name].uses_heatmap for name in methods)
    config = None
    for name in ('rf_manifest', 'input_config', 'scene_config'):
        value = getattr(a, name)
        if value and not Path(value).expanduser().is_file():
            raise FileNotFoundError(f'--{name.replace("_", "-")} file not found: {value}')
    if a.prepared:
        prepared = Path(a.prepared).expanduser().resolve()
        manifest = _read(prepared / 'manifest.json')
        config = _read(Path(manifest['pose_cache']) / 'manifest.json')['config']
        if a.rf_manifest or a.input_config or a.scene_config:
            raise ValueError('An existing --prepared experiment already fixes RF, input configuration, and scene.')
        if a.horizons_ms is not None and a.horizons_ms != manifest['horizons_ms']:
            raise ValueError('--horizons-ms differs from the existing prepared experiment; prepare a new one.')
        source = {'kind': 'prepared', 'path': str(prepared), 'cache': manifest['pose_cache'],
                  'horizons_ms': manifest['horizons_ms']}
    elif a.cache:
        cache = Path(a.cache).expanduser().resolve()
        config = _read(cache / 'manifest.json')['config']
        source = {'kind': 'cache', 'path': str(cache)}
    else:
        path = Path(a.config or DEFAULT_CONFIG).expanduser().resolve()
        config = load_config(path)
        source = {'kind': 'config', 'path': str(path)}
    recorded_import = source['kind'] == 'config' and 'recorded_heatmaps' in config and not a.rf_manifest and uses_heatmap
    if uses_heatmap and source['kind'] != 'prepared' and not a.rf_manifest and not recorded_import:
        raise ValueError('Heatmap methods require --rf-manifest for this exact cache, --prepared, or a recorded_heatmaps profile.')
    paired = uses_heatmap or source['kind'] == 'prepared' or bool(a.rf_manifest)
    if a.input_config and not paired:
        raise ValueError('--input-config configures heatmap input and requires a paired RF experiment.')
    if a.delivery_delay_ms and not paired:
        raise ValueError('--delivery-delay-ms requires RF data in a prepared experiment.')
    needs_baf = 'baf_roi' in methods or paired
    supplied = {}
    for method, name in CHECKPOINT_ARGUMENTS.items():
        value = getattr(a, name)
        if value:
            if method not in methods and not (method == 'baf_roi' and needs_baf):
                raise ValueError(f'Checkpoint supplied for an unselected method: {method}.')
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'Checkpoint not found: {path}')
            supplied[method] = str(path)
    horizons = source.get('horizons_ms', a.horizons_ms or config['evaluation']['horizons_ms'])
    plan = {'methods': methods, 'source': source, 'paired_evaluation': paired,
            'baf_dependency': needs_baf and 'baf_roi' not in methods,
            'recorded_heatmap_import': recorded_import, 'horizons_ms': horizons,
            'margins_deg': a.margins, 'reused_checkpoints': supplied,
            'default_config': str(DEFAULT_CONFIG), 'smoke_epochs': a.smoke_epochs,
            'heatmap_model': {'method': a.heatmap_adapter, 'experts': a.motion_experts,
                              'conditioning': a.motion_conditioning, 'modality': a.modality,
                              'objective': a.heatmap_objective or ('pose_roi' if a.heatmap_adapter == 'motion_moe' else 'roi_iou')},
            'evaluation': 'One shared ROI metric engine; common origins, targets, scene, horizons, and partition ordering.'}
    return plan, config


def _check_checkpoint(path, cache):
    import torch
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('cache_id') != cache.manifest['cache_id'] or bundle.get('normalization') != cache.stats:
        raise ValueError(f'Checkpoint must use the exact shared pose cache and normalization: {path}')


def _train_selected(a, out, plan, cfg, device):
    from baf_roi.data import Cache, prepare as prepare_pose
    from baf_roi_heatmap.contracts import InputConfig, ModelConfig, file_hash
    methods, source = plan['methods'], plan['source']
    source_kind = source['kind']
    prepared = Path(source['path']) if source_kind == 'prepared' else None
    rf_manifest = Path(a.rf_manifest).expanduser().resolve() if a.rf_manifest else None
    if source_kind == 'config':
        if 'recorded_heatmaps' in cfg:
            # Apply the same declared scene contract to standalone HMD runs.
            # profile() only validates configuration; it does not read RF files.
            from baf_roi_heatmap.recorded import profile
            cfg = profile(source['path'])
        # Materialize overrides outside data/ so recorded import still gets an empty directory.
        if a.seed is not None:
            cfg['training']['seed'] = a.seed
        if a.batch_size is not None:
            cfg['training']['batch_size'] = a.batch_size
        if a.horizons_ms is not None:
            cfg['evaluation']['horizons_ms'] = a.horizons_ms
        config_copy = out / 'resolved_input_config.json'
        _write(config_copy, cfg)
        if plan['recorded_heatmap_import']:
            from baf_roi_heatmap.recorded import prepare_recorded
            cfg, _ = prepare_recorded(config_copy, out / 'data')
            cache_path, rf_manifest = out / 'data/pose_cache', out / 'data/rf_manifest.json'
        else:
            cache_path = out / 'data/pose_cache'
            prepare_pose(cfg, cache_path)
    else:
        cache_path = Path(source['cache'] if source_kind == 'prepared' else source['path'])
    cache = Cache(cache_path)
    seed = int(cfg['training'].get('seed', 42) if a.seed is None else a.seed)
    batch = int(cfg['training'].get('batch_size', 64) if a.batch_size is None else a.batch_size)
    heat_epochs = a.heatmap_epochs or cfg.get('recorded_heatmaps', {}).get('epochs', 20)
    alpha_epochs = a.alpha_epochs or heat_epochs
    checkpoints = plan['reused_checkpoints'].copy()
    roles = {name: 'reused' for name in checkpoints}
    heatmap_config = None
    heatmap_objective = a.heatmap_objective or ('pose_roi' if a.heatmap_adapter == 'motion_moe' else 'roi_iou')
    heatmap_aggregation = a.heatmap_aggregation or ('recording' if a.heatmap_adapter == 'motion_moe' else 'origin')
    if 'baf_roi_heatmap' in checkpoints:
        import torch
        # Reusing a legacy checkpoint must not relabel its results with today's
        # motion_moe defaults in experiment.json. The evaluator verifies identity.
        reused = torch.load(checkpoints['baf_roi_heatmap'], map_location='cpu', weights_only=True)
        heatmap_config = asdict(ModelConfig(**reused['model_config']).validate())
        heatmap_objective = reused.get('objective', 'pose_roi')
        heatmap_aggregation = reused.get('aggregation', 'origin')
    if prepared:
        m = _read(prepared / 'manifest.json')
        baseline = str(Path(m['baseline_checkpoint']).resolve())
        if 'baf_roi' in checkpoints and file_hash(checkpoints['baf_roi']) != m['baseline_sha256']:
            raise ValueError('Supplied BAF differs from the frozen baseline in --prepared.')
        if file_hash(baseline) != m['baseline_sha256']:
            raise ValueError('The prepared baseline checkpoint changed after preparation.')
        checkpoints['baf_roi'] = baseline
        roles['baf_roi'] = 'reused_prepared_baseline'
    needs_baf = 'baf_roi' in methods or plan['paired_evaluation']
    if needs_baf and 'baf_roi' not in checkpoints:
        if seed != int(cache.manifest['config']['training'].get('seed', 42)):
            raise ValueError('BAF trains with its cache seed. For another seed, use a new --config/cache or reuse --baseline-checkpoint.')
        if batch != int(cache.manifest['config']['training'].get('batch_size', 64)):
            raise ValueError('BAF trains with its cache batch size. For another size, use a new --config/cache or reuse --baseline-checkpoint.')
        from baf_roi.train import train
        print('TRAIN baf_roi (also shared by both heatmap methods)', flush=True)
        checkpoints['baf_roi'] = str(train(cache_path, out / 'models/baf_roi', device, a.smoke_epochs))
        roles['baf_roi'] = 'trained'
    for method in ('baf_roi', 'motion_transformer_roi'):
        if method in checkpoints:
            _check_checkpoint(checkpoints[method], cache)
    if plan['paired_evaluation'] and not prepared:
        from baf_roi_heatmap.prepare import prepare
        from baf_roi_heatmap.contracts import read_scene_config
        if a.input_config:
            input_cfg = InputConfig(**_read(a.input_config))
        else:
            input_cfg = InputConfig(**cfg.get('recorded_heatmaps', {}).get('input_config', {}))
        prepared = out / 'prepared'
        print('PREPARE common origins and frozen BAF predictions', flush=True)
        prepare(cache_path, checkpoints['baf_roi'], rf_manifest, prepared, plan['horizons_ms'], input_cfg,
                device, batch, read_scene_config(a.scene_config) if a.scene_config else None)
    for method in methods:
        if method in checkpoints:
            continue
        print(f'TRAIN {method}', flush=True)
        if method == 'motion_transformer_roi':
            from motion_transformer_roi.train import train
            path = train(cache_path, out / ('models/' + method), device, a.smoke_epochs or a.mt_epochs, batch, seed)
        elif method == 'baf_roi_heatmap':
            from baf_roi_heatmap.train import train
            config = ModelConfig(method=a.heatmap_adapter, modality=a.modality, hidden=a.hidden,
                                 ablation=a.heatmap_ablation, experts=a.motion_experts, conditioning=a.motion_conditioning)
            heatmap_config = asdict(config)
            path = train(prepared, out / ('models/' + method), config, epochs=a.smoke_epochs or heat_epochs,
                         batch_size=batch, lr=a.lr, seed=seed, device=device,
                         objective=a.heatmap_objective, aggregation=a.heatmap_aggregation,
                         validation_limits={'coverage_drop': a.validation_coverage_drop,
                                            'translation_increase_m': a.validation_translation_increase_m,
                                            'rotation_increase_deg': a.validation_rotation_increase_deg})
        elif method == 'baf_roi_heatmap_alpha':
            from baf_roi_heatmap_alpha.train import train
            path = train(prepared, out / ('models/' + method), epochs=a.smoke_epochs or alpha_epochs,
                         batch_size=batch, lr=a.lr, seed=seed, device=device,
                         objective=a.heatmap_objective or 'roi_iou', aggregation=a.alpha_aggregation)
        else:
            raise ValueError(f'Missing BAF checkpoint after dependency preparation: {method}')
        checkpoints[method], roles[method] = str(Path(path).resolve()), 'trained'
    return {'format_version': 1, 'methods': methods, 'cache': str(Path(cache_path).resolve()),
            'cache_id': cache.manifest['cache_id'], 'prepared': str(prepared.resolve()) if prepared else None,
            'scene_config': str(Path(a.scene_config).resolve()) if a.scene_config else None,
            'horizons_ms': plan['horizons_ms'], 'checkpoints': checkpoints,
            'checkpoint_sha256': {name: file_hash(path) for name, path in checkpoints.items()},
            'checkpoint_roles': roles, 'seed': seed, 'batch_size': batch, 'device': device,
            'smoke_test': a.smoke_epochs is not None, 'smoke_epochs': a.smoke_epochs,
            'heatmap_adapter': heatmap_config['method'] if heatmap_config else a.heatmap_adapter,
            'heatmap_modality': heatmap_config['modality'] if heatmap_config else a.modality,
            'heatmap_model_config': heatmap_config,
            'heatmap_objective': heatmap_objective, 'heatmap_aggregation': heatmap_aggregation,
            'heatmap_motion_experts': heatmap_config['experts'] if heatmap_config else a.motion_experts,
            'heatmap_motion_conditioning': heatmap_config['conditioning'] if heatmap_config else a.motion_conditioning,
            'alpha_aggregation': a.alpha_aggregation,
            'complete': False, 'plan': plan}


def assert_matched_reports(reports):
    """A comparison must share identities, not merely the number of examples."""
    if not reports:
        raise ValueError('No methods were evaluated.')
    first_name, first = next(iter(reports.items()))
    keys = ('protocol_id', 'origin_keys_sha256', 'targets_sha256', 'rf_cohort_sha256',
            'delivery_delay_s', 'scene_id', 'split', 'horizons_ms')
    for name, report in reports.items():
        for key in keys:
            if key not in report:
                raise ValueError(f'{name} report is missing common evaluation identity {key}.')
            if report[key] != first[key]:
                raise ValueError(f'Cannot compare {first_name} with {name}: {key} differs.')


def write_summary(reports, output):
    assert_matched_reports(reports)
    rows = []
    reference = reports.get('baf_roi', {}).get('summaries', {})
    for name, report in reports.items():
        for key, values in report['summaries'].items():
            pooled = values['all']['pooled']
            horizon, margin = key.split('ms_margin')
            baseline = reference.get(key, {}).get('all', {}).get('pooled')
            row = {'method': name, 'horizon_ms': float(horizon), 'margin_deg': float(margin),
                   'selected_epoch': (report.get('checkpoint_selection') or {}).get('selected_epoch'),
                   'selection_status': (report.get('checkpoint_selection') or {}).get('status'),
                   'correction_applied_fraction': report.get('correction_applied_fraction'),
                   **{field: pooled.get(field) for field in (
                       'n_origins', 'iou_mean', 'f1_micro', 'coverage_micro', 'precision_micro',
                       'selected_mean', 'required_mean', 'fp_mean', 'fn_mean', 'translation_mean_m', 'rotation_mean_deg')},
                   'delta_iou_mean_vs_baf': pooled['iou_mean'] - baseline['iou_mean'] if baseline else None}
            rows.append(row)
    out = Path(output)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    first = next(iter(reports.values()))
    summary = {'methods': list(reports), 'protocol_id': first['protocol_id'],
               'origin_keys_sha256': first['origin_keys_sha256'], 'targets_sha256': first['targets_sha256'],
               'rf_cohort_sha256': first['rf_cohort_sha256'], 'delivery_delay_s': first['delivery_delay_s'],
               'scene_id': first['scene_id'], 'split': first['split'], 'horizons_ms': first['horizons_ms'],
               'primary_metrics': 'Pooled per-origin Mean IoU and pooled Micro F1; margin 0 is the unexpanded prediction.',
               'quality': {'ssim': None, 'status': 'not_evaluated',
                           'reason': 'No matched rendered frames or renderer are supplied. Coverage is a geometric proxy.'},
               'rows': rows}
    _write(out / 'summary.json', summary)
    return summary


def _evaluate_selected(a, experiment, methods, output, device):
    from roi_experiments.evaluation import evaluate, evaluate_pose
    from baf_roi_heatmap.contracts import read_scene_config, file_hash
    checkpoints = experiment['checkpoints']
    for name in methods:
        if name not in checkpoints:
            raise ValueError(f'This run has no {name} checkpoint; use run to train that method.')
        if file_hash(checkpoints[name]) != experiment['checkpoint_sha256'][name]:
            raise ValueError(f'Checkpoint was changed after this run: {name}.')
    horizons = a.horizons_ms or experiment['horizons_ms']
    scene_path = a.scene_config or experiment.get('scene_config')
    if experiment['prepared'] and (horizons != experiment['horizons_ms'] or (a.command == 'evaluate' and a.scene_config)):
        raise ValueError('Prepared runs fix their scene and horizons. Prepare a new experiment to change them.')
    batch_size = a.batch_size or experiment['batch_size']
    reports = {}
    for name in methods:
        print(f'EVALUATE {name} on the common {a.split} origins', flush=True)
        destination = Path(output) / 'evaluations' / a.split / name
        if experiment['prepared']:
            options = {}
            if name == 'baf_roi': options['baseline'] = 'baf'
            elif name == 'motion_transformer_roi': options['motion_transformer_checkpoint'] = checkpoints[name]
            elif name == 'baf_roi_heatmap': options['checkpoint'] = checkpoints[name]
            elif name == 'baf_roi_heatmap_alpha': options['alpha_checkpoint'] = checkpoints[name]
            report = evaluate(experiment['prepared'], destination, split=a.split, device=device,
                              batch_size=batch_size, margins=a.margins,
                              delivery_delay_s=a.delivery_delay_ms/1000, **options)
        else:
            if a.delivery_delay_ms:
                raise ValueError('--delivery-delay-ms requires RF data in a prepared experiment.')
            report = evaluate_pose(experiment['cache'], checkpoints[name], destination, method=name,
                                   split=a.split, device=device, batch_size=batch_size,
                                   horizons_ms=horizons, scene_config=read_scene_config(scene_path) if scene_path else None,
                                   margins=a.margins)
        reports[name] = report
    return write_summary(reports, output)


def run(a):
    _validate_options(a)
    if a.command == 'evaluate':
        experiment = _read(Path(a.run).expanduser().resolve() / 'experiment.json')
        if experiment.get('format_version') != 1:
            raise ValueError('Unsupported unified experiment manifest.')
        methods = resolve_methods(a.method, default=experiment['methods'])
        missing = set(methods) - experiment['checkpoints'].keys()
        if missing:
            raise ValueError(f'This run has no checkpoints for {sorted(missing)}.')
        if a.dry_run:
            return {'command': 'evaluate', 'methods': methods, 'run': str(Path(a.run).resolve()),
                    'horizons_ms': a.horizons_ms or experiment['horizons_ms'], 'margins_deg': a.margins}
        config = _read(Path(experiment['cache']) / 'manifest.json')['config']
        with resource_guard(config, a.output):
            device = resolve_device(a.device)
            import torch
            from baf_roi_heatmap.contracts import new_directory
            torch.set_num_threads(a.threads)
            out = new_directory(a.output)
            summary = _evaluate_selected(a, experiment, methods, out, device)
            _write(out / 'evaluation_run.json', {'source_run': str(Path(a.run).resolve()),
                                                'methods': methods, 'device': device, 'split': a.split})
            return {'output': str(out), 'summary': str(out / 'summary.csv'), 'methods': summary['methods']}
    methods = resolve_methods(a.method)
    plan, cfg = _source_plan(a, methods)
    if a.dry_run:
        return plan
    with resource_guard(cfg, a.output):
        device = resolve_device(a.device)
        import torch
        from baf_roi_heatmap.contracts import new_directory
        torch.set_num_threads(a.threads)
        out = new_directory(a.output)
        _write(out / 'plan.json', plan)
        experiment = _train_selected(a, out, plan, cfg, device)
        _write(out / 'experiment.json', experiment)
        _evaluate_selected(a, experiment, methods, out, device)
        experiment['complete'] = True
        _write(out / 'experiment.json', experiment)
        return {'output': str(out), 'summary': str(out / 'summary.csv'), 'methods': methods,
                'smoke_test': experiment['smoke_test']}


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        print(json.dumps(run(a), indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
