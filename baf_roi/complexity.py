"""Reproducible inference analysis of the existing, unmodified BAF runtime.

Dense MAC accounting is deliberately separate from end-to-end wall time. It is
not an assertion that NumPy/SciPy, nonlinearities, or transfers cost zero FLOPs.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .config import dump_json, validate_horizons
from .data import Series, sha256_file
from .models import build_models
from .pose import model_inputs
from .runtime import Predictor


MAC_SCOPE = ('Conv1d, Linear and LSTM matrix products only; one MAC = two FLOPs. '
             'Excludes bias, normalization, pooling, activations, recurrent elementwise gates, '
             'quaternion math, NumPy/SciPy, fusion, interpolation, ROI and transfers. '
             'These costs ARE included in end-to-end latency. Counts describe dense logical '
             'operations, not hardware instructions; convolution includes padded outputs before TCN chomp.')


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def summarize_times(samples_ms, batch_size, deadline_ms):
    x = np.asarray(samples_ms, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError('Need positive finite timing samples.')
    return {'iterations': len(x), 'mean_ms': float(x.mean()), 'std_ms': float(x.std()),
            'min_ms': float(x.min()), 'median_ms': float(np.median(x)),
            'p90_ms': float(np.percentile(x, 90)), 'p95_ms': float(np.percentile(x, 95)),
            'p99_ms': float(np.percentile(x, 99)), 'max_ms': float(x.max()),
            'histories_per_second': float(batch_size * 1000 / x.mean()),
            'amortized_ms_per_history': float(x.mean() / batch_size),
            'deadline_ms_per_call': deadline_ms,
            'deadline_miss_fraction': float(np.mean(x > deadline_ms)),
            'samples_ms': x.tolist()}


def benchmark(fn, device, batch_size, warmup, iterations, deadline_ms):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)
        samples = []
        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter_ns()
            fn()
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return summarize_times(samples, batch_size, deadline_ms)


def dense_macs(module, inputs, output):
    """Count actual output lengths (including causal convolution padding)."""
    if isinstance(module, nn.Conv1d):
        return output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
    if isinstance(module, nn.Linear):
        return output.numel() * module.in_features
    if isinstance(module, nn.LSTM):
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError('LSTM accounting requires an unpacked, batched sequence.')
        b, t = (x.shape[0], x.shape[1]) if module.batch_first else (x.shape[1], x.shape[0])
        directions = 2 if module.bidirectional else 1
        hidden = module.hidden_size
        recurrent = module.proj_size or hidden
        total = 0
        for layer in range(module.num_layers):
            width = module.input_size if layer == 0 else directions * recurrent
            total += b * t * directions * 4 * hidden * (width + recurrent)
            if module.proj_size:
                total += b * t * directions * hidden * module.proj_size
        return total
    raise TypeError(type(module).__name__)


def count_forecast(predictor, histories, horizons):
    rows, handles = {}, []
    uncounted = []
    for branch, model in predictor.models.items():
        for name, module in model.named_modules():
            key = f'{branch}.{name}'
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.LSTM)):
                rows[key] = {'module': key, 'type': type(module).__name__, 'calls': 0, 'macs': 0}

                def hook(m, args, out, key=key):
                    rows[key]['calls'] += 1
                    rows[key]['macs'] += int(dense_macs(m, args, out))

                handles.append(module.register_forward_hook(hook))
            elif list(module.parameters(recurse=False)):
                uncounted.append({'module': key, 'type': type(module).__name__})
    try:
        with torch.inference_mode():
            prediction = predictor.forecast_batch(histories, horizons)
    finally:
        for handle in handles:
            handle.remove()
    branch_counts = {name: sum(r['macs'] for r in rows.values() if r['module'].startswith(name + '.'))
                     for name in predictor.models}
    total = sum(branch_counts.values())
    return {'scope': MAC_SCOPE, 'dense_macs_per_call': total,
            'dense_flops_per_call': 2 * total, 'dense_macs_per_history': total // len(histories),
            'branch_macs_per_call': branch_counts, 'layers': list(rows.values()),
            'parameterized_modules_outside_mac_scope': uncounted}, prediction


def model_inventory(models):
    result = {}
    for name, model in models.items():
        params, buffers = list(model.parameters()), list(model.buffers())
        result[name] = {'parameters': sum(p.numel() for p in params),
                        'trainable_parameters': sum(p.numel() for p in params if p.requires_grad),
                        'parameter_bytes': sum(p.numel() * p.element_size() for p in params),
                        'buffer_bytes': sum(p.numel() * p.element_size() for p in buffers),
                        'parameter_dtypes': sorted({str(p.dtype) for p in params}),
                        'config': vars(model.config), 'architecture': str(model)}
    total = {key: sum(v[key] for v in result.values()) for key in
             ('parameters', 'trainable_parameters', 'parameter_bytes', 'buffer_bytes')}
    return {'branches': result, 'total': total}


def process_rss():
    """Current native process resident memory, where a standard OS API is available."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    elif sys.platform.startswith('linux'):
        try:
            return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            pass
    return None


def memory_probe(fn, device):
    """A separate pass: tracing overhead must never enter reported latency."""
    synchronize(device)
    cuda = torch.device(device).type == 'cuda'
    result = {'cpu_scope': 'Python-traced allocations during one warm call; NOT total CPU/native RSS. '
                          'Includes only allocations visible to tracemalloc; SciPy/PyTorch native memory '
                          'and allocator workspaces may be absent.',
              'cpu_native_peak_bytes': None,
              'rss_scope': 'Whole-process resident memory sampled every ~1 ms in a separate warm pass. '
                           'Includes interpreter, model, native allocators and prior caches. '
                           'Sampled peak is a lower bound; increments are not activation memory.'}
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        result['cuda_baseline_allocated_bytes'] = torch.cuda.memory_allocated(device)
        result['cuda_baseline_reserved_bytes'] = torch.cuda.memory_reserved(device)
    baseline = process_rss()
    rss = [] if baseline is None else [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(.001):
            value = process_rss()
            if value is not None:
                rss.append(value)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    tracemalloc.start()
    try:
        out = fn()
        synchronize(device)
        current, peak = tracemalloc.get_traced_memory()
        result.update(python_traced_retained_bytes=current, python_traced_peak_bytes=peak)
        if cuda:
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            result['cuda_incremental_peak_allocated_bytes'] = (
                result['cuda_peak_allocated_bytes'] - result['cuda_baseline_allocated_bytes'])
        # NumPy outputs only; aliases are counted once by object identity.
        arrays = {id(v): v for v in out.values() if isinstance(v, np.ndarray)}
        result['returned_numpy_array_bytes'] = sum(v.nbytes for v in arrays.values())
    finally:
        tracemalloc.stop()
        value = process_rss()
        if value is not None:
            rss.append(value)
        stop.set()
        sampler.join()
    result.update(rss_baseline_bytes=baseline, rss_sampled_peak_bytes=max(rss) if rss else None,
                  rss_sample_count=len(rss),
                  rss_sampled_increment_bytes=max(rss) - baseline if baseline is not None and rss else None)
    return result


def stage_probe(predictor, histories, horizons):
    """Non-overlapping diagnostic ranges; synchronization changes scheduling."""
    import baf_roi.runtime as runtime
    rows = {}

    def instrument(original, label):
        def wrapped(*args, **kwargs):
            synchronize(predictor.device)
            start = time.perf_counter_ns()
            out = original(*args, **kwargs)
            synchronize(predictor.device)
            row = rows.setdefault(label, {'calls': 0, 'total_ms': 0.})
            row['calls'] += 1
            row['total_ms'] += (time.perf_counter_ns() - start) / 1e6
            return out
        return wrapped

    with ExitStack() as stack, torch.inference_mode():
        for name in ('model_inputs', 'apply_decision_fusion', 'interpolate_predictions',
                     'quat_to_ypr', 'ypr_to_quat'):
            stack.enter_context(patch.object(runtime, name, instrument(getattr(runtime, name), name)))
        for name, model in predictor.models.items():
            stack.enter_context(patch.object(model, 'forward', instrument(model.forward, 'branch/' + name)))
        synchronize(predictor.device)
        start = time.perf_counter_ns()
        predictor.forecast_batch(histories, horizons)
        synchronize(predictor.device)
        total = (time.perf_counter_ns() - start) / 1e6
    return {'scope': 'Separate single diagnostic call with synchronized stage boundaries; '
                     'not interchangeable with uninstrumented latency. Remainder includes tensor '
                     'creation/transfers, denormalization, validation, EMA, history updates and instrumentation.',
            'total_ms': total, 'stages': rows,
            'remainder_ms': total - sum(row['total_ms'] for row in rows.values())}


def profile_call(predictor, fn, output, device):
    """Separate diagnostic trace; CPU NumPy stages receive explicit ranges."""
    import baf_roi.runtime as runtime
    import baf_roi.pose as pose
    from .roi import Scene
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def instrument(original, label):
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        return wrapped

    with ExitStack() as stack:
        for name in ('model_inputs', 'apply_decision_fusion', 'interpolate_predictions', 'quat_to_ypr'):
            stack.enter_context(patch.object(runtime, name, instrument(getattr(runtime, name), name)))
        # Module hooks name the branches without rewriting vendored source files.
        for name, model in predictor.models.items():
            stack.enter_context(patch.object(model, 'forward', instrument(model.forward, 'branch/' + name)))
        stack.enter_context(patch.object(Series, 'history', instrument(Series.history, 'observation_resampling')))
        stack.enter_context(patch.object(Scene, 'mask', instrument(Scene.mask, 'pose_to_roi')))
        stack.enter_context(patch.object(pose, 'ypr_to_quat', instrument(pose.ypr_to_quat, 'ypr_to_quat')))
        with torch.inference_mode(), torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.profiler.record_function('history_to_prediction'):
                fn()
                synchronize(device)
    prof.export_chrome_trace(str(output / 'trace.json'))
    (output / 'operators.txt').write_text(
        prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=100), encoding='utf-8')
    if torch.device(device).type == 'cuda':
        (output / 'operators_cuda.txt').write_text(
            prof.key_averages().table(sort_by='self_device_time_total', row_limit=100), encoding='utf-8')
    return 'trace.json'


def synthetic_history(batch_size):
    t = np.arange(30) / 30
    h = np.zeros((batch_size, 30, 6), dtype=np.float64)
    h[:, :, 0] = .1 * t
    h[:, :, 2] = 1.6
    h[:, :, 3] = 15 * t
    h[:, :, 4] = 2 * np.sin(t)
    return h


def prepare_input(predictor, history_path, batch_size):
    if history_path:
        raw = json.loads(Path(history_path).read_text(encoding='utf-8'))
        ts, pos, rot = (np.asarray(raw[k], np.float64) for k in
                        ('timestamps_s', 'positions', 'rotations_deg'))
        scale = .01 if predictor.data_config['position_unit'] == 'cm' else 1.
        history = Series(ts - ts[0], np.concatenate((pos * scale, rot), 1),
                         predictor.data_config['max_gap_ms'] / 1000).history(len(ts) - 1)
        return np.repeat(history[None], batch_size, axis=0), (ts, pos, rot), {
            'kind': 'provided_observations_repeated_across_batch',
            'path': str(Path(history_path).resolve()), 'sha256': sha256_file(history_path),
            'observed_samples': len(ts)}
    h = synthetic_history(batch_size)
    scale = 100 if predictor.data_config['position_unit'] == 'cm' else 1
    return h, (np.arange(30) / 30, h[0, :, :3] * scale, h[0, :, 3:]), {
        'kind': 'synthetic_history_repeated_across_batch', 'observed_samples': 30}


def environment(device):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'platform': platform.platform(), 'cpu': platform.processor(), 'torch': torch.__version__,
              'numpy': np.__version__, 'device': str(device), 'threads': torch.get_num_threads(),
              'interop_threads': torch.get_num_interop_threads(), 'cuda_runtime': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'cudnn_benchmark': torch.backends.cudnn.benchmark,
              'cudnn_deterministic': torch.backends.cudnn.deterministic,
              'matmul_precision': torch.get_float32_matmul_precision(),
              'git_revision': revision, 'git_dirty': dirty,
              'runtime_sha256': {name: sha256_file(Path(__file__).parent / name) for name in
                                 ('runtime.py', 'models.py', 'pose.py', 'upstream_ops.py', 'roi.py', 'complexity.py')}}
    vendor = Path(__file__).resolve().parents[1] / 'vendor/baf_pose'
    result['model_source_sha256'] = {name: sha256_file(vendor / name) for name in
        ('confidence/model.py', 'prediction/focused_branch/model.py', 'prediction/explorer_branch/model.py')}
    if torch.device(device).type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result['gpu'] = {'name': props.name, 'total_memory_bytes': props.total_memory,
                         'compute_capability': [props.major, props.minor]}
        result['cuda_matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
        result['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32
    return result


def write_summary(output, report):
    inventory = report['model']['total']
    lines = ['# BAF history 6DoF inference complexity', '',
             f"Weights: {report['weights']['kind']}", '',
             f"Parameters: {inventory['parameters']:,}; parameter bytes: {inventory['parameter_bytes']:,}.", '',
             'MAC scope: ' + MAC_SCOPE, '',
             'All horizons in one row share a single rollout. Parameters are counted once.', '',
             '| Batch | Horizons (ms) | Steps | Dense GMAC/call | Forecast p50/p95 (ms) | Histories/s |',
             '|---:|---|---:|---:|---:|---:|']
    for case in report['cases']:
        lat = case['latency']['forecast_batch']
        lines.append(f"| {case['batch_size']} | {case['horizons_ms']} | {case['rollout_steps']} | "
                     f"{case['operations']['dense_macs_per_call'] / 1e9:.6f} | "
                     f"{lat['median_ms']:.3f} / {lat['p95_ms']:.3f} | {lat['histories_per_second']:.2f} |")
    lines += ['', 'See report.json for online adapter / ROI timing, raw samples, per-layer counts, '
              'memory scope and environment. CPU native RSS is sampled; an exact native peak is not measured.', '',
              'Batched amortized latency is a throughput metric, not single-request response latency. '
              'Tail percentiles from small samples are descriptive, not statistical guarantees.', '']
    lines += ['## Model branches', '', '| Branch | Parameters | Parameter bytes | Buffer bytes |',
              '|---|---:|---:|---:|']
    for name, branch in report['model']['branches'].items():
        lines.append(f"| {name} | {branch['parameters']:,} | {branch['parameter_bytes']:,} | {branch['buffer_bytes']:,} |")
    lines += ['', '## All measured endpoints', '',
              'Isolated branches use device-resident inputs; other endpoints include the existing CPU pipeline.', '',
              '| Case | Endpoint | p50 ms | p95 ms | p99 ms | Histories/s | Deadline miss fraction |',
              '|---:|---|---:|---:|---:|---:|---:|']
    csv_rows = []
    for index, case in enumerate(report['cases']):
        for endpoint, stats in case['latency'].items():
            lines.append(f"| {index} | {endpoint} | {stats['median_ms']:.3f} | {stats['p95_ms']:.3f} | "
                         f"{stats['p99_ms']:.3f} | {stats['histories_per_second']:.2f} | {stats['deadline_miss_fraction']:.4f} |")
            csv_rows.append({'case': index, 'batch_size': case['batch_size'],
                             'horizons_ms': json.dumps(case['horizons_ms']), 'rollout_steps': case['rollout_steps'],
                             'dense_macs_full_forecast': case['operations']['dense_macs_per_call'],
                             'endpoint': endpoint,
                             **{k: v for k, v in stats.items() if k != 'samples_ms'}})
    with (output / 'measurements.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines += ['', '## Memory and stage diagnostics', '',
              'RSS is sampled whole-process resident memory; CUDA values are allocator statistics. '
              'Stage timings are an additional instrumented call.', '']
    for index, case in enumerate(report['cases']):
        mem = case['memory_forecast']
        lines += [f'### Case {index}', '',
                  f"Forecast output NumPy bytes: {mem['returned_numpy_array_bytes']:,}; "
                  f"Python-traced peak bytes: {mem['python_traced_peak_bytes']:,}; "
                  f"process sampled RSS peak bytes: {mem['rss_sampled_peak_bytes']}.", '']
        if 'cuda_peak_allocated_bytes' in mem:
            lines += [f"CUDA peak allocated / reserved bytes: {mem['cuda_peak_allocated_bytes']} / "
                      f"{mem['cuda_peak_reserved_bytes']}; allocated increment bytes: "
                      f"{mem['cuda_incremental_peak_allocated_bytes']}.", '']
        lines += ['| Diagnostic stage | Calls | Total ms |', '|---|---:|---:|']
        for name, stage in case['stage_diagnostic']['stages'].items():
            lines.append(f"| {name} | {stage['calls']} | {stage['total_ms']:.3f} |")
        lines += [f"| remainder | 1 | {case['stage_diagnostic']['remainder_ms']:.3f} |", '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze(args):
    if args.threads < 1 or args.iterations < 1 or args.warmup < 0 or any(b < 1 for b in args.batch_sizes):
        raise ValueError('Positive threads, iterations and batch sizes; nonnegative warmup required.')
    if not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        raise ValueError('deadline-ms must be finite and positive.')
    groups = [validate_horizons(g) for g in args.horizon_sets]
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu, cuda[:index], auto.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory to preserve earlier measurements.')
    output.mkdir(parents=True, exist_ok=True)
    synchronize(device)
    start = time.perf_counter()
    if args.checkpoint:
        predictor = Predictor.load(args.checkpoint, device)
        weights = {'kind': 'trained_checkpoint', 'path': str(Path(args.checkpoint).resolve()),
                   'sha256': sha256_file(args.checkpoint), 'file_bytes': Path(args.checkpoint).stat().st_size}
    else:
        predictor = Predictor(build_models(), {'mean': [0, 0, 0], 'std': [1, 1, 1]}, device=device)
        weights = {'kind': 'UNTRAINED_ARCHITECTURE_ONLY', 'seed': args.seed}
    synchronize(device)
    setup_ms = (time.perf_counter() - start) * 1000
    scene = None
    if args.scene_config:
        from .__main__ import read_scene_config
        from .roi import load_scene
        scene = load_scene(read_scene_config(args.scene_config))
    report = {'schema_version': 1, 'method': 'baf_roi', 'environment': environment(device),
              'weights': weights, 'model_setup_ms': setup_ms, 'model': model_inventory(predictor.models),
              'protocol': {'warmup': args.warmup, 'iterations': args.iterations, 'seed': args.seed,
                           'timing': 'perf_counter_ns wall time; CUDA synchronized at call boundaries; '
                                     'eval + inference_mode; no autocast/compile; setup and file I/O excluded',
                           'state': 'Independent decisions; no previous observed EMA state (stream_id=None).',
                           'memory': 'Separate warm pass; CUDA allocator totals include persistent model memory.',
                           'batching': 'Identical history replicated; forecast_batch is batched; observation API is B=1.'},
              'scene': scene.snapshot() if scene else None, 'cases': []}
    for batch in args.batch_sizes:
        histories, observations, metadata = prepare_input(predictor, args.history, batch)
        for horizons in groups:
            print(f'Analyzing batch={batch}, horizons={horizons}', flush=True)
            forecast = lambda: predictor.forecast_batch(histories, horizons)
            counts, prediction = count_forecast(predictor, histories, horizons)
            steps = math.ceil(max(horizons) * 30 / 1000 - 1e-10)
            latency = {}

            def measure(fn, b=batch):
                return benchmark(fn, device, b, args.warmup, args.iterations, args.deadline_ms)

            # First-call latency is intentionally not claimed: counting already warmed the model.
            latency['forecast_batch'] = measure(forecast)
            conf, pose = model_inputs(histories, predictor.stats)
            cx, px = torch.as_tensor(conf, device=device), torch.as_tensor(pose, device=device)
            for name, model in predictor.models.items():
                x = cx if name == 'confidence' else px
                latency['isolated_device_resident_' + name] = measure(lambda m=model, x=x: m(x))
            case = {'batch_size': batch, 'history_shape': list(histories.shape),
                    'horizons_ms': horizons, 'rollout_steps': steps, 'input': metadata,
                    'operations': counts, 'latency': latency,
                    'stage_diagnostic': stage_probe(predictor, histories, horizons),
                    'memory_forecast': memory_probe(forecast, device)}
            if batch == 1:
                online = lambda: predictor.predict_observations(*observations, horizons_ms=horizons)
                latency['observations_to_pose'] = measure(online)
            if scene:
                roi = lambda: scene.mask(prediction['positions_m'], prediction['quaternions_wxyz'])
                latency['roi_only'] = measure(roi)

                def complete():
                    out = forecast()
                    return dict(out, roi_mask=scene.mask(out['positions_m'], out['quaternions_wxyz']))

                latency['history_to_roi'] = measure(complete)
                case['memory_history_to_roi'] = memory_probe(complete, device)
                if batch == 1:
                    def online_roi():
                        out = online()
                        return dict(out, roi_mask=scene.mask(out['positions_m'], out['quaternions_wxyz']))
                    latency['observations_to_roi'] = measure(online_roi)
                case['roi_output_shape'] = [batch, len(horizons), len(scene.centers_m)]
            if args.profile:
                trace_dir = output / f'profile_{len(report["cases"]):03d}'
                trace_dir.mkdir()
                profile_fn = (online_roi if batch == 1 else complete) if scene else (online if batch == 1 else forecast)
                profile_call(predictor, profile_fn, trace_dir, device)
                case['profile_directory'] = trace_dir.name
            report['cases'].append(case)
            dump_json(output / 'report.json', report)
    write_summary(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint', help='Trusted v2 BAF checkpoint; no cache or retraining required.')
    weights.add_argument('--architecture-only', action='store_true', help='Explicit UNTRAINED smoke/structure mode.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1])
    parser.add_argument('--horizons-ms', dest='horizon_sets', nargs='+', type=float, action='append',
                        help='Repeat for a sweep. Each occurrence is ONE shared-rollout call.')
    parser.add_argument('--history', help='Same JSON observation schema as baf_roi predict; optional.')
    parser.add_argument('--scene-config', help='Optional confirmed scene, using the existing predict schema.')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--deadline-ms', type=float, default=1000 / 30)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--profile', action='store_true', help='Separate operator/Chrome traces, one per case.')
    args = parser.parse_args()
    if args.horizon_sets is None:
        args.horizon_sets = [[50, 100, 150, 200, 250]]
    try:
        analyze(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Analysis saved to {args.output}')


if __name__ == '__main__':
    main()
