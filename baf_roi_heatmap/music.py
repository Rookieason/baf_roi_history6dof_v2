"""Numeric AoA-ToF MUSIC adapted from the supplied 2026-10-27 MUSIC.py.

Preserves its Kronecker row order and spatial/frequency smoothing, but uses
trailing raw windows, explicit frequencies, bounded snapshot allocation and
no magnitude-only 'self_sanitize'. This module does NOT calibrate hardware.
"""
from pathlib import Path
import json
import time
import numpy as np
from .contracts import read_json, write_json, file_hash


def validate_csi(csi, timestamps_s, available_at_s, frequencies_hz):
    csi = np.asarray(csi)
    ts, ready, freq = (np.asarray(x, np.float64) for x in (timestamps_s, available_at_s, frequencies_hz))
    if csi.ndim != 4 or not np.iscomplexobj(csi) or not np.isfinite(csi).all():
        raise ValueError('CSI must be finite complex [packets, Tx, Rx, subcarriers].')
    if ts.shape != (len(csi),) or ready.shape != ts.shape or len(ts) < 2:
        raise ValueError('Need a timestamp and availability time per CSI packet.')
    if not np.isfinite(ts).all() or not np.isfinite(ready).all() or np.any(np.diff(ts) <= 0) or np.any(ready < ts):
        raise ValueError('Invalid raw timestamps/availability; use a common clock, never guessed fs.')
    if freq.shape != (csi.shape[-1],) or len(freq) < 2 or not np.isfinite(freq).all() or np.any(freq <= 0):
        raise ValueError('Supply actual positive subcarrier frequencies_hz.')
    spacing = np.diff(freq)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-5, atol=0.01):
        raise ValueError('This sliding-frequency smoother requires a uniform selected subcarrier grid.')
    return csi, ts, ready, freq


def steering_rows(theta_deg, tau_s, carrier_hz, frequencies_hz, rx_window, spacing_m, projection):
    """Rows follow angle-major, delay-minor; features follow Rx-major, frequency-minor."""
    theta = np.deg2rad(theta_deg)
    if projection == 'sin':
        angular = np.sin(theta)
    elif projection == 'legacy_cos':
        angular = 1 - np.cos(theta)  # preserved reference convention, NOT ordinary cos(theta)
    else:
        raise ValueError('projection must explicitly be sin or legacy_cos.')
    aa = np.exp(-2j * np.pi * carrier_hz * spacing_m / 3e8 * angular[:, None] * np.arange(rx_window))
    at = np.exp(-2j * np.pi * np.asarray(tau_s)[:, None] * frequencies_hz)
    rows = (aa[:, None, :, None] * at[None, :, None, :]).reshape(len(theta_deg) * len(tau_s), -1)
    return rows / np.sqrt(rows.shape[-1])


def smooth_covariance(window, rx_window, frequency_points, frequency_hop):
    """Accumulate Rx/frequency sliding snapshots without allocating the full snapshot cube."""
    _, _, receivers, carriers = window.shape
    offsets = np.arange(frequency_points) * frequency_hop
    if rx_window > receivers or offsets[-1] >= carriers or min(rx_window, frequency_points, frequency_hop) < 1:
        raise ValueError('MUSIC aperture exceeds input dimensions.')
    dimension = rx_window * frequency_points
    covariance = np.zeros((dimension, dimension), np.complex128)
    count = 0
    starts = np.arange(carriers - offsets[-1])
    for packet in window:
        for tx in packet:
            for rx in range(receivers - rx_window + 1):
                block = tx[rx:rx + rx_window]
                for start in range(0, len(starts), 1024):
                    indices = starts[start:start + 1024, None] + offsets
                    snapshots = block[:, indices].transpose(0, 2, 1).reshape(dimension, -1)
                    covariance += snapshots @ snapshots.conj().T
                    count += snapshots.shape[1]
    if count == 0:
        raise ValueError('No smoothing snapshots.')
    covariance /= count
    return (covariance + covariance.conj().T) / 2


def music_spectrum(covariance, steering, shape, signal_dim):
    """Fixed signal dimension is an explicit model-order hyperparameter."""
    n = covariance.shape[0]
    if not isinstance(signal_dim, int) or not 1 <= signal_dim < n:
        raise ValueError('signal_dim must be in [1, aperture_dimension-1].')
    if not np.isfinite(covariance).all() or np.trace(covariance).real <= 1e-20:
        raise ValueError('No usable dynamic covariance; do not emit arbitrary noise peaks.')
    _, vectors = np.linalg.eigh(covariance)
    noise = vectors[:, :n - signal_dim]
    denominator = np.maximum(np.sum(np.abs(steering.conj() @ noise) ** 2, axis=1), 1e-12)
    return (1 / denominator).reshape(shape)


def generate(raw_path, config_path, output_path):
    """Generate numeric spectra, raw-support bounds and delivery times, never PNG features."""
    output = Path(output_path)
    if output.suffix != '.npz':
        raise ValueError('Numeric heatmap output must end in .npz.')
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    cfg = read_json(config_path)
    with np.load(raw_path, allow_pickle=False) as data:
        required = ('csi', 'timestamps_s', 'available_at_s', 'frequencies_hz')
        if any(name not in data for name in required):
            raise ValueError(f'Raw NPZ must contain {required}.')
        csi, ts, ready, freq = (np.asarray(data[name]) for name in required)
    if csi.ndim != 4 or freq.shape != (csi.shape[-1],):
        raise ValueError('Raw CSI/frequency dimensions disagree.')
    indices = np.asarray(cfg.get('subcarrier_indices', list(range(csi.shape[-1]))))
    if indices.dtype.kind not in ('i', 'u'):
        raise ValueError('subcarrier_indices must be integers.')
    indices = indices.astype(int)
    if len(indices) < 2 or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= csi.shape[-1]:
        raise ValueError('subcarrier_indices must be ordered, unique and in bounds.')
    csi, freq = csi[..., indices], freq[indices]
    csi, ts, ready, freq = validate_csi(csi, ts, ready, freq)
    for key in ('window_packets', 'stride_packets', 'rx_window', 'frequency_points', 'frequency_hop', 'signal_dim'):
        if type(cfg[key]) is not int:
            raise ValueError(f'{key} must be an integer.')
    packets, stride = int(cfg['window_packets']), int(cfg['stride_packets'])
    rx, fp, hop = int(cfg['rx_window']), int(cfg['frequency_points']), int(cfg['frequency_hop'])
    theta, tau = np.asarray(cfg['theta_deg'], float), np.asarray(cfg['tau_s'], float)
    for axis in (theta, tau):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError('Search axes must be finite and strictly increasing.')
    if min(packets, stride, rx, fp, hop) <= 0 or packets < 2 or packets > len(csi):
        raise ValueError('Invalid packet/window sizes.')
    carrier, spacing, max_gap = (float(cfg[k]) for k in ('carrier_hz', 'antenna_spacing_m', 'max_raw_gap_s'))
    if not all(np.isfinite(v) and v > 0 for v in (carrier, spacing, max_gap)):
        raise ValueError('Carrier, spacing and max gap must be explicit positive values.')
    if (fp - 1) * hop >= len(freq) or rx > csi.shape[2]:
        raise ValueError('Configured aperture exceeds selected CSI.')
    mode = cfg.get('preprocess', 'window_mean')
    if mode not in ('window_mean', 'none', 'legacy_amplitude'):
        raise ValueError('preprocess=window_mean|none|legacy_amplitude.')
    delay = cfg.get('delivery_delay_s')
    if delay is not None and (not np.isfinite(delay) or delay < 0):
        raise ValueError('delivery_delay_s must be null or nonnegative.')
    steering = steering_rows(theta, tau, carrier, freq[np.arange(fp) * hop], rx, spacing, cfg['projection'])
    ends = np.arange(packets - 1, len(csi), stride)
    maps, valid, starts, stops, delivery, durations = [], [], [], [], [], []
    previous_delivery = -np.inf
    for end in ends:
        start = end - packets + 1
        tic = time.perf_counter()
        ok = np.max(np.diff(ts[start:end + 1])) <= max_gap
        window = csi[start:end + 1].astype(np.complex128, copy=True)
        if mode == 'legacy_amplitude':
            window = np.abs(window).astype(np.complex128)
        if mode != 'none':
            window -= window.mean(axis=0, keepdims=True)
        if ok:
            try:
                spectrum = music_spectrum(smooth_covariance(window, rx, fp, hop), steering,
                                          (len(theta), len(tau)), int(cfg['signal_dim']))
            except ValueError as exc:
                if 'No usable dynamic covariance' not in str(exc):
                    raise
                ok = False
        elapsed = time.perf_counter() - tic
        raw_ready = float(ready[start:end + 1].max())
        # Measured serial replay accounts for compute backlog; supplied delay is a declared delivery model.
        available = raw_ready + delay if delay is not None else max(raw_ready, previous_delivery) + elapsed
        previous_delivery = available
        maps.append(spectrum if ok else np.zeros((len(theta), len(tau))))
        valid.append(ok); starts.append(ts[start]); stops.append(ts[end]); delivery.append(available); durations.append(elapsed)
    verified = mode != 'legacy_amplitude' and all(cfg.get(k) is True for k in
                  ('phase_calibrated', 'frequency_mapping_verified', 'array_convention_verified'))
    metadata = {'format_version': 1, 'unit': 'linear', 'representation': 'aoa_tof',
                'physical_axes_verified': verified, 'clock_domain': cfg['clock_domain'],
                'availability_source': 'declared_delay' if delay is not None else 'measured_serial_replay',
                'generator': 'baf_roi_heatmap.music; adapted from 20261027 MUSIC.Azi_ToF',
                'raw_sha256': file_hash(raw_path), 'config': cfg}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, maps=np.asarray(maps, np.float32), valid=np.asarray(valid),
                        support_start_s=starts, support_end_s=stops, available_at_s=delivery,
                        theta_deg=theta, tau_s=tau, processing_s=durations,
                        metadata_json=np.array(json.dumps(metadata)))
    report = {'frames': len(maps), 'valid': int(sum(valid)), 'physical_axes_verified': verified,
              'median_compute_ms': float(np.median(durations) * 1000), 'output': str(output)}
    write_json(str(output) + '.audit.json', report)
    return report
