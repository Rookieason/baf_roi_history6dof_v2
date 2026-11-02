"""Compare the numerical AoA-ToF core against the user's original MUSIC.py.

Only numerical routines are invoked. Plot is stubbed deliberately; legacy plotting,
whole-record preprocessing, centered support and physical calibration are not endorsed.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi_heatmap.music import smooth_covariance, steering_rows, music_spectrum
from baf_roi_heatmap.contracts import file_hash, write_json


def check(reference_file, output):
    if Path(output).exists(): raise ValueError('Use a new report path.')
    sys.modules.setdefault('Plot', types.ModuleType('Plot'))
    spec = importlib.util.spec_from_file_location('reference_music', reference_file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = types.SimpleNamespace(f_0=5.57e9, BW=160e6, delta_f=2.5e6, antenna_spacing=.02,
        stream_win=3, freq_win=6, freq_hop=2, time_win=4, time_hop=1, time_sample_range=8, fs=100,
        stream_sample_range=4, num_Rx=4, freq_space=1, avg_frames=8, theta_min=-60,
        theta_max=60, theta_step=30, tau_min=0., tau_max=10e-9, tau_step=2e-9,
        num_sc=8, freq_sample_range=8, Sdim=2, projection='sin')
    rng = np.random.default_rng(112)
    x = rng.normal(size=(8, 2, 4, 8))+1j*rng.normal(size=(8, 2, 4, 8))
    results = []
    for projection in ('sin', 'cos'):
        args.projection = projection
        original = module.Azi_ToF(args)
        a, b = original.Rxx_smooth(x, 4), smooth_covariance(x, 3, 3, 2)
        np.testing.assert_allclose(a, b, atol=1e-12)
        rows = steering_rows(original.theta_grid, original.tau_grid, args.f_0,
                             args.f_0+np.arange(3)*2*args.delta_f, 3, args.antenna_spacing,
                             'sin' if projection == 'sin' else 'legacy_cos')
        np.testing.assert_allclose(rows, original.steering_matrix_chunk(original.tau_grid), atol=1e-12)
        _, _, p = original.cal_spectrum(a)
        q = music_spectrum(b, rows, p.shape, 2)
        np.testing.assert_allclose(p, q, rtol=1e-10, atol=1e-12)
        results.append({'projection': projection, 'max_covariance_error': float(np.max(abs(a-b))),
                        'max_spectrum_error': float(np.max(abs(p-q)))})
    report = {'status': 'PASS', 'reference_sha256': file_hash(reference_file), 'cases': results,
              'scope': 'Identical raw support; covariance, steering, spectrum. Not full legacy preprocessing or calibration.'}
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--reference-file', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(check(args.reference_file, args.output))
