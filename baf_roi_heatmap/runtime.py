"""Online adapter: original BAF state remains observed-only; RF correction is window-local."""
import time
from pathlib import Path
import numpy as np
import torch
from baf_roi.data import Series
from baf_roi.runtime import Predictor
from baf_roi.pose import model_inputs, quat_to_ypr
from .contracts import InputConfig, ModelConfig, file_hash, write_json
from .data import encode_window, ForecastDataset
from .model import ForecastAdapter
from .losses import numpy_prediction
from .prepare import fixed_forecast


class HeatmapPredictor:
    def __init__(self, baseline_checkpoint, adapter_checkpoint, device='cpu'):
        self.bundle = torch.load(adapter_checkpoint, map_location='cpu', weights_only=True)
        if self.bundle.get('format_version') != 1 or file_hash(baseline_checkpoint) != self.bundle['baseline_sha256']:
            raise ValueError('Adapter and baseline checkpoint identity mismatch.')
        self.base = Predictor.load(baseline_checkpoint, device)
        self.model = ForecastAdapter(ModelConfig(**self.bundle['model_config'])).to(device).eval()
        self.model.load_state_dict(self.bundle['model'], strict=True)
        self.cfg = InputConfig(**self.bundle['input_config']).validate()
        self.device = device

    def reset(self, stream_id=None):
        self.base.reset(stream_id)

    @torch.inference_mode()
    def predict_observations(self, timestamps_s, positions, rotations_deg, rf_series=None,
                             horizons_ms=None, stream_id=None, delivery_delay_s=0.):
        state = self.base.observed_state.copy()
        try:
            return self._predict(timestamps_s, positions, rotations_deg, rf_series, horizons_ms, stream_id, delivery_delay_s)
        except Exception:
            self.base.observed_state = state
            raise

    def _predict(self, timestamps_s, positions, rotations_deg, rf_series, horizons_ms, stream_id, delivery_delay_s):
        if not np.isfinite(delivery_delay_s) or delivery_delay_s < 0:
            raise ValueError('Delivery delay must be finite and nonnegative.')
        hs = list(self.bundle['horizons_ms'] if horizons_ms is None else horizons_ms)
        if not set(hs).issubset(self.bundle['horizons_ms']):
            raise ValueError('This adapter may only be evaluated at its trained horizons.')
        if rf_series is not None:
            if rf_series.grid_id != self.bundle['grid_id']:
                raise ValueError('Online RF physical grid differs from training.')
            if not rf_series.metadata.get('physical_axes_verified') and not self.bundle['allow_exploratory']:
                raise ValueError('Unverified online RF cannot silently replace calibrated training RF.')
        ts = np.asarray(timestamps_s, float)
        result = self.base.predict_observations(ts, positions, rotations_deg, hs, stream_id)
        scale = .01 if self.base.data_config['position_unit'] == 'cm' else 1.
        series = Series(ts-ts[0], np.concatenate((np.asarray(positions)*scale, rotations_deg), axis=1),
                        self.base.data_config['max_gap_ms']/1000)
        history = series.history(len(ts)-1)
        _, hmd = model_inputs(history, self.base.stats)
        window = encode_window(rf_series, float(ts[-1]), self.bundle['normalizer'], self.cfg, delivery_delay_s)
        result['rf_available'] = bool(window['rf_available'])
        result['adapted'] = False
        if self.model.config.modality == 'rf' and not window['rf_available']:
            return result
        if self.model.config.method == 'expert_gate':
            fp, fq = fixed_forecast(self.base, history[None], hs, 0.)
            ep, eq = fixed_forecast(self.base, history[None], hs, 1.)
        else:
            fp, fq, ep, eq = result['positions_m'], result['quaternions_wxyz'], result['positions_m'], result['quaternions_wxyz']
        inputs = {k: torch.as_tensor(v[None] if isinstance(v, np.ndarray) else [v], device=self.device)
                  for k, v in {'hmd': hmd, **window, 'horizon_s': np.asarray(hs, np.float32)/1000,
                               'base_p': result['positions_m'][0].astype(np.float32),
                               'base_q': result['quaternions_wxyz'][0].astype(np.float32),
                               'focused_p': fp[0].astype(np.float32), 'focused_q': fq[0].astype(np.float32),
                               'explorer_p': ep[0].astype(np.float32), 'explorer_q': eq[0].astype(np.float32),
                               'alpha': np.float32(result['alpha_t'][0])}.items()}
        prediction = self.model(inputs)
        p, q = numpy_prediction(prediction, result['positions_m'], result['quaternions_wxyz'])
        # Recursive intermediates are BAF-only. Do not expose them as an adapted trajectory.
        for key in ('rollout_positions_m', 'rollout_quaternions_wxyz', 'rollout_alpha'):
            result.pop(key, None)
        result.update(positions_m=p, quaternions_wxyz=q, rotations_ypr_deg=quat_to_ypr(q),
                      horizon_gate=prediction['gate'].cpu().numpy(), adapted=bool(prediction['applied'].any()))
        return result


def benchmark(prepared, adapter_checkpoint, output_path, device='cpu', count=50):
    if count < 1 or Path(output_path).exists():
        raise ValueError('Positive benchmark count and a new output path are required.')
    dataset = ForecastDataset(prepared, 'test')
    predictor = HeatmapPredictor(dataset.manifest['baseline_checkpoint'], adapter_checkpoint, device)
    def sample(index, stream):
        ri, oi = map(int, dataset.index[index]); r = dataset.records[ri]
        a = dataset.arrays(ri); raw = dataset.cache.arrays(r['cache_index'])
        h = raw['series'].history(int(a['anchors'][oi]))
        ts = float(a['origin_s'][oi])-np.arange(29, -1, -1)/30
        unit_scale = 100. if predictor.base.data_config['position_unit'] == 'cm' else 1.
        rf = dataset.rf_series(r['key'])
        if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
        tic = time.perf_counter()
        result = predictor.predict_observations(ts, h[:, :3]*unit_scale, h[:, 3:], rf, dataset.horizons, stream)
        dataset.scene.mask(result['positions_m'], result['quaternions_wxyz'])
        if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
        return (time.perf_counter()-tic)*1000
    for _ in range(5): sample(0, None)
    predictor.reset()
    values = [sample(i, dataset.records[int(dataset.index[i, 0])]['key']) for i in range(min(count, len(dataset)))]
    report = {'count': len(values), 'device': str(device), 'torch': str(torch.__version__), 'threads': torch.get_num_threads(),
              'parameters_adapter': sum(p.numel() for p in predictor.model.parameters()),
              'median_ms': float(np.median(values)), 'p95_ms': float(np.quantile(values, .95)),
              'scope': 'batch one: HMD resampling, BAF rollout, optional expert rollouts, RF lookup/encoding, adapter, hard ROI; warmup excluded',
              'excludes': 'RF acquisition, clock sync, MUSIC generation, disk I/O, network, renderer; not a full-system FPS guarantee'}
    write_json(output_path, report)
    return report
