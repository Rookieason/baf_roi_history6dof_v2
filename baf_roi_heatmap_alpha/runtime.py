"""Shared prepared-data inference and chronological online alpha-only inference."""
import numpy as np
import torch
from baf_roi.data import Series
from baf_roi.runtime import Predictor
from baf_roi.pose import quat_to_ypr
from baf_roi_heatmap.contracts import InputConfig, file_hash
from baf_roi_heatmap.data import encode_window
from baf_roi_heatmap.losses import numpy_prediction
from .model import AlphaConfig, AlphaFusion


def load_alpha(path, dataset, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 1 or bundle.get('method') != 'baf_roi_heatmap_alpha':
        raise ValueError('Expected a baf_roi_heatmap_alpha checkpoint.')
    if bundle.get('experiment_id') != dataset.manifest['experiment_id']:
        raise ValueError('Alpha checkpoint/prepared experiment mismatch.')
    baseline = dataset.manifest['baseline_checkpoint']
    if bundle.get('baseline_sha256') != dataset.manifest['baseline_sha256'] or file_hash(baseline) != bundle['baseline_sha256']:
        raise ValueError('Frozen baseline changed or belongs to another experiment.')
    base = Predictor.load(baseline, device)
    model = AlphaFusion(base, AlphaConfig(**bundle['model_config'])).to(device)
    model.gate.load_state_dict(bundle['gate'], strict=True)
    return model.eval(), bundle


def batch_histories(dataset, batch, device='cpu'):
    if 'history' in batch:
        return batch['history'].to(device=device, dtype=torch.float64)
    histories = []
    for ri, oi in zip(batch['record_index'], batch['origin_index']):
        ri, oi = int(ri), int(oi)
        record = dataset.records[ri]
        frozen = dataset.arrays(ri)
        raw = dataset.cache.arrays(record['cache_index'])
        histories.append(raw['series'].history(int(frozen['anchors'][oi])))
    return torch.as_tensor(np.stack(histories), dtype=torch.float64, device=device)


def predict_dataset_batch(model, dataset, batch, device='cpu'):
    """Targets intentionally never enter model.forward()."""
    inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
    return model(inputs, batch_histories(dataset, batch, device))


class HeatmapAlphaPredictor:
    """Online state is the original observed-only BAF EMA, independently per stream.

An alpha correction is held fixed only for the current forecast; it is never
committed as a new observation. A missing/stale RF window immediately returns
the original BAF result, independent of any earlier RF correction.
"""
    def __init__(self, baseline_checkpoint, alpha_checkpoint, device='cpu'):
        self.bundle = torch.load(alpha_checkpoint, map_location='cpu', weights_only=True)
        if self.bundle.get('format_version') != 1 or self.bundle.get('method') != 'baf_roi_heatmap_alpha':
            raise ValueError('Expected a baf_roi_heatmap_alpha checkpoint.')
        if file_hash(baseline_checkpoint) != self.bundle['baseline_sha256']:
            raise ValueError('Alpha and baseline checkpoint identity mismatch.')
        self.base = Predictor.load(baseline_checkpoint, device)
        self.model = AlphaFusion(self.base, AlphaConfig(**self.bundle['model_config'])).to(device).eval()
        self.model.gate.load_state_dict(self.bundle['gate'], strict=True)
        self.cfg = InputConfig(**self.bundle['input_config']).validate()
        self.device = device

    def reset(self, stream_id=None):
        self.base.reset(stream_id)

    @torch.inference_mode()
    def predict_observations(self, timestamps_s, positions, rotations_deg, rf_series=None,
                             horizons_ms=None, stream_id=None, delivery_delay_s=0.):
        state = self.base.observed_state.copy()
        try:
            return self._predict(timestamps_s, positions, rotations_deg, rf_series,
                                 horizons_ms, stream_id, delivery_delay_s)
        except Exception:
            self.base.observed_state = state
            raise

    def _predict(self, timestamps_s, positions, rotations_deg, rf_series,
                 horizons_ms, stream_id, delivery_delay_s):
        if not np.isfinite(delivery_delay_s) or delivery_delay_s < 0:
            raise ValueError('Delivery delay must be finite and nonnegative.')
        hs = list(self.bundle['horizons_ms'] if horizons_ms is None else horizons_ms)
        if not set(hs).issubset(self.bundle['horizons_ms']):
            raise ValueError('Use horizons included in the trained alpha experiment.')
        if rf_series is not None:
            if rf_series.grid_id != self.bundle['grid_id']:
                raise ValueError('Online RF physical grid differs from training.')
            if not rf_series.metadata.get('physical_axes_verified') and not self.bundle['allow_exploratory']:
                raise ValueError('Unverified online RF cannot replace calibrated training RF.')
        ts = np.asarray(timestamps_s, float)
        result = self.base.predict_observations(ts, positions, rotations_deg, hs, stream_id)
        window = encode_window(rf_series, float(ts[-1]), self.bundle['normalizer'], self.cfg, delivery_delay_s,
                               self.bundle.get('rf_age_basis', 'support_end'))
        result.update(rf_available=bool(window['rf_available']), adapted=False)
        if not window['rf_available']:
            return result
        scale = .01 if self.base.data_config['position_unit'] == 'cm' else 1.
        series = Series(ts-ts[0], np.concatenate((np.asarray(positions)*scale, rotations_deg), -1),
                        self.base.data_config['max_gap_ms']/1000)
        history = series.history(len(ts)-1)
        raw_inputs = {**window, 'horizon_s': np.asarray(hs, np.float32)/1000,
                      'base_p': result['positions_m'][0], 'base_q': result['quaternions_wxyz'][0],
                      'alpha': np.float32(result['alpha_t'][0])}
        inputs = {key: torch.as_tensor(np.asarray(value)[None], device=self.device)
                  for key, value in raw_inputs.items()}
        prediction = self.model(inputs, torch.as_tensor(history[None], device=self.device))
        if not prediction['applied'].any():
            return result
        p, q = numpy_prediction(prediction, result['positions_m'], result['quaternions_wxyz'])
        result.update(positions_m=p, quaternions_wxyz=q, rotations_ypr_deg=quat_to_ypr(q),
                      adapted=True, logit_delta=prediction['logit_delta'].cpu().numpy(),
                      horizon_gate=prediction['gate'].cpu().numpy())
        # alpha_t remains the observed BAF EMA; actual forecast weights are separate.
        for key in ('rollout_alpha', 'rollout_positions_m', 'rollout_quaternions_wxyz'):
            result[key] = prediction[key].cpu().numpy()
        return result
