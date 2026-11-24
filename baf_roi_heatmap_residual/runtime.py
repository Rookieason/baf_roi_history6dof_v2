"""CSI residuals inside BAF, with one assisted rollout per online decision.

Cached BAF endpoints are evaluation references only. They are never model
inputs. Online inference computes the observed confidence EMA, then runs the
CSI-conditioned experts directly. Missing RF and zero-initialized residuals
use the original BAF implementation for an exact baseline fallback.
"""
import numpy as np
import torch

from baf_roi.config import validate_horizons
from baf_roi.data import Series
from baf_roi.pose import quat_to_ypr
from baf_roi.runtime import Predictor
from baf_roi_heatmap.contracts import InputConfig, file_hash
from baf_roi_heatmap.data import encode_window
from baf_roi_heatmap_alpha.runtime import batch_histories
from .model import InternalResidualFusion, ResidualConfig


MODEL_INPUT_KEYS = ('maps', 'mask', 'timing', 'rf_available', 'horizon_s', 'alpha')


def _check_format(bundle):
    if bundle.get('format_version') != 1 or bundle.get('method') != 'baf_roi_heatmap_residual':
        raise ValueError('Expected a baf_roi_heatmap_residual checkpoint; post-hoc adapters are incompatible.')


def load_residual(path, dataset, device='cpu'):
    """Load only against the exact paired experiment and frozen BAF identity."""
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    _check_format(bundle)
    if bundle.get('experiment_id') != dataset.manifest['experiment_id']:
        raise ValueError('Internal residual checkpoint/prepared experiment mismatch.')
    baseline = dataset.manifest['baseline_checkpoint']
    if (bundle.get('baseline_sha256') != dataset.manifest['baseline_sha256']
            or file_hash(baseline) != bundle['baseline_sha256']):
        raise ValueError('Frozen baseline changed or belongs to another experiment.')
    expected = {
        'horizons_ms': dataset.horizons,
        'input_config': dataset.manifest['input_config'],
        'normalizer': dataset.manifest['normalizer'],
        'grid_id': dataset.rf['grid_id'],
        'rf_age_basis': dataset.rf_age_basis,
        'allow_exploratory': dataset.rf.get('allow_exploratory', False),
    }
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise ValueError(f'Internal residual checkpoint/prepared mismatch: {key}.')
    base = Predictor.load(baseline, device)
    model = InternalResidualFusion(base, ResidualConfig(**bundle['model_config'])).to(device)
    model.residual.load_state_dict(bundle['residual'], strict=True)
    return model.eval(), bundle


def predict_dataset_batch(model, dataset, batch, device='cpu'):
    """Whitelist causal inputs; exclude targets AND cached completed forecasts."""
    inputs = {key: batch['inputs'][key].to(device) for key in MODEL_INPUT_KEYS}
    return model(inputs, batch_histories(dataset, batch, device))


def reference_prediction(model, prediction, rf_available, original_base_p, original_base_q):
    """Apply exactly the online dispatch policy to prepared-data predictions.

    Missing RF (for the RF model) and globally zero output projections use the
    original baseline values. A trained residual may cancel for one particular
    input; online inference still uses its single Torch rollout in that case,
    so evaluation must keep that rollout too, even when ``applied`` is false.
    This preserves hard-ROI boundary parity without another model execution.
    """
    p = prediction['p'].detach().cpu().numpy().astype(np.float64)
    q = prediction['q'].detach().cpu().numpy().astype(np.float64)
    if torch.is_tensor(rf_available):
        rf_available = rf_available.detach().cpu().numpy()
    available = np.asarray(rf_available, dtype=bool)
    if available.shape != (len(p),):
        raise ValueError('Need one RF availability flag per prediction.')
    fallback = np.full(len(p), model.is_zero_residual(), dtype=bool)
    if model.config.modality == 'rf':
        fallback |= ~available
    return (np.where(fallback[:, None, None], original_base_p, p),
            np.where(fallback[:, None, None], original_base_q, q))


class HeatmapResidualPredictor:
    """Chronological online inference with observed-only, per-stream EMA state.

The model receives one causal RF context and produces its own entire assisted
trajectory. The state saved after success is the HMD-only EMA at the observed
origin; neither generated poses nor future rollout confidence are committed.
"""
    def __init__(self, baseline_checkpoint, residual_checkpoint, device='cpu'):
        self.bundle = torch.load(residual_checkpoint, map_location='cpu', weights_only=True)
        _check_format(self.bundle)
        if file_hash(baseline_checkpoint) != self.bundle['baseline_sha256']:
            raise ValueError('Internal residual and baseline checkpoint identity mismatch.')
        self.base = Predictor.load(baseline_checkpoint, device)
        self.model = InternalResidualFusion(self.base, ResidualConfig(**self.bundle['model_config'])).to(device).eval()
        self.model.residual.load_state_dict(self.bundle['residual'], strict=True)
        self.cfg = InputConfig(**self.bundle['input_config']).validate()
        self.device = device

    def reset(self, stream_id=None):
        self.base.reset(stream_id)

    @torch.inference_mode()
    def predict_observations(self, timestamps_s, positions, rotations_deg, rf_series=None,
                             horizons_ms=None, stream_id=None, delivery_delay_s=0.):
        # State is committed only at the end, after RF checks and all prediction
        # validation. A failed call can therefore be retried at the same time.
        return self._predict(timestamps_s, positions, rotations_deg, rf_series,
                             horizons_ms, stream_id, delivery_delay_s)

    def _predict(self, timestamps_s, positions, rotations_deg, rf_series,
                 horizons_ms, stream_id, delivery_delay_s):
        if not np.isfinite(delivery_delay_s) or delivery_delay_s < 0:
            raise ValueError('Delivery delay must be finite and nonnegative.')
        hs = validate_horizons(self.bundle['horizons_ms'] if horizons_ms is None else horizons_ms)
        if not set(hs).issubset(self.bundle['horizons_ms']):
            raise ValueError('Use horizons included in the trained internal residual experiment.')
        if rf_series is not None:
            if rf_series.grid_id != self.bundle['grid_id']:
                raise ValueError('Online RF physical grid differs from training.')
            if not rf_series.metadata.get('physical_axes_verified') and not self.bundle['allow_exploratory']:
                raise ValueError('Unverified online RF cannot replace calibrated training RF.')

        ts = np.asarray(timestamps_s, np.float64)
        pos = np.asarray(positions, np.float64)
        rot = np.asarray(rotations_deg, np.float64)
        if ts.ndim != 1 or len(ts) < 2 or pos.shape != (len(ts), 3) or rot.shape != pos.shape:
            raise ValueError('Need timestamps[N], positions[N,3], rotations_deg[N,3].')
        scale = .01 if self.base.data_config['position_unit'] == 'cm' else 1.
        series = Series(ts-ts[0], np.concatenate((pos*scale, rot), -1),
                        self.base.data_config['max_gap_ms']/1000)
        history = series.history(len(ts)-1)
        origin = float(ts[-1])
        previous = np.array([np.nan], dtype=np.float32)
        if stream_id is not None:
            state = self.base.observed_state.get(str(stream_id))
            if state is not None:
                if origin <= state[0]:
                    raise ValueError('Stream decisions must strictly increase; reset for a new trajectory.')
                if origin-state[0] <= self.base.data_config['max_gap_ms']/1000:
                    previous[0] = state[1]
        # Match Predictor.forecast_batch's float32 arithmetic and rounding.
        raw = np.asarray(self.base.confidence(history[None]), np.float32).reshape(-1)
        alpha = np.where(np.isnan(previous), raw,
                         self.base.mu*raw+(1-self.base.mu)*previous).astype(np.float32)
        if raw.shape != (1,) or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
            raise FloatingPointError('Invalid observed BAF confidence/EMA.')
        window = encode_window(rf_series, origin, self.bundle['normalizer'], self.cfg,
                               delivery_delay_s, self.bundle.get('rf_age_basis', 'support_end'))
        rf_available = bool(window['rf_available'])

        if (self.model.config.modality == 'rf' and not rf_available) or self.model.is_zero_residual():
            # This is an alternative execution path, not a prerequisite for the
            # assisted path. Preserve even hard-ROI boundary behavior exactly.
            result = self.base.forecast_batch(history[None], hs, initial_alpha=alpha)
            result.update(rf_available=rf_available, adapted=False,
                          execution_path='original_baf_fallback')
        else:
            values = {**window, 'horizon_s': np.asarray(hs, np.float32)/1000,
                      'alpha': alpha[0]}
            inputs = {key: torch.as_tensor(np.asarray(values[key])[None], device=self.device)
                      for key in MODEL_INPUT_KEYS}
            prediction = self.model(inputs, torch.as_tensor(history[None], device=self.device))
            p = prediction['p'].detach().cpu().numpy().astype(np.float64)
            q = prediction['q'].detach().cpu().numpy().astype(np.float64)
            if (not np.isfinite(p).all() or not np.isfinite(q).all()
                    or np.any(np.linalg.norm(q, axis=-1) < 1e-8)):
                raise FloatingPointError('Internal residual produced an invalid predicted pose.')
            result = {
                'horizons_ms': hs, 'positions_m': p, 'quaternions_wxyz': q,
                'rotations_ypr_deg': quat_to_ypr(q),
                'raw_confidence_t': raw.copy(), 'alpha_t': alpha.copy(),
                'rf_available': rf_available, 'adapted': bool(prediction['applied'].any()),
                'horizon_gate': prediction['gate'].detach().cpu().numpy(),
                'execution_path': 'internal_residual_single_rollout',
            }
            for key in ('rollout_alpha', 'rollout_positions_m', 'rollout_quaternions_wxyz'):
                result[key] = prediction[key].detach().cpu().numpy()
        if stream_id is not None:
            self.base.observed_state[str(stream_id)] = (origin, float(alpha[0]))
        return result
