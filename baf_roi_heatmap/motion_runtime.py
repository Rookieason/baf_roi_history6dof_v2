"""Online internal fusion; active CSI forecasts do not first run a BAF rollout."""
import numpy as np
import torch

from baf_roi.data import Series
from baf_roi.config import validate_horizons
from baf_roi.pose import quat_to_ypr
from baf_roi.runtime import Predictor, observed_weights
from .contracts import InputConfig, ModelConfig, file_hash
from .data import encode_window
from .model import build_forecaster


class MotionPredictor:
    def __init__(self, baseline_checkpoint, adapter_checkpoint, device='cpu'):
        self.bundle = torch.load(adapter_checkpoint, map_location='cpu', weights_only=True)
        if self.bundle.get('format_version') != 1 or file_hash(baseline_checkpoint) != self.bundle['baseline_sha256']:
            raise ValueError('Motion model and baseline checkpoint identity mismatch.')
        config = ModelConfig(**self.bundle['model_config']).validate()
        if config.method != 'motion_moe':
            raise ValueError('Expected a motion_moe checkpoint.')
        self.device = device
        self.base = Predictor.load(baseline_checkpoint, device)
        self.model = build_forecaster(config, baseline_checkpoint, device).eval()
        self.model.load_state_dict(self.bundle['model'], strict=True)
        self.cfg = InputConfig(**self.bundle['input_config']).validate()

    def reset(self, stream_id=None):
        self.base.reset(stream_id)

    @torch.inference_mode()
    def predict_observations(self, timestamps_s, positions, rotations_deg, rf_series=None,
                             horizons_ms=None, stream_id=None, delivery_delay_s=0.):
        if not np.isfinite(delivery_delay_s) or delivery_delay_s < 0:
            raise ValueError('Delivery delay must be finite and nonnegative.')
        hs = list(self.bundle['horizons_ms'] if horizons_ms is None else horizons_ms)
        if not hs or not set(hs).issubset(self.bundle['horizons_ms']):
            raise ValueError('Use horizons included in the trained motion experiment.')
        validate_horizons(hs)
        if rf_series is not None:
            if rf_series.grid_id != self.bundle['grid_id']:
                raise ValueError('Online RF physical grid differs from training.')
            if not rf_series.metadata.get('physical_axes_verified') and not self.bundle['allow_exploratory']:
                raise ValueError('Unverified online RF cannot replace calibrated training RF.')
        ts, pos, rot = np.asarray(timestamps_s, float), np.asarray(positions, float), np.asarray(rotations_deg, float)
        if ts.ndim != 1 or len(ts) < 2 or pos.shape != (len(ts), 3) or rot.shape != pos.shape:
            raise ValueError('Need timestamps[N], positions[N,3], rotations_deg[N,3].')
        window = encode_window(rf_series, float(ts[-1]), self.bundle['normalizer'], self.cfg,
                               delivery_delay_s, self.bundle.get('rf_age_basis', 'support_end'))
        if bool(self.model.baseline_only) or (self.model.config.modality == 'rf' and not window['rf_available']):
            result = self.base.predict_observations(ts, pos, rot, hs, stream_id)
            result.update(rf_available=bool(window['rf_available']), adapted=False)
            return result
        scale = .01 if self.base.data_config['position_unit'] == 'cm' else 1.
        series = Series(ts-ts[0], np.concatenate((pos*scale, rot), -1), self.base.data_config['max_gap_ms']/1000)
        history = series.history(len(ts)-1)
        state = self.base.observed_state.get(str(stream_id)) if stream_id is not None else None
        # Confidence is observed HMD only. Do not commit hypothetical rollout state.
        raw = self.base.confidence(history[None])
        alpha, next_state = observed_weights(raw, [float(ts[-1])], self.base.mu,
                                             self.base.data_config['max_gap_ms']/1000, state)
        p0 = np.broadcast_to(history[-1, :3], (len(hs), 3)).copy()
        # Fallback tensors are not consulted on this all-active online path.
        from baf_roi.pose import ypr_to_quat
        q0 = np.broadcast_to(ypr_to_quat(history[-1, 3:]), (len(hs), 4)).copy()
        values = {**window, 'horizon_s': np.asarray(hs, np.float32)/1000,
                  'alpha': alpha[0], 'base_p': p0, 'base_q': q0}
        inputs = {key: torch.as_tensor(np.asarray(value)[None], device=self.device) for key, value in values.items()}
        prediction = self.model(inputs, torch.as_tensor(history[None], device=self.device))
        p, q = prediction['p'].cpu().numpy(), prediction['q'].cpu().numpy()
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise FloatingPointError('Internal motion experts produced nonfinite poses.')
        result = {'horizons_ms': hs, 'positions_m': p, 'quaternions_wxyz': q, 'rotations_ypr_deg': quat_to_ypr(q),
                  'raw_confidence_t': raw, 'alpha_t': alpha,
                  'rf_available': bool(window['rf_available']), 'adapted': True,
                  'horizon_gate': prediction['gate'].cpu().numpy(),
                  'expert_weights': prediction['expert_weights'].cpu().numpy(),
                  'rf_reliability': prediction['rf_reliability'].cpu().numpy()}
        for key in ('rollout_alpha', 'rollout_positions_m', 'rollout_quaternions_wxyz', 'rollout_expert_weights'):
            result[key] = prediction[key].cpu().numpy()
        if stream_id is not None:
            self.base.observed_state[str(stream_id)] = next_state
        return result
