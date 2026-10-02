"""History-only inference. Forecast state is NEVER committed as observed state."""
from __future__ import annotations
import math
import numpy as np
import torch
from .models import build_models
from .pose import model_inputs, ypr_to_quat, quat_to_ypr, interpolate_predictions
from .upstream_ops import apply_decision_fusion
from .data import Series


class Predictor:
    def __init__(self, models, stats, mu=.6, device='cpu', data_config=None):
        self.device = torch.device(device)
        self.models = {k: v.to(self.device).eval() for k, v in models.items()}
        self.stats, self.mu = stats, float(mu)
        self.data_config = data_config or {'position_unit': 'm', 'max_gap_ms': 100}
        self.observed_state = {}  # stream_id -> (last_real_timestamp, alpha_at_that_time)

    @classmethod
    def load(cls, checkpoint, device='cpu'):
        # Only load trusted project checkpoints; no arbitrary pickled objects needed.
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if bundle.get('format_version') != 2:
            raise ValueError('Expected v2 author-centered checkpoint. Rebuild cache and retrain; v1 is not compatible.')
        models = build_models()
        for name in models:
            models[name].load_state_dict(bundle['models'][name], strict=True)
        result = cls(models, bundle['normalization'], bundle['config']['model']['temporal_mu'], device,
                     bundle['config']['data'])
        result.bundle = bundle
        return result

    @torch.inference_mode()
    def confidence(self, histories):
        conf, _ = model_inputs(histories, self.stats)
        return self.models['confidence'](torch.as_tensor(conf, device=self.device))['confidence'].cpu().numpy()

    @torch.inference_mode()
    def _step(self, histories):
        conf, pose = model_inputs(histories, self.stats)
        raw = self.models['confidence'](torch.as_tensor(conf, device=self.device))['confidence'].cpu().numpy()
        x = torch.as_tensor(pose, device=self.device)
        f = self.models['focused'](x)
        e = self.models['explorer'](x, None)
        mean, std = np.array(self.stats['mean'], np.float32), np.array(self.stats['std'], np.float32)
        return raw, f['p_hat_tr'].cpu().numpy() * std + mean, f['q_hat_norm'].cpu().numpy(), \
            e['p_hat_tr'].cpu().numpy() * std + mean, e['q_hat_norm'].cpu().numpy()

    def forecast_batch(self, histories, horizons_ms=(50, 100, 150, 200, 250), *, previous_alpha=None, initial_alpha=None):
        """Pure function of history, fixed trained weights, and optional CAUSAL alpha state.

        histories [B,30,6]: meters + dataset yaw,pitch,roll in degrees, at 30 Hz.
        previous_alpha: last OBSERVED decision's weight, or NaN to initialize with raw alpha.
        initial_alpha: weight ALREADY computed at this origin (used by ordered batched evaluation).
        No targets, labels, scene features, future poses, or timestamps enter a neural model.
        """
        if previous_alpha is not None and initial_alpha is not None:
            raise ValueError('Pass previous_alpha OR initial_alpha, not both.')
        h = np.asarray(histories, np.float64).copy()
        if h.ndim != 3 or h.shape[1:] != (30, 6) or not len(h) or not np.isfinite(h).all():
            raise ValueError('Need finite, nonempty history [B,30,6].')
        hs = list(horizons_ms)
        if not hs or len(set(hs)) != len(hs) or any(not np.isfinite(x) or not 50 <= x <= 250 for x in hs):
            raise ValueError('Distinct horizons must be within [50,250] ms.')
        steps = math.ceil(max(hs) * 30 / 1000 - 1e-10)
        ps, qs, alphas = [h[:, -1, :3].copy()], [ypr_to_quat(h[:, -1, 3:])], []
        alpha = None
        for step in range(steps):
            raw, fp, fq, ep, eq = self._step(h)
            raw = np.asarray(raw, np.float32).reshape(-1)
            if step == 0:
                raw_at_t = raw.copy()
                if initial_alpha is not None:
                    alpha = np.asarray(initial_alpha, np.float32).reshape(-1).copy()
                elif previous_alpha is not None:
                    prev = np.asarray(previous_alpha, np.float32).reshape(-1)
                    alpha = np.where(np.isnan(prev), raw, self.mu * raw + (1 - self.mu) * prev)
                else:
                    alpha = raw.copy()
            else:
                alpha = self.mu * raw + (1 - self.mu) * alpha
            if alpha.shape != (len(h),) or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
                raise ValueError('Bad alpha state: need one weight in [0,1] per history.')
            p, q, _ = apply_decision_fusion('full_slerp', alpha, raw, fp, ep, fq, eq, .5, .5)
            if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=-1) < 1e-8):
                raise FloatingPointError('Model produced an invalid predicted pose.')
            ps.append(p); qs.append(q); alphas.append(alpha.copy())
            # Only PREDICTED poses enter subsequent rollout windows; never ground truth.
            generated = np.concatenate((p, quat_to_ypr(q)), axis=-1)
            h = np.concatenate((h[:, 1:], generated[:, None]), axis=1)
        positions, quaternions = interpolate_predictions(np.stack(ps, 1), np.stack(qs, 1), hs)
        return {'horizons_ms': hs, 'positions_m': positions, 'quaternions_wxyz': quaternions,
                'rotations_ypr_deg': quat_to_ypr(quaternions), 'raw_confidence_t': raw_at_t,
                'alpha_t': alphas[0], 'rollout_alpha': np.stack(alphas, 1),
                'rollout_positions_m': np.stack(ps, 1), 'rollout_quaternions_wxyz': np.stack(qs, 1)}

    def reset(self, stream_id=None):
        if stream_id is None:
            self.observed_state.clear()
        else:
            self.observed_state.pop(str(stream_id), None)

    def predict_observations(self, timestamps_s, positions, rotations_deg,
                             horizons_ms=(50, 100, 150, 200, 250), stream_id=None):
        """Online adapter. Pass ONLY observations received by the decision time.
        positions use the checkpoint's recorded unit (cm or m). Timestamps in seconds.
        Per-stream calls must be chronological. Call at approximately 30 Hz.
        """
        ts = np.asarray(timestamps_s, np.float64)
        pos = np.asarray(positions, np.float64)
        rot = np.asarray(rotations_deg, np.float64)
        if ts.ndim != 1 or len(ts) < 2 or pos.shape != (len(ts), 3) or rot.shape != pos.shape:
            raise ValueError('Need timestamps[N], positions[N,3], rotations_deg[N,3].')
        ts_abs = float(ts[-1])
        scale = .01 if self.data_config['position_unit'] == 'cm' else 1.
        series = Series(ts - ts[0], np.concatenate((pos * scale, rot), 1), self.data_config['max_gap_ms'] / 1000)
        history = series.history(len(ts) - 1)
        prev = np.nan
        if stream_id is not None:
            state = self.observed_state.get(str(stream_id))
            if state is not None:
                if ts_abs <= state[0]:
                    raise ValueError('Stream decisions must strictly increase; reset for a new trajectory.')
                if ts_abs - state[0] <= self.data_config['max_gap_ms'] / 1000:
                    prev = state[1]
        result = self.forecast_batch(history[None], horizons_ms, previous_alpha=np.array([prev]))
        if stream_id is not None:
            # Commit the first observed-origin weight ONLY, not the final forecast weight.
            self.observed_state[str(stream_id)] = (ts_abs, float(result['alpha_t'][0]))
        return result
