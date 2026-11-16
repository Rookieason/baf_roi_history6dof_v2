"""A small heatmap gate inside the frozen BAF autoregressive prediction loop.

The observed BAF confidence EMA is unchanged. At an origin the same, causal RF
summary supplies a bounded log-odds offset to each *forecast* fusion weight.
Only the gate is trained; there is no position/rotation residual head, no new
pose encoder, no endpoint mixing, and no future RF input.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class AlphaConfig:
    hidden: int = 16
    max_logit_delta: float = 1.0

    def validate(self):
        if not isinstance(self.hidden, int) or self.hidden < 1:
            raise ValueError('hidden must be a positive integer.')
        if not math.isfinite(self.max_logit_delta) or not 0 < self.max_logit_delta <= 10:
            raise ValueError('max_logit_delta must lie in (0,10] for a bounded, finite float32 odds multiplier.')
        return self


FEATURE_NAMES = (
    'level_mean', 'level_spatial_std', 'background_abs_mean',
    'temporal_abs_mean', 'level_change', 'valid_fraction',
    'latest_age', 'level_peak_minus_mean',
)


def heatmap_statistics(inputs):
    """Eight robust summaries of the already-normalized causal RF window.

Invalid frames contribute neither pixels nor timing. Statistics describe RF
change/concentration; they are not calibrated speed or body-part estimates.
"""
    maps, mask = inputs['maps'], inputs['mask'].bool()
    valid = mask.to(maps.dtype)
    count = valid.sum(1).clamp_min(1)
    safe = torch.where(mask[:, :, None, None, None], maps, torch.zeros_like(maps))
    level = safe[:, :, 0]
    means = level.mean((-2, -1))
    stds = level.std((-2, -1), unbiased=False)
    peaks = level.amax((-2, -1)) - means
    bg = safe[:, :, 1].abs().mean((-2, -1))
    change = safe[:, :, 2].abs().mean((-2, -1))
    first = mask.to(torch.int64).argmax(1)
    last = mask.shape[1] - 1 - mask.flip(1).to(torch.int64).argmax(1)
    rows = torch.arange(len(maps), device=maps.device)
    nonempty = mask.any(1)
    trend = torch.where(nonempty, means[rows, last] - means[rows, first], 0.)
    age = torch.where(nonempty, inputs['timing'][rows, last, 0], 0.)
    average = lambda value: (value * valid).sum(1) / count
    return torch.stack((average(means), average(stds), average(bg),
                        average(change), trend, valid.mean(1), age,
                        average(peaks)), dim=-1)


class HeatmapAlphaGate(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = (config or AlphaConfig()).validate()
        self.head = nn.Sequential(nn.Linear(len(FEATURE_NAMES), self.config.hidden),
                                  nn.Tanh(), nn.Linear(self.config.hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        raw = self.head(heatmap_statistics(inputs)).squeeze(-1)
        delta = self.config.max_logit_delta * torch.tanh(raw)
        return torch.where(inputs['rf_available'].bool(), delta, torch.zeros_like(delta))


def correct_alpha(alpha, delta):
    """sigmoid(logit(alpha)+delta), including exact 0/1 endpoints.

This odds form needs no epsilon-clamp. Zero correction preserves alpha,
including saturated confidence values, and remains differentiable at zero.
"""
    odds_multiplier = torch.exp(delta)
    return alpha * odds_multiplier / (1 - alpha + alpha * odds_multiplier)


def ypr_to_quaternion(ypr):
    """Differentiable Rz(yaw) Ry(-pitch) Rx(roll), identical axis convention."""
    yaw, pitch, roll = (ypr * (math.pi / 360)).unbind(-1)
    pitch = -pitch
    cy, sy, cp, sp, cr, sr = (torch.cos(yaw), torch.sin(yaw), torch.cos(pitch),
                             torch.sin(pitch), torch.cos(roll), torch.sin(roll))
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)


def quaternion_to_ypr(q):
    """SciPy xyz Euler convention, including its gimbal-lock convention."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    sin_pitch = (2*(w*y-z*x)).clamp(-1, 1)
    # Avoid an infinite derivative at +/-pi/2 even on the unselected branch.
    pitch = torch.asin(sin_pitch.clamp(-1+1e-12, 1-1e-12))
    locked = sin_pitch.abs() > 1-5e-15
    roll = torch.atan2(torch.where(locked, torch.zeros_like(x), 2*(w*x+y*z)),
                       torch.where(locked, torch.ones_like(x), 1-2*(x*x+y*y)))
    yaw = torch.atan2(torch.where(locked, torch.zeros_like(x), 2*(w*z+x*y)),
                      torch.where(locked, torch.ones_like(x), 1-2*(y*y+z*z)))
    pitch = torch.where(locked, sin_pitch.sign()*math.pi/2, pitch)
    roll = torch.where(locked, 2*torch.atan2(x, w), roll)
    yaw = torch.where(locked, torch.zeros_like(yaw), yaw)
    # Euler angles are principal-valued as in scipy.spatial.transform.Rotation.
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    return torch.stack((yaw, -pitch, roll), -1) * (180/math.pi)


def slerp(q0, q1, weight):
    """Torch counterpart of baf_roi.upstream_ops.slerp_quaternion_wxyz."""
    q0, q1 = q0.float(), q1.float()
    q0 = q0 / (torch.linalg.vector_norm(q0, dim=-1, keepdim=True)+1e-8)
    q1 = q1 / (torch.linalg.vector_norm(q1, dim=-1, keepdim=True)+1e-8)
    weight = weight.float()[..., None]
    dot = (q0*q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0, 1)
    close = dot > .9995
    theta0 = torch.acos(dot.clamp(max=1-1e-6))
    sin_theta0 = torch.sin(theta0)
    theta = theta0*weight
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta)-dot*sin_theta/(sin_theta0+1e-8)
    s1 = sin_theta/(sin_theta0+1e-8)
    mixed = torch.where(close, (1-weight)*q0+weight*q1, s0*q0+s1*q1)
    return mixed / (torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)+1e-8)


class AlphaFusion(nn.Module):
    def __init__(self, baseline, config=None):
        super().__init__()
        self.config = (config or AlphaConfig()).validate()
        self.gate = HeatmapAlphaGate(self.config)
        self.experts = nn.ModuleDict(baseline.models)
        for parameter in self.experts.parameters():
            parameter.requires_grad_(False)
        self.experts.eval()
        self.mu = baseline.mu
        self.stats = baseline.stats
        self.register_buffer('mean', torch.tensor(self.stats['mean'], dtype=torch.float64))
        self.register_buffer('std', torch.tensor(self.stats['std'], dtype=torch.float64))

    def train(self, mode=True):
        super().train(mode)
        self.experts.eval()  # Frozen BatchNorm/dropout statistics are never updated.
        return self

    def _step(self, history):
        conf = torch.cat((history[..., :3], history[..., 5:6],
                          -history[..., 4:5], history[..., 3:4]), -1).float()
        pose = torch.cat(((history[..., :3]-self.mean)/self.std,
                          ypr_to_quaternion(history[..., 3:])), -1).float()
        raw = self.experts['confidence'](conf)['confidence'].reshape(-1)
        focused = self.experts['focused'](pose)
        explorer = self.experts['explorer'](pose, None)
        # Match the author's float32 de-normalization before geometry arithmetic.
        mean, std = self.mean.float(), self.std.float()
        return raw, focused['p_hat_tr']*std+mean, focused['q_hat_norm'], \
            explorer['p_hat_tr']*std+mean, explorer['q_hat_norm']

    def forward(self, inputs, histories):
        if histories.ndim != 3 or histories.shape[1:] != (30, 6):
            raise ValueError('Expected causal history [B,30,6], meters + yaw,pitch,roll degrees.')
        if not torch.isfinite(histories).all():
            raise ValueError('History must be finite.')
        horizon = inputs['horizon_s']
        if horizon.ndim != 2 or not torch.all(horizon == horizon[:1]):
            raise ValueError('All batch items must use the same horizon list.')
        hs = [float(x) for x in horizon[0].detach().cpu()]
        # The public input is float32 seconds; round back to decimal milliseconds
        # before choosing integer/fractional 30-Hz rollout indices.
        hs_ms = [round(x*1000, 4) for x in hs]
        if any(h < 50 or h > 1000 for h in hs_ms) or not hs_ms:
            raise ValueError('Horizons must lie in [50,1000] ms.')
        steps = math.ceil(max(hs_ms)*30/1000-1e-8)
        delta = self.gate(inputs)
        available = inputs['rf_available'].bool()
        applied = available & (delta != 0)
        penalty = (delta/self.config.max_logit_delta).square()
        if not self.training and not applied.any():
            return {'p': inputs['base_p'], 'q': inputs['base_q'],
                    'gate': inputs['alpha'][:, None].expand_as(horizon),
                    'penalty': penalty[:, None].expand_as(horizon),
                    'applied': applied[:, None].expand_as(horizon), 'logit_delta': delta}
        history = histories.double()
        positions = [history[:, -1, :3]]
        quaternions = [ypr_to_quaternion(history[:, -1, 3:])]
        weights = []
        alpha = inputs['alpha'].float()
        # cuDNN forbids backward through an eval-mode RNN. Native PyTorch permits
        # input gradients while retaining the frozen expert's evaluation behavior.
        context = torch.backends.cudnn.flags(enabled=False) if self.training and history.is_cuda else nullcontext()
        with context:
            for step in range(steps):
                raw, fp, fq, ep, eq = self._step(history)
                prior = alpha if step == 0 else self.mu*raw+(1-self.mu)*alpha
                alpha = correct_alpha(prior, delta)
                p = (1-alpha[:, None])*fp+alpha[:, None]*ep
                q = slerp(fq, eq, alpha)
                positions.append(p.double()); quaternions.append(q.double()); weights.append(alpha)
                generated = torch.cat((p.double(), quaternion_to_ypr(q.double())), -1)
                history = torch.cat((history[:, 1:], generated[:, None]), 1)
        ps, qs, gates = [], [], []
        for milliseconds in hs_ms:
            t = milliseconds*30/1000
            lo, hi = int(math.floor(t+1e-8)), int(math.ceil(t-1e-8))
            fraction = max(0., min(1., t-lo))
            ps.append((1-fraction)*positions[lo]+fraction*positions[hi])
            qs.append(slerp(quaternions[lo], quaternions[hi], alpha.new_full(alpha.shape, fraction)))
            # Corrected forecasts report the upper-step alpha, not endpoint mixture.
            gates.append(weights[hi-1])
        p, q = torch.stack(ps, 1), torch.stack(qs, 1).double()
        use = available if self.training else applied
        return {'p': torch.where(use[:, None, None], p, inputs['base_p'].double()),
                'q': torch.where(use[:, None, None], q, inputs['base_q']),
                # Match the original BAF diagnostic for fallback rows in every
                # batch composition; no applied correction means observed alpha_t.
                'gate': torch.where(applied[:, None], torch.stack(gates, 1),
                                    inputs['alpha'][:, None].expand_as(horizon)),
                'penalty': penalty[:, None].expand_as(horizon),
                'applied': applied[:, None].expand_as(horizon), 'logit_delta': delta,
                'rollout_alpha': torch.stack(weights, 1),
                'rollout_positions_m': torch.stack(positions, 1),
                'rollout_quaternions_wxyz': torch.stack(quaternions, 1)}

    def specification(self):
        return {'method': 'baf_roi_heatmap_alpha', 'config': asdict(self.config),
                'feature_names': list(FEATURE_NAMES),
                'trainable_parameters': sum(p.numel() for p in self.gate.parameters()),
                'frozen_parameters': sum(p.numel() for p in self.experts.parameters()),
                'alpha_convention': '0=Focused, 1=Explorer',
                'gate_diagnostic': 'Applied: alpha at upper rollout step of each horizon. Fallback: observed alpha_t, as in the BAF report.',
                'state': 'Observed HMD-only EMA unchanged; RF correction is local to one forecast.',
                'correction': 'Same bounded logit delta at each autoregressive fusion step.'}
