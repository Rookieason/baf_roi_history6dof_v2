"""Causal CSI residuals inside BAF, before either expert predicts a pose.

The original Focused LSTM and Explorer TCN produce their latent features. A
small CSI-conditioned residual is added to each latent, then the ORIGINAL
translation/rotation heads produce the next pose. The original confidence EMA
and pose fusion run once per autoregressive step. No completed BAF forecast is
an input to this model and there is no output-space pose correction.

The BAF parameters remain frozen for the controlled first experiment, while
autograd traverses their operations and the complete autoregressive rollout.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from baf_roi.config import validate_horizons
from baf_roi.models import focused as quaternion_ops
from baf_roi_heatmap.model import ablate_maps
from baf_roi_heatmap_alpha.model import (
    frozen_rnn_input_grad, quaternion_to_ypr, slerp, ypr_to_quaternion,
)


ABLATIONS = ('full', 'metadata_only', 'last_frame', 'spatial_mean',
             'angle_only', 'delay_only', 'abs_change')


@dataclass(frozen=True)
class ResidualConfig:
    hidden: int = 32
    bottleneck: int = 32
    max_relative_delta: float = .1
    modality: str = 'rf'
    ablation: str = 'full'

    def validate(self):
        for name in ('hidden', 'bottleneck'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f'{name} must be an integer of at least 4.')
        if (isinstance(self.max_relative_delta, bool)
                or not math.isfinite(self.max_relative_delta)
                or not 0 < self.max_relative_delta <= 1):
            raise ValueError('max_relative_delta must lie in (0,1].')
        if self.modality not in ('rf', 'hmd'):
            raise ValueError('modality must be rf or hmd.')
        if self.ablation not in ABLATIONS:
            raise ValueError(f'ablation must be one of {ABLATIONS}.')
        return self


class CausalHeatmapEncoder(nn.Module):
    """Retain a 2x2 spatial layout and the order of the valid observed RF frames.

    Input maps are already physically aligned and normalized using training
    recordings only. Invalid frames are sanitized BEFORE the CNN, including
    their timing, so masked NaNs cannot contaminate values or gradients.
    """
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.temporal = nn.GRUCell(16*2*2+2, hidden)

    def forward(self, maps, mask, timing):
        if (maps.ndim != 5 or maps.shape[2] != 3 or not maps.shape[1]
                or min(maps.shape[-2:]) < 2):
            raise ValueError('maps must be nonempty [B,T,3,height,width].')
        if mask.shape != maps.shape[:2] or timing.shape != (*maps.shape[:2], 2):
            raise ValueError('RF mask/timing must match the map batch and frame count.')
        mask = mask.bool()
        maps = torch.where(mask[:, :, None, None, None], maps, torch.zeros_like(maps))
        timing = torch.where(mask[..., None], timing, torch.zeros_like(timing))
        if not torch.isfinite(maps).all() or not torch.isfinite(timing).all():
            raise ValueError('Valid RF maps and their timing must be finite.')
        dtype = self.spatial[0].weight.dtype
        maps, timing = maps.to(dtype=dtype), timing.to(dtype=dtype)
        batch, frames = maps.shape[:2]
        spatial = self.spatial(maps.reshape(batch*frames, *maps.shape[2:]))
        features = torch.cat((spatial.reshape(batch, frames, -1), timing), -1)
        state = features.new_zeros((batch, self.hidden))
        for step in range(frames):
            update = self.temporal(features[:, step], state)
            state = torch.where(mask[:, step, None], update, state)
        return state


class LatentResidualHead(nn.Module):
    """A bounded relative latent residual, initialized as the identity path.

    scale = max(RMS(z).detach(), 1e-6) accounts for the different scales of
    Focused (LayerNorm) and Explorer (no LayerNorm). The per-coordinate bound
    is beta*scale. Except for the explicit near-zero floor, this also gives
    ||delta_z||_2 <= beta*||z||_2. This is NOT a meter/degree error bound.
    """
    def __init__(self, latent_dim, config):
        super().__init__()
        self.max_relative_delta = config.max_relative_delta
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.head = nn.Sequential(
            nn.Linear(latent_dim+config.hidden+1, config.bottleneck), nn.Tanh(),
            nn.Linear(config.bottleneck, latent_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, latent, context, elapsed_s, available):
        elapsed = latent.new_full((len(latent), 1), float(elapsed_s))
        features = torch.cat((self.norm(latent), context.to(latent), elapsed), -1)
        normalized = torch.tanh(self.head(features))
        normalized = torch.where(available[:, None], normalized, torch.zeros_like(normalized))
        # Detach only the residual scale, never the latent/forecast computation.
        scale = latent.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        delta = self.max_relative_delta*scale*normalized
        return latent+delta, {
            'penalty': normalized.square().mean(-1),
            'applied': (delta != 0).any(-1),
            'relative_norm': self.max_relative_delta*normalized.square().mean(-1).sqrt(),
        }


class ResidualFeatures(nn.Module):
    """All and only the trainable CSI encoder and internal residual heads."""
    def __init__(self, config, latent_dims):
        super().__init__()
        self.config = config
        self.encoder = CausalHeatmapEncoder(config.hidden)
        self.heads = nn.ModuleDict({name: LatentResidualHead(size, config)
                                    for name, size in latent_dims.items()})

    def encode(self, inputs):
        maps, mask, timing = inputs['maps'], inputs['mask'].bool(), inputs['timing']
        if maps.ndim != 5 or mask.shape != maps.shape[:2] or timing.shape != (*maps.shape[:2], 2):
            raise ValueError('RF maps, mask and timing have incompatible shapes.')
        # Sanitize before ablation as spatial averaging must not see padding NaNs.
        maps = torch.where(mask[:, :, None, None, None], maps, torch.zeros_like(maps))
        timing = torch.where(mask[..., None], timing, torch.zeros_like(timing))
        if self.config.modality == 'hmd':
            # Same structure/capacity; CSI and RF acquisition metadata are absent.
            maps, mask, timing = torch.zeros_like(maps), torch.zeros_like(mask), torch.zeros_like(timing)
        else:
            maps, mask = ablate_maps(maps, mask, self.config.ablation)
        return self.encoder(maps, mask, timing)

    def is_zero(self):
        return all(not torch.count_nonzero(parameter).item()
                   for head in self.heads.values()
                   for parameter in head.head[-1].parameters())


class InternalResidualFusion(nn.Module):
    """One BAF rollout with CSI acting before its expert output heads."""
    def __init__(self, baseline, config=None):
        super().__init__()
        self.config = (config or ResidualConfig()).validate()
        self.experts = nn.ModuleDict(baseline.models)
        for parameter in self.experts.parameters():
            parameter.requires_grad_(False)
        self.experts.eval()
        self.mu, self.stats = float(baseline.mu), baseline.stats
        if not math.isfinite(self.mu) or not 0 <= self.mu <= 1:
            raise ValueError('The original BAF temporal_mu must be in [0,1].')
        self.register_buffer('mean', torch.tensor(self.stats['mean'], dtype=torch.float64))
        self.register_buffer('std', torch.tensor(self.stats['std'], dtype=torch.float64))
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or not (self.std > 0).all()):
            raise ValueError('BAF translation normalization must be finite, three-dimensional and positive.')
        dims = {name: self.experts[name].translation_head[0].in_features
                for name in ('focused', 'explorer')}
        self.residual = ResidualFeatures(self.config, dims)

    def train(self, mode=True):
        super().train(mode)
        self.experts.eval()  # Preserve pretrained dropout/normalization behavior.
        return self

    def is_zero_residual(self):
        """Sufficient exact identity check for the online runtime's baseline path.

        Nonzero parameters can sometimes cancel for a particular input; that
        case deliberately still runs the single assisted forecast.
        """
        return self.residual.is_zero()

    def _expert_step(self, name, pose, context, elapsed_s, available):
        expert = self.experts[name]
        if name == 'focused':
            with frozen_rnn_input_grad(expert, self.training and pose.is_cuda and torch.is_grad_enabled()):
                _, (hidden, _) = expert.encoder(pose)
            latent = hidden[-1]
            if expert.use_layer_norm:
                latent = expert.latent_norm(latent)
        else:
            if expert.config.aux_dim != 0:
                raise ValueError('The reproduced BAF Explorer requires aux_dim=0.')
            encoded = expert.tcn(expert.input_proj(pose.transpose(1, 2)))
            latent = encoded[:, :, -1]

        # The sole CSI insertion point: before BOTH original expert output heads.
        assisted, diagnostics = self.residual.heads[name](latent, context, elapsed_s, available)
        p_normalized = pose[:, -1, :3]+expert.translation_head(assisted)
        current_q = quaternion_ops.quaternion_normalize(pose[:, -1, 3:7])
        delta_q = quaternion_ops.quaternion_normalize(expert.rotation_head(assisted))
        q = quaternion_ops.quaternion_multiply(delta_q, current_q)
        q = quaternion_ops.quaternion_normalize(q)
        q = quaternion_ops.quaternion_align_sign(q, current_q)
        p = p_normalized*self.std.float()+self.mean.float()
        return p, q, diagnostics

    def _step(self, history, context, elapsed_s, available):
        confidence_input = torch.cat((history[..., :3], history[..., 5:6],
                                      -history[..., 4:5], history[..., 3:4]), -1).float()
        pose = torch.cat(((history[..., :3]-self.mean)/self.std,
                          ypr_to_quaternion(history[..., 3:])), -1).float()
        raw = self.experts['confidence'](confidence_input)['confidence'].reshape(-1)
        fp, fq, focused = self._expert_step('focused', pose, context, elapsed_s, available)
        ep, eq, explorer = self._expert_step('explorer', pose, context, elapsed_s, available)
        return raw, fp, fq, ep, eq, focused, explorer

    def forward(self, inputs, histories):
        """Consume only observed history/RF, requested horizons, and observed EMA.

        `base_p`, `base_q`, targets, labels and completed expert trajectories are
        intentionally never read. Forward always executes one rollout, including
        during training with zero-initialized residuals (gradients stay alive).
        """
        if (histories.ndim != 3 or histories.shape[1:] != (30, 6)
                or not len(histories) or not torch.isfinite(histories).all()):
            raise ValueError('Expected finite, nonempty HMD history [B,30,6].')
        batch = len(histories)
        horizon = inputs['horizon_s']
        if (horizon.ndim != 2 or horizon.shape[0] != batch or not horizon.shape[1]
                or not torch.isfinite(horizon).all()
                or not torch.all(horizon == horizon[:1])):
            raise ValueError('Every batch item must use the same nonempty finite horizon list.')
        # Recover decimal milliseconds from float32 seconds before step indexing.
        hs = validate_horizons([round(float(v)*1000, 4) for v in horizon[0].detach().cpu()])
        steps = math.ceil(max(hs)*30/1000-1e-8)
        alpha = inputs['alpha'].float()
        available = inputs['rf_available'].bool()
        if (alpha.shape != (batch,) or available.shape != (batch,)
                or not torch.isfinite(alpha).all() or not ((alpha >= 0) & (alpha <= 1)).all()):
            raise ValueError('Need one finite observed alpha in [0,1] and RF availability per history.')
        if inputs['maps'].shape[0] != batch:
            raise ValueError('RF and HMD batch sizes must match.')
        if self.config.modality == 'hmd':
            available = torch.ones_like(available)
        context = self.residual.encode(inputs)  # Exactly once per observed origin.
        history = histories.double()
        positions = [history[:, -1, :3]]
        quaternions = [ypr_to_quaternion(history[:, -1, 3:])]
        weights, penalties, changed, relative_norms = [], [], [], []
        initial_alpha = alpha
        for step in range(steps):
            raw, fp, fq, ep, eq, focused, explorer = self._step(history, context, (step+1)/30., available)
            if step == 0:
                raw_at_t = raw
            else:
                alpha = self.mu*raw+(1-self.mu)*alpha
            if (not torch.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all()):
                raise FloatingPointError('BAF produced invalid confidence during rollout.')
            p = (1-alpha[:, None])*fp+alpha[:, None]*ep
            q = slerp(fq, eq, alpha)
            if (not torch.isfinite(p).all() or not torch.isfinite(q).all()
                    or not (torch.linalg.vector_norm(q, dim=-1) > 1e-8).all()):
                raise FloatingPointError('The assisted BAF produced an invalid pose.')
            positions.append(p.double())
            quaternions.append(q.double())
            weights.append(alpha)
            penalties.append((focused['penalty']+explorer['penalty'])/2)
            changed.append(focused['applied'] | explorer['applied'])
            relative_norms.append(torch.stack((focused['relative_norm'], explorer['relative_norm']), -1))
            generated = torch.cat((p.double(), quaternion_to_ypr(q.double())), -1)
            history = torch.cat((history[:, 1:], generated[:, None]), 1)

        cumulative_penalty = torch.stack(penalties, 1).cumsum(1)
        cumulative_changed = torch.stack(changed, 1).to(torch.int64).cumsum(1) > 0
        ps, qs, gates, horizon_penalties, applied = [], [], [], [], []
        for milliseconds in hs:
            t = milliseconds*30/1000
            lo, hi = int(math.floor(t+1e-8)), int(math.ceil(t-1e-8))
            fraction = max(0., min(1., t-lo))
            ps.append((1-fraction)*positions[lo]+fraction*positions[hi])
            qs.append(slerp(quaternions[lo], quaternions[hi], alpha.new_full(alpha.shape, fraction)).double())
            gates.append(weights[hi-1])
            horizon_penalties.append(cumulative_penalty[:, hi-1]/hi)
            applied.append(cumulative_changed[:, hi-1])
        applied = torch.stack(applied, 1)
        # Shared evaluation calls an unapplied gate the observed-origin alpha;
        # the actual per-step trajectory remains available in rollout_alpha.
        gate = torch.where(applied, torch.stack(gates, 1),
                           initial_alpha[:, None].expand_as(horizon))
        return {
            'p': torch.stack(ps, 1), 'q': torch.stack(qs, 1),
            'gate': gate, 'penalty': torch.stack(horizon_penalties, 1),
            'applied': applied, 'raw_confidence_t': raw_at_t,
            'alpha_t': initial_alpha, 'rollout_alpha': torch.stack(weights, 1),
            'rollout_positions_m': torch.stack(positions, 1),
            'rollout_quaternions_wxyz': torch.stack(quaternions, 1),
            'rollout_relative_residual_norm': torch.stack(relative_norms, 1),
        }

    def specification(self):
        return {
            'method': 'baf_roi_heatmap_residual', 'config': asdict(self.config),
            'trainable_parameters': sum(p.numel() for p in self.residual.parameters()),
            'frozen_parameters': sum(p.numel() for p in self.experts.parameters()),
            'insertion': 'Expert latent features, before original translation and rotation heads at every step.',
            'rf_encoder': 'Shared 3-channel CNN with a 2x2 spatial grid and masked temporal GRU; once per observed origin.',
            'residual': 'z_assisted = z + beta * max(RMS(z).detach(), 1e-6) * tanh(A([LN(z), RF_context, elapsed_s])).',
            'penalty': 'Mean squared normalized latent residual, averaged across both experts and steps up to each horizon.',
            'bound': 'Relative feature bound, not a bound on position/rotation error or full-rollout drift.',
            'initialization': 'Zero final residual projections reproduce the original expert computation.',
            'alpha_convention': '0=Focused, 1=Explorer; original confidence and EMA formula unchanged.',
            'gate_diagnostic': 'Applied: alpha at upper rollout step. Fallback: observed-origin alpha; rollout_alpha always reports actual step weights.',
            'state': 'Only observed HMD confidence EMA is persistent. Predicted poses and RF context are forecast-local.',
            'gradient': 'Only residual modules update; gradients traverse original frozen heads, encoders and rollout.',
            'hmd_control': 'Same module structure with maps/mask/timing all zero, residual active at every origin.',
            'input_exclusions': ['completed_BAF_forecast', 'future_pose', 'future_RF', 'ROI_labels'],
        }
