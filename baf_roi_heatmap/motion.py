"""CSI-conditioned motion experts INSIDE the BAF autoregressive loop.

The original LSTM/TCN encoders initialize head-motion expertise. Joint HMD/CSI
features condition their latent features BEFORE the next-pose heads. The heads,
conditioners, RF encoder and router learn from future pose/ROI, not from the old
head-only binary labels. Additional experts have no assigned body-part semantics.
No completed BAF forecast is corrected. Cached BAF outputs serve only fallback
and training/validation comparisons.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import torch
from torch import nn

from baf_roi.config import validate_horizons
from baf_roi_heatmap_alpha.model import (frozen_rnn_input_grad, quaternion_to_ypr,
                                        slerp, ypr_to_quaternion)
from .contracts import ModelConfig
from .model import ablate_maps, quaternion_product, unit_quaternion


class MotionContext(nn.Module):
    """Encode observed RF once; recompute head-motion context at each step."""
    def __init__(self, hidden):
        super().__init__()
        self.hmd = nn.GRU(7, hidden, batch_first=True)
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.temporal = nn.GRUCell(66, hidden)
        self.reliability = nn.Sequential(nn.Linear(2*hidden+3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.joint = nn.Sequential(nn.Linear(2*hidden+2, hidden), nn.Tanh())

    def radio(self, inputs, config):
        maps, mask = ablate_maps(inputs['maps'], inputs['mask'].bool(), config.ablation)
        batch, frames = maps.shape[:2]
        if config.modality == 'hmd':
            return maps.new_zeros(batch, config.hidden), maps.new_zeros(batch, 2)
        # Invalid pixels/timestamps cannot influence valid frames or gradients.
        maps = torch.where(mask[:, :, None, None, None], maps, torch.zeros_like(maps))
        timing = torch.where(mask[:, :, None], inputs['timing'], torch.zeros_like(inputs['timing']))
        features = self.spatial(maps.reshape(-1, *maps.shape[2:])).reshape(batch, frames, -1)
        features = torch.cat((features, timing), -1)
        state = features.new_zeros(batch, config.hidden)
        for j in range(frames):
            update = self.temporal(features[:, j], state)
            state = torch.where(mask[:, j, None], update, state)
        last = frames-1-mask.flip(1).long().argmax(1)
        age = timing[torch.arange(batch, device=maps.device), last, 0]
        quality = torch.stack((mask.float().mean(1), age), -1)
        available = inputs['rf_available'].bool()
        return (torch.where(available[:, None], state, torch.zeros_like(state)),
                torch.where(available[:, None], quality, torch.zeros_like(quality)))

    def forward(self, pose, radio, quality, elapsed, config):
        _, state = self.hmd(pose)
        head = state[-1]
        step = head.new_full((len(head), 1), elapsed)
        if config.modality == 'hmd':
            reliability = head.new_zeros(len(head))
        else:
            reliability = torch.sigmoid(self.reliability(torch.cat((head, radio, quality, step), -1))).squeeze(-1)
        joint = self.joint(torch.cat((head, radio*reliability[:, None], step, quality[:, :1]), -1))
        return joint, reliability


class ConditionedExpert(nn.Module):
    """Warm-start an author's expert; condition latent BEFORE pose reconstruction.

    Encoders remain frozen to limit capacity on the small recording cohort.
    Joint mode trains latent FiLM and native pose heads. Router-only freezes the
    whole expert and bypasses FiLM, isolating improvements from routing alone.
    """
    def __init__(self, original, family, hidden, conditioning):
        super().__init__()
        self.family, self.conditioning = family, conditioning
        self.branch = deepcopy(original)
        width = self.branch.config.hidden_dim
        self.film = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2*width))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        for parameter in self.branch.parameters():
            parameter.requires_grad_(False)
        if conditioning == 'joint':
            for head in (self.branch.translation_head, self.branch.rotation_head):
                for parameter in head.parameters():
                    parameter.requires_grad_(True)
        else:
            for parameter in self.film.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Deterministic author dropout, including while learning native heads.
        self.branch.eval()
        return self

    def forward(self, pose, context):
        if self.family == 'focused':
            with frozen_rnn_input_grad(self.branch.encoder,
                                      self.training and pose.is_cuda and torch.is_grad_enabled()):
                _, (state, _) = self.branch.encoder(pose)
            latent = state[-1]
            if self.branch.use_layer_norm:
                latent = self.branch.latent_norm(latent)
        else:
            latent = self.branch.tcn(self.branch.input_proj(pose.transpose(1, 2)))[:, :, -1]
        penalty = context.new_zeros(len(pose))
        if self.conditioning == 'joint':
            scale, shift = self.film(context).chunk(2, -1)
            scale, shift = .1*torch.tanh(scale), .1*torch.tanh(shift)
            latent = latent*(1+scale)+shift
            penalty = scale.square().mean(-1)+shift.square().mean(-1)
        p = pose[:, -1, :3]+self.branch.translation_head(latent)
        dq = unit_quaternion(self.branch.rotation_head(latent))
        q = unit_quaternion(quaternion_product(dq, unit_quaternion(pose[:, -1, 3:])))
        return p, q, penalty


def fuse_experts(positions, quaternions, weights):
    """LERP positions and deterministic, sign-invariant cumulative SLERP.

    For two experts this is the author's BAF fusion. For K>2 the declared order
    is Focused, Explorer, then latent experts; this is not a Euclidean quaternion
    average or an order-independent quaternion barycenter.
    """
    p = (positions*weights[..., None]).sum(1)
    q, mass = quaternions[:, 0], weights[:, 0]
    for j in range(1, weights.shape[1]):
        total = mass+weights[:, j]
        q = slerp(q, quaternions[:, j], weights[:, j]/total.clamp_min(1e-8))
        mass = total
    return p, q


class MotionFusion(nn.Module):
    def __init__(self, baseline, config=None):
        super().__init__()
        self.config = (config or ModelConfig(method='motion_moe')).validate()
        if self.config.method != 'motion_moe':
            raise ValueError('MotionFusion requires method=motion_moe.')
        self.stats, self.mu = baseline.stats, baseline.mu
        self.confidence = deepcopy(baseline.models['confidence'])
        for parameter in self.confidence.parameters():
            parameter.requires_grad_(False)
        self.context = MotionContext(self.config.hidden)
        self.motion_experts = nn.ModuleList([
            ConditionedExpert(baseline.models[family], family, self.config.hidden, self.config.conditioning)
            for family in ['focused']+['explorer']*(self.config.experts-1)])
        self.router = nn.Sequential(nn.Linear(self.config.hidden+2, self.config.hidden), nn.Tanh(),
                                    nn.Linear(self.config.hidden, self.config.experts))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.register_buffer('mean', torch.tensor(self.stats['mean'], dtype=torch.float64))
        self.register_buffer('std', torch.tensor(self.stats['std'], dtype=torch.float64))
        # Epoch zero is an explicit, exactly preserved BAF candidate. Training
        # switches this off; its value is persisted in checkpoints.
        self.register_buffer('baseline_only', torch.tensor(True))
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.confidence.eval()
        return self

    def forward(self, inputs, histories):
        if histories.ndim != 3 or histories.shape[1:] != (30, 6) or not torch.isfinite(histories).all():
            raise ValueError('Expected finite causal history [B,30,6], meters + yaw,pitch,roll.')
        horizons = inputs['horizon_s']
        if horizons.ndim != 2 or not torch.all(horizons == horizons[:1]):
            raise ValueError('All batch items must share horizons.')
        hs = [round(float(v)*1000, 4) for v in horizons[0].detach().cpu()]
        validate_horizons(hs)
        batch, count = len(histories), len(hs)
        use = (inputs['rf_available'].bool() if self.config.modality == 'rf'
               else torch.ones(batch, dtype=torch.bool, device=histories.device))
        if bool(self.baseline_only):
            use = torch.zeros_like(use)
        prior = inputs['alpha'].float()
        fallback_weights = torch.stack([1-prior, prior]+[torch.zeros_like(prior)]*(self.config.experts-2), -1)
        if not use.any():
            return {'p': inputs['base_p'], 'q': inputs['base_q'],
                    'gate': prior[:, None].expand(-1, count),
                    'penalty': prior.new_zeros(batch, count),
                    'applied': use[:, None].expand(-1, count),
                    'expert_weights': fallback_weights[:, None].expand(-1, count, -1),
                    'rf_reliability': prior.new_zeros(batch, count)}
        radio, quality = self.context.radio(inputs, self.config)
        history = histories.double()
        ps, qs = [history[:, -1, :3]], [ypr_to_quaternion(history[:, -1, 3:])]
        weights, penalties, reliabilities = [], [], []
        alpha = prior
        for step in range(math.ceil(max(hs)*30/1000-1e-8)):
            conf = torch.cat((history[..., :3], history[..., 5:6], -history[..., 4:5], history[..., 3:4]), -1).float()
            pose = torch.cat(((history[..., :3]-self.mean)/self.std, ypr_to_quaternion(history[..., 3:])), -1).float()
            raw = self.confidence(conf)['confidence'].reshape(-1)
            alpha = prior if step == 0 else self.mu*raw+(1-self.mu)*alpha
            context, reliability = self.context(pose, radio, quality, step/30, self.config)
            candidate = [expert(pose, context) for expert in self.motion_experts]
            p_candidates = torch.stack([p for p, _, _ in candidate], 1)*self.std.float()+self.mean.float()
            q_candidates = torch.stack([q for _, q, _ in candidate], 1)
            # Split the original Explorer prior across its warm-start descendants.
            # Endpoints remain finite; this is a routing prior, not a calibrated
            # behavioral probability or a forced old binary target.
            a = alpha.clamp(1e-5, 1-1e-5)
            logits = torch.stack([torch.log1p(-a)]+[torch.log(a)-math.log(self.config.experts-1)]*(self.config.experts-1), -1)
            offset = self.config.max_logit_delta*torch.tanh(self.router(torch.cat((context, alpha[:, None],
                                                       alpha.new_full((batch, 1), step/30)), -1)))
            weight = torch.softmax(logits+offset, -1)
            p, q = fuse_experts(p_candidates, q_candidates, weight)
            alpha = weight[:, 1:].sum(-1)
            ps.append(p.double()); qs.append(q.double()); weights.append(weight)
            penalties.append(torch.stack([v for _, _, v in candidate], -1).mean(-1)+offset.square().mean(-1))
            reliabilities.append(reliability)
            generated = torch.cat((p.double(), quaternion_to_ypr(q.double())), -1)
            history = torch.cat((history[:, 1:], generated[:, None]), 1)
        positions, rotations, horizon_weights, horizon_reliability = [], [], [], []
        for h in hs:
            t = h*30/1000
            lo, hi = math.floor(t+1e-8), math.ceil(t-1e-8)
            fraction = max(0., min(1., t-lo))
            positions.append((1-fraction)*ps[lo]+fraction*ps[hi])
            rotations.append(slerp(qs[lo], qs[hi], prior.new_full((batch,), fraction)).double())
            horizon_weights.append(weights[hi-1]); horizon_reliability.append(reliabilities[hi-1])
        w = torch.stack(horizon_weights, 1)
        w = torch.where(use[:, None, None], w, fallback_weights[:, None])
        return {'p': torch.where(use[:, None, None], torch.stack(positions, 1), inputs['base_p'].double()),
                'q': torch.where(use[:, None, None], torch.stack(rotations, 1), inputs['base_q'].double()),
                'gate': w[..., 1:].sum(-1),
                'penalty': torch.stack(penalties, 1).mean(1)[:, None].expand(-1, count)*use[:, None],
                'applied': use[:, None].expand(-1, count), 'expert_weights': w,
                'rf_reliability': torch.stack(horizon_reliability, 1)*use[:, None],
                'rollout_positions_m': torch.stack(ps, 1), 'rollout_quaternions_wxyz': torch.stack(qs, 1),
                'rollout_expert_weights': torch.stack(weights, 1),
                'rollout_alpha': torch.stack(weights, 1)[..., 1:].sum(-1)}

    def specification(self):
        return {'config': asdict(self.config),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'parameters': sum(p.numel() for p in self.parameters()),
                'expert_names': ['focused', 'explorer']+[f'latent_motion_{j}' for j in range(2, self.config.experts)],
                'training': 'Frozen author temporal encoders; joint mode trains native pose heads and latent conditioners.',
                'semantics': 'Warm-start families, not validated full-body action or body-part classes.',
                'rf': 'Observed origin window only; no generated/future RF; context reused with elapsed forecast time.',
                'fusion': 'Position weighted sum; ordered cumulative SLERP; each fused pose enters the next step.'}
