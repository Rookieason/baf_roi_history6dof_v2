"""python -m baf_roi_heatmap --help"""
import argparse
import json
import torch
from .contracts import InputConfig, ModelConfig, read_json, read_scene_config


def parser():
    root = argparse.ArgumentParser(description='Causal CSI-assisted ROI experiments; no real-data gains are assumed.')
    root.add_argument('--threads', type=int, default=2)
    commands = root.add_subparsers(dest='command', required=True)
    p = commands.add_parser('manifest-template'); p.add_argument('--cache', required=True); p.add_argument('--maps-directory', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('generate'); p.add_argument('--raw', required=True); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--cache', required=True); p.add_argument('--baseline', required=True)
    p.add_argument('--rf-manifest', required=True); p.add_argument('--output', required=True); p.add_argument('--input-config')
    p.add_argument('--horizons', type=float, nargs='+', default=[50, 100, 150, 200, 250, 400, 600, 800, 1000])
    p.add_argument('--scene-config'); p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=64)
    p = commands.add_parser('train'); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    p.add_argument('--method', choices=['residual', 'expert_gate'], default='residual'); p.add_argument('--modality', choices=['rf', 'hmd'], default='rf')
    p.add_argument('--ablation', choices=['full', 'spatial_mean', 'last_frame', 'angle_only', 'delay_only', 'abs_change', 'metadata_only'], default='full')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001); p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cpu')
    p.add_argument('--roi-weight', type=float, default=.25)
    p.add_argument('--hidden', type=int, default=32); p.add_argument('--max-translation-m', type=float, default=.2)
    p.add_argument('--max-rotation-deg', type=float, default=20.); p.add_argument('--max-logit-delta', type=float, default=2.)
    p = commands.add_parser('evaluate'); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    choice = p.add_mutually_exclusive_group(required=True); choice.add_argument('--checkpoint'); choice.add_argument('--baseline', choices=['baf', 'focused', 'explorer', 'fixed'])
    p.add_argument('--split', choices=['valid', 'test'], default='test'); p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--margins', type=float, nargs='+', default=[0.]); p.add_argument('--delivery-delay-ms', type=float, default=0.); p.add_argument('--policy')
    p = commands.add_parser('policy'); p.add_argument('--validation', required=True); p.add_argument('--output', required=True); p.add_argument('--coverage', type=float, default=.95)
    p = commands.add_parser('compare'); p.add_argument('--left', required=True); p.add_argument('--right', required=True); p.add_argument('--output', required=True)
    p.add_argument('--bootstrap-samples', type=int, default=2000); p.add_argument('--seed', type=int, default=42)
    p = commands.add_parser('oracle'); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'valid', 'test'], default='valid'); p.add_argument('--grid-points', type=int, default=21)
    p = commands.add_parser('benchmark'); p.add_argument('--prepared', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--count', type=int, default=50)
    p = commands.add_parser('plot'); p.add_argument('--evaluations', nargs='+', required=True); p.add_argument('--output', required=True)
    return root


def main(argv=None):
    a = parser().parse_args(argv)
    if a.threads < 1: raise ValueError('--threads must be positive.')
    torch.set_num_threads(a.threads)
    if a.command == 'manifest-template':
        from .manifest import manifest_template
        result = manifest_template(a.cache, a.maps_directory, a.output)
    elif a.command == 'generate':
        from .music import generate
        result = generate(a.raw, a.config, a.output)
    elif a.command == 'prepare':
        from .prepare import prepare
        cfg = InputConfig(**read_json(a.input_config)) if a.input_config else InputConfig()
        result = prepare(a.cache, a.baseline, a.rf_manifest, a.output, a.horizons, cfg, a.device,
                         a.batch_size, read_scene_config(a.scene_config) if a.scene_config else None)
        result = {'experiment_id': result['experiment_id'], 'records': len(result['records'])}
    elif a.command == 'train':
        from .train import train
        result = {'checkpoint': train(a.prepared, a.output, ModelConfig(a.method, a.modality, a.ablation, a.hidden, a.max_translation_m, a.max_rotation_deg, a.max_logit_delta),
                                     a.epochs, a.batch_size, a.lr, a.seed, a.device, a.roi_weight)}
    elif a.command == 'evaluate':
        from .evaluation import evaluate
        result = evaluate(a.prepared, a.output, a.checkpoint, a.baseline or 'baf', a.split, a.device,
                          a.batch_size, a.margins, a.delivery_delay_ms/1000, a.policy)
    elif a.command == 'policy':
        from .evaluation import select_policy
        result = select_policy(a.validation, a.output, a.coverage)
    elif a.command == 'compare':
        from .evaluation import compare
        result = compare(a.left, a.right, a.output, a.bootstrap_samples, a.seed)
    elif a.command == 'oracle':
        from .diagnostics import oracle
        result = oracle(a.prepared, a.output, a.split, a.grid_points)
    elif a.command == 'benchmark':
        from .runtime import benchmark
        result = benchmark(a.prepared, a.checkpoint, a.output, a.device, a.count)
    else:
        from .diagnostics import plot
        result = plot(a.evaluations, a.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
