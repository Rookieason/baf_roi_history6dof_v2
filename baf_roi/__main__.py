from pathlib import Path
import argparse
import json
import sys
import torch
from .config import load_config, dump_json


def read_scene_config(path):
    path = Path(path).resolve()
    obj = json.loads(path.read_text(encoding='utf-8'))
    cfg = obj.get('scene', obj)
    for name in ('adaptive_config_path', 'partition_centers_file'):
        if cfg.get(name):
            p = Path(cfg[name]).expanduser()
            cfg[name] = str(p.resolve() if p.is_absolute() else (path.parent / p).resolve())
    return cfg


def main():
    parser = argparse.ArgumentParser(description='BAF-Pose -> ROI, history 6DoF only.')
    parser.add_argument('--threads', type=int, default=4, help='CPU intra-op threads, not training data workers.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('train'); p.add_argument('--cache', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto'); p.add_argument('--smoke-epochs', type=int, default=None,
        help='Explicit testing override for ALL four stages. Do not call this a full experiment.')
    p = sub.add_parser('evaluate'); p.add_argument('--cache', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--scene-config'); p.add_argument('--device', default='auto')
    p.add_argument('--split', default='test', choices=['valid', 'test']); p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--no-save-masks', action='store_true')
    p.add_argument('--horizons-ms', nargs='+', type=float,
                   help='Evaluation-only horizons in [50,1000] ms; defaults to the cache manifest.')
    p = sub.add_parser('predict'); p.add_argument('--checkpoint', required=True); p.add_argument('--history', required=True)
    p.add_argument('--output', required=True); p.add_argument('--scene-config'); p.add_argument('--device', default='auto')
    p.add_argument('--horizons-ms', nargs='+', type=float, default=[50, 100, 150, 200, 250])
    sub.add_parser('audit')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    torch.set_num_threads(args.threads)
    device = getattr(args, 'device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.command == 'audit':
        from .audit import verify
        print(json.dumps(verify(), indent=2))
    elif args.command == 'prepare':
        from .data import prepare
        prepare(load_config(args.config), args.output)
    elif args.command == 'train':
        from .train import train
        print(train(args.cache, args.output, device, args.smoke_epochs))
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        report = evaluate(args.cache, args.checkpoint, args.output, device, args.split, args.batch_size,
                          read_scene_config(args.scene_config) if args.scene_config else None, not args.no_save_masks,
                          horizons_ms=args.horizons_ms)
        print(json.dumps(report['metrics'], indent=2))
    elif args.command == 'predict':
        from .runtime import Predictor
        from .roi import load_scene
        raw = json.loads(Path(args.history).read_text(encoding='utf-8'))
        predictor = Predictor.load(args.checkpoint, device)
        result = predictor.predict_observations(raw['timestamps_s'], raw['positions'], raw['rotations_deg'], args.horizons_ms)
        report = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in result.items()}
        report['position_unit_output'] = 'm'
        report['rotation_order_output'] = 'yaw,pitch,roll; +X forward, +Z up'
        if args.scene_config:
            scene = load_scene(read_scene_config(args.scene_config))
            report['roi_mask'] = scene.mask(result['positions_m'], result['quaternions_wxyz']).astype(int).tolist()
            report['partition_ids'] = scene.partition_ids
            report['scene_id'] = scene.snapshot()['scene_id']
        dump_json(args.output, report)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
