"""Optional package-level CLI; the common roi_experiments runner is preferred."""
import argparse
import json
import torch
from .model import AlphaConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description='Heatmap changes only actual BAF forecast alpha.')
    parser.add_argument('--threads', type=int, default=2)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu'); p.add_argument('--roi-weight', type=float, default=.25)
    p.add_argument('--hidden', type=int, default=16); p.add_argument('--max-logit-delta', type=float, default=1.)
    p.add_argument('--objective', choices=['roi_iou', 'pose_roi'], default='roi_iou')
    p.add_argument('--aggregation', choices=['recording', 'origin'], default='recording')
    p = sub.add_parser('evaluate')
    p.add_argument('--prepared', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--split', choices=['valid', 'test'], default='test')
    p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--margins', type=float, nargs='+', default=[0.]); p.add_argument('--delivery-delay-ms', type=float, default=0.)
    p.add_argument('--policy')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive.')
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.command == 'train':
        from .train import train
        result = {'checkpoint': train(args.prepared, args.output, AlphaConfig(args.hidden, args.max_logit_delta),
                                      args.epochs, args.batch_size, args.lr, args.seed, device, args.roi_weight,
                                      objective=args.objective, aggregation=args.aggregation)}
    else:
        from baf_roi_heatmap.evaluation import evaluate
        result = evaluate(args.prepared, args.output, alpha_checkpoint=args.checkpoint,
                          split=args.split, device=device, batch_size=args.batch_size,
                          margins=args.margins, delivery_delay_s=args.delivery_delay_ms/1000, policy_path=args.policy)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
