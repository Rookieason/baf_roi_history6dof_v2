"""Read-only validation of one configured recording, without building a cache."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baf_roi.data import discover_recordings, read_recording, make_split
from baf_roi_heatmap.recorded import profile, session_settings, inspect_session, verify_generation_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/recorded_20261031.json')
    parser.add_argument('--session', default='20260330-082303_record-5-minutes')
    args = parser.parse_args()
    cfg = profile(args.config)
    recordings = discover_recordings(cfg['data']['dataset_root'], cfg['data'])
    split = make_split(recordings, cfg['split'])
    print('SPLIT', {k: len(v) for k, v in split.items()}, flush=True)
    data_cfg = {**cfg['data'], **cfg['data'].get('session_overrides', {}).get(args.session, {})}
    for path in recordings[args.session]:
        t, _, audit = read_recording(path, data_cfg)
        print('HMD', path.name, 'duration_s', float(t[-1]-t[0]), audit, flush=True)
    dataset = Path(cfg['data']['dataset_root'])
    settings = session_settings(cfg['recorded_heatmaps'], args.session)
    print('MAP_AUDIT', inspect_session(dataset, args.session, settings)[-1], flush=True)
    print('GENERATION', verify_generation_sample(dataset, args.session, settings), flush=True)


if __name__ == '__main__':
    main()
