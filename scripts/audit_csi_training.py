"""Audit prepared CSI supervision and raw receive-clock alignment without changing data."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi.data import Cache
from baf_roi.pose import ypr_to_quat
from baf_roi.roi import Scene
from baf_roi_heatmap.contracts import InputConfig
from baf_roi_heatmap.data import HeatmapSeries
from baf_roi_heatmap.losses import soft_visibility
from baf_roi_heatmap.model import quaternion_product, rotation_vector_quaternion, unit_quaternion


def quantiles(values):
    a = np.asarray(values, float)
    return dict(zip(('min', 'p50', 'p95', 'max'), np.percentile(a, [0, 50, 95, 100]).tolist())) if a.size else None


def hmd_receive_timing(raw_root, session):
    receive, sample, serial, queue = [], [], [], []
    for path in sorted((raw_root/'db'/session/'sensor.full.50010').glob('*.csv')):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                payload = json.loads(row['payload_json'])
                receive.append(float(row['recv_unix']))
                sample.append(payload['monotonic_sample_time_us']/1e6)
                serial.append(payload['sample_serial'])
                queue.append(float(row['recv_unix'])-payload['rx_ts_ms']/1000)
    if not sample:
        return None
    order = np.argsort(sample)
    sample, receive = np.asarray(sample)[order], np.asarray(receive)[order]
    jitter = (receive-receive[0])-(sample-sample[0])
    jitter -= np.percentile(jitter, 1)
    return {'rows': len(sample), 'repeated_sample_serials': len(serial)-len(set(serial)),
            'recorded_sample_intervals_s': quantiles(np.diff(sample)),
            'relative_sample_to_receive_jitter_s': quantiles(jitter),
            'receiver_to_recorder_s': quantiles(queue),
            'scope': 'Relative receive/sample cadence; absolute capture-to-receive clock offset is not calibrated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw-root')
    args = parser.parse_args()
    root = Path(args.prepared)
    m = json.loads((root/'manifest.json').read_text())
    cache = Cache(m['pose_cache'])
    s = m['scene']; scene = Scene(np.array(s['centers_m']), s['partition_ids'], s['config'])
    cfg = InputConfig(**m['input_config'])
    rf = {r['key']: r for r in m['rf_manifest']['records']}
    records = defaultdict(list)
    errors = defaultdict(int)
    timings = defaultdict(lambda: defaultdict(list))
    torch.set_num_threads(2)
    for r in m['records']:
        with np.load(root/'frozen'/f"{r['key']}.npz") as archive:
            a = {k: archive[k] for k in archive.files}
        raw = cache.arrays(r['cache_index'])
        targets = np.stack([raw['series'].at(raw['series'].t[int(anchor)] + np.array(m['horizons_ms'])/1000)
                            for anchor in a['anchors']])
        errors['target_position_mismatch'] += int(np.count_nonzero(abs(targets[..., :3]-a['target_p']) > 1e-9))
        tq = ypr_to_quat(targets[..., 3:])
        errors['target_rotation_mismatch'] += int(np.count_nonzero(abs((tq*a['target_q']).sum(-1)) < 1-1e-9))
        errors['target_mask_mismatch'] += int(np.count_nonzero(scene.mask(a['target_p'], a['target_q']) != a['target_mask']))
        predicted = scene.mask(a['base_p'], a['base_q']); truth = a['target_mask']
        intersection = (predicted & truth).sum(-1); union = (predicted | truth).sum(-1)
        iou = np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union!=0).mean(1)
        records[r['session']].append({'split': r['split'], 'count': len(iou), 'iou_sum': float(iou.sum())})
    sessions = {key: {'split': value[0]['split'], 'origins': sum(r['count'] for r in value),
                      'baseline_iou': sum(r['iou_sum'] for r in value)/sum(r['count'] for r in value)}
                for key, value in records.items()}
    for split in ('train', 'valid'):
        subset = [r for r in m['records'] if r['split']==split]
        # Gradients of a shared, zero-initialized pose correction isolate the
        # training objective from the encoders and expose opposing supervision.
        accumulated = {name: torch.zeros(6, dtype=torch.float64) for name in ('pose', 'roi')}
        totals = defaultdict(float); n = 0
        for r in subset:
            with np.load(root/'frozen'/f"{r['key']}.npz") as a:
                for start in range(0, len(a['anchors']), 16):
                    end = start+16
                    p, q, tp, tq = [torch.tensor(a[k][start:end], dtype=torch.float64) for k in ('base_p', 'base_q', 'target_p', 'target_q')]
                    truth = torch.tensor(a['target_mask'][start:end], dtype=torch.float64)
                    correction = torch.zeros(6, dtype=torch.float64, requires_grad=True)
                    p = p+correction[:3]
                    q = unit_quaternion(quaternion_product(rotation_vector_quaternion(correction[3:]).expand_as(q), q))
                    translation = torch.nn.functional.smooth_l1_loss(p/.1, tp/.1)
                    rotation = (4*(1-(q*unit_quaternion(tq)).sum(-1).clamp(-1,1).square())/np.deg2rad(10)**2).mean()
                    probability = soft_visibility(p,q,scene)
                    hit = (probability*truth).sum(-1); fp=(probability*(1-truth)).sum(-1); fn=((1-probability)*truth).sum(-1)
                    roi = torch.where(truth.sum(-1)>0, 1-(hit+1e-6)/(hit+.3*fp+.7*fn+1e-6), probability.mean(-1)).mean()
                    size = len(p); n += size
                    for key, loss in [('translation',translation),('rotation',rotation),('roi',roi)]: totals[key]+=float(loss.detach())*size
                    for key, loss in [('pose',translation+rotation),('roi',roi)]:
                        accumulated[key] += torch.autograd.grad(loss,correction,retain_graph=True)[0]*size
        for key in totals: totals[key]/=n
        gradients={key:(g/n).tolist() for key,g in accumulated.items()}
        pose, roi = accumulated.values()
        cosine=float(torch.dot(pose,roi)/(pose.norm()*roi.norm()).clamp_min(1e-12))
        timings[split]['loss'] = dict(totals)
        timings[split]['shared_residual_gradients'] = gradients
        timings[split]['pose_roi_gradient_cosine'] = cosine
        print('OBJECTIVE',split,dict(timings[split]),flush=True)
    raw_root = Path(args.raw_root) if args.raw_root else None
    clock_report = {}
    for session in sessions:
        entries = [r for r in m['records'] if r['session']==session]
        r = rf[entries[0]['key']]
        series = HeatmapSeries(r['file'], r['clock_offset_s'], r['clock_uncertainty_s'], True)
        stat=defaultdict(list); violations=0
        for record in entries:
            with np.load(root/'frozen'/f"{record['key']}.npz") as a:
                for origin in a['origin_s']:
                    queries, ids=series.indices(float(origin),cfg)
                    for query,idx in zip(queries,ids):
                        if idx<0: continue
                        violations += int(series.ready[idx]+series.uncertainty>query+1e-9 or series.start[idx]-series.uncertainty<origin-cfg.raw_history_s-1e-9)
                    if ids[-1]>=0:
                        idx=ids[-1];stat['latest_support_end_age_s'].append(origin-series.end[idx]);stat['latest_support_span_s'].append(series.end[idx]-series.start[idx])
        result={k:quantiles(v) for k,v in stat.items()}; result['causal_violations']=violations
        if raw_root:
            result['hmd_receive_timing'] = hmd_receive_timing(raw_root, session)
            receivers={}; offsets=[]
            for nic in range(1,5):
                receiver={}
                for path in sorted((raw_root/'db'/session/f'csi.rx.{nic}').glob('*.csv')):
                    with path.open() as stream:
                        for row in csv.DictReader(stream):
                            stamp=float(row['recv_unix']); name=Path(row['array_saved']).name
                            receiver[name]=min(stamp,receiver.get(name,stamp))
                            if row.get('rx_system_ns'): offsets.append(stamp-int(row['rx_system_ns'])/1e9)
                receivers[str(nic)]=receiver
            matched=raw_root/'intermediates'/session/'matched_csi'/f'{session}_matched.csv'
            with matched.open() as stream: rows=list(csv.DictReader(stream))
            measurement=[]
            with np.load(r['file']) as maps:
                for j,index in enumerate(maps['source_map_index']):
                    stamps=[receivers[str(nic)].get(rows[int(index)][f'nic{nic}_array_saved'],np.nan) for nic in range(1,5)]
                    if np.isfinite(stamps).any():measurement.append(series.end[j]-float(np.nanmax(stamps)))
            result['support_end_minus_center_receive_s']=quantiles(measurement)
            result['collector_minus_radio_clock_s']=quantiles(offsets)
        clock_report[session]=result
        print('CLOCK',session,result,flush=True)
    result={'supervision_errors':dict(errors),'sessions':sessions,'objectives':dict(timings),'clock':clock_report,
            'note':'No held-out test targets used to optimize weights or choose hyperparameters.'}
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('OUTPUT',destination,flush=True)


if __name__=='__main__': main()
