"""A tiny REAL-data interface/gradient check, never a trained model or benchmark."""
import argparse
import copy
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from baf_roi.config import fingerprint
from baf_roi.data import Cache
from baf_roi.models import build_models
from baf_roi_heatmap.contracts import InputConfig, ModelConfig, read_json, write_json
from baf_roi_heatmap.data import ForecastDataset, HeatmapSeries
from baf_roi_heatmap.model import ForecastAdapter
from baf_roi_heatmap.losses import task_loss
from baf_roi_heatmap.prepare import prepare


def main(validation, output):
    source = Path(validation); out = Path(output)
    if out.exists():
        raise ValueError('Use a new smoke output directory.')
    out.mkdir(parents=True)
    torch.set_num_threads(2); torch.manual_seed(42)
    cache = Cache(source/'pose_cache'); rf = read_json(source/'rf_manifest.json')
    cfg = InputConfig(**cache.manifest['config']['recorded_heatmaps']['input_config'])
    selected, entries = [], []
    for split in ('train','valid','test'):
        for ri, record in enumerate(cache.records):
            if record['split'] != split:
                continue
            a = cache.arrays(ri); entry = next(r for r in rf['records'] if r['key']==record['key'])
            series = HeatmapSeries(entry['file'], uncertainty_s=.001, allow_exploratory=True)
            chosen = []
            for oi, anchor in enumerate(a['anchors']):
                if a['t'][anchor]+.25 > a['t'][-1]:
                    continue
                _, ids = series.indices(record['epoch_start_s']+a['t'][anchor],cfg)
                if ids[-1]>=0 and (ids>=0).mean()>=cfg.min_valid_fraction:
                    chosen.append(oi)
                if len(chosen)==4:
                    break
            if len(chosen)<4:
                continue
            target = out/'pose_cache'/record['key']; target.mkdir(parents=True)
            for name in ('t','pose','unwrapped'):
                shutil.copy2(source/'pose_cache'/record['key']/(name+'.npy'),target/(name+'.npy'))
            for name in ('anchors','labels','next_labels','targets','scores'):
                np.save(target/(name+'.npy'),a[name][chosen],allow_pickle=False)
            r = copy.deepcopy(record); r['counts']['input_windows']=len(chosen)
            selected.append(r); entries.append(entry); break
        else:
            raise ValueError(f'No RF-available smoke samples in {split}')
    manifest = copy.deepcopy(cache.manifest); manifest.pop('cache_id')
    manifest['trajectories']=selected
    manifest['split']={s:[r['session'] for r in selected if r['split']==s] for s in ('train','valid','test')}
    manifest['cache_id']=fingerprint(manifest)
    write_json(out/'pose_cache/manifest.json',manifest)
    rf['records']=entries; rf['hmd_cache_id']=manifest['cache_id']; write_json(out/'rf_manifest.json',rf)
    # Random frozen experts exist only to check interfaces on twelve origins.
    torch.save({'format_version':2,'models':{k:m.state_dict() for k,m in build_models().items()},
                'normalization':cache.stats,'config':manifest['config'],'cache_id':manifest['cache_id'],
                'warning':'SMOKE ONLY: randomly initialized experts, not a trained checkpoint'},out/'UNTRAINED_SMOKE_EXPERTS.pt')
    prepare(out/'pose_cache',out/'UNTRAINED_SMOKE_EXPERTS.pt',out/'rf_manifest.json',out/'prepared',
            horizons=[50,100,150,200,250],input_config=cfg,batch_size=4)
    dataset=ForecastDataset(out/'prepared','train')
    batch=next(iter(DataLoader(dataset,batch_size=4)))
    assert bool(batch['inputs']['rf_available'].all())
    model=ForecastAdapter(ModelConfig(method='residual',modality='rf'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad();loss=task_loss(model(batch['inputs']),batch['supervision'],dataset.scene)
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    gradient=float(model.spatial[0].weight.grad.abs().max())
    assert gradient>0,'CSI encoder receives no gradient'
    with torch.no_grad():
        altered={**batch['inputs'],'maps':torch.zeros_like(batch['inputs']['maps'])}
        sensitivity=float((model(batch['inputs'])['p']-model(altered)['p']).abs().max())
    assert sensitivity>0,'Prediction does not depend on CSI map values'
    report={'status':'PASS','scope':'12 real-data origins, random frozen experts, two optimizer steps only',
            'hmd_shape':list(batch['inputs']['hmd'].shape),'map_shape':list(batch['inputs']['maps'].shape),
            'finite_loss':float(loss.detach()),'csi_encoder_max_abs_gradient':gradient,'map_perturbation_effect_m':sensitivity}
    write_json(out/'smoke_report.json',report); print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.validation,a.output)
