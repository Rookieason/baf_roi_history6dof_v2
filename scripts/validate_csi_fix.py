"""Full-training-set ablations; select weights on validation only, with cached inputs."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi_heatmap.contracts import write_json, new_directory
from baf_roi_heatmap.data import ForecastDataset
from roi_experiments.runner import resolve_device, resource_guard


class Materialized:
    """Exactly the public dataset's samples, retained in RAM to avoid ZIP thrashing."""
    def __init__(self, root, split):
        self.original = ForecastDataset(root, split)
        self.samples = [self.original[i] for i in range(len(self.original))]
        print('MATERIALIZED',split,len(self.samples),flush=True)

    def __getattr__(self, name): return getattr(self.original,name)
    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--repaired',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--device',default='auto')
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--variants',nargs='+',choices=['objective','balanced','center','center_origin','alpha'],default=['objective','balanced','center','alpha'])
    a=p.parse_args();torch.set_num_threads(2)
    out=new_directory(a.output);device=resolve_device(a.device)
    config=json.loads((Path(a.original)/'manifest.json').read_text())
    config=json.loads((Path(config['pose_cache'])/'manifest.json').read_text())['config']
    cached={}; current=None; results=[]
    def dataset(root,split):
        key=(str(root),split)
        if key not in cached:cached[key]=Materialized(root,split)
        return cached[key]
    with resource_guard(config,out):
        for variant in a.variants:
            prepared=a.original if variant in ('objective','balanced') else a.repaired
            if prepared!=current:
                cached.clear();gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
                current=prepared
            module=importlib.import_module('baf_roi_heatmap_alpha.train' if variant=='alpha' else 'baf_roi_heatmap.train')
            print('VARIANT',variant,'epochs',a.epochs,flush=True)
            with patch.object(module,'ForecastDataset',dataset):
                checkpoint=module.train(prepared,out/variant,epochs=a.epochs,device=device,lr=a.lr,
                                        objective='roi_iou',aggregation='origin' if variant in ('objective','center_origin') else 'recording')
            training=json.loads((out/variant/'training.json').read_text())
            results.append({'variant':variant,'checkpoint':checkpoint,'prepared':prepared,
                            **{k:training[k] for k in ('selected_epoch','selected_validation_iou','baseline_validation_iou',
                                'best_trained_epoch','best_trained_validation_iou','selection_status','objective','aggregation')}})
            write_json(out/'ablation_summary.json',results)
            print('RESULT',results[-1],flush=True)


if __name__=='__main__':main()
