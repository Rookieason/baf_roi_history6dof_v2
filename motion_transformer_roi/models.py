"""Isolated imports of verbatim author model and loss sources."""
from pathlib import Path
import hashlib
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_original(name):
    key = '_roi_motion_upstream_' + name
    if key not in sys.modules:
        path = ROOT / 'vendor/motion_transformer' / (name + '.py')
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


model = load_original('model')
utility = load_original('common_utility')


def source_hashes():
    return {name: hashlib.sha256((ROOT / 'vendor/motion_transformer' / name).read_bytes()).hexdigest()
            for name in ('model.py', 'common_utility.py')}


def build_model(config=None):
    return model.MotionTransformerPredictor(model.MotionTransformerConfig(**(config or {})))
