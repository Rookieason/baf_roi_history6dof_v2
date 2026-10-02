"""Load original, unmodified BAF-Pose model/loss files without import collisions."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]

def load_original(name: str, relative_path: str):
    key = '_roi_upstream_' + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / 'vendor/baf_pose' / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

confidence = load_original('confidence', 'confidence/model.py')
focused = load_original('focused', 'prediction/focused_branch/model.py')
explorer = load_original('explorer', 'prediction/explorer_branch/model.py')

def build_models():
    # run.sh does NOT pass --use_aux. Its actual Explorer is aux_dim=0,
    # despite model.py's factory default of four auxiliary features.
    return {
        'confidence': confidence.build_confidence_estimator(input_dim=6, seq_len=30),
        'focused': focused.FocusedResidualLSTM(focused.FocusedBranchConfig()),
        'explorer': explorer.build_explorer_branch(pose_dim=7, aux_dim=0),
    }
