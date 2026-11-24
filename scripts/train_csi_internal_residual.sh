#!/usr/bin/env bash
# Main comparison on this branch: BAF, Motion Transformer, internal CSI residual.
# The shared launcher starts one background run and prints its output/log paths.
set -euo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/train_csi_20261031.sh" \
  --method baf_roi motion_transformer_roi baf_roi_heatmap_residual "$@"
