#!/usr/bin/env bash
set -euo pipefail

uv run m2/only_vehicle_endpoint.py -c configs/self_localization_turning.py

src_dir="results/self_localization_turning/2/step2-ego"
dst_dir="figs/7"
mkdir -p "$dst_dir"

cp "$src_dir/position.png" "$dst_dir/"
cp "$src_dir/position_zoomed.png" "$dst_dir/"
cp "$src_dir/theta.png" "$dst_dir/"
cp "$src_dir/theta_zoomed.png" "$dst_dir/"
cp "$src_dir/vel.png" "$dst_dir/"