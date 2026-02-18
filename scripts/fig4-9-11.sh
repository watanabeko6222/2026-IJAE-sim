#!/usr/bin/env bash
set -euo pipefail

uv run m2/only_vehicle_endpoint.py -c configs/self_localization_turning.py

src_dir="results/self_localization_turning/2/step2-ego"
dst_dir_9="figs/4-9"
dst_dir_10="figs/4-10"
dst_dir_11="figs/4-11"
mkdir -p "$dst_dir_9"
mkdir -p "$dst_dir_10"
mkdir -p "$dst_dir_11"

cp "$src_dir/position.png" "$dst_dir_9/"
cp "$src_dir/position_zoomed.png" "$dst_dir_9/"
cp "$src_dir/theta.png" "$dst_dir_10/"
cp "$src_dir/theta_zoomed.png" "$dst_dir_10/"
cp "$src_dir/vel.png" "$dst_dir_11/"