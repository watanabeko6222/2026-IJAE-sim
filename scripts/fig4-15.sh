#!/usr/bin/env bash
set -euo pipefail

uv run straight_passing_ablation.py -s 42

src_dir="results/straight_passing_ablation/42/section5"
dst_dir="figs/4-15"
mkdir -p "$dst_dir"

cp "$src_dir/abs_theta.png" "$dst_dir/"
cp "$src_dir/abs_vel.png" "$dst_dir/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir/"
