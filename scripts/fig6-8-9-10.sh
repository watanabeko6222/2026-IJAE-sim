#!/usr/bin/env bash
set -euo pipefail

uv run straight_passing.py -s 42

src_dir="results/straight_passing/42/section3"
dst_dir="figs/6"
mkdir -p "$dst_dir"

cp "$src_dir/ego_theta.png" "$dst_dir/"
cp "$src_dir/ego_vel.png" "$dst_dir/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir/"

src_dir="results/straight_passing/42/section4"
dst_dir="figs/8"
mkdir -p "$dst_dir"

cp "$src_dir/rel_theta.png" "$dst_dir/"
cp "$src_dir/rel_vel.png" "$dst_dir/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir/"

src_dir="results/straight_passing/42/section5"
dst_dir="figs/9"
mkdir -p "$dst_dir"

cp "$src_dir/abs_theta.png" "$dst_dir/"
cp "$src_dir/abs_vel.png" "$dst_dir/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir/"

src_dir="results/straight_passing/42/section5"
dst_dir="figs/10"
mkdir -p "$dst_dir"

cp "$src_dir/x_uncertainty2.png" "$dst_dir/"