#!/usr/bin/env bash
set -euo pipefail

uv run straight_passing.py -s 42

dst_dir_12="figs/4-12"
dst_dir_13="figs/4-13"
dst_dir_14="figs/4-14"
mkdir -p "$dst_dir_12"
mkdir -p "$dst_dir_13"
mkdir -p "$dst_dir_14"

src_dir="results/straight_passing/42/section4"
cp "$src_dir/rel_theta.png" "$dst_dir_12/"
cp "$src_dir/rel_vel.png" "$dst_dir_12/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir_12/"

src_dir="results/straight_passing/42/section5"
cp "$src_dir/abs_theta.png" "$dst_dir_13/"
cp "$src_dir/abs_vel.png" "$dst_dir_13/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir_13/"

cp "$src_dir/x_uncertainty2.png" "$dst_dir_14/"