#!/usr/bin/env bash
set -euo pipefail

uv run straight_passing.py -s 42

src_dir="results/straight_passing/42/section3"
dst_dir_6="figs/4-6"
dst_dir_7="figs/4-7"
dst_dir_8="figs/4-8"
mkdir -p "$dst_dir_6"
mkdir -p "$dst_dir_7"
mkdir -p "$dst_dir_8"

cp "$src_dir/ego_theta.png" "$dst_dir_7/"
cp "$src_dir/ego_vel.png" "$dst_dir_8/"
cp "$src_dir/sum_xy_2d_2.png" "$dst_dir_6/"