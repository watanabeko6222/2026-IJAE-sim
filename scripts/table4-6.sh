#!/usr/bin/env bash
set -euo pipefail

seed=1
iter=1000

uv run straight_passing_mote_carlo.py --seed "$seed" --iter "$iter"
uv run straight_passing_mote_carlo_ablation.py --seed "$seed" --iter "$iter"

dst_dir="tables/4-6"
mkdir -p "$dst_dir"

src_base="results/straight_passing_mote_carlo/${seed}/content_ratio.csv"
src_ablation="results/straight_passing_mote_carlo_ablation/${seed}/content_ratio.csv"

cp "$src_base" "$dst_dir/content_ratio_base.csv"
cp "$src_ablation" "$dst_dir/content_ratio_ablation.csv"

echo "Saved table inputs to $dst_dir"
echo "  - $dst_dir/content_ratio_base.csv"
echo "  - $dst_dir/content_ratio_ablation.csv"
