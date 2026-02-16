# 実行環境
ubuntu 22.04 LTS

## 環境の構築
uv のインストール:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

依存関係のセットアップ:
```bash
uv sync --frozen
```

## 図の作成

Fig.6, Fig.8, Fig.9, Fig.10:
```bash
bash ./scripts/fig6-8-9-10.sh
```

Fig.7:
```bash
bash ./scripts/fig7.sh
```

Fig.11:
```bash
bash ./scripts/fig11.sh
```

## 表の作成

Table 4:
```bash
bash ./scripts/table4.sh
```

## 主な出力先

シミュレーション結果:
- `results/straight_passing/42/section3/`
- `results/straight_passing/42/section4/`
- `results/straight_passing/42/section5/`
- `results/straight_passing_ablation/42/section5/`
- `results/self_localization_turning/2/step2-ego/`
- `results/straight_passing_mote_carlo/1/content_ratio.csv`
- `results/straight_passing_mote_carlo_ablation/1/content_ratio.csv`

論文用にコピーした図:
- `figs/6/`
- `figs/7/`
- `figs/8/`
- `figs/9/`
- `figs/10/`
- `figs/11/`

論文用にコピーした表:
- `tables/4/content_ratio_base.csv`
- `tables/4/content_ratio_ablation.csv`
