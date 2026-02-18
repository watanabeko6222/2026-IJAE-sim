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

Fig.4-6, Fig.4-7, Fig.4-8:
```bash
bash ./scripts/fig4-6-8.sh
```

Fig.4-9, Fig.4-10, Fig.4-11:
```bash
bash ./scripts/fig4-9-11.sh
```

Fig.4-12, Fig.4-13, Fig.4-14:
```bash
bash ./scripts/fig4-12-14.sh
```

Fig.4-15:
```bash
bash ./scripts/fig4-15.sh
```

## 表の作成

Table 4-6:
```bash
bash ./scripts/table4-6.sh
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
- `figs/*/`

論文用にコピーした表:
- `tables/4-6/content_ratio_base.csv`
- `tables/4-6/content_ratio_ablation.csv`
