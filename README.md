# 実行環境
ubuntu 22.04 LTS

## 環境の構築
uvのインストール
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

仮想環境の構築
```bash
uv sync --frozen
```

## 図表作成

Fig.4 を作成する場合
```bash
bash ./scripts/fig4.sh
```

Fig.5 を作成する場合
```bash
bash ./scripts/fig5.sh
```

Fig.6 を作成する場合
```bash
bash ./scripts/fig6.sh
```

## Table 作成

Table 3 を作成する場合
```bash
bash ./scripts/table3.sh
```

Table 4 を作成する場合
```bash
bash ./scripts/table4.sh
```

Table 5 を作成する場合
```bash
bash ./scripts/table5.sh
```

## 出力先
- 生データ・中間生成物: `results/section{3|4|5}/100/`
- 論文用にコピーした図:
  - `figs/4/` (section3)
  - `figs/5/` (section4)
  - `figs/6/` (section5)
- 論文用にコピーした表:
  - `tables/3/last_sigma.csv` (section3)
  - `tables/4/last_sigma.csv` (section4)
  - `tables/5/last_sigma.csv` (section5)