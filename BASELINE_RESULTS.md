# Baseline Results

运行命令：

```bash
python3 scripts/run_baselines.py
```

指标为 `(Pearson + Spearman) / 2`。`cv0` 只用于 debug；CV-G/CV-GCluster/CV-E/CV-GE 才用于判断泛化能力。`genomic_ridge` 是最多 512 个 genotype-only marker 的 primal ridge，作为低秩 GBLUP-like 基线，不等同于完整 marker GBLUP。

| Crop | Trait | CV0 environment mean | CV-G environment mean | CV-GCluster environment mean | CV-E genotype mean | CV-GE genomic ridge |
|---|---|---:|---:|---:|---:|---:|
| Maize | Trait_1 | 0.5518 | 0.5530 | 0.5029 | 0.1692 | 0.1399 |
| Rice | Trait_1 | 0.5408 | 0.5409 | 0.5054 | 0.0796 | 0.0023 |
| Wheat | Trait_1 | 0.7478 | 0.7485 | 0.7454 | 0.0736 | -0.0370 |
| Soybean | Trait_1 | 0.6590 | 0.6594 | 0.6393 | -0.1826 | 0.5621 |
| Soybean | Trait_2 | 0.3864 | 0.3851 | 0.2983 | -0.0664 | 0.7981 |

## Interpretation

* Maize/Rice/Wheat 的 CV-G 分数主要来自已知环境效应；同一模型在 CV-E 和严格 CV-GE 明显下降，不能把 CV-G 分数当作官方 blind-test 能力。
* Wheat 的严格 CV-GE genomic ridge 为负，说明高基因型缺失和环境/材料双重 shift 下需要更强的缺失建模与 G×E baseline。
* Soybean 的 genomic ridge 对两个性状都超过均值基线，但 Trait_1 的 CV-GE 仍低于 CV-G，Trait_2 的结果需用 RKHS/GBLUP 和独立 OOF 复核，不能直接视为最终模型优势。
* 环境均值只在训练和验证共享环境时有意义；CV-E 中未知环境只能回退到训练信息，负分是环境 shift 的诊断而不是代码错误。

完整指标和每条 OOF 预测：

* `results/metrics/baseline_metrics.csv`
* `results/metrics/baseline_fold_metrics.csv`（每折 score、worst/best fold 的来源）
* `results/metrics/baseline_summary.md`
* `results/folds/split_summary.csv`（每种划分的 train/validation/purge 覆盖）
* `results/oof/{crop}/{protocol}/baseline_oof.csv`
* `results/folds/{crop}/{protocol}.csv`
