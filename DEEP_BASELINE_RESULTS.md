# Deep baseline results

运行时间：2026-09-22。模型：`deep_gxe_net` / DeepGxE-Net-L，约 1.69M 参数，最多 4,096 个 marker，天气聚合特征，训练折内预处理，inner early stopping。

## 输入/输出

每一条输入记录是一个 `genotype_id × environment_id` 表型观测。模型输入为：

1. 训练折内筛选、插补和标准化后的 marker dosage `[batch, P]`；
2. 原始 dosage 缺失 mask `[batch, P]`；
3. 环境天气特征 `[batch, F]`：`n_days`，以及每个天气变量的 mean/std/min/max/first/last/missing fraction。

模型输出是该记录指定 `Trait_*` 的一个标量预测。每个性状单独训练；OOF 文件保存真实值、预测值、行/材料/环境 ID、fold、参数量和 selected-marker 数量。

## Performance

Competition score 为 `(Pearson + Spearman) / 2`。所有分数都是完整 outer OOF 汇总。

| Crop | Trait | CV-G | CV-GCluster | CV-GE | CV-E |
|---|---|---:|---:|---:|---:|
| Maize | Trait_1 | 0.6268 | 0.6117 | 0.0320 | 0.0456 |
| Rice | Trait_1 | 0.6810 | 0.6655 | 0.3208 | 0.2456 |
| Wheat | Trait_1 | 0.7531 | 0.7549 | 0.3823 | 0.5040 |
| Soybean | Trait_1 | 0.8433 | 0.8186 | 0.6335 | 0.6756 |
| Soybean | Trait_2 | 0.8994 | 0.8693 | 0.8387 | 0.8568 |

参考：当前均值/岭回归基线的 CV-GE score 为 Maize `0.1399`、Rice `0.0023`、Wheat `-0.0370`、Soybean Trait_1 `0.5621`、Soybean Trait_2 `0.7981`。因此深度模型在 Rice/Wheat/Soybean 的严格 CV-GE 有增益，但 Maize CV-GE 仍低于 genomic-ridge，不能直接 promotion。

## Distribution and leakage checks

- CV-G：5 folds；所有作物 genotype overlap 为 0。
- CV-GCluster：按 genotype-only cluster 留出；所有 genotype overlap 为 0。
- CV-GE：validation 同时是 unseen genotype 和 unseen environment；所有 genotype/environment overlap 为 0。
- CV-E：environment overlap 为 0；genotype overlap 非零是预期的，因为这是只留出环境的协议。
- Wheat 的 phenotype/weather ID 映射已修复为双向 source-ID 到 canonical-ID 映射，天气特征不再因 `LocX_YYYY`/`YYYY-LocX` 差异变成全缺失。
- 每折的 marker 缺失率、环境标准化均值 shift、overlap、runtime 和参数量见 `results/metrics/deep*_distribution.csv`。

## Artifact locations

| Artifact | Path |
|---|---|
| CV-G metrics | `results/metrics/deep_metrics.csv` |
| CV-GCluster/CV-GE metrics | `results/metrics/deep_strict_metrics.csv` |
| CV-E metrics | `results/metrics/deep_cv_e_metrics.csv` |
| OOF predictions | `results/oof/{crop}/{protocol}/deep*_oof.csv` |
| Training history | `results/metrics/deep*_training_history.csv` |
| Distribution diagnostics | `results/metrics/deep*_distribution.csv` |
| Canonical fold manifests | `results/folds/{crop}/{protocol}.csv` |

推荐模型选择规则：先看 CV-GCluster/CV-GE 的 mean、fold SD 和 worst fold，再看 CV-E；不能以 CV-G 单独高分作为提交依据。Maize 需要优先做环境特征增强、伪盲测和与 genomic-ridge 的 OOF 集成，而不是继续无约束增大网络。
