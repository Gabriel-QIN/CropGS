# Blind-test 泛化优化报告

日期：2026-09-23。目标：以官方 blind test 为最终目标；CV-G 高分不作为单独 promotion 依据。

## 1. 为什么玉米、水稻在新环境划分中下降

### 1.1 环境样本数远小于环境特征维数

| Crop | Environments | Basic weather features | Dynamic features | 可识别最大秩 |
|---|---:|---:|---:|---:|
| Maize | 14 | 120 | 307 | 13 |
| Rice | 7 | 29 | 73 | 6 |
| Wheat | 12 | 29 | 73 | 11 |
| Soybean | 31 | 36 | 91 | 30 |

同一环境的数百/上千条材料记录并不等于数百个独立天气样本；天气 encoder 实际只看到 7 个 Rice 环境或 14 个 Maize 环境。高维动态统计很容易记住环境，不能可靠外推。

### 1.2 环境主效应很强，但盲测要求预测未知环境的尺度

环境均值解释的表型方差比例约为 Maize `30.1%`、Rice `30.8%`、Wheat `56.5%`、Soybean Trait_1 `44.1%`。独立测试中 Maize hard-E 的 pooled score 为 `-0.0925`，但 environment-macro score 为 `0.3728`：模型在单个环境内部仍有材料排序能力，主要失败在不同未知环境之间的均值/尺度校准。

### 1.3 原始 CV-GE 没有专门针对极端环境

原始 CV-GE 将环境按记录数平衡分组，适合平均风险估计，但不会保证最远天气环境成为单独 stress test。新的独立集显式按 genotype PCA 距离和天气距离分层，因此比随机 CV-GE 更接近真实分布偏移。

### 1.4 当前 marker block 没有物理位置先验

发布数据没有 chromosome/position。Transformer/CNN 的连续 marker block 只使用 CSV 稳定顺序，不能保证相邻 marker 在生物学上相邻。扩大网络会放大错误归纳偏置。参数量消融也证实 large 模型不稳定。

### 1.5 Wheat/Soybean 的“高分”也要拆开看

Wheat 独立 easy-G 为 `0.7551`，但 hard-G×E 只有 `0.2145`，并非真正稳定。Soybean Trait_2 hard-G×E 为 `0.8633`，目前最稳；Trait_1 为 `0.6003`。Soybean hard-G-only 只有 24 条，按预注册规则直接跳过，不报告相关性。

## 2. 冻结的独立测试设计

选择只使用 genotype PCA、天气特征和观测覆盖，不读取任何 Trait 值。

| Role | 定义 | 解释 |
|---|---|---|
| development | 非 holdout genotype × 非天气离群环境 | 唯一允许训练、CV 和调参的数据 |
| test_easy_g | PCA 中心附近的新 genotype × 已知类型环境 | Easy：新材料但遗传分布内插 |
| test_hard_g | PCA 远端 genotype × 已知类型环境 | 遗传外推 |
| test_hard_e | development genotype × 天气离群环境 | 环境外推 |
| test_hard_ge | PCA 远端 genotype × 天气离群环境 | Hard：材料和环境同时外推 |
| purge_mixed | 中心 genotype × 天气离群环境 | 定义混合，purge，不参与评分 |

最低有效样本数固定为 `100`；低于 100 标记 `skipped_insufficient_n`，不训练、不计算 Pearson/Spearman、不进入比较。

development-only 的 CV-G、CV-GCluster、CV-E、CV-GE 清单也已统一冻结在 `results/folds/{crop}/development_*.csv`。逐折检查发现 Wheat LOEO 有 2 折、Soybean LOEO 有 4 折不足 100 条，均标记跳过；所有 CV-G/CV-GCluster/CV-GE 折达到门槛。诊断见 `results/folds/development_fold_summary.csv`。

清单：[independent_test_summary.csv](results/folds/independent_test_summary.csv) 和 `results/folds/{crop}/independent_test.csv`。

## 3. 架构、参数量和特征消融

消融只使用 frozen development 数据的严格 CV-GE；没有读取独立测试标签。

| Model / representation | Params | Maize score | Maize worst | Rice score | Rice worst |
|---|---:|---:|---:|---:|---:|
| Gated, basic, base | 1.70M | **0.2738** | **0.3099** | 0.2422 | 0.1412 |
| Gated, dynamic, base | 1.72M | 0.2293 | 0.0926 | 0.2104 | 0.1936 |
| Reaction norm, dynamic, small | 0.35M | 0.0834 | -0.3350 | 0.1439 | 0.1811 |
| Reaction norm, dynamic, base | 1.57M | 0.1542 | -0.0913 | 0.2461 | **0.2077** |
| Reaction norm, dynamic, large | 11.19M | 0.0398 | -0.4782 | 0.2929 | -0.3793 |
| Gated, basic, low-rank-6 | 1.68M | 0.2208 | -0.1071 | **0.3151** | 0.1652 |
| Reaction norm, dynamic, low-rank-6 | 1.53M | 0.2661 | 0.2878 | 0.2924 | 0.0052 |

结论：

1. Large 不是更好。11.19M 模型 worst-fold 在两个作物都为负，直接淘汰。
2. 动态天气特征直接展开会过拟合；Maize/Rice 都不如 basic gated baseline。
3. 训练折内 low-rank weather 对 Rice 的平均 CV-GE 有明显提升，但 Maize 的 gated 版本不稳定。
4. 当前 promotion 候选按作物分开：Maize 保留 gated-basic-base 与 reaction-dynamic-lowrank 的 OOF 集成候选；Rice 保留 gated-basic-lowrank 与 reaction-dynamic-base 的稳健集成候选。

进一步做了跨 fold 权重选择的 convex OOF ensemble。Maize ensemble score `0.2729`，没有超过 gated-basic-base 的 `0.2738`；Rice ensemble `0.2462`，低于 gated-basic-lowrank 的 `0.3151`。因此当前两个深度架构的简单线性融合不 promotion；下一步应引入归纳偏置不同的 full-marker GBLUP/RKHS，而不是继续混合相似神经网络。

## 4. 独立 easy/hard 结果（已使用一次，现封存）

Competition score = `(Pearson + Spearman) / 2`。

| Crop / Trait | Easy-G | Hard-G | Hard-E | Hard-G×E |
|---|---:|---:|---:|---:|
| Maize Trait_1 | 0.6346 | 0.3500 | -0.0925 | -0.1449 |
| Rice Trait_1 | 0.6521 | 0.5660 | 0.3063 | 0.2172 |
| Wheat Trait_1 | 0.7551 | 0.7281 | 0.3582 | 0.2145 |
| Soybean Trait_1 | 0.8562 | skipped (n=24) | 0.5349 | 0.6003 |
| Soybean Trait_2 | 0.8503 | skipped (n=24) | 0.8767 | 0.8633 |

这些结果只能作为一次性泛化审计，后续不能据此反复调参。下一轮选择只使用 development nested CV；最终验证应等待新的 pseudo-blind 或官方预测数据。

## 5. 冠军导向的下一阶段模型

### 5.1 稳健混合专家，而不是单一大网络

提交候选采用 distance-gated ensemble：

- additive expert：full-marker GBLUP / Bayesian ridge，负责未知环境下的稳定遗传主效应；
- nonlinear genotype expert：低秩 marker/PCA MLP 或有位置元数据后的 chromosome encoder；
- environment expert：低参数 raw-weather TCN/phase encoder，输出环境均值与尺度；
- G×E expert：reaction-norm low-rank interaction；
- OOD gate：根据 genetic distance、weather distance、fold variance 自动降低不可靠 G×E expert 权重。

Maize hard-E 表明必须把“环境均值/尺度预测”和“环境内材料排序”拆成两个任务，不能继续只扩大融合网络。

### 5.2 与竞赛指标一致的训练目标

候选 loss：标准化 Huber + within-environment Pearson surrogate + pairwise rank loss。环境均值/尺度 head 单独使用 LOEO 训练；相关性 loss 只在 inner-training batch 内计算，不能读取 outer/test 标签。

### 5.3 Promotion gate

模型必须同时满足：

1. development CV-GCluster、CV-E、CV-GE 均不退化；
2. worst-fold 为正，并优于当前作物 baseline；
3. 三个固定 seed 均值提升，提升大于 seed 标准差；
4. environment-macro 与 pooled score 同时报告，二者严重背离时不得 promotion；
5. 参数量增加必须带来 strict-CV 增益；否则选择更小模型；
6. 独立测试现已封存，不再用于本轮参数选择。

## 6. 主要产物

- 划分器：`src/validation/independent_test.py`
- 清单生成：`scripts/build_independent_test.py`
- 深度模型：`src/models/deep/genotype_environment_net.py`
- CV/消融入口：`scripts/run_deep_baseline.py`
- 一次性测试入口：`scripts/run_independent_test.py`
- 消融结果：`results/metrics/ablation_*`
- 独立结果：`results/metrics/independent_*_final_metrics.csv`
- Cross-fitted ensemble：`results/oof/{maize,rice}/cv_ge/ensemble_crossfit_oof.csv`
