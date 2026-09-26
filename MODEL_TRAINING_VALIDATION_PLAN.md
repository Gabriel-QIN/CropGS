# AI 育种挑战赛：模型训练与验证计划

> 版本：2026-09-22 · 阶段：P0 Data Audit、P1 leakage-aware baseline、P4 DeepGxE-Net-L 首轮训练完成
>
> 总目标：`Blind-test generalization > validation score inflation`。任何模型只有在严格的 unseen genotype、unseen environment 和 genotype × environment stress test 下稳定，才进入最终提交候选。

## 0. 当前审计结论

审计命令：

```bash
python3 scripts/data_audit.py \
  --data-root Dataset \
  --reports-dir reports \
  --figures-dir reports/figures
```

审计脚本对基因型矩阵逐标记计算编码、缺失率、MAF、杂合率、单态/多态标记、重复行/列，并对材料做 PCA、聚类和抽样遗传相似度分析。结果写入：

```text
reports/
├── data_audit.json
├── cross_crop_summary.md
├── maize_data_audit.md
├── rice_data_audit.md
├── wheat_data_audit.md
├── soybean_data_audit.md
└── figures/
    ├── cross_crop_dataset_overview.svg
    └── {maize,rice,wheat,soybean}/
        ├── phenotype_environment_distribution.svg
        ├── genotype_environment_presence.svg
        ├── genotype_qc_distribution.svg
        └── genotype_pca.svg
```

### 0.1 四作物真实数据规模

| 作物 | 基因型材料 | SNP | 表型记录 | 环境 | 性状 | 基因型缺失 | 环境 ID 链接 |
|---|---:|---:|---:|---:|---:|---:|---|
| Maize | 1,879 | 50,000 | 12,854 | 14 | 1 | 0.636% | 14/14 直接匹配 |
| Rice | 1,000 | 60,000 | 5,881 | 7 | 1 | 0% | 7/7 直接匹配 |
| Wheat | 1,953 | 10,297 | 5,108 | 12 | 1 | 20.230% | 0/12 直接；12/12 规范化匹配 |
| Soybean | 11,604 | 41,756 | 12,007 | 31 | 2 | 0.738% | 31/31 直接匹配 |

### 0.2 影响验证设计的结构事实

| 作物 | 材料跨环境覆盖 | 重要 QC / 解释 |
|---|---|---|
| Maize | 1,445 个材料出现在 7 个环境，415 个出现在 6 个环境；其余为 8、12、13、14 个环境 | 4,080 个单态标记；3 个杂交组合全基因型缺失；检测到 1 组完全重复材料列。 |
| Rice | 247 个材料覆盖 7 个环境，459 个覆盖 6 个环境；3 个材料仅出现 1 个环境 | 12,187 个单态标记；无基因型缺失；E1 只有 276 条表型记录。 |
| Wheat | 703 个材料仅出现 1 个环境，665 个出现 3 个，526 个出现 4 个；最多 8 个 | 基因型缺失率高且不均匀；1 个材料全缺失、多个材料接近全缺失；5 个单态标记；表型环境为 `Loc3_2015`，气象环境为 `2015-Loc3`，必须版本化映射。 |
| Soybean | 11,266/11,604 个材料仅出现 1 个环境；296 个出现 2 个；仅 42 个出现至少 3 个 | 429 个单态、12 个全缺失标记；`Sample_20087` 全基因型缺失；`Variable_2` 有 3,077 个气象缺失；Trait_1 有 20 个缺失值。 |

审计 PCA 使用最多 3,000 个多态标记作为结构诊断，不把 PCA/cluster 标签直接当作模型特征。基于实际表型行的严格 5 折可行性检查如下：

| 作物 | CV-G 每折表型行数（稳定哈希诊断） | CV-GE 严格交集每折表型行数 |
|---|---|---|
| Maize | 2,352–2,703 | 405–594 |
| Rice | 1,021–1,343 | 172–340 |
| Wheat | 966–1,107 | 165–234 |
| Soybean | 2,377–2,424 | 431–511 |

CV-GE 的验证记录定义为同时属于该折 holdout genotype group 和 holdout environment group 的交集；训练集会移除所有 holdout genotype 或 holdout environment 的记录，交集之外的记录作为 purge，不参与该折训练/评分。

## Part 1 — Dataset inventory

### 1.1 文件与主键

| 作物 | 文件 | 基因型主键 | 表型主键 | 环境主键 | 结论 |
|---|---|---|---|---|---|
| Maize | `Dataset/Maize/Maize/{Genotypes,Phenotypes,Environment}.csv` | `Genotypes.csv` 首列 `SNP_Name`，列名为 `Sample_*/Sample_*` 杂交组合 | `Environment + Hybrid` | `Environment` | 三表直接链接；1 组重复材料列，重复键表型为 0。 |
| Rice | `Dataset/Rice/Rice/{Genotypes,Phenotypes,Environment}.csv` | 首列 `Marker`，列名 `Sample_1`… | `Environment + Sample` | `Environment` | 三表直接链接；0 基因型缺失，E1 低覆盖。 |
| Wheat | `Dataset/Wheat/Wheat/{Genotypes,Phenotypes,Environment}.csv` | 首列 `Marker`，列名 `Sample0001`… | `Environment + Sample` | `Environment` | 材料直接链接；环境需将 `LocX_YYYY` 映射为 `YYYY-LocX`。 |
| Soybean | `Dataset/Soybean/Soybean/{Genotypes,Phenotypes,Environment}.csv` | 首列 `SNP_Name`，列名 `Sample_*` | `Environment + Sample` | `Environment` | 三表直接链接；表型非常稀疏，气象与 Trait_1 有缺失。 |

### 1.2 缺失、重复和可用性规则

1. **材料 ID**：以基因型表列名和表型 ID 的精确交集为准。审计中四作物均为全交集；不得按列序号或字符串截断连接。
2. **环境 ID**：Wheat 使用一个显式映射文件 `data/interim/environment_id_map.csv`，必须包含原始 ID、规范 ID、位置、年份、匹配依据和人工复核状态。未通过映射校验前不运行 Wheat 的 CV-E/CV-GE。
3. **表型缺失**：Soybean `Trait_1` 的 20 条缺失记录从该性状的训练/评分表移除，并记录行 ID；不得用全数据均值填补目标值。`Trait_2` 无缺失。
4. **基因型全缺失材料**：Maize 的 3 个材料、Wheat 的 1 个全缺失且多个接近全缺失材料、Soybean 的 `Sample_20087` 不进入需要基因型预测的主训练表；保留在 QC 清单并生成 `prediction_confidence=low` 诊断。删除必须是 fold 内规则，不能根据验证标签删除。
5. **单态/重复标记**：单态标记不提供遗传区分度，但是否剔除必须在训练折内拟合并写入配置；完全重复标记只保留一个代表列，代表选择按 marker ID 稳定排序。
6. **气象缺失**：Soybean `Variable_2` 的缺失率和每环境缺失模式写入报告。气象特征工程必须保留缺失指示变量；插补器只能在训练折拟合，或者使用事先冻结的物理规则，不得用表型信息。

## Part 2 — Leakage audit

| 操作/信息 | 状态 | 可接受边界 |
|---|---|---|
| 按 `genotype_id` 分组，将同一材料所有环境记录放在同一侧 | **Safe / 必须** | CV-G、CV-GE 的硬约束。 |
| 按 environment 分组 | **Safe / 必须** | CV-E、CV-GE；Wheat 先做版本化 ID 映射。 |
| 只使用 genotype ID、SNP ID、原始环境日期做键连接 | **Safe** | 不读取目标表型；保留审计哈希。 |
| 用全基因型计算 QC、重复列、marker 编码一致性 | **Conditionally safe** | 只作为数据完整性和预先冻结的 genotype-only 规则；若用于模型特征/选择，必须在 outer training fold 重算。 |
| 用全基因型做 PCA/GRM/cluster 诊断 | **Conditionally safe** | 可用于审计、预定义 stress-test 清单；作为输入特征或调参依据时改为 outer-train 内拟合，并将 validation 投影到 train。 |
| 基于 genotype-only 的固定 MAF/missingness 规则 | **Conditionally safe** | 可事先冻结；更严格版本在每个 training fold 重算阈值并只保留训练折可用 marker。 |
| genotype imputation / scaling / PCA | **Conditionally safe** | 拟合参数只来自 training fold；验证/官方测试只应用训练参数。缺失指示变量一并传入。 |
| weather 聚合特征 | **Conditionally safe** | 仅使用预测时可获得的逐日气象；聚合器和插补器不使用表型，分布标准化在 training fold 拟合。 |
| 用全数据 GWAS、相关性、Pearson、遗传效应筛 SNP | **Unsafe** | 严重 phenotype leakage；必须在每个 inner training fold 内执行。 |
| 用所有表型计算环境均值、材料均值或 BLUP 后再 CV | **Unsafe** | 目标统计量会携带 validation label；只能用该 fold 的训练表型。 |
| 同一材料不同环境拆到 train/validation | **Unsafe for CV-G/CV-GE** | 只可作为 CV0 debug，不得用于模型选择。 |
| 根据 validation 分数手工反复挑特征/折/模型 | **Unsafe** | 用 nested CV、冻结候选矩阵和一次性 pseudo-blind test 替代。 |
| 查看官方测试的无标签 SNP/天气分布 | **Conditionally safe** | 仅做 schema、allele alignment、距离和置信度 QC；不得据此反复改模型或使用隐藏标签。 |

## Part 3 — Validation design

所有 fold 文件都必须落盘到 `results/folds/{crop}/`，字段至少为：`row_id, genotype_id, environment_id, trait, protocol, fold, role, purge_reason`。每次生成记录 `seed=20260922`、代码版本和数据文件 MD5。

### CV0 — Random observation split（仅 debug）

逐行随机 80/20，只用于检查代码、loss、输出格式和 OOF 拼接。结果标记为 `debug_only=true`，不进入 leaderboard、超参选择或最终报告。

### CV-G — unseen genotype，主验证

* 分组键：Maize 的 `Hybrid`，其他作物的 `Sample`。
* 5 折；同一材料的所有环境记录必须在同一折。
* 推荐生成器：按 `sha256("20260922|g|" + genotype_id) % 5` 产生稳定初始分组，然后检查每折表型行数、环境覆盖和材料覆盖；若需要平衡，只允许用不读取表型值的确定性 greedy group assignment，并重新导出 manifest。
* 训练折内拟合所有 imputation、scaling、PCA、marker selection、GWAS、模型和 ensemble 权重。
* 主报告：global pooled、environment macro、coverage-stratified（材料出现 1、2、≥3 环境）Pearson/Spearman/MSE。

这是当前四作物共同的主 protocol，但 Soybean 的大量单环境材料意味着 G×E 参数不能依赖材料重复观测，必须同时报告单环境材料子集与跨环境材料子集。

### CV-GCluster — unseen genetic cluster，核心 stress test

1. 审计阶段已用多态标记 PCA 和 k-means 得到结构诊断：Maize/Wheat/Soybean 为 5 cluster，Rice 为 3 cluster；cluster 样本数量分别为 Maize `338/405/375/400/361`、Rice `386/300/314`、Wheat `436/475/210/388/444`、Soybean `2187/2818/957/2318/3324`。
2. 这些 cluster 只用于结构诊断和固定 stress-test manifest，不使用表型；模型开发中的超参、特征和 cluster 数量必须在 outer training 内重新拟合。
3. 每次整 cluster 作为 validation，训练集不得包含该 cluster 的任何材料记录；报告每 cluster 的样本数、遗传距离、worst-fold 和距离分层分数。
4. 对 Wheat 的高缺失材料，cluster assignment 必须附带 missingness mask 和 out-of-support 标记，防止缺失模式代替遗传结构。

### CV-E — unseen environment，LOEO 主协议

环境组很少，不能用普通随机 k-fold 伪造未知环境，固定使用 Leave-One-Environment-Out：

* Maize：14 folds；Rice：7 folds；Wheat：12 folds（先按规范化环境 ID）；Soybean：31 folds。
* 每折整环境进入 validation，训练中不出现该环境的任何表型。
* 预测时允许使用该环境可观测的天气协变量，但不能使用该环境的 phenotype mean、材料均值或任何后验信息。
* 小样本环境保留并单独标记：Soybean `5MN90=29`、`MN2011_12=42`、`4MN87=49`；不让它们单独决定模型排序，报告置信区间和有效样本数。
* Rice 的 E1 仅 276 条记录，单独作为低覆盖 environment stress test。

评分同时给出 LOEO macro、按样本数 weighted 和按 trait macro，避免大环境淹没小环境。

### CV-GE — unseen genotype + unseen environment，严格交集

采用 5 个 genotype group fold 和 5 个 environment group fold：

```text
g_fold = sha256("20260922|g|" + genotype_id) % 5
env_fold = 按环境表型记录数降序、依次放入当前负载最小 fold
validation = (g_fold == f) AND (env_fold == f)
train      = (g_fold != f) AND (env_fold != f)
purge      = 其余记录，不参与训练或该折评分
```

这样 validation 中的材料和环境都没有在 training 中出现。已用真实表型行检查，严格交集每折约为 Maize `405–594`、Rice `172–340`、Wheat `165–234`、Soybean `431–511`。如果某个新版本数据使任一折有效行数低于预设阈值，停止该协议并重新设计，不用随机补样。

### Pseudo blind test

在开发开始前生成并冻结 `results/folds/{crop}/pseudo_blind_manifest.csv`：

* 开发集约 85%，pseudo-blind 约 15%；材料按 genotype-only 遗传距离优先选择远端材料，环境按天气极端/低覆盖优先，但选择规则、seed 和清单一次性冻结。
* pseudo-blind 的表型在模型开发、特征选择、ensemble 权重和阈值选择期间不可读取。
* 在所有候选方案冻结后只评估一次，比较 CV-G、CV-GCluster、CV-E、CV-GE 与 pseudo-blind 的方向是否一致。
* 评估后不回头调参；若结果冲突，记录为 model risk，不把 pseudo-blind 重新并回训练集。

### Nested CV 和 OOF

* classical/ML：外层 CV-G 与 CV-GCluster，内层 3 折 CV-G 做超参和特征选择；CV-E/CV-GE 作为外部 stress test。
* 深度模型：计算昂贵时可固定少量预注册配置，并用 approximate nested CV；不得用 outer validation 做 early stopping 或模型选择。
* 每条 OOF 预测保存：`crop, trait, row_id, genotype_id, environment_id, true, pred, model, feature_set, protocol, outer_fold, seed`。
* ensemble 权重只用训练折 OOF 优化，并在 outer validation 上一次应用；不读取 pseudo-blind/test label。

## Part 4 — Baseline matrix

| Model | Input | G×E support | Expected advantage | Main risk | Compute | Priority |
|---|---|---|---|---|---|---|
| Mean / environment mean | y / environment ID | Environment mean only | sanity check、检查评分管道 | 目标统计量泄漏 | very low | P1 |
| Ridge / Elastic Net | fold-safe dosage、LD-pruned SNP、PCA | 可加环境 one-hot/交互 | 小样本、高维稳定 | 线性、对结构敏感 | low–medium | P2 |
| rrBLUP / GBLUP | fold-safe marker matrix / GRM | additive G；reaction norm 支持 G×E | GS 强基线、可解释 | GRM 计算和缺失处理 | medium–high | P2 |
| Reaction-norm GBLUP | G + environment kernel | `K_G ⊗ K_E` 或 Hadamard | 直接建模 G×E | 环境 kernel 失配 | high | P2 |
| RKHS / G×E RKHS | dosage/PCA + environment | nonlinear G、G×E | 非线性和缓冲效应 | 核参数、计算量 | high | P2 |
| Bayesian Ridge / LASSO / BayesA/B | markers | additive；可扩展 trait | 稀疏/小 n 大 p | 迭代慢、先验敏感 | medium–high | P2 |
| Random Forest | PCA/LD-pruned markers + environment | weak G×E via interaction features | 非线性 sanity check | 50K SNP 直接输入过拟合 | medium | P3 |
| XGBoost / LightGBM / CatBoost | PCA/selected SNP + environment | explicit interactions | tabular 强基线 | phenotype-based feature leakage | medium–high | P3 |
| MLP / DeepGS-like | standardized dosage + mask | concat / gated G×E | 学习高维交互 | 小样本、容量过大 | GPU | P4 |
| 1D-CNN / chromosome CNN | marker blocks | local marker patterns | 参数共享 | 无 chromosome position 元数据 | GPU | P4（位置补齐后） |
| Multimodal G×E model | dosage blocks + engineered/raw weather | FiLM/bilinear/cross-attention | environment-conditioned G | 复杂、易过拟合 | multi-GPU | P5 |
| OOF linear/rank ensemble | OOF predictions | inherits candidates | 稳定化 Pearson+Spearman | 权重泄漏/过拟合 | low | P6 |

如果 GBLUP/RKHS 在 CV-GCluster 或 CV-GE 上不能被复杂模型稳定超过，不采用复杂模型作为最终方案。

## Part 5 — Main model architecture

复杂模型只在 classical baseline 稳定后实现。由于当前发布文件没有 chromosome/position 字段，首版使用稳定 marker block；拿到官方位置元数据后再启用 chromosome-aware encoder。当前实现位于 `src/models/deep/genotype_environment_net.py`，训练入口为 `scripts/run_deep_baseline.py`。

### 5.1 已实现的 DeepGxE-Net-L

输入是记录级别的 `genotype_id × environment_id`：训练折内筛选的最多 4,096 个 marker dosage、对应 missing mask，以及由天气序列聚合得到的 `n_days + 每个变量的 mean/std/min/max/first/last/missing_fraction`。marker block 卷积和 3 层 Transformer 生成 genotype embedding；天气 MLP 生成 environment embedding；sigmoid gate 建模紧凑的 G×E 交互。默认约 1.69M 可训练参数，单性状单模型输出一个标量预测。

过拟合控制已经编码在运行入口中：marker QC、缺失插补、标准化和天气特征处理全部只在 outer training fold 拟合；inner training-only split 选择早停 epoch；最终模型用 outer training 全部记录按冻结 epoch 重训；AdamW weight decay、dropout、Huber loss、梯度裁剪和固定 seed 同时启用。每折的 group overlap、missing fraction、天气分布 shift、参数量和 runtime 写入 `results/metrics/deep*_distribution.csv`。

首轮结果：CV-G 仅作为主开发协议，严格 CV-GE 才是 blind-test 风险信号。当前完整结果分别落在 `results/metrics/deep_metrics.csv`（CV-G）、`results/metrics/deep_strict_metrics.csv`（CV-GCluster/CV-GE）和 `results/metrics/deep_cv_e_metrics.csv`（LOEO CV-E）。复杂模型只有在严格协议的 worst-fold 和 OOD 分数不恶化时才应进入提交候选。

已完成的首轮数值汇总见 `DEEP_BASELINE_RESULTS.md`。当前判断是：DeepGxE-Net-L 在 Rice/Wheat/Soybean 的严格 CV-GE 超过 genomic-ridge，但 Maize CV-GE 为 `0.0320`，低于 genomic-ridge 的 `0.1399`；模型不能整体 promotion，必须按作物和协议做 OOD 风险门控。

2026-09-23 已加入 phenotype-blind 独立测试的 easy/hard 分层、reaction-norm 架构、small/base/large 参数量消融、dynamic weather 和 fold-fitted low-rank weather 表示。完整分析与一次性独立测试结果见 `GENERALIZATION_OPTIMIZATION_REPORT.md`。独立测试现已封存，后续选择只能使用 development nested CV。

```text
Genotype dosage G:        [B, P]
Genotype missing mask M:  [B, P]
                              │ concatenate / mask
                         marker blocks (L=256)
                              │ linear 2 -> 128
                         block tokens: [B, ceil(P/256), 128]
                              │ mean/attention pooling
                         genotype embedding z_g: [B, 128]

Engineered environment:  [B, F_env]
Raw weather sequence:    [B, T, V] + padding mask
                              │ MLP 64 / TCN 64
                         environment embedding z_e: [B, 64 or 128]

z_g, z_e ── FiLM/bilinear gate ── z_ge: [B, 256]
                                  │ MLP + dropout
                         trait head: [B, n_traits]
```

实际输入上限按作物单独冻结：Maize `P≈45,920`、Rice `P≈47,813`、Wheat `P≈10,292`（缺失 mask 必须保留）、Soybean `P≈41,315`；工程特征维度为气象变量数 × 聚合统计，不能把不同作物的变量语义强行共享。天气序列使用环境内实际 `T` 和 padding mask，不把缺失日期当作真实 0。

第一版 loss 只比较 MSE、Huber；只有 OOF 证明有效才加入 Pearson/rank 正则。每个作物/性状独立输出，不默认共享一套跨作物网络。

## Part 6 — Experiment schedule

| 阶段 | 交付物 | 进入下一阶段的门槛 |
|---|---|---|
| P0 Data audit | 四作物报告、JSON、SVG、ID mapping | 已完成；Wheat mapping 已生成，下一步做链接测试。 |
| P1 Validation | fold manifests、CV-G/CV-GCluster/CV-E/CV-GE、OOF schema、均值/基因组 ridge baseline | 所有 fold 无 genotype/environment overlap；泄漏测试通过；已完成首轮 baseline。 |
| P2 Classical GS | mean、GBLUP、reaction norm、RKHS、Bayesian baseline | 在 CV-G 和 CV-GE 上有可复现 OOF；报告稳定性，不只看均值。 |
| P3 ML | Ridge/Elastic Net/tree boosting | 只使用 fold-safe PCA/LD/marker selection；比较 compute/performance。 |
| P4 DL | MLP、低容量 CNN | 至少与 GBLUP/RKHS 在 strict CV 上公平比较。 |
| P5 Multimodal | G + engineered weather + raw weather + G×E ablations | 完成 genotype-only、environment-only、G+E、G+E+G×E 逐层消融。 |
| P6 OOF ensemble | nonnegative sum-to-one、z-score、rank ensemble | 权重只由 training OOF 得到；没有 test label。 |
| P7 Pseudo blind | 一次性冻结评估 | 与严格 CV 方向一致，或风险被明确记录。 |
| P8 Final | 全训练表型重训、官方预测、代码/配置/环境锁定 | 只应用冻结 preprocessing，做输入 QC 和预测置信度。 |

模型 promotion 同时报告 `mean score / fold std / worst fold / OOD score / runtime`。不能用自定义综合分数掩盖某个协议失败。

## Part 7 — Compute estimate

当前机器审计到的资源为约 112 CPU、1 TiB RAM、8 张 NVIDIA A800 80 GB。原始四作物数据约 1.8 GB。

| 工作 | CPU | RAM | GPU | 存储 | 主要瓶颈 |
|---|---:|---:|---:|---:|---|
| P0 audit（当前脚本） | 8–32 核 | 2–8 GB 峰值 | 不需要 | 报告/图约几十 MB | 大豆 4.8 亿个基因型单元的 CSV 解析 |
| GRM / GBLUP | 16–64 核 | 4–32 GB；Soybean GRM 需关注 n² | 不需要 | 每 crop 数百 MB | `O(n²p)` 及重复 outer/inner folds |
| RKHS / reaction norm | 16–64 核 | 8–64 GB | 可选 | 1–10 GB | kernel 分解和多环境交互 |
| Ridge/boosting | 8–32 核 | 8–64 GB | 可选 | 1–10 GB | 高维特征和 nested search |
| MLP/CNN | 8–32 核供数 | 16–128 GB | 1 张 A800 足够，ensemble 可并行 2–8 张 | 10–100 GB checkpoints | 小 n/大 p 下容量与调参过拟合 |
| 多模态模型 | 16–64 核 | 32–128 GB | 1–4 张 A800 | 50–200 GB | raw weather 序列、重复 OOF 和模型保存 |

不把虚假的精确时间作为计划门槛；每个实验记录 wall time、CPU/RAM/GPU memory、数据版本和 seed。

## Part 8 — Blind-test strategy

1. **冻结协议**：先冻结 ID mapping、缺失处理、marker QC、特征集合候选、fold manifest 和 scoring 实现。
2. **训练与选择**：只用 development 表型；outer/inner CV 生成 OOF，比较 strict CV-G/CV-GCluster/CV-E/CV-GE；保存每一条 OOF 预测。
3. **集成**：在每个 outer training fold 的 OOF 上学习 nonnegative、sum-to-one 的 raw/z-score/rank ensemble 权重；验证折只做一次前向预测。
4. **Pseudo-blind**：冻结所有选择后只评估一次，保留结果，不回调参数。
5. **官方数据发布后**：
   * 校验样本 ID、marker ID、allele alignment、缺失比例和遗传距离；
   * 只应用训练阶段冻结的 imputer/scaler/PCA/marker 清单；
   * 用全部允许的训练表型重训最终模型；
   * 生成每个 crop/trait 的预测、ensemble 方差、fold 方差和 genetic/environment distance；
   * 检查输出行数、ID 顺序、精度和提交格式；不使用隐藏标签。
6. **置信度**：将 ensemble variance、fold prediction variance、genetic distance、environment distance 作为诊断，不用事后删掉难预测样本来提高分数。

## Part 9 — Risks and mitigations

| 风险 | 当前证据 | 缓解措施 |
|---|---|---|
| Overfitting | 小 n 大 p；复杂模型候选多 | GBLUP/RKHS 先行、nested CV、容量上限、一次 pseudo-blind。 |
| Genotype leakage | 多环境重复材料；Maize 有重复材料列 | 全部 genotype group CV；审计重复列并固定处理。 |
| Environment leakage | 环境均值很容易泄漏；Wheat ID 不一致 | LOEO；环境统计只在 train 算；Wheat mapping 版本化。 |
| Population structure | PCA PC1/PC2 解释比例和相似度分布显著 | CV-GCluster、遗传距离分层、报告 worst cluster。 |
| Relatedness | 部分材料高度相似，随机材料折会过于乐观 | cluster/distance holdout，不能只报 CV0。 |
| Environment shift | Soybean 31 环境且样本数 29–929 | LOEO、低样本环境单列置信区间、天气距离诊断。 |
| Genotype shift | Wheat 缺失重，Maize 有全缺失杂交组合 | missingness mask、训练内 imputation、out-of-support 标记。 |
| Sample imbalance | Soybean 11,266 个材料只有一个环境；Rice E1 低覆盖 | coverage-stratified 指标、macro/weighted 同时报。 |
| Phenotype noise | 多环境表型未必平衡，Trait_1 有缺失 | 去重、重复策略预注册、robust loss 只用 OOF 验证。 |
| Small n / large p | Rice 1,000×60K，Maize 1,879×50K | MAF/LD/PCA fold-safe benchmark，控制 DL 容量。 |
| Computational cost | Soybean genotype CSV 约 1.45 GB，nested CV 昂贵 | 分层实验、固定 benchmark、记录 compute/performance。 |
| Metadata absence | 当前没有 chromosome/position 字段 | 首版不用 chromosome CNN；拿到元数据后单独做增量实验。 |

## 项目目录设计

原始发布数据继续保留在现有 `Dataset/`，不复制、不改写。项目代码和生成物按以下边界组织：

```text
AI_breeding/
├── Dataset/                         # 原始发布数据，只读
├── data/
│   ├── raw/                         # 原始数据入口说明/MD5，不复制大文件
│   ├── interim/                     # ID mapping、QC manifest、fold 前中间表
│   └── processed/                   # fold 内生成的特征；禁止提交临时全量标签特征
├── configs/
│   ├── audit.yaml
│   ├── validation.yaml
│   ├── crops/maize.yaml
│   ├── crops/rice.yaml
│   ├── crops/wheat.yaml
│   └── crops/soybean.yaml
├── src/
│   ├── data/                        # CSV reader、schema、ID linkage、manifest
│   ├── preprocessing/               # dosage、missing mask、weather、PCA/LD
│   ├── validation/                  # CV-G/GCluster/E/GE、nested CV、purge
│   ├── models/
│   │   ├── classical/               # rrBLUP、GBLUP、reaction norm、RKHS
│   │   ├── ml/                      # Ridge、Elastic Net、boosting
│   │   ├── deep/                    # MLP、CNN、低容量 DeepGS
│   │   └── multimodal/              # G×E weather model
│   ├── evaluation/                  # Pearson、Spearman、MSE、OOF schema
│   └── ensemble/                    # OOF raw/z/rank ensemble、uncertainty
├── scripts/
│   ├── data_audit.py                # 当前已实现
│   ├── build_environment_mapping.py
│   ├── build_folds.py
│   ├── run_baselines.py
│   ├── run_nested_cv.py
│   └── make_submission.py
├── notebooks/                       # 仅探索和可视化，不作为生产入口
├── results/
│   ├── folds/{crop}/
│   ├── oof/{crop}/{protocol}/
│   ├── predictions/{crop}/
│   ├── metrics/{crop}/
│   ├── ablations/{crop}/
│   └── checkpoints/                 # 大文件不入版本库，记录 manifest
├── reports/                         # 审计、模型报告、图表
├── tests/                           # schema、linkage、fold leakage、metric tests
├── experiments.csv                  # 每次实验的可复现索引
├── MODEL_TRAINING_VALIDATION_PLAN.md
└── README.md
```

### 目录使用规则

* `Dataset/` 只读；脚本通过配置引用路径，不在运行中覆盖原文件。
* `data/interim/` 保存可重建的 mapping/fold manifest，并记录输入 MD5、seed、代码版本。
* `results/oof/` 是模型比较的唯一来源；任何 ensemble 都必须能追溯到 OOF 文件。
* `notebooks/` 不承载最终训练逻辑；生产运行从 `configs/` + `scripts/` 开始。
* 每个实验记录 `experiment_id, crop, trait, protocol, model, feature_set, seed, fold, Pearson, Spearman, CompetitionScore, MSE, RMSE, runtime, memory, notes`。

## 下一步

当前应执行 P1，而不是直接训练深度模型：

1. 校验已生成的 Wheat `data/interim/environment_id_map.csv` 并写入映射测试。
2. 实现四种 validation manifest，运行 overlap/leakage 单元测试。
3. 固定 phenotype missing、全缺失材料、单态标记和 weather 缺失处理策略。
4. 已完成 mean 与 low-rank genomic ridge baseline；下一步在相同 OOF 协议上补跑正式 GBLUP、reaction-norm GBLUP、RKHS，保存 OOF 后再进入 ML/DL。
