# MultiLabelSMOTE 验收与回归报告

日期：2026-11-23

## 1. 新增能力

新增 `imblearn.over_sampling.MultiLabelSMOTE`，把多标签指示矩阵按
**标签集合（label-set combination）** 作为重采样单位；同一组合内部的样本才互相
做 SMOTE 插值，因此合成样本的标签集合必然是训练集中出现过的组合。

新增/修改文件：

| 文件 | 作用 |
| --- | --- |
| `imblearn/over_sampling/_multilabel.py` | 多标签校验、标签集合提取、按组合分组（稠密/稀疏统一） |
| `imblearn/over_sampling/_multilabel_smote.py` | `MultiLabelSMOTE` 采样器 |
| `imblearn/over_sampling/tests/test_multilabel_smote.py` | 80 个专用测试 |
| `scripts/acceptance_multilabel.py` | 可复现验收脚本，产出 `output/multilabel_acceptance.*` |
| `imblearn/over_sampling/__init__.py` | 导出 `MultiLabelSMOTE` |
| `imblearn/utils/estimator_checks.py` | 多标签采样器跳过 1D 目标的通用检查；DataFrame 列名检查补 2D 目标 |
| `imblearn/utils/_test_common/instance_generator.py` | 注册实例与通用检查的 xfail 说明 |
| `doc/over_sampling.rst`、`doc/references/over_sampling.rst`、`doc/whats_new/v0.15.rst` | 用户指南、API 与 changelog |

旧单标签采样器（SMOTE / RandomOverSampler / ADASYN / SMOTE-NC / SMOTE-N / SVM-SMOTE 等）
的入口、参数与数值结果均未改动。

## 2. 需求覆盖

### 一、需要覆盖的形态

| # | 需求 | 实现与验证 |
| --- | --- | --- |
| 1 | 稠密/稀疏指示矩阵，稀疏不得先转稠密 | `check_multilabel_target` 对 `csr/csc/coo/lil/dok/bsr/dia`（含 scipy sparse array 与 pandas sparse DataFrame）全程保持稀疏；`label_sets` 直接读 CSR `indptr/indices`。`test_sparse_target_is_never_densified` 在 `toarray/todense` 被 monkeypatch 成抛错时仍能完成校验与分组 |
| 2 | 按组合统计频次、补长尾 | `group_by_label_set` 以标签元组为键；`test_rare_combination_grown_to_budget` 断言 `(0,2)` 从 8 补到 30 |
| 3 | 合成标签集合必须训练集真实存在 | 组合内插值 + `indicator_block` 仅使用该组合；`test_synthetic_label_set_seen_in_training`（稠密/稀疏）、`test_no_unseen_label_combination_can_be_built` |
| 4 | 每标签预算与整体预算同时可控 | `label_budget`（硬缺口 `raise` / 软缺口 `warn`，写入 `label_budget_shortfall_`）、`total_budget`（最大余数按比例缩放）。`test_label_budget_does_not_flood_other_label` 验证打标签 0 的预算不会淹没标签 1 |
| 5 | 单正标签 / 全零 / 多标签行行为确定 | 单正标签正常成组；全零行默认保留（`zero_label_rows='keep'`）、`'resample'` 时作为 `()` 组补足；`test_wide_label_matrix_many_labels_per_row` 覆盖一行 10 个标签 |
| 6 | 只出现一次的组合 keep/merge/skip | `singleton_strategy`，`test_singleton_strategy_keep_merge_skip` 分别断言保留/借邻近组合插值/删除该行 |
| 7 | 行列对齐、dtype、索引不变 | `test_output_alignment_dtype_and_shape` 断言原始行逐位不变、`sample_indices_` 前缀为 `0..n-1`；`test_dtype_preserved` 覆盖 int8/int32/int64/float32/bool；DataFrame 列名与索引保持 |
| 8 | 下游分类器可直接消费 | `output_sparse=False` 时稀疏输入直接产出稠密 `y`；Pipeline + `MultiOutputClassifier`、`OneVsRestClassifier` 直接吃采样结果（含稀疏目标） |
| 9 | 固定种子逐位一致 | `test_fixed_seed_bit_for_bit_deterministic`（稠密/稀疏）、`output/determinism_evidence.json` 给出 SHA-256 |

### 二、不能让步的地方

| # | 约束 | 落点 |
| --- | --- | --- |
| 1 | X/y 逐行同源，重排同时作用于两边 | 原始行用同一 `original_positions` 索引 X 与 y；合成行先追加 X 再追加对应 y 块；`_check_alignment` 运行时自查行数与列数 |
| 2 | 合成过程不得读取验证集/未来批次标签 | `_synthesise` 只接收训练批次的 `base_indices/pool_indices`；`test_generated_rows_ignore_held_out_data` 验证改变留出集不改变输出 |
| 3 | 稀疏内存不随标签数平方增长 | 标签元组长度 ≤ 该行 nnz；`_groups_from_keys` 在稠密 padding 超预算时改用每行 bytes key；合成块只存 `n_rows × len(label_set)` 个 1；百万行 × 50 标签分组峰值约 143 MB（`output/multilabel_acceptance.json`） |
| 4 | 旧单标签路径不变 | 仅新增模块与导出；`test_single_label_samplers_still_public`、`test_single_label_sampler_rejects_multilabel_and_keeps_results`；原有采样器测试全部通过（见第 5 节） |

### 三、失败时

| # | 需求 | 落点 |
| --- | --- | --- |
| 1 | 非法取值/行数不一致/全零行过多报位置与计数 | 非法值报出总数、取值、前 5 个 `(row, column)`；行列不一致报双方行数；`zero_label_rows_threshold` 报出全零行数与占比及前 5 行 |
| 2 | 预算无法达到目标时说明缺口 | `label_budget` 缺口以 `short by N synthetic sample(s)` 逐标签列出（raise）或告警并写入 `label_budget_shortfall_`，不静默少采 |
| 3 | 稀疏格式不支持时给可执行建议 | 非数值稀疏 dtype 建议 `y.astype(np.int8)`；校验失败建议 `y.tocsr()`；1D 目标建议改用 SMOTE / `MultiLabelBinarizer` 转 2D |

### 四、已有行为

- 单标签采样器数值结果与随机性不变（第 5 节测试与基线一致）。
- 采样链与 Pipeline 组合继续可用（`MultiLabelSMOTE` 是 `BaseSampler` 子类，`fit_resample` 可直接放入 `imblearn.pipeline.Pipeline`）。
- 不引入新依赖：仅使用 numpy / scipy / scikit-learn / sklearn-compat。

## 3. 确定性逐位对比（`random_state=42`）

两次独立 `fit_resample`（全新采样器实例、同一整数种子）：

```
dense  X equal: True   max|ΔX| = 0.0
dense  y equal: True   max|Δy| = 0.0
sparse X equal: True
sparse y equal: True
X sha256 run1 = a321ba99e6b4d66c74f128c135aad1519f977ba218f4c62d01c98a5252a66c90
X sha256 run2 = a321ba99e6b4d66c74f128c135aad1519f977ba218f4c62d01c98a5252a66c90
y sha256 run1 = ff5da7fecd509f919210de25ca779473cc0e2e827caaefb006752dea1fa3875f
y sha256 run2 = ff5da7fecd509f919210de25ca779473cc0e2e827caaefb006752dea1fa3875f
```

完整数据见 `output/determinism_evidence.json`。

## 4. 采样结果摘要（长尾数据集，62 行 × 4 标签）

原始组合计数：`(0,)=30, (1,)=20, (0,2)=8, (3,)=1, ()=3`

`auto` 策略（每个非空组合补到 30）：

```
synthetic per combination: {'(0,)': 0, '(1,)': 10, '(0, 2)': 22, '(3,)': 0}
稀有组合 (0,2): 8 -> 30
输出行数: 62 -> 94（稠密与稀疏一致）
```

`(3,)` 只出现一次，默认 `singleton_strategy='keep'`：保留原样本、不合成，并发出一条
说明性 `UserWarning`；`'merge'` 则借最近的宿主组合插 29 条，`'skip'` 删除该行。

预算：

```
total_budget=10           -> 恰好 10 个合成样本
label_budget={2:5} (warn) -> 含标签 2 的合成样本恰好 5，缺口 {2: 17} 明确上报
```

## 5. 原有采样器测试结果

环境：Python 3.14，numpy 2.5.5，scipy 1.18.1，scikit-learn 1.10.dev0。

屏蔽本机 scipy 1.18 的既有弃用告警后：

```
pytest imblearn/over_sampling imblearn/under_sampling imblearn/combine
=> 273 passed, 1 skipped        (output/sampler_tests.log + .xml)
```

默认配置（不屏蔽告警）：

```
=> 7 failed, 260 passed, 1 skipped   (output/sampler_tests_default.log + .xml)
```

这 7 个失败全部与 SMOTEN 有关，且已在**未改动的基线**上逐一复现（`git checkout` 后同命令
同样 7 个失败）：根因是 scipy 1.18 弃用 `KDTree.distance_matrix`，被仓库
`filterwarnings = ["error::DeprecationWarning"]` 升级为错误，与本次改动无关。
失败清单见 `output/sampler_tests_default_junit.xml`。

全仓回归（屏蔽预存弃用告警）：

```
pytest imblearn -n 8
=> 1 failed, 2647 passed, 70 skipped, 37 xfailed, 5 xpassed
```

唯一失败 `test_check_param_validation[BalancedRandomForestClassifier]` 在基线上同样失败
（scikit-learn 1.10.dev0 与该估计器参数约束不兼容），本次改动无关。
5 个 `xpassed` 也都是基线已有的宽松 xfail（RUSBoost / NearMiss 的 FIXME 以及需要联网的
`test_fetch`），不属于新增采样器。

新增 `MultiLabelSMOTE` 的通用 estimator 检查：18 passed，26 xfailed（全部是“需要 2D
多标签目标、1D 单标签检查结构性不适用”并已注明原因），1 skipped；专用测试 80 个全部通过。

## 6. 输出文件

- `output/multilabel_acceptance.json`：验收机器可读结果
- `output/multilabel_acceptance_summary.txt`：验收摘要
- `output/determinism_evidence.json`：固定种子逐位对比与 SHA-256
- `output/sampler_tests.log` / `output/sampler_tests_junit.xml`：屏蔽弃用告警后的采样器测试
- `output/sampler_tests_default.log` / `output/sampler_tests_default_junit.xml`：默认配置结果
- `output/full_suite.log`：全仓回归日志
