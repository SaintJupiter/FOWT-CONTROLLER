# 论文—代码—证据一致性审计（第一步）

审计日期：2026-07-10  
论文：`/Users/saintyoung/Desktop/小论文/6.28短时风况预测辅助的浮式风机主动压载调节算法.docx`  
仓库：`/Users/saintyoung/Desktop/FOWT-CONTROLLER-main`

## 1. 结论先行

当前论文的预测模型精度结果可以追溯，但表 2 的核心控制收益不能归因于论文所述的“LSTM 预测 + 事件准入 + 姿态融合 + 候选动作规划”。表 2 的 29.04% 节泵和 32.17% 启停减少来自一个加宽姿态反馈死区的控制 profile；运行时预测 provider 被明确设为 `None`。

因此，本次没有直接删除或重写原程序。若此时按论文叙述“超级精简”，很可能把错误的因果归因固化进导师版。第一步应先冻结可信算法身份，再生成独立、可复现的导师提交包。

推荐用仓库中已经存在的“实际启用预测 provider”链作为行为刻画起点：10 个工况、每个 6 h，LSTM provider 实际运行并生成 10 份 planner 日志。该链累计泵量由 9016.34 m³ 降至 8474.37 m³，表面差值为 6.01%；`t>5°` 减少 64 s，`t>7.5°` 减少 44 s，`t>10°` 不增加。但它也不是严格的“只替换预测源”对照：current-only 的姿态融合增益为 0.40，learned 为 0.45。仓库缺少 current-only gain=0.45 的 6 h 运行，因此 6.01% 不能全部归因于预测，更不能直接替换为论文 headline。

## 2. 备份与操作边界

- 已在任何代码改动前创建归档：`backups/pre_simplification_code_20260710.tar.gz`（265 MB）。
- SHA-256：`128724663907397c054793013e8bace11fbc184a1554b3c7f7736e5fefed5b95`。
- 校验文件：`backups/pre_simplification_code_20260710.sha256`。
- 归档包含代码、配置、文档、论文修订材料及未跟踪的小型文件；排除 `.git`、虚拟环境、缓存，以及约 88 GB 的 `data/`、`results/`、`outputs/` 大型可再生数据与结果目录。
- 当前工作树原本已有大量用户修改和未跟踪文件。本次不覆盖、不回退、不删除这些内容。

## 3. 最高优先级证据冲突

### 3.1 论文表 2 数值来源

论文中的下列数值与文件完全一致：

- 对照泵量：146843.9607 m³；
- “预测辅助”泵量：104193.9911 m³；
- 节泵：29.0444%；
- 启停：108899 次降至 73869 次，即 32.167%。

来源：

- `outputs/wind_prediction/selector_positive_add40_6h_v1/combined_positive_summary_170case_6h/positive_combined_total.csv`
- `scripts/analysis/summarize_selector_positive_add40_combined_v1.py`

该 170 组不是一个冻结的单次 170-case run，而是从旧 101 组中筛选 50 个 positive case，再与新 120 组 positive case 拼接为 170 个。仓库没有对应单一 170-case 运行协议、统一哈希或端到端 manifest。

### 3.2 实际控制器身份

两批来源运行均使用：

```text
primary_control_profile = dc_preserving_deadband_engineered_v1
```

但 runner 对这一 profile 明确执行：

```python
provider = None
```

代码证据：

- `scripts/analysis/run_prediction_primary_casebook.py:6035-6045`
- `scripts/analysis/run_prediction_primary_casebook.py:6894-6940`
- `scripts/analysis/run_prediction_primary_casebook.py:6941-6957`

实际变化是 PID 姿态死区被设为约 1.5°；随后 `run_closed_loop_case` 收到的 `preview_trim_provider` 为 `None`。两批 summary 中所有 case 的 `primary_candidate_ratio`、`primary_applied_ratio` 和 `primary_delta_mean_kg` 均为 0，输出目录也没有 planner 日志。

结论：29.04%/32.17% 是“加宽死区/目标释放的反馈经济性策略”结果，不是预测辅助规划结果。协议中的 `forecast_source_effective=lstm_dual_head_preview` 只是错误泄漏的元数据，不能证明预测模型参与控制。

### 3.3 姿态代价没有在论文主结论中充分披露

该 170 组 positive-only 拼接虽然节泵，但后处理显示：

- `t>2°` 增加 64908 s；
- `t>3°` 增加 5625 s；
- `t>4°` 增加 1308 s；
- 汇总中的 `t>5°`、`t>7.5°`、`t>10°` 也分别增加约 372 s、16 s、3 s。

因此，这一结果不仅算法身份错误，还存在筛选偏差和服务姿态代价披露不足的问题。

## 4. 论文主张与程序对应关系

| 论文主张 | 程序/证据 | 判定 | 必须修正的边界 |
|---|---|---:|---|
| 过去 120 min 预测未来 60 min | `data/processed/.../metadata.json`；`scripts/modeling/train_ballast_lstm.py`；`src/wind_prediction/forecast_adapter.py` | 一致 | 12 个 10 min 输入点的时间戳跨度是 110 min，可表述为覆盖 12 个 10 min 时段 |
| LSTM 双头输出 6 步风矢量与 8 个事件概率 | 训练脚本和冻结 `lstm_config.json` | 一致 | 模型结构与输出维度可追溯 |
| 风速 MAE 0.775、t+10 0.492、t+60 0.985、方向 MAE 6.812、F1 0.631 | 冻结 regression/event metrics CSV | 一致 | 表 1 可保留 |
| 未来 60 min 划分 0–20、20–40、40–60 min | `src/wind_prediction/ballast_planner.py:15-19,457-469` | 一致 | 实现是三段均值代理，不是逐个 10 min 步控制 |
| 事件概率作为分段准入量 `q_k` | `ballast_planner_provider.py:4913-4915` | 不一致 | 当前函数固定返回 `[1,1,1]`，基础分段需求没有被事件概率准入或缩放 |
| 预测风况形成等效姿态需求 | `ballast_planner.py:357-376` | 部分一致 | 是 `(V/12)^2` 和风向映射的启发式压力代理，不是经 6DOF 标定的等效力矩/姿态模型 |
| 预测需求与实时 pitch/roll 融合 | `ballast_planner.py:198-300`；provider `:6518-6524` | 代码存在 | 论文 170 组运行没有 provider，因而没有启用这条链 |
| 5 个动作、三段组合、共 125 个候选 | `ballast_planner.py:21,544-760,846`；provider `:6535-6562` | 结构存在 | 170 组没有运行；候选预演只是代理状态更新，不是 6DOF rollout |
| 候选中检查舱容、泵能力、姿态安全边界 | `ballast_planner.py:476-486,574-662` | 部分一致 | 有舱容和可选 safety floor；没有按段模拟泵率、ramp、dwell 和可达量，也没有真实 6DOF 安全预演 |
| 首动作转换为压载目标 | provider `:7331-7356` | 代码存在 | 170 组目标实际来自反馈 PID |
| 水泵受能力、舱容、启停滞环和 ramp 约束 | `archive/legacy_fowt_control/core_model.py:255-260,484-635` | 一致 | 这是 plant 执行层，不是候选规划预演层 |
| 六自由度等效时域模型 | `archive/legacy_fowt_control/core_model.py:199-204,686-770` | 部分一致 | 可称“简化等效 6DOF”；不可暗示为全耦合高保真水动力模型 |
| 170×6 h = 1020 h 预测辅助验证 | 两批 positive-only 结果拼接 | 不一致 | 时长算术成立，但控制器无预测 provider，且不是单一冻结 run |
| 指标包含 peak、RMS、超阈值暴露 | runner 和后处理 | 部分一致 | 170 主汇总没有 RMS 字段或正式结果；应删除 RMS 承诺或补算 |
| 预测可作用工况占测试集 33.77% | `mine_regime_candidate_library_v3.py` 与 prevalence 表可复算 | 口径不一致 | 92,570/274,122=33.7696% 合并了 validation+test，并使用真实未来 oracle 标签；test-only 同口径为 49,918/140,173=35.6117%，两者都不是 learned runtime 在线识别率 |

## 5. 实际启用 provider 的行为刻画链

仓库已有一组实际启用 provider 的冻结对照：

| 项目 | current-only | learned LSTM |
|---|---:|---:|
| profile | `rawenv_holdpause_barrier_reliefcap_adaptive_v1` | 同左 |
| posture-state gain | 0.40 | 0.45 |
| case 数 | 10 | 10 |
| 单 case 时长 | 6 h | 6 h |
| 累计泵量 | 9016.3416 m³ | 8474.3733 m³ |
| `t>5°` | 36431 s | 36367 s |
| `t>7.5°` | 327 s | 283 s |
| `t>10°` | 10 s | 10 s |

表面节泵差值为 6.01096%。learned 目录中存在 10 份 planner 日志，summary 的 candidate/applied 指标非零，说明预测规划链确实运行；但 gain 不相等意味着该结果同时包含“预测源变化”和“姿态融合增益调参”的影响，不能作为严格预测消融。

证据：

- `outputs/wind_prediction/guard10_6h_current_only_baseline_20260606/run_protocol.json`
- `outputs/wind_prediction/guard10_6h_production_learned_gain045_20260606/run_protocol.json`
- `outputs/wind_prediction/guard10_6h_production_learned_gain045_20260606/planner_logs/`
- `outputs/wind_prediction/paper_20260606_execution/production_near_6h_audit.md`
- `outputs/wind_prediction/paper_20260606_execution/claim_evidence_map.md`

限制：这仍是 10-case、6 h 的 production-near casebook，不是长期部署验证；当前 production provider 仍包含大量实验覆盖层，事件概率的基础准入仍未按论文公式生效。使用该链作为论文证据前，至少需要补跑同 profile、同 gain=0.45、仅 forecast source 不同的 current-only 对照。

33.77% 的来源也可以追溯到：

- `scripts/analysis/mine_regime_candidate_library_v3.py:1-7,337-405`；
- `outputs/wind_prediction/regime_conditioned_policy_development_v1/regime_mining_v3/raw_tables/regime_mining_v3_prevalence_all_splits.csv`。

该脚本自身明确说明它使用 actual future wind 生成 oracle 分类，只用于离线工况挖掘，不证明 learned runtime 识别。论文可把它改写为“validation+test 离线 oracle 候选片段占比 33.77%”；若坚持“测试集”，应使用 35.61%，并仍需注明这是 oracle 标签而不是在线识别率。

## 6. 复杂度与可信度审计

### 6.1 规模

- `src/wind_prediction/`：17 个 Python 文件，16,944 行。
- `scripts/`：338 个 Python 文件，143,952 行。
- 其中 `scripts/analysis/`：262 个文件，109,603 行。
- `src/wind_prediction/ballast_planner_provider.py`：11,140 行，占 `src/wind_prediction` 的 65.7%。
- `BallastPlannerPreviewProvider.__init__`：约 289 个参数、745 个 `self` 属性。
- provider `compute`：约 2,183 行；最终日志/返回对象有 200 余个显式字段。
- `scripts/analysis/run_prediction_primary_casebook.py`：8,060 行；参数解析约 2,052 行、309 个 CLI 参数；主流程约 4,436 行。
- registry 中约 79 个 profile：1 个 production、4 个 diagnostic、5 个 historical、69 个 isolated experiment。

### 6.2 隐藏依赖与测试缺口

- 主入口通过 `sys.path` 直接导入 `archive/legacy_fowt_control`；所以 `archive/` 不是可直接删除的历史垃圾。
- 关键依赖包括 `run_validation.py`、`controllers_extras.py`、`core_model.py`、`controllers.py`、`defaults.py`、`wind_env.py`。
- 当前没有正规的 `tests/` 目录；名称含 `test` 的文件主要是实验脚本，不是回归测试。
- `scripts/analysis/check_attitude_allocation_contract.py` 当前导入一个不存在的 `build_attitude_allocation_matrix`，该检查本身已失效。
- `configs/production_controller_v1.json` 已把 production 状态标成 `draft_frozen_for_architecture_cleanup`，也承认 provider 过大和参数过多。

这些结构性问题解释了为什么协议元数据、实际 provider 身份、结果归因能够长期漂移而没有被自动发现。

## 7. 第一阶段验证结果

已执行的只读/非破坏性检查：

- 全部 `src/` 和 `scripts/` Python 文件语法编译通过；
- `wind_prediction` 包可导入；
- dataset manifest 检查通过；
- forecast contract 的 oracle、persistence、current-only、learned 四种来源检查通过；
- 现有三案例 summary smoke checker 通过；
- attitude allocation contract checker 失败，原因是检查脚本导入的函数不存在。

注意：语法通过不等于算法正确。当前最缺的是“运行协议必须与实际 provider/模型身份一致”的回归测试和“论文主张—证据文件”自动核验。

## 8. 精简候选方案

### 方案 A：以真实预测链生成独立导师版（推荐）

保留 LSTM adapter、三段压力代理、姿态融合、125 候选核心、简化 6DOF plant、水泵约束、单一 production 配置、单一 paired runner 和证据生成器。删除/隔离 69 个 isolated profile、绝大多数 overlay、历史 selector/positive-only 汇总和与正文无关的分析脚本。

优点：与论文主题一致，可把已启用 provider 的 10-case 链作为 characterization 起点；可以通过小型测试锁定算法身份。代价：必须先补跑 gain 完全一致的 current-only 对照；论文表 2 随后重写，170 组需要用真实 provider 重新跑，或者诚实降级为经过严格匹配的 10-case 结果。

### 方案 B：保留 29.04%，把论文改成宽死区反馈经济性策略

导师版只需保留反馈 MIMO/PID、deadband/target-release、水泵执行和 6DOF 仿真；LSTM 与 125 候选规划不再作为表 2 因果机制。

优点：最容易复现现有 170 组数字。代价：论文题目、摘要、方法和贡献都要大幅改写，已不再是“短时风况预测辅助”算法。

### 方案 C：先修复论文所述机制，再重跑完整 170 组

先使事件概率 `q_k` 真正参与分段准入，补齐候选泵率/可达量/姿态约束，冻结小型实现和测试，再用无筛选 paired 170-case protocol 重跑。

优点：方法、代码和证据最完整。代价：工作量和计算量最大，最终收益不会等于现有 29.04%。

## 9. 推荐的精简原则

在选择方案 A/B/C 之前不删除原仓库。选择后新建一个独立的导师提交目录，通过 deletion test 决定是否保留文件：

1. 是否直接参与选定算法的运行时调用链？
2. 是否直接生成论文正文中保留的表、图或指标？
3. 是否是复现所需的冻结配置、模型接口或最小数据契约？
4. 是否是验证算法身份、安全约束或结果完整性的测试？

四项均为“否”的内容不进入导师版；实验遗留保留在原仓库/备份中，不混入提交包。

导师版至少应强制具备：

- 一个明确的算法身份和一个 production 配置；
- 一个训练/加载契约和一个 paired evaluation 入口；
- 一个结果 manifest，记录代码版本、配置哈希、模型哈希、case 列表和实际 provider 身份；
- 覆盖 forecast shape、事件准入、姿态融合、候选约束、provider 启用、paired 汇总的自动测试；
- 运行协议与实际执行对象不一致时立即失败，而不是只写误导性 metadata。

## 10. 论文文档的附带排版问题

对 34 页 DOCX 进行了渲染检查：

- 第 32 页为空白页；
- 参考文献末尾存在空的 `[21]`；
- 第 7–8 页仍有橙色下划线/修订痕迹；
- 标题在分页处断开。

这些不影响本次代码审计结论，但提交前应清理。

## 11. 下一步决策门

在程序精简前必须选择算法身份。推荐选择方案 A：以实际启用 provider 的链为核心，另建导师版，不破坏当前仓库；先补跑严格匹配的 current-only gain=0.45 对照，得到可归因的 10-case 数字，再决定是否重跑无筛选 170 组。在此之前，不应把 6.01% 写成纯预测收益。
