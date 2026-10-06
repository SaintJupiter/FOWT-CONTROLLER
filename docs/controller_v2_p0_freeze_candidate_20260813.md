# Controller V2 P0基线冻结候选清单

更新日期：2026年8月13日  
状态：候选清单已形成，阻断项尚未关闭，未暂存、提交或推送  
适用范围：科研版V2的框架起点，不代表平台物理模型或控制性能已经验证

## 1. 当前判断

当前工作区具备形成P0基线的基础，但不能整体提交。工作区仍有数百个未跟踪文件，并混有旧论文产物、一次性分析脚本、本机工具配置和历史试验结果。P0必须按白名单冻结，禁止使用`git add -A`。

现有证据支持以下有限结论：

- 原有389项仓库测试于2026年8月13日重新运行并全部通过。补充证据清单及其破坏性校验后，当前415项测试再次全部通过。
- 4类固定六小时工况的8条控制链轨迹均完成，现存20个输出文件与清单哈希一致。
- 开放环审计的12项方向、质量属性和独立进排水检查均通过，5个输出文件与清单哈希一致。
- 同状态预测内容探针能够仅改变未来风矢量，并引起候选排序和可执行目标变化。

上述内容只证明当前文件组合能够运行、信息链路已经接通并满足已有检查。正式六小时链路仍直接使用`archive/legacy_fowt_control/core_model.py`，其中的状态推进只能按参考姿态附近的小角度线性近似理解。`research_incremental_v1`通过扣除参考载荷建立增量零点，并未求解重量、浮力和系泊预张力共同作用下的真实静平衡。因此不得据此调节控制参数或形成节泵、姿态及完整六自由度物理结论。由于v9和v6均来自脏工作区，它们只能作为历史链路证据，不能代替最终干净候选上的复现运行。

## 2. 运行身份

| 项目 | 当前值 |
| --- | --- |
| 分支 | `codex/controller-architecture-v2` |
| 运行证据对应HEAD | `0aa3e125cc254407fb1fef8911aace6b33139dcc` |
| 证据生成时工作区 | 脏工作区，清单另存实际运行文件哈希 |
| Python | `3.12.13` |
| NumPy | `2.4.3` |
| PyTorch | `2.13.0.dev20260530` |
| pandas | `3.0.2` |
| SciPy | `1.17.1` |
| openpyxl | `3.1.5` |

证据生成时的Git提交不能单独重建运行状态。冻结依据必须是候选文件内容、外部数据和模型哈希，而不是只记录HEAD。

## 3. Git候选文件组

### A. V2控制核心与配置

```text
src/wind_prediction/__init__.py
src/wind_prediction/controller.py
src/wind_prediction/action_plan.py
src/wind_prediction/ballast_allocation.py
src/wind_prediction/controller_chain_runner.py
src/wind_prediction/controller_configuration.py
src/wind_prediction/controller_core.py
src/wind_prediction/controller_plant_adapter.py
src/wind_prediction/controller_replay_adapter.py
src/wind_prediction/controller_runtime.py
src/wind_prediction/dataset_manifest.py
src/wind_prediction/execution_rollout.py
src/wind_prediction/forecast_action_policy.py
src/wind_prediction/forecast_adapter.py
src/wind_prediction/forecast_contract.py
src/wind_prediction/forecast_evidence.py
src/wind_prediction/replay_dataset.py

configs/controller_core_v2.json
configs/controller_chain_smoke_v2.json

scripts/validation/run_controller_core_smoke.py
scripts/validation/run_controller_v2_chain_smoke.py
```

### B. 当前平台桥接

```text
archive/legacy_fowt_control/core_model.py
archive/legacy_fowt_control/defaults.py
archive/legacy_fowt_control/wind_env.py
scripts/validation/run_platform_open_loop_audit.py
src/wind_prediction/ballast_mass_properties.py
src/wind_prediction/input_files.py
```

前三个文件虽然位于`archive`，但仍是当前V2链路的实际运行依赖，不能按普通历史代码排除。

以下两个Excel未被Git跟踪，应作为外部物理输入记录，不能混入普通源码白名单：

```text
archive/legacy_fowt_control/data/副本水平刚度曲线.xlsx
archive/legacy_fowt_control/data/Thrust force at hub height 风机叶片及轮毂相关数据.xlsx
```

`configs/reference_platforms/volturnus_s_openfast_v1_1_16.json`可作为公开参考数据证据保留，但其身份必须继续标记为`evidence-only`，不能借此宣称P1参考平台已经确定。

### C. 测试与回归保护

测试按两个范围理解。23个文件直接覆盖V2控制核心、平台桥接和六小时链路，构成聚焦测试范围。全部`tests/test_*.py`和`tests/fixtures/`用于仓库回归，保留V1兼容代码与V2代码共同运行的检查。机器清单只收录一次测试文件，避免同一路径跨文件组重复。当前仅保存仓库回归结果，聚焦测试的独立命令和结果仍是P0待补证据。仓库回归还依赖Provider兼容代码、历史配置、分析入口和外部参考资产，现有清单尚不是可脱离仓库独立运行的测试包。

旧版Provider兼容重构涉及`ballast_planner_provider.py`、拆分后的Provider模块及相应测试。该组变更用于保留已投稿版本的兼容性，不属于V2控制核心，应作为独立的原子文件组审查和提交，不能零散混入A组。

### D. 规范与证据说明

```text
docs/controller_v2_framework.md
docs/controller_v2_pause_20260812.md
docs/platform_open_loop_audit.md
docs/controller_architecture_audit_20260811.md
docs/research_v2_execution_governance.md
docs/创新大论文规划.md
docs/controller_v2_p0_freeze_candidate_20260813.md
README.md
requirements.txt
```

## 4. 外部模型、数据与运行结果

以下内容不直接进入普通Git提交，应通过外部归档、Release附件、Git LFS或校内存储保存，并由哈希索引关联。

### 模型

```text
outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1/
```

该目录约440KB，现有链路清单记录7个模型及评价文件的逐文件哈希。

### 运行所需数据

```text
data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1/metadata.json
data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1/scaler_train.json
data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1/sample_index.csv.gz
data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1/X_test.npy
data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1/y_uv_raw_test.npy
data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1/y_event_test.npy
data/processed/wind_ml_10min/fino1_platform_10min/canonical_observations_10min.csv.gz
```

这组最小数据约494MB。现有v9清单遗漏了`y_uv_raw_test.npy`和`y_event_test.npy`，但`ReplayDataset`会实际加载二者。P0总索引必须补充以下哈希：

```text
a0d3256bd37be410454f1e7ddfe2183248710a801338c34710e454040a8639fc  y_uv_raw_test.npy
fc86594f7523d183c5400fb54859fe9443ee684150060de2a5b1facff2b1543c  y_event_test.npy
```

在P0冻结前不修改v9运行脚本，以免破坏其与现存运行源码哈希的对应关系。该清单缺口在P0总索引中补充，后续版本再修正运行器的自动取证范围。

### 运行证据

```text
outputs/controller_v2_chain_smoke_reference_incremental_v9_20260813/
outputs/platform_open_loop_audit_reference_incremental_v6_20260813/
```

关键清单哈希：

```text
7a33019201ba1215170d44ed767556045263b80597c19af0e190008ab6c6bee4  v9/manifest.json
fc2fa53127641c1ca0997b2ae4d171b490c3d0f5bfe03601b17185e2949d1759  v6/audit_summary.json
```

两个目录分别约21MB和136KB。现有结果可继续作为历史证据归档，但最终P0必须在干净候选上重新运行四类六小时链路和开放环审计。归档还必须同时保存其引用的模型和最小数据，不能只保存输出目录。

## 5. 2026年8月13日测试记录

执行命令：

```bash
PYTHONPATH=src .venv312/bin/python3.12 -m unittest discover -s tests
```

原有测试结果：

```text
Ran 389 tests in 19.498s
OK
```

测试过程中出现4条“系泊文件缺失并采用线性回退”的预期警告，对应缺失输入和回退路径测试，不属于测试失败。

完成证据清单的身份绑定、Git状态复核、通配符展开复核、严格断言和测试记录校验后，再次执行相同命令：

```text
Ran 415 tests in 21.546s
OK
```

机器可读结果保存于：

```text
manifests/p0/controller_v2_p0_candidate_v1/test_result.json
manifests/p0/controller_v2_p0_candidate_v1/unittest_full.log
```

机器结果和完整测试输出均进入候选清单。该记录仍来自当前脏工作区，只能用于P0候选审查。
测试记录用于证明已保存结果在冻结后未被改变，不能替代测试命令的实际执行。最终P0仍需在干净候选上重新运行测试并直接生成该记录。

## 6. 机器清单

P0候选范围由`configs/evidence/controller_v2_p0_candidate_v1.json`声明，生成结果保存为`manifests/p0/controller_v2_p0_candidate_v1/manifest.json`。清单逐文件记录大小、SHA-256、Git状态、通配符展开结果和外部存储状态，并重新检查测试记录、v9链路及v6开放环结果中的关键断言。文件记录、验证状态和Git状态均进入清单身份计算，校验时重新读取当前仓库和证据文件。

当前所有声明文件均存在，内容完整性校验通过，但声明范围含未提交与已修改文件，外部归档位置仍以`pending`标记，因此状态为`incomplete`。这一状态是预期结果，只有干净候选、外部资产位置和复现运行全部确定后才允许转为`passed`。

校验时必须同时给出经过研究者确认的清单ID，不能只指定清单路径。该要求用于防止通过缩小文件范围并重新生成清单来改变冻结对象。候选阶段每次内容调整都会产生新ID，只有最终冻结ID需要在提交记录或外部归档索引中单独固定。

## 7. 必须排除的内容

以下目录和文件不进入P0基线提交：

```text
.claude/
.omx/
.taskmaster/
pkcs11.txt
audit_output/
paper_revision/
artifacts/
scripts/paper_figures/
backups/
recovery/
cleanup_archive/
outputs/
data/
models/
references/
6.4opus.md
6.5 deep-research-report-.md
6.6gpt.md
```

其中`outputs/`和`data/`的P0必要内容按第4节外部归档，不作为普通源码提交。论文图表、旧审计产物和单工况试验结果不因“可能以后有用”而进入V2基线。

## 8. 提交前门禁

提交前必须完成以下检查：

1. 根据本清单形成精确暂存白名单，禁止整体暂存。
2. 检查暂存文件中是否存在本机绝对路径、凭据、本机工具配置和大文件。
3. 将V2控制核心、平台桥接、旧版兼容重构和文档分别形成可解释的文件组。
4. 在拟提交文件集合确定后，从干净工作树重新运行仓库全量测试、四类六小时链路和开放环审计。
5. 对两个外部证据目录、模型和最小数据建立统一的机器可读总索引。
6. 由研究者确认提交范围、二进制物理输入和外部证据保存方式后，才允许暂存、提交和推送。

## 9. P0剩余工作

当前已经完成文件盘点、证据核验、反方排除审查、候选范围整理和机器清单初版。P0尚未完成的内容包括：

- 确定外部物理输入、模型、数据和结果证据的唯一保存位置与获取方式。
- 生成可安装的环境锁定记录，补足开发版PyTorch来源。
- 对旧版Provider兼容重构进行单独审查，避免其与V2核心互相污染。
- 补齐仓库回归所依赖的兼容代码、历史配置和外部参考资产清单，并保存聚焦测试的独立结果。
- 在干净提交候选上重新执行全量测试、四类六小时链路和开放环检查。
- 由研究者决定是否提交、推送以及采用何种方式保存约515MB外部证据与数据。

在上述事项完成前，P0状态为`候选，待阻断项关闭`。P1的资料整理可以并行准备，但不修改平台参数和控制逻辑。
