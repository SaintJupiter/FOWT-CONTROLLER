# 预测驱动主动压载：可迁移控制方法的扩展检索与比较

日期：2026-10-06。性质：选题阶段的范围检索和方法比较，不是穷尽性系统综述，也不是已确定的新算法方案。

## 1. 当前结论与研究边界

**应该扩大比较范围。此前从两篇全文迅速收敛到时间不确定预览，证据不足以支持排除其他方法。** 本轮将比较扩展到预测信任机制、跨周期决策导向学习、经济区域控制、混合执行时序、后悔最优控制、稀疏控制、预览价值和闭环前馈等方向。

这不是同时开发八套算法。当前产出是有出处、能比较具体数学构造的候选库；下一步从其中选少量相互竞争的机制做纸面迁移。Gostin、Hoang和Lucia继续保留，但均不再是预先指定的唯一母体。

北极星不变：**相同平台、设备能力和姿态要求下，改善姿态与实际累计泵量的权衡。**

\[
V_{\mathrm{abs}}=\sum_{i=1}^{3}\int_0^T |q_i^{\mathrm{actual}}(t)|\,\mathrm dt.
\]

这不是净舱量变化，不自动等于电能或寿命。允许姿态带、RMS、峰值和超限时间需要分别定义；降低其中一项不能掩盖另一项恶化。评价带由研究任务确定，不能为了得到节泵结果反复放宽。窗口结束时报告实际舱量和泵状态，避免把暂存水量或推迟动作误当永久收益。

## 2. 检索与证据口径

- 重点年份为2023年至2026年10月6日；保留少量直接相关的2022年方法作为连接。正式发表年份、在线年份与预印本版本分开记录。
- 检索领域包括风机预览控制、储能与能源调度、水系统泵调度、变浮力、受约束控制与在线学习。按问题结构检索，不只搜“active ballast MPC”。
- 关键词组包括 learning-augmented control / untrusted predictions / convex reparameterization；decision-focused forecasting / intertemporal gradient；zone economic control / lexicographic optimization；minimum dwell time / switched MPC；regret optimal control；maximum hands-off；preview information value；optimal feedforward。
- 核心依据来自出版社、会议论文集、作者机构库和作者预印本。搜索摘要用于定位，不用于补写未读定理。没有把无法核查的聚合站或生成式论文摘要作为依据。
- 阅读状态：**F**=此前本地逐页全文核对；**M**=已读公开全文的相关方法、假设或算例章节，未逐证全部证明；**A**=出版社/作者摘要、元数据或章节节选，尚不足以迁移定理。
- 两位审查Agent分别检索执行/水系统和决策导向预测，主对话检索控制理论、预览与应用对标并做交叉核查。并非独立专家评审，不把多Agent一致当作事实证据。
- 本轮没有修改控制代码、训练预测器或运行新仿真。新增PDF本地下载遇到网络解析失败，公开全文通过网页读取；不声称已经将以下全部PDF归档。已有Hoang/Gostin本地副本不变。

## 3. 与本地程序的真实起点对齐

| 已有能力 | 本轮核对位置 | 对选题的限制 |
|---|---|---|
| 三舱独立与海水交换，水量累积，真实泵执行状态可传播 | [执行层](../src/wind_prediction/execution_rollout.py)、[平台压载状态](../src/fowt_platform/ballast_snapshot.py) | 不能假定存在舱间管网；固定总量与两轴精确力矩条件下，不能默认还有终点分配冗余 |
| 连续预览QP，含绝对水量变化的上图形式及多个姿态/动作成本 | [QP](../src/wind_prediction/preview_mpc.py)、[设计](../src/wind_prediction/preview_mpc_design.py) | 再加一个L1泵量项不构成新能力；当前已有v4，不能沿用旧v2进度当作代码现状 |
| 继续/释放/新目标均可比较，首块用实际执行重评估，后续可从达到的状态重新规划 | [控制周期](../src/wind_prediction/preview_mpc_control_cycle.py) | 不能以“程序完全没有执行反馈/释放策略”制造研究缺口；这些仍基于当前单条预览，不等于完整条件策略 |
| 合格候选主要按统一加权时域成本选择，近似同成本再比较实际泵量等 | 同上，`_select_candidate` | 可行性筛选加成本排序不等于全过程的词典序区域经济优化；但换成词典序本身也已有成熟文献 |
| LSTM及监督、物理风载转换已存在 | [训练脚本](../scripts/modeling/train_ballast_lstm.py)、[来源绑定预览](../src/wind_prediction/source_bound_rotor_preview.py) | 脚本支持的监督不等于当前权重实际采用；当前权重身份见10月5日诊断。不能把任意DFL输出继续冒充真实风矢量 |

当前还存在转子性能表适用域、执行阈值设备依据和广泛动态验证边界。这些会影响后续结论，但本轮不借此重新启动模型大修。程序修复不包装成控制算法创新。

特别核对：[连续实验执行配置](../scripts/validation/preview_mpc_experiment_runtime.py)使用1 m3/min流量、20 s最短开机、12 s最短停机、5 s执行子步，而控制周期为600 s。**泵量积累慢，不等于驻留或爬升跨越多个控制周期。** 因此目前不能据R10预设“长驻留承诺”是本项目的关键困难，也不能把软件配置视为设备实测参数。

## 4. 八类可以竞争的方法

下表的“可能改进”是本项目迁移判断，不是原论文已经证明的主动压载收益。

| 方法族 | 具体借鉴对象 | 改动的核心计算 | 可能服务的收益 | 当前判断 |
|---|---|---|---|---|
| A. 有依据地使用不可靠预测 | Li 2022；Shen 2024；Li 2025，见R01-R03 | 控制策略参数化、预测信任的在线更新及性能比较 | 好预测能发挥作用，坏预测造成的控制损失可解释/受限 | 值得重点精读；不能退化成旧概率准入换名字 |
| B. 跨周期决策导向预测 | Peršak-Anjos、Zhang、Yeh等，R04-R09 | 训练梯度穿过优化和状态累积，或按决策风险校准 | 学习哪些预测误差真正影响后续动作 | 可深改训练链，但不是自动获得新控制律；物理语义和可微性是硬问题 |
| C. 区域经济目标与执行承诺 | Chen-Lazar、Jv、Chen等，R10-R12 | 先满足任务优先级，再安排资源；将当前模式承诺延续到未来 | 减少无必要精确跟踪，避免名义未来动作不可执行 | 与泵量最直接；须固定姿态任务，不能把放宽要求算算法优势 |
| D. 时间误差、目标可行性与条件决策 | Gostin、Hoang、Leung，R14-R16 | 误差传播/约束收紧，参考管理，或信息分叉后的策略 | 防止预测时序错误或可用预览不足造成不合适的提前动作 | 继续比较，不再优先锁定；三篇不是可随意叠加的模块 |
| E. 后悔最优控制 | Didier、Martin等，R17-R18 | 优化相对明确全信息参照的最坏性能差，而非只看名义平均 | 在不同扰动规律下平衡性能损失 | 算法层次深，线性/二次与终端结构要求强；未必最容易服务LSTM预览 |
| F. 稀疏/最大停机控制 | Ikeda，R19；变浮力对照R13 | 优化非零动作持续时间或按需启停幅值 | 泵停机时间和执行活动可成为直接决策对象 | 停得久不等于少泵；不能将L0、L1、启停次数混用 |
| G. 预览长度的控制价值 | Liu-Ozay，R20 | 分析有限预览下可保持约束的状态集合及其增益 | 解释何时更长预览不再显著扩展可调节范围 | 很好的理论/适用性支撑，单独未必构成新压载算法 |
| H. 基于既有闭环的最优前馈 | Weich、Hegazy等，R21-R22 | 将反馈响应纳入前馈动作优化，明确不同目标之间的代价 | 减少前馈与反馈互相抵消，合理利用预览 | 不必重建所有控制环节；快速变桨结果不能直接迁移到慢泵 |

R23场景选择、R24异常预测监测是辅助方法；R25-R26用于防止忽略同领域已有工作。它们不因较新就自动成为主线。

## 5. 值得深挖的具体构造，而不是算法名称

### 5.1 候选A：预测信任如何成为可分析的控制机制

R02不是直接把两个控制器的输出任意混合，而是借助扰动响应控制的凸重参数化，再进行控制策略组合；原文特别针对简单混合可能破坏稳定性的问题。R03则用已经揭示的预测误差进行延迟在线置信更新。

**对本项目的候选问题：** 能否依据风载预测误差对当前及后续姿态/泵量的影响，更新“采用多大程度的预览作用”，而不是用固定准入阈值？作用对象必须明确是控制策略、预测估计还是目标参数，不能直接将真实外力乘可靠度。

真正需要迁移的是策略参数化、延迟信息更新和相应性能分析。只增加一个经验权重，仍与旧方法很接近。

关键边界：R02所用稳定线性动力学条件不由本系统自动满足，舱量含积分状态；R03要求光滑性、优化正则性及很强的共同可行性条件。真实泵模式、爬升、L1代价和约束切换不允许直接引用其保证。假设不满足可能意味着不适合迁移，不能自动视为可做创新的“空白”。

### 5.2 候选B：从单次预测损失转为跨周期状态后果

R04的关键是训练时保留“早期预测改变状态，状态再影响后续决策”的梯度路径，而不只对一次QP反传。示意为：

\[
\frac{\mathrm du_t}{\mathrm d\vartheta}
=\frac{\partial u_t}{\partial\hat w_t}\frac{\mathrm d\hat w_t}{\mathrm d\vartheta}
+\frac{\partial u_t}{\partial z_t}\frac{\mathrm dz_t}{\mathrm d\vartheta}.
\]

此式是所借方法的结构说明，不是本项目原创。它与慢泵累计水量有关，但原文的关键假设是**不确定性只进入目标函数，状态转移/可行集不受其直接影响，且假定相对完全补救**。我们的风载直接影响姿态动态和约束，所以不能照搬储能训练图。

此外，原文使用的是逐次更新的确定性策略训练，不等于场景树多阶段反馈。训练输出可能成为有用但有偏的决策表示；若继续经物理风载转换，必须保持可解释风况身份，或明确把新增输出命名为决策参数。

R05提供另一种切入：利用下游对偶边际价值构造学习信号。R06提供风险校准与训练结合的参照。但QP与混合泵执行的敏感性、时间序列相关性、风险单调性都需另行处理。不能让“新学习模块”成为无法解释的第二层控制器。

### 5.3 候选C：将姿态任务与慢执行承诺直接写进决策

R11-R12的词典序结构可示意为：先求最低任务偏差 \(J_z^*\)，再在不超过 \(J_z^*+\varepsilon\) 的条件下优化资源成本。这与给所有目标乘权重后相加不同，也不同于现有候选近似同成本后的泵量排序。

但它本身不是原创；若从追踪零倾角改成允许某一姿态带，必须让所有基线使用同一任务，并继续报告带内姿态差异。R12的44.08%节能比较含点跟踪与区间跟踪差异，不能照搬成纯算法提升。其BO调整内部目标带，不是直接调整评价允许带；也不能误说所有收益都来自放宽评价。

R12仍有硬近似：预测时域内扰动保持常值、状态相关输入界在一次优化内冻结；DeePC基础引理要求可控LTI及持续激励，非线性算例另采用正则化。BO离线选择内部目标带。这些都不能直接提供本项目连续重预测和混合执行下的保证；也没有必要为了借词典序结构而把已有平台模型替换成DeePC。

R10更具体地研究最小驻留时间：把当前模式尚未完成的驻留承诺延续到未来，并用相应终端结构处理后续可延续性。对本项目可问：**六块预览中的名义动作，是否在预测更新和真实泵过渡之后仍能执行？** 新贡献若存在，应是适合目标跟踪泵的表达或性质，而不是再次把泵状态放进状态向量。

原文主要解决计算及可行性结构，假设瞬时切换；并未给本项目证明节泵。软件驻留/阈值也需有设备或研究假设依据，不能人为加严泵限制来凸显新算法。

按上述现有秒级驻留配置，R10暂降为执行终端结构的参考，不列为已经发现的主要改进机会。若将来有设备依据证明跨周期承诺显著，再评估它是否值得承担核心贡献。

### 5.4 候选D：原来的时间误差路线保留为平行候选

Gostin提供完整误差集合传播与约束收紧，不是只有时间窗口概念。其有限任务、可实现反馈和终端构造与滚动LSTM不相同；共同时间偏移减少允许轨迹，也不自动优于逐时刻窗口。Hoang的概率预测与有限分支是另一种近似，Leung处理可调参考与终端可行性，均需要明确输入身份。

因此当前不把三者叠加。它们与A-C一起比较“具体改哪一步、有什么可成立的性质、是否服务真实泵量与姿态”，而不是预设时间误差就是最重要的问题。此前全文意见仍见[专项精读](recent_preview_control_fulltext_review_20261006.md)。

## 6. 其余方法为什么保留、为什么不马上主推

- **后悔最优R17-R18：** 确实改变控制设计准则，不只是换名字；但其全信息最优参照不是把记录未来送给当前控制器所得结果。相同任务下还需定义成本、扰动集合和终端条件，不会消除所有权重选择。
- **最大停机R19：** 优化非零控制的时长，L1替代只有在相应条件下等价。已有程序已经罚绝对水量；停机更久可能通过更大流量实现，不必更省水，也不保证少启停。
- **变浮力R13：** 提供一条非MPC参考，利用自然运动并调整动作幅度。但原文单轴、恒扰及快执行器近似与本项目不同；借控制律必须重建三舱耦合和预览的作用，不能只拿一个启停判断式。
- **预览价值R20：** 在其条件下分析安全可行集合随预览增长的变化，主要针对已知窗口内的准确扰动信息。不能直接推出带LSTM误差时“最佳预测长度”。
- **最优前馈R21-R22：** 与物理风机最接近，但执行通道是变桨/发电机，不是压载。借其闭环协调计算可以深入；只说“加入前馈”则重复小论文已有能力。
- **场景选择R23：** 主成分和敏感性并非空泛思路，但单调性前提与风向、换向未必相容；目前三泵六块没有天然证明计算瓶颈。
- **运行时异常监测R24：** 可以检查预测域外状态并切换备援，但异常阈值本身不是新核心算法；备援是否适合慢泵当前状态仍需证明。

## 7. 26篇候选与对标文献登记

以下按方法族编号，不按“越新越好”排序。M只表示关键章节已核查，不表示所有定理、实验和代码均复现。无正式发表记录的条目明确标预印本。

| 编号 | 论文、年份与身份 | 原文具体贡献/本次用途 | 来源与阅读 |
|---|---|---|---|
| R01 | Li等，**Robustness and Consistency in Linear Quadratic Control with Untrusted Predictions**，2022，POMACS 6(1) | 在线调整预测信任，分析可靠预测表现与错误预测下的竞争性能；LQ条件限制 | [作者页](https://tongxin.me/research/papers/robust-lqc/)，[DOI](https://doi.org/10.1145/3508038)，A |
| R02 | Shen、Wierman、Qu，**Combining model-based controller and ML advice via convex reparameterization**，2024，L4DC，PMLR 242:679-693，会议 | 扰动响应重参数化后组合策略，给出有界性/后悔分析；稳定动力学等假设需对应 | [论文与公开PDF](https://proceedings.mlr.press/v242/shen24a.html)，M |
| R03 | Tongxin Li，**Learning-Augmented Control: Adaptively Confidence Learning for Competitive MPC**，2025，arXiv预印本，未核实正式发表 | 延迟在线置信学习与竞争比；强可行性、光滑性及KKT正则条件 | [v1全文](https://arxiv.org/html/2507.14595v1)，M |
| R04 | Peršak、Anjos，**Decision-Focused Forecasting: A Differentiable Multistage Optimisation Architecture**，2024首稿/2025 v2，预印本 | 状态路径梯度；原问题不确定性仅进入目标函数，不等于扰动驱动的姿态约束控制 | [v2全文](https://arxiv.org/html/2405.14719v2)，M |
| R05 | Zhang、Jia、Wen、Bian、Shi，**Toward Value-Oriented Renewable Energy Forecasting: An Iterative Learning Approach**，2025，IEEE TSG 16(2):1962-1974，2024在线 | 以调度LP的对偶边际价值构造迭代预测训练；不能直接照搬到QP和时序泵执行 | [作者全文](https://arxiv.org/html/2309.00803v3)，[DOI](https://doi.org/10.1109/TSG.2024.3503554)，M |
| R06 | Yeh、Christianson、Wierman、Yue，**Conformal Risk Training: End-to-End Optimization of Conformal Risk Control**，2025，NeurIPS，会议 | 风险校准与端到端训练，包含OCE/CVaR；交换性、损失界、适当单调性及独立校准要求 | [会议全文](https://proceedings.neurips.cc/paper_files/paper/2025/file/6559542f75b4452ebaaf82094c7defb7-Paper-Conference.pdf)，M |
| R07 | Stratigakos、Pineda、Morales，**Decision-focused linear pooling for probabilistic forecast combination**，2025，International Journal of Forecasting 41(3):1112-1125 | 用决策成本学习概率预测组合权重，可依赖上下文；不是控制器成本权重，需保留联合时序分布 | [机构全文](https://spiral.imperial.ac.uk/bitstreams/0674bffd-73e3-4f4d-98d4-3f69c0409d1d/download)，[DOI](https://doi.org/10.1016/j.ijforecast.2024.11.006)，M |
| R08 | Wu、Yi、Xu、Anderson，**Online Energy Storage Arbitrage under Imperfect Predictions: A Conformal Risk-Aware Approach**，2026，ACM e-Energy:408-423，2025预印本 | 校准价值预测的保守程度；保证针对有条件的时序误差代理，不是直接利润或姿态保证 | [作者v2](https://arxiv.org/html/2511.01032v2)，[DOI](https://doi.org/10.1145/3744255.3798116)，M |
| R09 | Wahdany、Schmitt、Cremer，**More than accuracy: end-to-end wind power forecasting that optimises the energy system**，2023，EPSR 221:109384 | 经风功率转换/DC-OPF的KKT微分；其具体损失和主要结果偏系统成本预测误差，不能当作实际控制节省的直接证据 | [出版全文](https://publications.rwth-aachen.de/record/956021/files/956021.pdf)，[DOI](https://doi.org/10.1016/j.epsr.2023.109384)，M |
| R10 | Yutao Chen、Mircea Lazar，**An efficient MPC algorithm for switched systems with minimum dwell time constraints**，2022，Automatica 143:110453 | 模式分块、松弛/取整/重优化、剩余驻留承诺和l步终端结构；递归delta可行，不是任意扰动下精确保证 | [机构全文](https://pure.tue.nl/ws/portalfiles/portal/302807216/1-s2.0-S0005109822003089-main.pdf)，[DOI](https://doi.org/10.1016/j.automatica.2022.110453)，M |
| R11 | Jv、Wang、Zhang、Yin、Liu，**Lexicographic optimization for economic model predictive control with zone tracking**，2023，CHERD 200:646-654 | 多区域与经济任务的优先级优化；不能仅从摘要引用稳定性条件 | [出版页](https://www.sciencedirect.com/science/article/pii/S0263876223007517)，[DOI](https://doi.org/10.1016/j.cherd.2023.11.041)，A |
| R12 | Xiaoqiao Chen等，**Economic zone data-enabled predictive control for connected open water systems**，2026，Water Research 291:125181，2025在线/预印本 | 混合整数区域DeePC、词典序、BO内部目标带；含泵输入不连通集合，但非本项目滚动预测保证 | [作者全文](https://arxiv.org/html/2510.03043v1)，[出版页](https://www.sciencedirect.com/science/article/pii/S0043135425020846)，[DOI](https://doi.org/10.1016/j.watres.2025.125181)，M |
| R13 | Pinto、Carneiro、de Almeida、Cruz，**Depth Control of Variable Buoyancy Systems: A Low Energy Approach Using a VSC with a Variable-Amplitude Law**，2025，Actuators 14(10):491 | 非MPC变结构启停与变幅律；单轴、恒扰、快执行器近似；不是慢三舱预览控制的现成解 | [开放全文](https://www.mdpi.com/2076-0825/14/10/491)，M |
| R14 | Gostin、Koeln，**Robust Model Predictive Control with Temporally-Uncertain Disturbance Preview Information**，2024，ACC:2488-2493，会议 | 时间窗口误差传播、收紧约束和有限任务终端构造 | [DOI](https://doi.org/10.23919/ACC60939.2024.10644827)，[本地索引](../references/multistage_control/README.md)，F |
| R15 | Hoang等，**Probabilistic Forecasting for Multi-Stage Nonlinear Model Predictive Control**，2025，CCTA:281-287，会议 | 灰箱概率预测与有限深度多阶段NMPC；算例分支时域NR=1，含尺度选择 | [DOI](https://doi.org/10.1109/CCTA53793.2025.11151328)，[本地索引](../references/multistage_control/README.md)，F |
| R16 | Leung、Kolmanovsky，**Feasibility governor for MPC with disturbance preview information**，2024，Systems & Control Letters 185:105735 | 修改参考以满足预览/终端可行条件；不能将扰动风载当可随意削弱的参考 | [出版页](https://www.sciencedirect.com/science/article/pii/S0167691124000239)，A，本轮未重新全文核对 |
| R17 | Didier、Zeilinger，**Generalised Regret Optimal Controller Synthesis for Constrained Systems**，2023，IFAC-PapersOnLine 56(2):2576-2582，会议 | 系统级参数化和受约束regret合成；线性时变、二次成本和扰动集合 | [作者全文](https://arxiv.org/html/2211.08101v2)，[DOI](https://doi.org/10.1016/j.ifacol.2023.10.1341)，M |
| R18 | Martin、Furieri、Dorfler、Lygeros、Ferrari-Trecate，**On the Guarantees of Minimizing Regret in Receding Horizon**，2025，IEEE TAC 70(3):1547-1562，2024在线 | 滚动regret策略与终端构造，在其条件下给递归可行性、稳定性及后悔界 | [出版摘要](https://ieeexplore.ieee.org/document/10684161/)，[作者预印本](https://arxiv.org/abs/2306.14561)，M（预印本相关章节） |
| R19 | Takuya Ikeda，**Nonconvex Optimization Problems for Maximum Hands-Off Control**，2025，IEEE TAC 70(3):1905-1912，2024在线/预印本 | 非凸代价与最短非零控制时长的等价条件；LTI、输入界、终点条件，不能默认适用混合泵 | [作者预印本](https://arxiv.org/abs/2402.10402)，[DOI](https://doi.org/10.1109/TAC.2024.3474061)，M |
| R20 | Zexiang Liu、Necmiye Ozay，**Quantifying the Value of Preview Information for Safety Control**，2025，IEEE TAC 70(7):4484-4499 | 分析有限预览可控不变集相对无限预览的差距；准确窗口预览及线性/凸集合等条件 | [作者全文](https://arxiv.org/html/2303.10660)，[DOI](https://doi.org/10.1109/TAC.2024.3524462)，M |
| R21 | Weich、Schlipf、Burth，**Lidar-Assisted Optimal Feedforward Control of Wind Turbines**，2026，J. Phys.: Conf. Ser. 3224:052028，会议 | 以受约束NMPC最优前馈增强工业反馈，优化转矩更新和集体变桨速率 | [出版页](https://doi.org/10.1088/1742-6596/3224/5/052028)，A |
| R22 | Hegazy等，**The potential of wave feedforward control for floating wind turbines: a wave tank experiment**，2024，Wind Energy Science 9:1669-1688 | 缩尺水池与软件在环比较不同前馈目标；展示响应改善与执行活动的权衡 | [开放全文](https://wes.copernicus.org/articles/9/1669/2024/)，M |
| R23 | Mdoe、Jaschke，**Sensitivity-based scenario selection for multi-stage MPC along principal components**，2025，Computers & Chemical Engineering 194:108992 | 主成分与优化敏感性选场景；单调性等条件，不能默认风向耦合满足 | [出版页](https://www.sciencedirect.com/science/article/pii/S0098135424004101)，A |
| R24 | Contreras、Shorinwa、Schwager，**Safe, Out-of-Distribution-Adaptive MPC with Conformalized Neural Network Ensembles**，2025，L4DC，PMLR 283:194-207，会议 | 预测集合、运行时域外监测与可达性备援，原应用含移动障碍预测 | [会议页](https://proceedings.mlr.press/v283/contreras25a.html)，A |
| R25 | Lyu、Li、Zan、Zhong，**A Coupled Framework for Short-Term Mooring Tension Prediction and Ballast Control for Floating Offshore Wind Turbines**，2026，JMSE 14(18):1735 | FAST-AQWA、短时响应预测和MPC压载；与本项目风预测及独立海水交换口径不同，用于同领域重叠检查 | [开放原文](https://www.mdpi.com/2077-1312/14/18/1735)，M（方法相关部分） |
| R26 | Meng、Lyu、Zhang、Ai、Song，**Coupled analysis and performance evaluation of a semi-submersible floating wind turbine with active ballasting system**，2026，Ocean Engineering 343:123142 | 主动压载耦合模型和PID等应用验证对标，不是新预览算法母体 | [机构页与公开PDF](https://discovery.ucl.ac.uk/id/eprint/10216279/)，[DOI](https://doi.org/10.1016/j.oceaneng.2025.123142)，A（本轮主要元数据与摘要） |

## 8. 按师兄的方式，怎样从文献变成自己的工作

正确借鉴不是“挑一篇年代新的论文照搬”，也不是必须宣称成熟母体存在普遍缺陷。应做到：

1. **明确母体的可执行构造。** 指出原控制律、优化层、学习更新、终端集合或策略参数化，不停留在“鲁棒”“智能”。
2. **明确本对象的差异。** 风载进入动态与约束、舱量累积、真实泵过渡、预测滚动更新究竟在哪个假设或计算中产生困难；已有代码已解决的部分不能重复提出。
3. **明确自己改的一处计算。** 原来怎么算、新方案怎么算、为什么需要修改；不是把几篇论文依次接在框图中。
4. **明确相应性质。** 可是可行性、误差界、闭环性能、近似误差、收敛或有依据的性能解释。不要求机械凑多个定理，也不把标准母体的定理当成自己的。
5. **明确真实收益维度。** 同等姿态任务下少泵、同等泵量下姿态更好、或必要的执行可信度改善。仅计算更快且原问题本来算得动，不足以主导当前论文。
6. **允许有据可查的组合贡献。** 若组合解决单一方法不能正确处理的接口且作用可分辨，可以形成研究贡献；“组合”并非天然无效。但引用数量不构成创新证据。

本轮没有证明任何拟议改动在全部相关文献中原创。每个候选真正形成机制后，还需围绕该机制进行近邻文献排重，而不是现在给八个方向分别起新算法名称。

## 9. 下一步安排：先竞争式精读，不先开八套实验

**第一组：直接改变控制决策的候选。** 并列精读R02/R03的预测信任结构、R11/R12的任务优先级、R14的时间误差传播；R18的后悔准则保留为另一种设计取向。每类输出不超过两页的“原构造-本系统对应-具体缺口-候选改动-可分析性质”。未取得全文的R11先不承诺其理论。R10依据时间尺度核对结果降为支撑参考。

**第二组：训练接口候选。** 先比较R05的预测依赖约束及补救价值、R07的约束不确定性分布接口；R04作为跨周期梯度补充，R06/R08作为风险校准参照。这一组与控制律改造竞争，不默认一起加入。先写清真实扰动在哪里进入、补救何时且能否执行、不可行如何评价，再讨论可微层和网络训练。

**保留组：** R17-R20用于替代设计准则和理论支撑，R21-R22用于闭环前馈与强基线，R23-R24仅在需要时采用，R25-R26持续用于同领域对标。

精读筛选依据是物理契合、原文实际能力、现有代码重复程度、可推导性、调参/黑箱负担及潜在控制收益，不按年份、公式数量或论文宣传的百分比排名。暂不设置人为分数表，避免把主观打分伪装成客观选题结果。

本轮广筛到这份比较表收束，不以“不断增加篇数”作为进展。下一轮完成上述代表机制的同格式比较后：若只能重现现有能力、收益依赖改变硬件/放宽任务、或关键假设无法合理对应，则降为基线/辅助或暂不采用；若能写出一处自洽的核心改动及可检验性质，则保留1-2个方案供选择。未发现适合方案时应说明具体缺口再定向检索，不宣称已选出最优路线，也不无限重开整个算法库。

完成同口径比较后，选一个主要机制，最多配一个必要支撑。然后才做数学原型及必要的实现检查；不再用少量窗口给研究方向判生死，也不靠理论合理性提前宣布收益。

## 10. 交叉审查保留的不同意见

- 执行侧审查更看好R10的执行承诺和R12的经济任务结构，但明确R10的收益主要是计算与可行性，不能预设三泵存在计算瓶颈。
- 学习侧审查最初把R04列为结构最贴近的参考。主对话核对后指出“不确定性只进成本、状态确定”的硬假设，因此它只能作为跨周期梯度的优先精读对象，不能直接升格为最佳可部署母体。
- 主对话进一步核对实际泵配置后，将R10的长驻留迁移优先级下调：20/12秒与600秒控制块并非同一时间尺度，不能仅因“慢泵”就认定其主要机制有必要。
- R12内部控制目标带的BO与评价允许带不是一回事；必须同时避免“把调参收益全说成新算法”和“把全部收益全说成放宽约束”两种误读。
- 新论文中的状态可行、概率风险、竞争比、delta可行和真实平台姿态安全是不同性质，报告中不互相替代。

最终状态：**候选范围已实质扩大；已有多条可进入公式与算法层的借鉴方向；尚未选定最终母体，尚未形成新算法或获得新控制收益。**
