# 纵摇静态偏差的MoorDyn开关诊断

更新日期：2026年8月21日
状态：`已量化MoorDyn模块开关影响，并以同源静态装配定位旧候选的主要刚度偏差`

## 问题与结论

此前在相同纯纵摇静态力矩下，当前低阶候选模型在2°至10°范围内的终态纵摇均比OpenFAST小约6%。本轮不调整任何低阶参数，只将公开OpenFAST模型中的`CompMooring`由3切换为0，并以同一组\(\pm10\ \mathrm{MNm}\)预载的中心差分读出局部纵摇斜率。

MoorDyn开启时，OpenFAST的条件终态纵摇斜率为\(2.73377\ \mathrm{GN\,m/rad}\)；关闭MoorDyn后为\(2.49283\ \mathrm{GN\,m/rad}\)。两者相差\(0.24093\ \mathrm{GN\,m/rad}\)。当前低阶候选中，加入辅助`.frc`系泊项前后的矩阵敏感性为\(0.24047\ \mathrm{GN\,m/rad}\)，数值相近。

这个结果量化了MoorDyn模块开关对终态读数的量级，并表明当前低阶候选相对MoorDyn开启的OpenFAST仍偏硬\(0.15838\ \mathrm{GN\,m/rad}\)。候选矩阵中辅助`.frc`系泊项的敏感性与OpenFAST模块开关影响接近，但两者的静态约束并不相同，不能据此把\(0.15838\ \mathrm{GN\,m/rad}\)直接解释为某个系泊系数。随后建立的同源静态装配则表明，旧候选在**无系泊静态部分**的主要偏差来自重量项对象范围不一致；完整MoorDyn配置仍需在独立系泊口径确定后复核。

## 对照口径

采用冻结的IEA 15 MW/VolturnUS-S公开输入副本（`v1.1.16`）。AeroDyn、InflowWind、ServoDyn、波浪和海流关闭，ElastoDyn与HydroDyn保留。两个配置各自使用独立的零预载基线，并对正、负纯纵摇HydroDyn `AddF0`预载分别进行临时松弛和600 s原始阻尼释放。最终读数取末150 s均值。

对每个配置，记相对本配置零预载基线的正、负纵摇增量为\(\Delta\theta_+\)、\(\Delta\theta_-\)，则使用

\[
K_{\mathrm{eff}}=
\frac{M}{\left(\Delta\theta_+-\Delta\theta_-\right)/2}.
\]

该中心差分可削弱零载姿态偏置和偶次响应对结果的影响。它测得的是允许其他自由度自行调整后的**条件终态纵摇斜率**，不是MoorDyn局部\(K_{55}\)、单独的重力刚度，也不是可逐项相加的矩阵分解。

## 结果

| OpenFAST配置 | \(\Delta\theta_+\)（°） | \(\Delta\theta_-\)（°） | 奇对称响应（°） | 偶分量（°） | \(K_{\mathrm{eff}}\)（GN·m/rad） | 尾段判据 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MoorDyn开启 | +0.209750 | -0.209421 | 0.209586 | +0.000164 | 2.73377 | 通过 |
| MoorDyn关闭 | +0.229684 | -0.230000 | 0.229842 | -0.000158 | 2.49283 | 通过 |

MoorDyn关闭时，释放末窗的纵摇读数稳定，但其零预载吃水、水平位置和艏摇均会改变。相应的surge、heave和yaw尾段均值及范围已完整保存在机器可读结果中。因此，表中两行不能解释为在完全相同静态约束下的“有无系泊刚度”，只能用于判断模块开关对终态纵摇读数的整体影响。

当前低阶候选的完整受约束静态刚度为\(2.89215\ \mathrm{GN\,m/rad}\)，比MoorDyn开启的OpenFAST高\(5.79\%\)。候选矩阵移除辅助`.frc`系泊项后的数值\(2.65168\ \mathrm{GN\,m/rad}\)仅保留为矩阵敏感性记录，**不与MoorDyn关闭的OpenFAST做一一对应比较**：前者人为消除了自由surge/yaw坐标，后者仍由OpenFAST自行演化，两个静态问题并不相同。因而，本轮不能宣布“系泊项正确”或“约6%完全来自静水、重量项”。

## 已定位的参考状态混用

旧候选的重量项来自辅助二阶`.frc`文件中的\(20.1\ \mathrm{Mt}\)、\(z_{\mathrm{CG}}=-2.32\ \mathrm{m}\)，对应纵摇重量恢复\(0.45746\ \mathrm{GN\,m/rad}\)。为避免继续手算近似，本轮新增`source_consistent_reference.py`，直接从冻结压缩包中的ElastoDyn、塔架质量表、叶片质量表和一阶WAMIT文件读取数据。质量清单包括平台、塔架、机舱、偏航轴承、轮毂和三支叶片；叶片质量在静态一阶矩中明确集中于转子顶点，不作为完整转子惯量模型使用。

该同源清单得到总质量\(20.25244\ \mathrm{Mt}\)、\(z_{\mathrm{CG}}=-1.53519\ \mathrm{m}\)，对应重量恢复\(0.30501\ \mathrm{GN\,m/rad}\)。将该重量项与同一份一阶WAMIT静水矩阵组合，并只保留heave--pitch静水耦合后，得到无系泊等效纵摇刚度\(2.49923\ \mathrm{GN\,m/rad}\)。与MoorDyn关闭的OpenFAST读数\(2.49283\ \mathrm{GN\,m/rad}\)相比，差值为\(0.00640\ \mathrm{GN\,m/rad}\)，约为\(0.26\%\)。

相比之下，旧候选移除辅助`.frc`系泊项后的无系泊矩阵刚度为\(2.65168\ \mathrm{GN\,m/rad}\)，比同一OpenFAST条件高\(0.15885\ \mathrm{GN\,m/rad}\)。其中\(0.15245\ \mathrm{GN\,m/rad}\)可由旧`.frc`重心与同源ElastoDyn重心导致的重量恢复差解释。这一量级与此前2°至10°范围内的固定偏硬高度一致，表明将一阶WAMIT静水恢复与辅助`.frc`整机重心混用，是当前最需要排除的基线来源差异；它不是通过修改单个经验刚度参数应当处理的问题。

这一结论限定在无系泊静态恢复标度。该装配显式确认ElastoDyn平台参考点位于SWL（`PtfmRefzt=0`）且HydroDyn采用`WAMITULEN=1 m`，再按该同一参考点组合一阶WAMIT矩阵。MoorDyn开启时的完整响应仍受系泊预张力、自由度调整和零载平衡影响，不能把0.26%的静态标度差直接当作完整系统的最终拟合结果。

## 为什么2°至10°一直约为6%

此前六个终态载荷点中，2°、3°、5°、7°和10°均通过尾段稳定性检查。OpenFAST分别给出2.117°、3.176°、5.297°、7.420°和10.607°，低阶模型相对误差为\(-5.86\%\)至\(-6.07\%\)，平均\(-5.95\%\)。同一范围内，OpenFAST反算刚度仅由\(2.7321\)降至\(2.7266\ \mathrm{GN\,m/rad}\)，变化约\(0.20\%\)。

因此，当前偏差更像一项近似固定的静态恢复刚度基线偏高，而不是在5°、7°或10°附近突然出现的大角度非线性失效。12°点尾段未通过稳定性判据，只作为量程边界保留，不参与这一结论。

## 与现有900 h控制样本的关系

现有150组、每组6 h的历史控制快照中，姿态指标使用\(\max(|\mathrm{pitch}|,|\mathrm{roll}|)\)。在预测辅助算法的旧模拟输出中，900 h内该量大于2°、3°、5°、7.5°和10°的时间占比分别为9.199%、4.383%、1.089%、0.206%和0.109%。按区间计，0°至2°约90.801%，2°至3°约4.817%，3°至5°约3.294%，5°至7.5°约0.882%，7.5°至10°约0.097%，10°以上约0.109%。

这些比例只描述旧控制模拟中的指标覆盖范围，不能当作真实海况概率，也不能替代OpenFAST验证。它们仅说明：2°至5°是当前样本中最常进入的中等姿态区间，7.5°和10°更适合作为安全边界核对；新平台模型应优先在2°、3°和5°附近完成一致装配，再保留7°和10°作为范围检查。

## 下一步

不应根据本轮数值直接把某一项刚度调小\(0.158\ \mathrm{GN\,m/rad}\)。同源静态重量项已实现为独立模块，旧`reference.py`保持不变作为历史复现依据。2°、5°和10°的无系泊端点核对已通过，详见`openfast_source_consistent_static_pitch_sweep_audit_20260821.md`。下一步应为新低阶平台建立独立的系泊表示和完整动力学惯量口径，再在MoorDyn开启条件下复核；在此之前不把当前低阶模型用于控制性能结论。

## 可复核材料

- 运行入口：[run_openfast_pitch_restoring_component_audit.py](../scripts/validation/run_openfast_pitch_restoring_component_audit.py)
- 同源静态装配：[source_consistent_reference.py](../src/fowt_platform/source_consistent_reference.py)
- 本轮机器可读结果：[openfast_pitch_restoring_component_audit_20260821.json](../artifacts/openfast_pitch_restoring_component_audit_20260821.json)
- 倾角量程比较：[openfast_large_angle_pitch_end_state_sweep_audit_20260821.md](openfast_large_angle_pitch_end_state_sweep_audit_20260821.md)
- 同源静态端点核对：[openfast_source_consistent_static_pitch_sweep_audit_20260821.md](openfast_source_consistent_static_pitch_sweep_audit_20260821.md)
- 小力矩一致性核对：[openfast_moment_angle_consistency_audit_20260821.md](openfast_moment_angle_consistency_audit_20260821.md)
