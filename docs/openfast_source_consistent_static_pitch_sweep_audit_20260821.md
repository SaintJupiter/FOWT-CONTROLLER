# 同源静态纵摇端点核对

更新日期：2026年8月21日
状态：`通过；仅作为无系泊静态恢复标度核对`

## 目的与边界

`source_consistent_reference.py`从冻结的ElastoDyn、塔架质量表、叶片质量表和一阶WAMIT文件建立静态质量与恢复项，不读取辅助二阶`.frc`质量、重心或系泊矩阵。该装配得到总质量\(20.25244\ \mathrm{Mt}\)、\(z_{\mathrm{CG}}=-1.53519\ \mathrm{m}\)，并给出无系泊纵摇刚度\(2.49923\ \mathrm{GN\,m/rad}\)。

为检查该标度在有限倾角范围内是否仍成立，关闭AeroDyn、InflowWind、ServoDyn、波浪、海流和MoorDyn，仅保留ElastoDyn与HydroDyn。对2°、5°和10°三种目标增量，按同源静态刚度计算纯纵摇`AddF0`力矩。每个载荷先进行临时松弛定位，再在原始阻尼下释放600 s，取末150 s相对零预载基线的纵摇均值；尾段判据仅检查纵摇和横摇的范围及相邻窗口均值漂移。

该核对不验证系泊、阻尼、完整六自由度动力响应或控制算法；它只回答一阶WAMIT静水恢复与同源ElastoDyn重力项能否给出一致的纵摇静态量级。

## 结果

| 同源静态目标（°） | 施加纵摇力矩（MN·m） | OpenFAST终态增量（°） | 差值（°） | 相对差值（%） | 纵摇、横摇尾段判据 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 87.240 | 2.006 | -0.006 | -0.29 | 通过 |
| 5 | 218.099 | 5.019 | -0.019 | -0.38 | 通过 |
| 10 | 436.198 | 10.053 | -0.053 | -0.53 | 通过 |

三点误差均小于0.6%，且未随角度明显放大。与旧候选在同类范围内约6%的固定偏差相比，这一结果支持“旧候选的基线偏硬与质量、重心和静水恢复的来源混用有关”的判断。它不意味着新模块已经完成完整平台动力学建模；系泊、动态惯量和阻尼仍需按独立来源接入。

## 可复核材料

- 运行入口：[run_openfast_source_consistent_static_pitch_sweep.py](../scripts/validation/run_openfast_source_consistent_static_pitch_sweep.py)
- 机器可读结果：[openfast_source_consistent_static_pitch_sweep_20260821.json](../artifacts/openfast_source_consistent_static_pitch_sweep_20260821.json)
- 同源静态装配：[source_consistent_reference.py](../src/fowt_platform/source_consistent_reference.py)
- 来源混用诊断：[openfast_pitch_restoring_source_decomposition_audit_20260821.md](openfast_pitch_restoring_source_decomposition_audit_20260821.md)
