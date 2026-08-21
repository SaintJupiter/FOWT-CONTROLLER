# 静态恢复对齐运行装配的纵摇端点核对

更新日期：2026年8月21日
状态：`通过；仅验证MoorDyn开启条件下的静态纵摇恢复量级`

## 修正内容

历史候选将一阶WAMIT静水恢复与辅助`.frc`中的整机质量、重心组合，导致完整条件静态纵摇恢复偏硬。新装配采用冻结ElastoDyn与一阶WAMIT文件建立质量、重心、静水恢复和附加质量，保留辅助`.frc`中的完整惯量张量与局部线性系泊矩阵，并以显式来源标签`static_restoring_aligned_with_aux_frc_inertia_mooring`标识其边界。

该装配不是完整同源六自由度动力学模型。其目的仅是修复已经定位的静态恢复来源混用，并为当前三舱压载短链提供一致的零压载参考状态。

## 端点结果

关闭气动、控制器、风、浪和流，保留HydroDyn与MoorDyn。对纯纵摇HydroDyn `AddF0`静态预载进行临时松弛和600 s原始阻尼释放，读数相对匹配的零预载基线计算。

| 低阶目标纵摇增量（°） | 低阶模型（°） | OpenFAST（°） | 低阶相对OpenFAST差值（%） | 尾段判据 |
| ---: | ---: | ---: | ---: | --- |
| 2 | 2.000 | 2.006 | -0.28 | 通过 |
| 5 | 5.000 | 5.018 | -0.36 | 通过 |
| 10 | 10.000 | 10.047 | -0.47 | 通过 |

三点误差均小于0.5%，相对于历史候选在相同范围内约5.5%至6.1%的系统性偏小，静态恢复基线已得到修正。该结论不外推为惯量、阻尼、系泊非线性、风浪时域响应或控制性能的一致性证明。

## 可复核材料

- 运行装配：[source_consistent_reference.py](../src/fowt_platform/source_consistent_reference.py)
- 运行入口：[run_openfast_large_angle_pitch_sweep.py](../scripts/validation/run_openfast_large_angle_pitch_sweep.py)
- 机器可读结果：[openfast_static_restoring_aligned_pitch_sweep_20260821.json](../artifacts/openfast_static_restoring_aligned_pitch_sweep_20260821.json)
- 历史偏差定位：[openfast_pitch_restoring_source_decomposition_audit_20260821.md](openfast_pitch_restoring_source_decomposition_audit_20260821.md)
