# 静态恢复项对齐后的纵摇自由释放核对

更新日期：2026年8月21日
状态：`完成局部纵摇时间尺度核对，不作为阻尼、完整六自由度或闭环控制验证`

## 条件

本检查关闭风、波浪、海流、气动和控制器，仅保留刚性平台六自由度、HydroDyn与MoorDyn。OpenFAST和低阶模型均从参考平衡附近施加`1 deg`初始纵摇并自由释放。低阶侧采用`static_restoring_aligned_with_aux_frc_inertia_mooring`装配，阻尼矩阵显式取零。

## 结果

| 指标 | OpenFAST | 低阶模型 | 差异 |
| --- | ---: | ---: | ---: |
| 主周期（s） | 28.50 | 27.65 | -0.85 s（-2.98%） |

两侧纵摇响应的初始恢复方向一致。相对历史混合装配的`26.88 s`，当前周期差由约`5.7%`收敛至`3.0%`以内。

## 边界

该结果只针对无环境、`1 deg`局部扰动下的纵摇自由响应。低阶模型未拟合阻尼，因此不比较衰减包络，也不据此说明横摇、纵荡、耦合响应、波浪工况、平均风运行点或压载闭环控制已经得到验证。

## 可复核材料

- 运行入口：[run_openfast_aligned_pitch_free_response.py](../scripts/validation/run_openfast_aligned_pitch_free_response.py)
- 单元测试：[test_openfast_aligned_pitch_free_response.py](../tests/test_openfast_aligned_pitch_free_response.py)
- 机器可读结果：[openfast_aligned_pitch_free_response_20260821.json](../artifacts/openfast_aligned_pitch_free_response_20260821.json)
