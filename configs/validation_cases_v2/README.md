# 科研版V2分层验证工况

三组清单仅依据FINO1测试集中的风速和水平风矢量构造，未读取累计泵量、姿态响应或其他控制结果。每个工况持续6小时，三层工况互不重叠，且任意两个入选工况的起始时刻至少相隔48小时。旧版170组工况及现有烟雾测试时段未参与抽样。

## 使用边界

- `stage10`用于参数方向筛选，覆盖增强、回落或反转、持续振荡和低扰动。
- `stage20`与`stage10`完全独立，用于检查改进是否依赖少量样本。其结果不回流调参。
- `stage30`仅在参数冻结后运行，用于形成阶段性证据。
- 三层均不是论文最终的大范围验证集合，不得根据节泵结果替换其中的工况。

## 工况定义

- 增强：工况末段平均风速较初段明显升高，且首尾平均来流方向未发生大幅转向。
- 回落或反转：工况末段风速明显降低，或首尾平均风矢量发生较大方向变化。CSV中的`subtype`进一步区分`relief`和`reversal`。
- 持续振荡：首尾风速净变化不大，但平滑风矢量的累计变化、风速范围和转折次数较高。
- 低扰动：风速净变化、风速范围、风矢量累计变化和方向变化均处于测试集较低区间。

具体阈值、固定随机种子、源数据哈希和清单哈希见`selection_manifest.json`。

## stage10

- future_relief_or_reversal：v2_stage10_rev_02、v2_stage10_rel_07、v2_stage10_rel_08
- low_disturbance：v2_stage10_low_01、v2_stage10_low_09
- oscillation：v2_stage10_osc_03、v2_stage10_osc_04
- strengthening：v2_stage10_enh_05、v2_stage10_enh_06、v2_stage10_enh_10

## stage20

- future_relief_or_reversal：v2_stage20_rel_02、v2_stage20_rel_04、v2_stage20_rev_07、v2_stage20_rel_09、v2_stage20_rev_15
- low_disturbance：v2_stage20_low_05、v2_stage20_low_14、v2_stage20_low_16、v2_stage20_low_17、v2_stage20_low_20
- oscillation：v2_stage20_osc_03、v2_stage20_osc_10、v2_stage20_osc_11、v2_stage20_osc_12、v2_stage20_osc_19
- strengthening：v2_stage20_enh_01、v2_stage20_enh_06、v2_stage20_enh_08、v2_stage20_enh_13、v2_stage20_enh_18

## stage30

- future_relief_or_reversal：v2_stage30_rel_04、v2_stage30_rev_07、v2_stage30_rev_09、v2_stage30_rel_11、v2_stage30_rev_13、v2_stage30_rel_17、v2_stage30_rev_20、v2_stage30_rel_29
- low_disturbance：v2_stage30_low_02、v2_stage30_low_06、v2_stage30_low_15、v2_stage30_low_21、v2_stage30_low_23、v2_stage30_low_27、v2_stage30_low_28
- oscillation：v2_stage30_osc_03、v2_stage30_osc_05、v2_stage30_osc_12、v2_stage30_osc_18、v2_stage30_osc_22、v2_stage30_osc_24、v2_stage30_osc_26
- strengthening：v2_stage30_enh_01、v2_stage30_enh_08、v2_stage30_enh_10、v2_stage30_enh_14、v2_stage30_enh_16、v2_stage30_enh_19、v2_stage30_enh_25、v2_stage30_enh_30

## 重叠审计

- 三层内部及相互之间的时间重叠数：0。
- 实际最小起始时间间隔：72小时。
- 与旧版170组及既有烟雾测试时段的时间重叠数：0。

当前数据足以形成三层清单。若后续增加新的类别或提高独立性要求，应从同一测试集的完整6小时记录中补充，并沿用相同的结果盲选、时间去重和参数冻结规则。
