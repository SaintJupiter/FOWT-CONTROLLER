# 多阶段控制母体参考资料

整理日期：2026-10-06。用途是精读成熟方法、核对公式和假设，并与当前主动压载代码建立对应。获取全文不等于已完成精读，也不代表新算法或收益已经成立。

**当前定位更新：本目录是已取得资料的历史分类，不代表研究必须采用多阶段控制。** 根据最新要求，方法比较已扩展到预测信任、跨周期决策学习、经济区域控制、执行时序、后悔控制和前馈等方向，见[扩展文献比较](../../docs/transferable_control_literature_map_20261006.md)。Gostin与Hoang继续保留，但不再排他地确定下一步路线。

## 获取状态

| 资料 | 本地文件 | 状态 |
|---|---|---|
| Lucia, Finkler, Engell (2013), Multi-stage nonlinear model predictive control applied to a semi-batch polymerization reactor under uncertainty | 尚无PDF；建议取得后命名为 `01_Lucia_Finkler_Engell_2013_Multi_stage_NMPC.pdf` | 已核对出版信息，尚未取得全文。不能用摘要或另一篇论文代替原文。 |
| Lucia, Engell (2015), Potential and Limitations of Multi-stage Nonlinear Model Predictive Control | [02_Lucia_Engell_2015_Potential_and_Limitations_preprint.pdf](02_Lucia_Engell_2015_Potential_and_Limitations_preprint.pdf) | 已下载公开会议预印本，6页；题名、作者及会议身份匹配，各页可提取文本。不是期刊终版，尚未逐式精读。 |
| do-mpc, Basics of model predictive control | [03_do_mpc_theory_5_1_2_snapshot_20261006.html](03_do_mpc_theory_5_1_2_snapshot_20261006.html) | 已保存官方网页快照，页面显示文档版本5.1.2。含多阶段数学形式；未打包外部样式、图片和脚本，离线排版可能不完整。不是研究论文。 |
| Hoang等 (2025), Probabilistic forecasting for multi-stage nonlinear model predictive control | [04_Hoang_et_al_2025_Probabilistic_Forecasting_Multi_Stage_NMPC.pdf](local_pdfs/04_Hoang_et_al_2025_Probabilistic_Forecasting_Multi_Stage_NMPC.pdf) | 用户提供CCTA出版版，7页，已全文阅读及公式/表格核对；本地授权副本，不上传公开仓库。 |
| Gostin, Koeln (2024), Robust Model Predictive Control with Temporally-Uncertain Disturbance Preview Information | [05_Gostin_Koeln_2024_Temporally_Uncertain_Disturbance_Preview.pdf](local_pdfs/05_Gostin_Koeln_2024_Temporally_Uncertain_Disturbance_Preview.pdf) | 用户提供ACC出版版，6页，已全文阅读及独立数学边界审查；本地授权副本，不上传公开仓库。 |

近期两篇的公式、假设、实际算例、迁移限制及与本地代码的对应见[全文精读与迁移评估](../../docs/recent_preview_control_fulltext_review_20261006.md)。不是仅凭摘要的推荐；也不表示已形成新算法或取得控制收益。

## 来源与版本

### 核心母体论文：2013

- 作者：Sergio Lucia, Tiago Finkler, Sebastian Engell。
- 期刊：Journal of Process Control, 23(9), 1306-1319, October 2013。
- DOI：[10.1016/j.jprocont.2013.08.008](https://doi.org/10.1016/j.jprocont.2013.08.008)。
- [出版商页面](https://www.sciencedirect.com/science/article/pii/S0959152413001686)。当前页面提供机构访问或购买入口，本轮未取得可直接下载的公开全文。
- 后续通过学校图书馆机构权限、作者公开接受稿或作者提供的副本取得全文。不要绕过访问控制。若取得作者稿，应在此登记实际版本，不冒称出版终版。

### 边界与适用性论文：2015

- 作者：Sergio Lucia, Sebastian Engell。
- 正式发表：IFAC-PapersOnLine, 48(8), 1015-1020, 2015。
- DOI：[10.1016/j.ifacol.2015.09.101](https://doi.org/10.1016/j.ifacol.2015.09.101)。
- 本地版本：ADCHEM 2015会议预印本，2015年6月7-10日，Whistler, Canada。预印本页码与正式出版页码不能混用。
- [实际下载源](https://skoge.folk.ntnu.no/prost/proceedings/adchem2015/media/papers/0233.pdf)：NTNU公开会议资料库。
- 文件大小：268132 bytes；SHA-256：`6126dca4c019868983a4bbf24cbe9477650adb4a13faf92c004a793eca78c3bb`。

### 官方实现参考

- [实际保存源](https://www.do-mpc.com/en/latest/theory_mpc.html)。`latest`以后可能更新，本地文件是2026-10-06保存的页面，不代表永久最新。
- 文件大小：47979 bytes；SHA-256：`5513ceeb6026c7766dca22dead1d0d7d5d1df6a5ab4fe1a2877c4d5d631bb316`。
- 理论和实现参考不等于决定本项目必须采用do-mpc；当前求解器、平台和泵执行层不因归档资料而更换。

### 用户提供的近期全文

- Hoang等：IEEE CCTA 2025，281-287，DOI [10.1109/CCTA53793.2025.11151328](https://doi.org/10.1109/CCTA53793.2025.11151328)。SHA-256：`b7332e4bc601b73035910730654790e8595673109c851a3e5327bff1c8854598`。
- Gostin与Koeln：ACC 2024，2488-2493，DOI [10.23919/ACC60939.2024.10644827](https://doi.org/10.23919/ACC60939.2024.10644827)。SHA-256：`0c880354db52a332014ed96c5c3ee5a42afa131a1974babedadcbd9f33f66704`。
- 原始文件来自用户Downloads目录，经逐字节复制保存，未改写PDF。`local_pdfs/.gitignore`将这两份机构授权副本排除于默认Git跟踪之外。

## 精读时需要留下的内容

逐项记录“原文页码/公式编号 - 原文含义和成立假设 - 本系统对应变量或代码 - 保留/修改/不采用”。尤其关注：

1. 决策变量及状态转移；压载目标与实际舱量不能混为一谈。
2. 场景树中的信息揭示；共享信息节点只能拥有共同动作，不能提前获知完整未来。
3. 场景权重的身份；鲁棒情形、概率加权情形及预测概率校准分开。
4. 阶段/终端代价、约束覆盖及滚动执行方式。
5. 定理的假设和结论；反应器算例的收益不能直接当作主动压载收益。
6. 与当前确定性预览QP和实际泵执行层的差异；成熟母体能力与拟新增机制分开。

当前研究顺序采用广泛比较与理论分析：先对不同成熟构造进行同口径精读，再明确一个对象特定改动；暂不指定唯一母体。少量窗口用于检查实现，不作为获得研究资格的门槛；实际收益和普适性留待方法明确后的统一验证。多阶段是候选之一，不预设它必然提升或本身就是原创。

## 文件使用

保留原始PDF和HTML，不在原文件中改写公式。后续阅读笔记另存Markdown，并明确其是原文解释还是本项目设计。下载资料仅用于本地研究；对外分享或上传公开仓库前，应另行确认各文件许可，不因公开可访问就默认可以再分发。
