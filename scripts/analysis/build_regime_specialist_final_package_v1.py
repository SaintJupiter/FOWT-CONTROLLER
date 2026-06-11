#!/usr/bin/env python3
"""Build the final paper-ready package for regime-conditioned specialists."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
PAPER = ROOT / "paper_ready"
REGISTRY = PAPER / "regime_specialist_registry_final.json"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text())

    rows: list[list[object]] = []
    for item in registry["specialists"]:
        evidence = item.get("evidence", {})
        saving = ""
        if "matched_24_case_saving_pct" in evidence:
            saving = f"{evidence['matched_24_case_saving_pct']:.2f}% matched; {evidence.get('mixed_subset_saving_pct', 0):.2f}% mixed subset"
        elif "clean_start_32_case_saving_pct" in evidence:
            saving = (
                f"{evidence['clean_start_32_case_saving_pct']:.2f}% clean-start; "
                f"{evidence['broader_64_case_saving_pct']:.2f}% broader"
            )
        elif "dual_dispatcher_boundary_saving_pct" in evidence:
            saving = f"{evidence['dual_dispatcher_boundary_saving_pct']:.2f}% boundary harm"
        else:
            saving = str(evidence.get("role", "not quantified"))
        rows.append(
            [
                item["regime"],
                item["profile"],
                item["status"],
                item["trigger_basis"],
                item["mechanism"],
                saving,
                item["caveat"],
            ]
        )

    with (PAPER / "final_regime_specialist_claim_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "regime",
                "profile",
                "status",
                "trigger_basis",
                "mechanism",
                "evidence",
                "caveat",
            ]
        )
        writer.writerows(rows)

    md = [
        "# Final Regime Specialist Claim Table",
        "",
        "This table is generated from `regime_specialist_registry_final.json` and is the compact paper-facing view of the current controller state.",
        "",
        _markdown_table(
            ["Regime", "Profile", "Status", "Trigger", "Mechanism", "Evidence", "Caveat"],
            rows,
        ),
        "",
        "## Final Architecture Claim",
        "",
        "The final controller should be described as a regime-conditioned specialist library, not as one global pump-saving controller.",
        "",
        "- v1.6 hard safety floor remains the always-on backbone.",
        "- `relief_decay_episode_auto_v1` is the mature automatic specialist.",
        "- `neutral_mhs_clean_budget_v1` is the second automatic candidate.",
        "- `direction_reversal_catchup_boundary` is an abstain/veto regime.",
        "- h120 60-120min information is retained as supervisory context.",
        "- `dual_specialist_pump_saving_v1` is an integration prototype, not the final automatic policy.",
        "",
    ]
    (PAPER / "final_regime_specialist_claim_table.md").write_text("\n".join(md))

    zh = """# 最终分工况策略说明

这份说明由 `regime_specialist_registry_final.json` 生成，用来固定当前最适合写进论文的控制器结构。

## 一句话结论

当前不应再写成“一个全局自动节泵控制器”。更稳妥的表述是：系统由 v1.6 硬安全底座 + 若干分工况专家组成；每个专家只在自己能识别、能解释、已验证的工况里启动。

## 当前可用的工况专家

| 工况 | 当前角色 | 自动启动依据 | 节泵表现 | 主要含义 |
| --- | --- | --- | --- | --- |
| 瞬时峰值后回落 | 成熟自动主线 | 预测显示近期压力/姿态需求会先高后降 | 18.13% matched；混合集里该子集 25.18% | 不追一个马上会自己回落的临时峰值 |
| 中等偏高且稳定、初始姿态很干净 | 第二自动候选 | 预测显示中等偏高但稳定，且当前姿态接近水平 | 31.48% clean-start；更宽泛样本 11.97% | 少做没必要的经济层跟踪，但必须依赖“干净初始姿态” |
| 方向反转/追赶补偿边界 | 禁止节泵/回到 v1.6 | 预测或状态显示反转、catch-up、边界行为 | 双专家混合在这里 -8.15% | 这类不是机会，是边界负例 |
| h120 远端信息 | 监督提示 | 60-120min 远端风险、回落、再增强、方向变化 | advisory only | 保留在系统里，但不直接驱动泵控 |

## 为什么不能直接合并成一个总开关

两个正向专家各自有效，但边界工况会把全局自动合并打坏。混合验证显示总体能省一些泵，但在方向反转/catch-up 边界上会误触发，导致负收益和安全余量成本。所以最终论文主线应是“分工况专家库”，不是一个没有边界的总策略。

## 当前执行原则

- v1.6 hard floor、fallback、recovery 不动。
- 正向专家只在自己工况内启动。
- 边界工况默认 abstain，回到 v1.6。
- 20%左右的结果只在特定工况或机会域内陈述，不写成全局平均。
- h120 继续输出和记录，但定位是远端监督/预警，不声称直接贡献闭环节泵。
"""
    (PAPER / "regime_specialist_final_summary_zh.md").write_text(zh)

    playbook = """# Regime Specialist Execution Playbook

Use this file when reproducing the current paper-facing controllers.

## A0 Baseline

```bash
.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \\
  --cases-csv <CASEBOOK.csv> \\
  --out-dir <OUT_A0> \\
  --primary-label a0_v16 \\
  --primary-control-profile rawenv_holdpause_barrier_reliefcap_adaptive_v1 \\
  --primary-only --duration-s <SECONDS> --skip-figures
```

## Mature Specialist: Transient Peak / Future Decay

Use when the casebook is defined by transient-peak/future-decay opportunity rules.

```bash
.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \\
  --cases-csv <RELIEF_DECAY_CASEBOOK.csv> \\
  --out-dir <OUT_RELIEF_DECAY> \\
  --primary-label relief_decay_episode_auto_v1 \\
  --primary-control-profile relief_decay_episode_auto_v1 \\
  --forecast-source learned \\
  --primary-only --duration-s <SECONDS> --skip-figures
```

Expected role: automatic specialist, not global controller.

## Second Candidate: Neutral Moderate-High Steady Clean-Start

Use when the casebook is defined by neutral-MHS wind shape plus clean current posture.

```bash
.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \\
  --cases-csv <NEUTRAL_MHS_CASEBOOK.csv> \\
  --out-dir <OUT_NEUTRAL_MHS> \\
  --primary-label neutral_mhs_clean_budget_v1 \\
  --primary-control-profile neutral_mhs_clean_budget_v1 \\
  --forecast-source learned \\
  --primary-only --duration-s <SECONDS> --skip-figures
```

Expected role: second automatic candidate. The current-state clean-start gate is essential.

## Integration Prototype: Not Final

```bash
.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \\
  --cases-csv <MIXED_CASEBOOK.csv> \\
  --out-dir <OUT_DUAL> \\
  --primary-label dual_specialist \\
  --primary-control-profile dual_specialist_pump_saving_v1 \\
  --forecast-source learned \\
  --primary-only --duration-s <SECONDS> --skip-figures
```

Expected role: diagnostic only. Mixed validation found positive aggregate saving but boundary/catch-up harm, so do not use as the final paper controller.

## Boundary Rule

Direction reversal / catch-up cases are abstain cases. They should be used as negative controls and boundary figures, not as pump-saving opportunities.
"""
    (PAPER / "regime_specialist_execution_playbook.md").write_text(playbook)


if __name__ == "__main__":
    main()
