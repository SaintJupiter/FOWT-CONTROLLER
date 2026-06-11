"""Representative [60,120] far-event base rate (read-only).

The enriched eval sets (guard10/broader20) gave 42-60% base rate. This computes
the TRUE base rate on the full, non-curated FINO1 population (912k rows, 2005-2025)
using the IDENTICAL label the detector was scored against:

  far_horizon_norm[block] = norm_term(pressure_proxy_vec(uv_block)) = (mean_speed/12)^2
                            clipped to 1.5   (deadband factors cancel exactly)
  far-event(t) = 1 if max over far blocks {60-80,80-100,100-120min} >= 1.0
               <=> max far-block mean wind speed >= 12 m/s

Outputs: overall + per-split representative base rate; per-case spread on the
enriched sets (from oracle planner_logs); implied deployment precision for the
detector's current operating points at the representative base rate.
"""
import glob, os, re
import numpy as np, pandas as pd

DS = "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
PROBE = "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
WIND_REF = 12.0
CAP = 1.5
THETA = 1.0
FAR_STEPS = [(6, 8), (8, 10), (10, 12)]   # 60-80, 80-100, 100-120 min
FARCOLS = ["far_horizon_norm_60_80", "far_horizon_norm_80_100", "far_horizon_norm_100_120"]
# detector best operating points from PR analysis (prec@recall>=0.5)
OPS = {"guard10": (0.80, 0.20), "broader20": (0.76, 0.33)}   # (recall, fpr)


def block_norm(uv_block):
    # uv_block: [steps,2] -> mean per-step speed -> (mean_speed/12)^2 capped
    speed = np.sqrt(uv_block[..., 0] ** 2 + uv_block[..., 1] ** 2)
    mean_speed = speed.mean(axis=-1)
    return np.clip((np.maximum(mean_speed, 0.0) / WIND_REF) ** 2, 0.0, CAP)


def far_event_labels(uv):
    # uv: [N,12,2] -> label per row
    norms = np.stack([block_norm(uv[:, s:e, :]) for (s, e) in FAR_STEPS], axis=1)  # [N,3]
    return (norms.max(axis=1) >= THETA).astype(int)


def representative():
    print("=== Representative [60,120] far-event base rate (full FINO1 population) ===")
    tot_pos = tot_n = 0
    for split in ("train", "validation", "test"):
        uv = np.load(f"{DS}/y_uv_raw_{split}.npy")          # [N,12,2] actual future winds
        lab = far_event_labels(uv)
        print(f"  {split:11} n={len(lab):7d}  base_rate={lab.mean()*100:5.1f}%")
        tot_pos += lab.sum(); tot_n += len(lab)
    base = tot_pos / tot_n
    print(f"  {'OVERALL':11} n={tot_n:7d}  base_rate={base*100:5.1f}%")
    return base


def cid(p):
    return re.sub(r"_prediction_primary_econ_planner_log\.csv$", "", os.path.basename(p))


def enriched_per_case(ds):
    rows = []
    for f in sorted(glob.glob(f"{PROBE}/{ds}_v16_oracle_baseline/planner_logs/*.csv")):
        d = pd.read_csv(f)
        lab = (d[FARCOLS].max(axis=1) >= THETA).astype(int)
        rows.append((cid(f), len(lab), int(lab.sum()), lab.mean()))
    df = pd.DataFrame(rows, columns=["case", "buckets", "far_event_buckets", "case_base_rate"])
    return df


def implied_precision(recall, fpr, b):
    return recall * b / (recall * b + fpr * (1 - b))


def main():
    rep = representative()
    print()
    enriched = {}
    for ds in ("guard10", "broader20"):
        df = enriched_per_case(ds)
        enriched[ds] = df
        print(f"=== {ds} enriched per-case [60,120] far-event rate ===")
        print(f"  set base_rate (pooled buckets) = {df.far_event_buckets.sum()/df.buckets.sum()*100:.1f}%")
        print(f"  per-case spread: min={df.case_base_rate.min()*100:.0f}%  "
              f"median={df.case_base_rate.median()*100:.0f}%  max={df.case_base_rate.max()*100:.0f}%  "
              f"cases_with_0={int((df.case_base_rate==0).sum())}/{len(df)}")
        print(df.sort_values("case_base_rate", ascending=False)
              .assign(case_base_rate=lambda x: (x.case_base_rate*100).round(0))
              .to_string(index=False))
        print()
    print("=== Implied deployment precision at representative base rate ===")
    print(f"representative base rate b = {rep*100:.1f}%")
    for ds, (rec, fpr) in OPS.items():
        for b, tag in [(rep, "representative"),
                       (enriched[ds].far_event_buckets.sum()/enriched[ds].buckets.sum(), "enriched(eval)")]:
            print(f"  {ds:9} (rec={rec*100:.0f}%,fpr={fpr*100:.0f}%)  b={b*100:4.1f}% [{tag:13}] "
                  f"-> implied precision = {implied_precision(rec,fpr,b)*100:.0f}%")
    pd.concat([enriched[d].assign(dataset=d) for d in enriched], ignore_index=True)\
      .to_csv(f"outputs/wind_prediction/h120_supervisory_advisory_v1/far_event_per_case.csv", index=False)


if __name__ == "__main__":
    main()
