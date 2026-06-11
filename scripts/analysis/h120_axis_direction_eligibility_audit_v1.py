"""Step 1 (read-only): direction-consistency audit of the addressable prepare window.

Does NOT touch the controller. For each case it:
  1. Rebuilds the addressable prepare window on the 1Hz timeseries:
       C1 posture in [4,5) deg   (posture = max(|pitch|,|roll|))
       C2 worsening              (posture[t] - posture[t-60] > 0.05 deg)
       C3 pump idle              (pump_total_rate_m3_min < 0.5)
  2. Forward-fills (merge_asof backward) the 600s far-horizon DIRECTION signals
     onto each 1Hz second:
       far_horizon_direction_shift, far_horizon_reversal,
       far_horizon_dir_shift_deg, far_horizon_far_max (pressure rise magnitude)
  3. Classifies each eligible second as direction-clean iff
       direction_shift == 0 AND reversal == 0.
  4. Also reports dominant-axis stability (pitch vs roll) inside the window,
     since axis-aware micro needs a well-defined axis to act on.

Pass bar (decided before running, fail-fast):
  - direction-clean fraction of eligible window >= 0.50, AND
  - dominant-axis-stable fraction >= 0.50,
  on the set that actually has a window (broader20). Otherwise Step 2 is moot.
"""
import glob, os, re
import numpy as np
import pandas as pd

PROBE = "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
OUT = "outputs/wind_prediction/h120_axis_direction_eligibility_audit_v1"
os.makedirs(OUT, exist_ok=True)

LO, HI = 4.0, 5.0          # C1 posture band
WORSEN_LAG, WORSEN_EPS = 60, 0.05   # C2
PUMP_IDLE = 0.5            # C3
FLOOR = 5.0                # floor domain reference (time>5)

DIR_COLS = ["current_time_s", "far_horizon_direction_shift", "far_horizon_reversal",
            "far_horizon_dir_shift_deg", "far_horizon_far_max", "far_horizon_far_min"]


def case_id(path):
    b = os.path.basename(path)
    return re.sub(r"_prediction_primary_econ_(timeseries|planner_log)\.csv$", "", b)


def audit_run(run_dir, dataset):
    ts_files = {case_id(p): p for p in glob.glob(f"{run_dir}/timeseries/*.csv")}
    pl_files = {case_id(p): p for p in glob.glob(f"{run_dir}/planner_logs/*.csv")}
    rows = []
    for cid, tsf in sorted(ts_files.items()):
        plf = pl_files.get(cid)
        t = pd.read_csv(tsf)
        posture = np.maximum(t["pitch_deg"].abs(), t["roll_deg"].abs())
        pump = t["pump_total_rate_m3_min"]
        c1 = (posture >= LO) & (posture < HI)
        worsen = posture - posture.shift(WORSEN_LAG)
        c2 = worsen > WORSEN_EPS
        c3 = pump < PUMP_IDLE
        elig = (c1 & c2 & c3).fillna(False)

        # axis dominance inside band
        pitch_dom = t["pitch_deg"].abs() >= t["roll_deg"].abs()

        # forward-fill direction signals from 600s planner buckets onto 1Hz
        p = pd.read_csv(plf)
        avail = [c for c in DIR_COLS if c in p.columns]
        p = p[avail].sort_values("current_time_s")
        merged = pd.merge_asof(
            pd.DataFrame({"t_s": t["t_s"].values}).sort_values("t_s"),
            p.rename(columns={"current_time_s": "t_s"}),
            on="t_s", direction="backward")
        ds = merged["far_horizon_direction_shift"].fillna(1).astype(int).values
        rv = merged["far_horizon_reversal"].fillna(1).astype(int).values
        dshift = merged["far_horizon_dir_shift_deg"].values
        farmax = merged.get("far_horizon_far_max", pd.Series(np.nan, index=merged.index)).values

        clean = (ds == 0) & (rv == 0)
        e = elig.values
        n_elig = int(e.sum())
        n_clean = int((e & clean).sum())
        # dominant-axis-stable: within eligible window, fraction in the majority axis
        if n_elig > 0:
            pd_e = pitch_dom.values[e]
            axis_major = max(pd_e.mean(), 1 - pd_e.mean())
            dshift_e = dshift[e]
            farmax_e = farmax[e]
        else:
            axis_major = np.nan
            dshift_e = np.array([])
            farmax_e = np.array([])

        rows.append(dict(
            dataset=dataset, case=cid, total_s=len(t),
            time_over_floor=int((posture > FLOOR).sum()),
            elig_s=n_elig,
            elig_clean_s=n_clean,
            clean_frac=(n_clean / n_elig) if n_elig else np.nan,
            axis_stable_frac=axis_major,
            pitch_dom_frac=(pitch_dom.values[e].mean() if n_elig else np.nan),
            dshift_mean=(float(np.nanmean(dshift_e)) if n_elig else np.nan),
            dshift_p90=(float(np.nanpercentile(dshift_e, 90)) if n_elig else np.nan),
            farmax_mean_clean=(float(np.nanmean(farmax[e & clean])) if n_clean else np.nan),
        ))
    return pd.DataFrame(rows)


def summarize(df, dataset):
    tot_elig = df.elig_s.sum()
    tot_clean = df.elig_clean_s.sum()
    tot_floor = df.time_over_floor.sum()
    # eligible-weighted axis stability
    w = df.elig_s.replace(0, np.nan)
    axis_w = float(np.nansum(df.axis_stable_frac * w) / np.nansum(w)) if tot_elig else np.nan
    return dict(
        dataset=dataset, cases=len(df), total_s=int(df.total_s.sum()),
        elig_s=int(tot_elig),
        elig_clean_s=int(tot_clean),
        clean_frac=(tot_clean / tot_elig) if tot_elig else np.nan,
        time_over_floor=int(tot_floor),
        clean_over_floor_ratio=(tot_clean / tot_floor) if tot_floor else np.nan,
        axis_stable_frac_w=axis_w,
        cases_with_clean=int((df.elig_clean_s > 0).sum()),
    )


all_cases, summaries = [], []
for dataset, run in [("guard10", f"{PROBE}/guard10_v16_oracle_baseline"),
                     ("broader20", f"{PROBE}/broader20_v16_oracle_baseline")]:
    df = audit_run(run, dataset)
    all_cases.append(df)
    summaries.append(summarize(df, dataset))

case_df = pd.concat(all_cases, ignore_index=True)
sum_df = pd.DataFrame(summaries)
case_df.to_csv(f"{OUT}/direction_eligibility_case_table.csv", index=False)
sum_df.to_csv(f"{OUT}/direction_eligibility_summary.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 40)
print("=== SUMMARY ===")
print(sum_df.to_string(index=False))
print("\n=== PER-CASE (eligible cases only) ===")
print(case_df[case_df.elig_s > 0].sort_values(["dataset", "elig_clean_s"], ascending=[True, False])
      .to_string(index=False))
