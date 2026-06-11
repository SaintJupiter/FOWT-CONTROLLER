"""Target-B PR / operating-point analysis on the EXISTING far-block signal (read-only).

Task: at planner time t, predict whether a high-pressure event occurs in the
[t+60, t+120]min window -- the region a 60-min forecast structurally cannot see.

  label(t)  = 1 if max(oracle.{60_80,80_100,100_120}(t)) >= theta_event
              (oracle far blocks == actual future pressure)
  score(t)  = max(learned.{60_80,80_100,100_120}(t))
              (learned far blocks only; no 0-60 leakage)

No model is trained. We sweep a single threshold on the existing learned far
score and report the precision-recall frontier + best operating points, to test
whether the AUC-0.85 far-horizon skill converts to usable advisory precision.
"""
import glob, os, re
import numpy as np, pandas as pd

ORACLE = "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/{ds}_v16_oracle_baseline"
LEARNED = "outputs/wind_prediction/h120_supervisory_advisory_v1/{ds}_v16_learned_h120"
OUT = "outputs/wind_prediction/h120_supervisory_advisory_v1"
FARBLOCKS = ["far_horizon_norm_60_80", "far_horizon_norm_80_100", "far_horizon_norm_100_120"]
FAR_OFFSETS = [60, 80, 100]   # block start (min) for lead estimate
THETA_EVENT = 1.0


def cid(p):
    return re.sub(r"_prediction_primary_econ_planner_log\.csv$", "", os.path.basename(p))


def logs(run):
    return {cid(p): pd.read_csv(p) for p in glob.glob(f"{run}/planner_logs/*.csv")}


def build(ds, theta):
    o = logs(ORACLE.format(ds=ds))
    l = logs(LEARNED.format(ds=ds))
    lab, scr, lead = [], [], []
    for c in o:
        if c not in l:
            continue
        od = o[c].set_index("current_time_s")
        ld = l[c].set_index("current_time_s")
        idx = od.index.intersection(ld.index)
        act = od.loc[idx, FARBLOCKS].values   # actual future pressure in [60,120]
        prd = ld.loc[idx, FARBLOCKS].values   # learned prediction
        for ar, pr in zip(act, prd):
            label = int(ar.max() >= theta)
            lab.append(label)
            scr.append(float(pr.max()))
            # lead = earliest far-block offset whose ACTUAL crosses theta (true events only)
            if label:
                hits = [FAR_OFFSETS[k] for k in range(3) if ar[k] >= theta]
                lead.append(min(hits) if hits else np.nan)
            else:
                lead.append(np.nan)
    return np.array(lab), np.array(scr), np.array(lead)


def pr_frontier(lab, scr):
    P = lab.sum()
    N = len(lab) - P
    pts = []
    for tau in np.unique(np.round(scr, 3)):
        fire = scr >= tau
        tp = int((fire & (lab == 1)).sum())
        fp = int((fire & (lab == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / P if P else float("nan")
        fpr = fp / N if N else float("nan")
        pts.append((tau, prec, rec, fpr, tp + fp))
    return pd.DataFrame(pts, columns=["tau", "precision", "recall", "fpr", "fires"]), P, N


def best_points(fr):
    f = fr.dropna(subset=["precision", "recall"]).copy()
    f = f[f.fires > 0]
    f["f1"] = 2 * f.precision * f.recall / (f.precision + f.recall).replace(0, np.nan)
    out = {}
    out["max_f1"] = f.loc[f.f1.idxmax()] if len(f) else None
    rec50 = f[f.recall >= 0.5]
    out["prec@rec>=0.5"] = rec50.loc[rec50.precision.idxmax()] if len(rec50) else None
    return out


def main():
    print(f"Target-B PR / operating point  (theta_event={THETA_EVENT}, window [60,120]min)\n")
    rows = []
    for ds in ("guard10", "broader20"):
        lab, scr, lead = build(ds, THETA_EVENT)
        fr, P, N = pr_frontier(lab, scr)
        base = P / (P + N)
        print(f"===== {ds} =====")
        print(f"samples={P+N}  positives(event in[60,120])={P}  base_rate={base*100:.0f}%  "
              f"mean_lead(true events)={np.nanmean(lead):.0f}min")
        # frontier snapshot
        print(f"{'tau':>5} {'prec':>6} {'recall':>7} {'fpr':>6} {'fires':>6}")
        for _, r in fr.iterrows():
            if r.fires > 0:
                print(f"{r.tau:>5.2f} {r.precision*100:>5.0f}% {r.recall*100:>6.0f}% {r.fpr*100:>5.0f}% {int(r.fires):>6}")
        bp = best_points(fr)
        for name, r in bp.items():
            if r is not None:
                print(f"  [{name}] tau={r.tau:.2f}  precision={r.precision*100:.0f}%  "
                      f"recall={r.recall*100:.0f}%  fpr={r.fpr*100:.0f}%")
                rows.append(dict(dataset=ds, op=name, base_rate=base, tau=r.tau,
                                 precision=r.precision, recall=r.recall, fpr=r.fpr,
                                 mean_lead_min=np.nanmean(lead)))
        print()
        fr.assign(dataset=ds).to_csv(f"{OUT}/pr_frontier_{ds}.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/pr_best_operating_points.csv", index=False)
    print(f"saved -> {OUT}/pr_frontier_*.csv, pr_best_operating_points.csv")


if __name__ == "__main__":
    main()
