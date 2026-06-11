"""Target-B supervisory advisory lead-time validation (read-only).

Event (target B) = a DISTURBANCE/high-pressure onset in the *actual* pressure
timeline, independent of controlled posture. Ground truth comes from the ORACLE
run's near block (oracle forecast == actual future), so the event is the physical
pressure event, not the controller-masked attitude breach.

Advisory fires at planner time t when the forecast predicts event-level pressure
(>= theta) in some horizon block:
  - 60-advisory: max over blocks covering [0,60]min
  - 120-advisory: max over all blocks covering [0,120]min

For each actual onset T_e, lead = T_e - earliest advisory fire within its horizon.
We compare, on the SAME ground-truth onsets:
  - learned-60 vs learned-120  (does the longer horizon keep a tail of extra lead?)
  - oracle-120                  (the ceiling, and the learned-vs-oracle gap)

Decision: target B survives iff learned-120 preserves a non-trivial 60-90min tail
of unique lead over learned-60 at acceptable precision.
"""
import glob, os, re, sys
import numpy as np, pandas as pd

OUT = "outputs/wind_prediction/h120_supervisory_advisory_v1"
ORACLE = "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/{ds}_v16_oracle_baseline"
LEARNED = OUT + "/{ds}_v16_learned_h120"

NEAR = ["far_horizon_norm_0_20", "far_horizon_norm_20_40", "far_horizon_norm_40_60"]
FAR = NEAR + ["far_horizon_norm_60_80", "far_horizon_norm_80_100", "far_horizon_norm_100_120"]
TRUTH_COL = "far_horizon_norm_0_20"   # oracle near block == actual near-future pressure
DEDUP_S = 1800.0


def cid(p):
    return re.sub(r"_prediction_primary_econ_planner_log\.csv$", "", os.path.basename(p))


def load_logs(run):
    return {cid(p): pd.read_csv(p) for p in glob.glob(f"{run}/planner_logs/*.csv")}


def onsets(truth_df, theta):
    """Actual-pressure event onsets on the 600s grid."""
    d = truth_df.sort_values("current_time_s")
    t = d["current_time_s"].values
    p = d[TRUTH_COL].values
    out = []
    last = -1e12
    for i in range(len(p)):
        crossed = p[i] >= theta and (i == 0 or p[i - 1] < theta)
        if crossed and (t[i] - last) >= DEDUP_S:
            out.append(t[i])
            last = t[i]
    return out


def fires(adv_df, cols, theta):
    d = adv_df.sort_values("current_time_s")
    sig = d[cols].max(axis=1).values
    return d["current_time_s"].values[sig >= theta]


def lead(onset_t, fire_t, horizon_s):
    cand = [ft for ft in fire_t if 0 <= (onset_t - ft) <= horizon_s]
    return (onset_t - min(cand)) / 60.0 if cand else 0.0


def precision(fire_t, ons, horizon_s):
    if len(fire_t) == 0:
        return float("nan")
    hit = sum(1 for ft in fire_t if any(0 <= (o - ft) <= horizon_s for o in ons))
    return hit / len(fire_t)


def analyze(ds, theta):
    truth = load_logs(ORACLE.format(ds=ds))
    learned = load_logs(LEARNED.format(ds=ds))
    rows = []
    fl60 = fl120 = fo120 = []
    L = {"l60": [], "l120": [], "o120": []}
    fires_l60 = fires_l120 = 0
    ons_total = 0
    prec_l60n = prec_l60d = prec_l120n = prec_l120d = 0
    for c, tdf in truth.items():
        ons = onsets(tdf, theta)
        ons_total += len(ons)
        if c not in learned or not ons:
            continue
        ldf = learned[c]
        f_l60 = fires(ldf, NEAR, theta)
        f_l120 = fires(ldf, FAR, theta)
        f_o120 = fires(tdf, FAR, theta)
        # precision accumulation (per fire, hit if onset within horizon ahead)
        prec_l60d += len(f_l60); prec_l120d += len(f_l120)
        prec_l60n += sum(1 for ft in f_l60 if any(0 <= (o - ft) <= 3600 for o in ons))
        prec_l120n += sum(1 for ft in f_l120 if any(0 <= (o - ft) <= 7200 for o in ons))
        for o in ons:
            L["l60"].append(lead(o, f_l60, 3600))
            L["l120"].append(lead(o, f_l120, 7200))
            L["o120"].append(lead(o, f_o120, 7200))
    a = {k: np.array(v) for k, v in L.items()}
    n = len(a["l60"])
    if n == 0:
        return None
    dl = a["l120"] - a["l60"]
    return dict(
        ds=ds, theta=theta, onsets=ons_total, matched=n,
        learn60_med=np.median(a["l60"]), learn60_max=a["l60"].max(),
        learn120_med=np.median(a["l120"]), learn120_max=a["l120"].max(),
        oracle120_max=a["o120"].max(),
        dlead_med=np.median(dl), dlead_max=dl.max(),
        frac_120_earlier=float((dl > 1e-6).mean()),
        frac_120_ge30=float((dl >= 30).mean()),
        learn_vs_oracle_taildrop=a["o120"].max() - a["l120"].max(),
        prec_l60=(prec_l60n / prec_l60d) if prec_l60d else float("nan"),
        prec_l120=(prec_l120n / prec_l120d) if prec_l120d else float("nan"),
    )


def main():
    print("Target-B supervisory advisory lead (learned vs oracle ceiling)\n")
    for theta in (0.9, 1.0, 1.1):
        print(f"########## theta = {theta} ##########")
        for ds in ("guard10", "broader20"):
            r = analyze(ds, theta)
            if r is None:
                print(f"[{ds}] no matched onsets"); continue
            print(
                f"[{ds}] onsets={r['onsets']:3d}  "
                f"learn60(med/max)={r['learn60_med']:.0f}/{r['learn60_max']:.0f}  "
                f"learn120(med/max)={r['learn120_med']:.0f}/{r['learn120_max']:.0f}  "
                f"oracle120_max={r['oracle120_max']:.0f}  "
                f"Δlearn(med/max)={r['dlead_med']:.0f}/{r['dlead_max']:.0f}  "
                f"120-earlier={r['frac_120_earlier']*100:.0f}%(≥30min {r['frac_120_ge30']*100:.0f}%)  "
                f"prec60/120={r['prec_l60']*100:.0f}/{r['prec_l120']*100:.0f}%  "
                f"learn-vs-oracle tail drop={r['learn_vs_oracle_taildrop']:.0f}min"
            )
        print()


if __name__ == "__main__":
    main()
