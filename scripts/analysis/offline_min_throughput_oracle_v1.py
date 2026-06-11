#!/usr/bin/env python3
"""Offline minimum-throughput oracle for ballast pump-saving ceilings.

This is a read-only analysis script. It compares existing closed-only and
primary/deadband time series, then estimates whether a linearized
minimum-throughput oracle has material room below the current deadband result.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse


RHO_KG_M3 = 1025.0
DEFAULT_TANK_CAPACITY_KG = 1850.0 * RHO_KG_M3
TANK_COLS = ("tank1_kg", "tank2_kg", "tank3_kg")
TARGET_TANK_COLS = ("target_tank1_kg", "target_tank2_kg", "target_tank3_kg")
POSTURE_COLS = ("pitch_deg", "roll_deg")
PUMP_COLS = ("pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min")
TOTAL_PUMP_COL = "pump_total_rate_m3_min"
READ_COLS = {"t_s", *TANK_COLS, *TARGET_TANK_COLS, *POSTURE_COLS, *PUMP_COLS, TOTAL_PUMP_COL}


@dataclass(frozen=True)
class CasePair:
    case_id: str
    timestamp: str
    label: str
    closed_path: Path
    primary_path: Path
    summary_row: dict[str, object]


@dataclass
class FitResult:
    B: np.ndarray
    rank: int
    condition: float
    rmse_pitch: float
    rmse_roll: float
    n_samples: int
    source: str
    usable: bool
    reason: str


@dataclass
class LpResult:
    pump_m3: float
    effective_pump_m3: float
    tail_to_closed_m3: float
    status: str
    terminal_error_kg: float
    terminal_reference_error_kg: float
    posture_violation_max_deg: float
    oracle_est_time_gt3_s: float
    oracle_est_time_gt4_s: float
    oracle_est_time_gt5_s: float
    max_abs_pitch_deg: float
    max_abs_roll_deg: float
    m_path: object = None
    t_path: object = None
    m_closed_path: object = None
    posture_est_path: object = None
    posture_closed_path: object = None


def _timestamp_stem(ts: object) -> str:
    return str(ts).replace(":", "").replace(" ", "_")


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _read_timeseries(path: Path) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in READ_COLS)
            break
        except TimeoutError as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        assert last_exc is not None
        raise last_exc
    missing = [c for c in ("t_s", *TANK_COLS, *POSTURE_COLS) if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    return df


def _infer_dt_s(df: pd.DataFrame, fallback: float = 1.0) -> float:
    if "t_s" not in df.columns or len(df) < 2:
        return float(fallback)
    t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float(fallback)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return float(fallback)
    return float(np.median(dt))


def _time_over_limit_s(df: pd.DataFrame, limit_deg: float) -> float:
    if len(df) == 0:
        return 0.0
    pitch = pd.to_numeric(df.get("pitch_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    roll = pd.to_numeric(df.get("roll_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    axis = np.maximum(np.abs(pitch), np.abs(roll))
    if "t_s" in df.columns and len(df) >= 2:
        t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(t)
        med = _infer_dt_s(df)
        weights = np.concatenate([np.where(np.isfinite(dt) & (dt > 0.0), dt, med), [med]])
    else:
        weights = np.full(len(df), _infer_dt_s(df), dtype=float)
    weights = weights[: len(df)]
    return float(np.sum(weights[axis > float(limit_deg)]))


def _max_abs_axis(df: pd.DataFrame) -> tuple[float, float]:
    pitch = pd.to_numeric(df.get("pitch_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    roll = pd.to_numeric(df.get("roll_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float(np.max(np.abs(pitch))) if pitch.size else 0.0, float(np.max(np.abs(roll))) if roll.size else 0.0


def _last_vec(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return df[cols].iloc[-1].to_numpy(dtype=float)


def _first_vec(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return df[cols].iloc[0].to_numpy(dtype=float)


def _pump_work_from_timeseries(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    if TOTAL_PUMP_COL in df.columns:
        rate = pd.to_numeric(df[TOTAL_PUMP_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        cols = [c for c in PUMP_COLS if c in df.columns]
        if not cols:
            return 0.0
        rate = (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            .sum(axis=1)
        )
    if "t_s" in df.columns and len(df) >= 2:
        t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(t)
        med = _infer_dt_s(df)
        weights = np.concatenate([np.where(np.isfinite(dt) & (dt > 0.0), dt, med), [med]])
    else:
        weights = np.full(len(df), _infer_dt_s(df), dtype=float)
    weights = weights[: len(rate)]
    return float(np.sum(rate * weights / 60.0))


def _summary_pump(row: dict[str, object], key: str, df: pd.DataFrame) -> float:
    value = _as_float(row.get(key), default=math.nan)
    if np.isfinite(value):
        return float(value)
    return _pump_work_from_timeseries(df)


def _find_timeseries(
    timeseries_dir: Path,
    case_id: str,
    timestamp: str,
    primary_label_substring: str = "",
) -> tuple[Path, Path]:
    prefix = f"{case_id}_{_timestamp_stem(timestamp)}_"
    closed = sorted(timeseries_dir.glob(f"{prefix}closed_only_timeseries.csv"))
    if not closed:
        raise FileNotFoundError(f"closed-only time series not found for {prefix}")
    candidates = sorted(
        p
        for p in timeseries_dir.glob(f"{prefix}*_timeseries.csv")
        if not p.name.endswith("_closed_only_timeseries.csv")
    )
    if primary_label_substring:
        candidates = [p for p in candidates if primary_label_substring in p.name]
    if not candidates:
        raise FileNotFoundError(f"primary time series not found for {prefix}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"multiple primary time series for {prefix}: {names}")
    return closed[0], candidates[0]


def _load_case_pairs(
    run_dir: Path,
    max_cases: int,
    primary_label_substring: str,
) -> list[CasePair]:
    summary_path = run_dir / "casebook_summary.csv"
    timeseries_dir = run_dir / "timeseries"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not timeseries_dir.exists():
        raise FileNotFoundError(timeseries_dir)

    summary = pd.read_csv(summary_path)
    if "case_id" not in summary.columns or "timestamp" not in summary.columns:
        raise KeyError(f"{summary_path} must contain case_id and timestamp columns")
    if max_cases > 0:
        summary = summary.head(int(max_cases)).copy()

    pairs: list[CasePair] = []
    for _, row in summary.iterrows():
        case_id = str(row["case_id"])
        timestamp = str(row["timestamp"])
        closed_path, primary_path = _find_timeseries(
            timeseries_dir=timeseries_dir,
            case_id=case_id,
            timestamp=timestamp,
            primary_label_substring=primary_label_substring,
        )
        pairs.append(
            CasePair(
                case_id=case_id,
                timestamp=timestamp,
                label=str(row.get("label", "")),
                closed_path=closed_path,
                primary_path=primary_path,
                summary_row=row.to_dict(),
            )
        )
    return pairs


def _resample_trace(df: pd.DataFrame, resample_s: float, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns and c != "t_s"]
    work = df[["t_s", *cols]].copy()
    work["t_s"] = pd.to_numeric(work["t_s"], errors="coerce")
    work = work.dropna(subset=["t_s"]).sort_values("t_s")
    work = work.groupby("t_s", as_index=False).last()
    if len(work) == 0:
        raise ValueError("cannot resample an empty trace")

    t = work["t_s"].to_numpy(dtype=float)
    start = float(t[0])
    end = float(t[-1])
    if resample_s <= 0.0:
        grid = t
    else:
        grid = np.arange(start, end + 1e-9, float(resample_s), dtype=float)
        if grid.size == 0 or abs(float(grid[0]) - start) > 1e-9:
            grid = np.insert(grid, 0, start)
        if float(grid[-1]) < end - 1e-6:
            grid = np.append(grid, end)

    out: dict[str, np.ndarray] = {"t_s": grid}
    for col in cols:
        values = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(t)
        if np.sum(mask) == 0:
            out[col] = np.full_like(grid, np.nan, dtype=float)
        elif np.sum(mask) == 1:
            out[col] = np.full_like(grid, values[mask][0], dtype=float)
        else:
            out[col] = np.interp(grid, t[mask], values[mask])
    return pd.DataFrame(out)


def _paired_resampled(
    closed: pd.DataFrame,
    primary: pd.DataFrame,
    resample_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = (*TANK_COLS, *POSTURE_COLS)
    c = _resample_trace(closed, resample_s=resample_s, cols=cols)
    p = _resample_trace(primary, resample_s=resample_s, cols=cols)
    start = max(float(c["t_s"].iloc[0]), float(p["t_s"].iloc[0]))
    end = min(float(c["t_s"].iloc[-1]), float(p["t_s"].iloc[-1]))
    if end < start:
        raise ValueError("paired traces have no overlapping time span")
    if resample_s <= 0.0:
        grid = np.intersect1d(c["t_s"].to_numpy(dtype=float), p["t_s"].to_numpy(dtype=float))
    else:
        grid = np.arange(start, end + 1e-9, float(resample_s), dtype=float)
        if grid.size == 0 or float(grid[-1]) < end - 1e-6:
            grid = np.append(grid, end)

    def regrid(src: pd.DataFrame) -> pd.DataFrame:
        out = {"t_s": grid}
        t = src["t_s"].to_numpy(dtype=float)
        for col in cols:
            out[col] = np.interp(grid, t, src[col].to_numpy(dtype=float))
        return pd.DataFrame(out)

    return regrid(c), regrid(p)


def _fit_from_xy(
    X: np.ndarray,
    Y: np.ndarray,
    source: str,
    max_condition: float,
    min_rank: int,
    min_samples: int,
) -> FitResult:
    if X.size == 0 or Y.size == 0:
        return FitResult(
            B=np.full((2, 3), np.nan),
            rank=0,
            condition=math.inf,
            rmse_pitch=math.nan,
            rmse_roll=math.nan,
            n_samples=0,
            source=source,
            usable=False,
            reason="no_fit_samples",
        )
    finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X = X[finite]
    Y = Y[finite]
    mag = np.linalg.norm(X, axis=1)
    keep = mag > 1.0
    X = X[keep]
    Y = Y[keep]
    if X.shape[0] < min_samples:
        return FitResult(
            B=np.full((2, 3), np.nan),
            rank=0,
            condition=math.inf,
            rmse_pitch=math.nan,
            rmse_roll=math.nan,
            n_samples=int(X.shape[0]),
            source=source,
            usable=False,
            reason="too_few_fit_samples",
        )
    coef, _, rank, svals = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    resid = pred - Y
    rmse = np.sqrt(np.mean(resid * resid, axis=0))
    if len(svals) == 0 or int(rank) <= 0:
        condition = math.inf
    else:
        condition = float(svals[0] / max(svals[int(rank) - 1], 1e-12))
    usable = int(rank) >= int(min_rank) and np.isfinite(condition) and condition <= max_condition
    reason = "ok" if usable else "ill_conditioned_or_low_rank"
    return FitResult(
        B=coef.T,
        rank=int(rank),
        condition=condition,
        rmse_pitch=float(rmse[0]),
        rmse_roll=float(rmse[1]),
        n_samples=int(X.shape[0]),
        source=source,
        usable=bool(usable),
        reason=reason,
    )


def _fit_response_model(
    closed_rs: pd.DataFrame,
    primary_rs: pd.DataFrame,
    source: str,
    max_condition: float,
    min_rank: int,
    min_samples: int,
) -> FitResult:
    X = primary_rs[list(TANK_COLS)].to_numpy(dtype=float) - closed_rs[list(TANK_COLS)].to_numpy(dtype=float)
    Y = primary_rs[list(POSTURE_COLS)].to_numpy(dtype=float) - closed_rs[list(POSTURE_COLS)].to_numpy(dtype=float)
    return _fit_from_xy(
        X=X,
        Y=Y,
        source=source,
        max_condition=max_condition,
        min_rank=min_rank,
        min_samples=min_samples,
    )


def _fit_family_response_model(
    pairs: list[CasePair],
    resample_s: float,
    max_condition: float,
    min_rank: int,
    min_samples: int,
) -> FitResult:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for pair in pairs:
        closed = _read_timeseries(pair.closed_path)
        primary = _read_timeseries(pair.primary_path)
        closed_rs, primary_rs = _paired_resampled(closed, primary, resample_s=resample_s)
        xs.append(primary_rs[list(TANK_COLS)].to_numpy(dtype=float) - closed_rs[list(TANK_COLS)].to_numpy(dtype=float))
        ys.append(primary_rs[list(POSTURE_COLS)].to_numpy(dtype=float) - closed_rs[list(POSTURE_COLS)].to_numpy(dtype=float))
    if not xs:
        return _fit_from_xy(
            X=np.empty((0, 3)),
            Y=np.empty((0, 2)),
            source="family",
            max_condition=max_condition,
            min_rank=min_rank,
            min_samples=min_samples,
        )
    return _fit_from_xy(
        X=np.vstack(xs),
        Y=np.vstack(ys),
        source="family",
        max_condition=max_condition,
        min_rank=min_rank,
        min_samples=min_samples,
    )


def _weighted_time_over_limit(posture: np.ndarray, t_s: np.ndarray, limit_deg: float) -> float:
    if posture.size == 0:
        return 0.0
    axis = np.max(np.abs(posture), axis=1)
    if len(t_s) >= 2:
        dt = np.diff(t_s)
        med = float(np.median(dt[np.isfinite(dt) & (dt > 0.0)])) if np.any(np.isfinite(dt) & (dt > 0.0)) else 0.0
        weights = np.concatenate([np.where(np.isfinite(dt) & (dt > 0.0), dt, med), [med]])
    else:
        weights = np.ones(len(posture), dtype=float)
    return float(np.sum(weights[axis > float(limit_deg)]))


def _solve_lp_oracle(
    closed_rs: pd.DataFrame,
    primary_rs: pd.DataFrame,
    fit: FitResult,
    posture_limit_deg: float,
    posture_bound_mode: str,
    posture_margin_deg: float,
    terminal_tank_tol_kg: float,
    terminal_reference: str,
    tank_capacity_kg: float,
    pump_rate_limit_m3_min: float,
    rho_kg_m3: float,
    trust_region_mode: str = "off",
    trust_region_margin_kg: float = 0.0,
) -> LpResult:
    if not fit.usable:
        return LpResult(
            pump_m3=math.nan,
            effective_pump_m3=math.nan,
            tail_to_closed_m3=math.nan,
            status=f"skipped_fit_{fit.reason}",
            terminal_error_kg=math.nan,
            terminal_reference_error_kg=math.nan,
            posture_violation_max_deg=math.nan,
            oracle_est_time_gt3_s=math.nan,
            oracle_est_time_gt4_s=math.nan,
            oracle_est_time_gt5_s=math.nan,
            max_abs_pitch_deg=math.nan,
            max_abs_roll_deg=math.nan,
        )

    t_s = closed_rs["t_s"].to_numpy(dtype=float)
    m_closed = closed_rs[list(TANK_COLS)].to_numpy(dtype=float)
    posture_closed = closed_rs[list(POSTURE_COLS)].to_numpy(dtype=float)
    m_primary = primary_rs[list(TANK_COLS)].to_numpy(dtype=float)
    posture_primary = primary_rs[list(POSTURE_COLS)].to_numpy(dtype=float)
    n_steps = int(len(t_s))
    n_intervals = max(0, n_steps - 1)
    if n_steps < 2:
        return LpResult(
            pump_m3=math.nan,
            effective_pump_m3=math.nan,
            tail_to_closed_m3=math.nan,
            status="skipped_too_few_time_steps",
            terminal_error_kg=math.nan,
            terminal_reference_error_kg=math.nan,
            posture_violation_max_deg=math.nan,
            oracle_est_time_gt3_s=math.nan,
            oracle_est_time_gt4_s=math.nan,
            oracle_est_time_gt5_s=math.nan,
            max_abs_pitch_deg=math.nan,
            max_abs_roll_deg=math.nan,
        )

    n_m = 3 * n_steps
    n_u = 3 * n_intervals
    n_vars = n_m + 2 * n_u

    def m_idx(k: int, i: int) -> int:
        return 3 * k + i

    def up_idx(k: int, i: int) -> int:
        return n_m + 3 * k + i

    def un_idx(k: int, i: int) -> int:
        return n_m + n_u + 3 * k + i

    c = np.zeros(n_vars, dtype=float)
    if n_u > 0:
        c[n_m:] = 1.0 / float(rho_kg_m3)

    # Tank-mass bounds.  Default: full tank.  Trust region: restrict each tank to the
    # range the deadband/closed traces actually visited (m_closed + observed delta
    # envelope), so the LP cannot exploit the fitted linear posture model OUTSIDE the
    # range where it has data (the rank-2 tank->posture map has a null space the LP
    # otherwise rides to "redistribute ballast for free").
    if str(trust_region_mode).strip().lower() in ("observed_delta", "observed", "on"):
        obs_delta = m_primary - m_closed
        obs_lo = obs_delta.min(axis=0)
        obs_hi = obs_delta.max(axis=0)
        margin = float(trust_region_margin_kg)
        bounds = []
        for k in range(n_steps):
            for i in range(3):
                lo = max(0.0, float(m_closed[k, i] + obs_lo[i] - margin))
                hi = min(float(tank_capacity_kg), float(m_closed[k, i] + obs_hi[i] + margin))
                bounds.append((lo, max(hi, lo)))
    else:
        bounds = [(0.0, float(tank_capacity_kg)) for _ in range(n_m)]
    dt = np.diff(t_s)
    for k in range(n_intervals):
        step_bound = max(0.0, float(pump_rate_limit_m3_min) * float(rho_kg_m3) * float(dt[k]) / 60.0)
        for _ in range(3):
            bounds.append((0.0, step_bound))
    for k in range(n_intervals):
        step_bound = max(0.0, float(pump_rate_limit_m3_min) * float(rho_kg_m3) * float(dt[k]) / 60.0)
        for _ in range(3):
            bounds.append((0.0, step_bound))

    eq_rows = 3 + 3 * n_intervals
    a_eq = sparse.lil_matrix((eq_rows, n_vars), dtype=float)
    b_eq = np.zeros(eq_rows, dtype=float)
    row = 0
    for i in range(3):
        a_eq[row, m_idx(0, i)] = 1.0
        b_eq[row] = float(m_closed[0, i])
        row += 1
    for k in range(n_intervals):
        for i in range(3):
            a_eq[row, m_idx(k + 1, i)] = 1.0
            a_eq[row, m_idx(k, i)] = -1.0
            a_eq[row, up_idx(k, i)] = -1.0
            a_eq[row, un_idx(k, i)] = 1.0
            row += 1

    ub_rows = 6 + 4 * n_steps
    a_ub = sparse.lil_matrix((ub_rows, n_vars), dtype=float)
    b_ub = np.zeros(ub_rows, dtype=float)
    row = 0
    terminal_mode = str(terminal_reference).strip().lower()
    if terminal_mode == "closed":
        terminal = m_closed[-1, :]
    elif terminal_mode == "deadband":
        terminal = m_primary[-1, :]
    else:
        raise ValueError(f"unsupported terminal_reference={terminal_reference}")
    for i in range(3):
        a_ub[row, m_idx(n_steps - 1, i)] = 1.0
        b_ub[row] = float(terminal[i] + terminal_tank_tol_kg)
        row += 1
        a_ub[row, m_idx(n_steps - 1, i)] = -1.0
        b_ub[row] = float(-terminal[i] + terminal_tank_tol_kg)
        row += 1

    B = np.asarray(fit.B, dtype=float)
    bound_mode = str(posture_bound_mode).strip().lower()
    if bound_mode == "absolute":
        posture_bounds = np.full_like(posture_closed, float(posture_limit_deg), dtype=float)
    elif bound_mode == "closed_envelope":
        posture_bounds = np.maximum(
            float(posture_limit_deg),
            np.abs(posture_closed) + float(posture_margin_deg),
        )
    elif bound_mode == "deadband_envelope":
        posture_bounds = np.maximum(
            float(posture_limit_deg),
            np.abs(posture_primary) + float(posture_margin_deg),
        )
    else:
        raise ValueError(f"unsupported posture_bound_mode={posture_bound_mode}")
    for k in range(n_steps):
        closed_m_k = m_closed[k, :]
        for axis_idx in range(2):
            coeff = B[axis_idx, :]
            closed_offset = float(posture_closed[k, axis_idx] - np.dot(coeff, closed_m_k))
            for i in range(3):
                a_ub[row, m_idx(k, i)] = float(coeff[i])
            b_ub[row] = float(posture_bounds[k, axis_idx] - closed_offset)
            row += 1
            for i in range(3):
                a_ub[row, m_idx(k, i)] = float(-coeff[i])
            b_ub[row] = float(posture_bounds[k, axis_idx] + closed_offset)
            row += 1

    result = linprog(
        c=c,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    status = f"{int(result.status)}:{result.message}"
    if not result.success:
        return LpResult(
            pump_m3=math.nan,
            effective_pump_m3=math.nan,
            tail_to_closed_m3=math.nan,
            status=status,
            terminal_error_kg=math.nan,
            terminal_reference_error_kg=math.nan,
            posture_violation_max_deg=math.nan,
            oracle_est_time_gt3_s=math.nan,
            oracle_est_time_gt4_s=math.nan,
            oracle_est_time_gt5_s=math.nan,
            max_abs_pitch_deg=math.nan,
            max_abs_roll_deg=math.nan,
        )

    x = np.asarray(result.x, dtype=float)
    m = x[:n_m].reshape(n_steps, 3)
    moved_m3 = float(np.sum(x[n_m:]) / float(rho_kg_m3))
    posture_est = posture_closed + (m - m_closed) @ B.T
    axis_abs = np.abs(posture_est)
    violation = float(max(0.0, np.max(axis_abs - posture_bounds)))
    tail_to_closed_m3 = float(np.sum(np.abs(m[-1, :] - m_closed[-1, :])) / float(rho_kg_m3))
    terminal_error = float(np.max(np.abs(m[-1, :] - m_closed[-1, :])))
    terminal_reference_error = float(np.max(np.abs(m[-1, :] - terminal)))
    return LpResult(
        pump_m3=moved_m3,
        effective_pump_m3=moved_m3 + tail_to_closed_m3,
        tail_to_closed_m3=tail_to_closed_m3,
        status=status,
        terminal_error_kg=terminal_error,
        terminal_reference_error_kg=terminal_reference_error,
        posture_violation_max_deg=violation,
        oracle_est_time_gt3_s=_weighted_time_over_limit(posture_est, t_s, 3.0),
        oracle_est_time_gt4_s=_weighted_time_over_limit(posture_est, t_s, 4.0),
        oracle_est_time_gt5_s=_weighted_time_over_limit(posture_est, t_s, 5.0),
        max_abs_pitch_deg=float(np.max(axis_abs[:, 0])),
        max_abs_roll_deg=float(np.max(axis_abs[:, 1])),
        m_path=m,
        t_path=t_s,
        m_closed_path=m_closed,
        posture_est_path=posture_est,
        posture_closed_path=posture_closed,
    )


def _pct_saving(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or reference <= 0.0 or not np.isfinite(candidate):
        return math.nan
    return float((reference - candidate) / reference * 100.0)


def _case_summary_row(
    pair: CasePair,
    modes: set[str],
    family_fit: FitResult | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    closed = _read_timeseries(pair.closed_path)
    primary = _read_timeseries(pair.primary_path)

    closed_pump = _summary_pump(pair.summary_row, "closed_pump_work_m3", closed)
    primary_pump = _summary_pump(pair.summary_row, "primary_pump_work_m3", primary)

    closed_start = _first_vec(closed, TANK_COLS)
    closed_end = _last_vec(closed, TANK_COLS)
    primary_start = _first_vec(primary, TANK_COLS)
    primary_end = _last_vec(primary, TANK_COLS)
    net_lower_bound_m3 = float(np.sum(np.abs(primary_end - primary_start)) / float(args.rho_kg_m3))
    closed_endpoint_net_lower_bound_m3 = float(
        np.sum(np.abs(closed_end - closed_start)) / float(args.rho_kg_m3)
    )
    deadband_gap_to_net_bound_m3 = float(primary_pump - net_lower_bound_m3)
    deadband_gap_pct_of_deadband = (
        float(deadband_gap_to_net_bound_m3 / primary_pump * 100.0) if primary_pump > 0.0 else math.nan
    )

    closed_max_pitch, closed_max_roll = _max_abs_axis(closed)
    primary_max_pitch, primary_max_roll = _max_abs_axis(primary)

    fit = FitResult(
        B=np.full((2, 3), np.nan),
        rank=0,
        condition=math.inf,
        rmse_pitch=math.nan,
        rmse_roll=math.nan,
        n_samples=0,
        source="not_requested",
        usable=False,
        reason="not_requested",
    )
    lp = LpResult(
        pump_m3=math.nan,
        effective_pump_m3=math.nan,
        tail_to_closed_m3=math.nan,
        status="not_requested",
        terminal_error_kg=math.nan,
        terminal_reference_error_kg=math.nan,
        posture_violation_max_deg=math.nan,
        oracle_est_time_gt3_s=math.nan,
        oracle_est_time_gt4_s=math.nan,
        oracle_est_time_gt5_s=math.nan,
        max_abs_pitch_deg=math.nan,
        max_abs_roll_deg=math.nan,
    )

    if "v1_lp" in modes:
        closed_rs, primary_rs = _paired_resampled(closed, primary, resample_s=float(args.resample_s))
        per_case_fit = _fit_response_model(
            closed_rs=closed_rs,
            primary_rs=primary_rs,
            source="per_case",
            max_condition=float(args.max_fit_condition),
            min_rank=int(args.min_fit_rank),
            min_samples=int(args.min_fit_samples),
        )
        if per_case_fit.usable:
            fit = per_case_fit
        elif family_fit is not None and family_fit.usable:
            fit = FitResult(
                B=family_fit.B,
                rank=family_fit.rank,
                condition=family_fit.condition,
                rmse_pitch=family_fit.rmse_pitch,
                rmse_roll=family_fit.rmse_roll,
                n_samples=family_fit.n_samples,
                source="family_fallback",
                usable=True,
                reason=f"per_case_{per_case_fit.reason}",
            )
        else:
            fit = per_case_fit
        lp = _solve_lp_oracle(
            closed_rs=closed_rs,
            primary_rs=primary_rs,
            fit=fit,
            posture_limit_deg=float(args.posture_limit_deg),
            posture_bound_mode=str(args.posture_bound_mode),
            posture_margin_deg=float(args.posture_margin_deg),
            terminal_tank_tol_kg=float(args.terminal_tank_tol_kg),
            terminal_reference=str(args.terminal_reference),
            tank_capacity_kg=float(args.tank_capacity_kg),
            pump_rate_limit_m3_min=float(args.pump_rate_limit_m3_min),
            rho_kg_m3=float(args.rho_kg_m3),
            trust_region_mode=str(getattr(args, "trust_region_mode", "off")),
            trust_region_margin_kg=float(getattr(args, "trust_region_margin_kg", 0.0)),
        )

    oracle_pump = lp.pump_m3
    oracle_effective_pump = lp.effective_pump_m3
    return {
        "case_id": pair.case_id,
        "timestamp": pair.timestamp,
        "label": pair.label,
        "closed_pump_m3": closed_pump,
        "deadband_pump_m3": primary_pump,
        "net_lower_bound_m3": net_lower_bound_m3,
        "lp_oracle_pump_m3": oracle_pump,
        "lp_oracle_effective_pump_m3": oracle_effective_pump,
        "lp_oracle_tail_to_closed_m3": lp.tail_to_closed_m3,
        "deadband_saving_pct_vs_closed": _pct_saving(closed_pump, primary_pump),
        "oracle_saving_pct_vs_closed": _pct_saving(closed_pump, oracle_pump),
        "oracle_effective_saving_pct_vs_closed": _pct_saving(closed_pump, oracle_effective_pump),
        "oracle_incremental_saving_pct_vs_deadband": _pct_saving(primary_pump, oracle_pump),
        "oracle_effective_incremental_saving_pct_vs_deadband": _pct_saving(primary_pump, oracle_effective_pump),
        "oracle_incremental_saving_pp_vs_closed": (
            _pct_saving(closed_pump, oracle_pump) - _pct_saving(closed_pump, primary_pump)
            if np.isfinite(_pct_saving(closed_pump, oracle_pump))
            and np.isfinite(_pct_saving(closed_pump, primary_pump))
            else math.nan
        ),
        "deadband_gap_to_net_bound_m3": deadband_gap_to_net_bound_m3,
        "deadband_gap_pct_of_deadband": deadband_gap_pct_of_deadband,
        "closed_endpoint_net_lower_bound_m3": closed_endpoint_net_lower_bound_m3,
        "deadband_gap_to_closed_endpoint_net_bound_m3": float(primary_pump - closed_endpoint_net_lower_bound_m3),
        "terminal_tank_error_kg": lp.terminal_error_kg,
        "terminal_reference_error_kg": lp.terminal_reference_error_kg,
        "deadband_terminal_tank_error_kg": float(np.max(np.abs(primary_end - closed_end))),
        "posture_limit_deg": float(args.posture_limit_deg),
        "posture_bound_mode": str(args.posture_bound_mode),
        "posture_margin_deg": float(args.posture_margin_deg),
        "terminal_reference": str(args.terminal_reference),
        "trust_region_mode": str(getattr(args, "trust_region_mode", "off")),
        "trust_region_margin_kg": float(getattr(args, "trust_region_margin_kg", 0.0)),
        "lp_status": lp.status,
        "lp_posture_violation_max_deg": lp.posture_violation_max_deg,
        "fit_rank": fit.rank,
        "fit_condition": fit.condition,
        "fit_rmse_pitch": fit.rmse_pitch,
        "fit_rmse_roll": fit.rmse_roll,
        "fit_samples": fit.n_samples,
        "fit_source": fit.source,
        "fit_reason": fit.reason,
        "closed_time_gt3_s": _time_over_limit_s(closed, 3.0),
        "deadband_time_gt3_s": _time_over_limit_s(primary, 3.0),
        "oracle_est_time_gt3_s": lp.oracle_est_time_gt3_s,
        "closed_time_gt4_s": _time_over_limit_s(closed, 4.0),
        "deadband_time_gt4_s": _time_over_limit_s(primary, 4.0),
        "oracle_est_time_gt4_s": lp.oracle_est_time_gt4_s,
        "closed_time_gt5_s": _time_over_limit_s(closed, 5.0),
        "deadband_time_gt5_s": _time_over_limit_s(primary, 5.0),
        "oracle_est_time_gt5_s": lp.oracle_est_time_gt5_s,
        "closed_max_abs_pitch_deg": closed_max_pitch,
        "closed_max_abs_roll_deg": closed_max_roll,
        "deadband_max_abs_pitch_deg": primary_max_pitch,
        "deadband_max_abs_roll_deg": primary_max_roll,
        "oracle_est_max_abs_pitch_deg": lp.max_abs_pitch_deg,
        "oracle_est_max_abs_roll_deg": lp.max_abs_roll_deg,
        "closed_timeseries": str(pair.closed_path),
        "primary_timeseries": str(pair.primary_path),
    }


def _aggregate(rows: pd.DataFrame, run_group: str) -> pd.DataFrame:
    n_cases = int(len(rows))
    closed_total = float(rows["closed_pump_m3"].sum()) if n_cases else 0.0
    deadband_total = float(rows["deadband_pump_m3"].sum()) if n_cases else 0.0
    net_total = float(rows["net_lower_bound_m3"].sum()) if n_cases else 0.0
    closed_endpoint_net_total = (
        float(rows["closed_endpoint_net_lower_bound_m3"].sum())
        if n_cases and "closed_endpoint_net_lower_bound_m3" in rows.columns
        else math.nan
    )

    success = rows["lp_oracle_pump_m3"].notna() & rows["lp_status"].astype(str).str.startswith("0:")
    success_rows = rows[success].copy()
    failed_lp_cases = int(n_cases - len(success_rows))
    success_closed_total = float(success_rows["closed_pump_m3"].sum()) if len(success_rows) else math.nan
    success_deadband_total = float(success_rows["deadband_pump_m3"].sum()) if len(success_rows) else math.nan
    lp_total = float(success_rows["lp_oracle_pump_m3"].sum()) if len(success_rows) else math.nan
    lp_effective_total = (
        float(success_rows["lp_oracle_effective_pump_m3"].sum())
        if len(success_rows) and "lp_oracle_effective_pump_m3" in success_rows.columns
        else math.nan
    )
    lp_tail_total = (
        float(success_rows["lp_oracle_tail_to_closed_m3"].sum())
        if len(success_rows) and "lp_oracle_tail_to_closed_m3" in success_rows.columns
        else math.nan
    )
    extra = success_rows["deadband_pump_m3"] - success_rows["lp_oracle_pump_m3"] if len(success_rows) else pd.Series(dtype=float)
    effective_extra = (
        success_rows["deadband_pump_m3"] - success_rows["lp_oracle_effective_pump_m3"]
        if len(success_rows) and "lp_oracle_effective_pump_m3" in success_rows.columns
        else pd.Series(dtype=float)
    )
    positive_extra = extra[extra > 0.0]
    positive_effective_extra = effective_extra[effective_extra > 0.0]
    top_case_share = (
        float(positive_extra.max() / positive_extra.sum()) if len(positive_extra) and positive_extra.sum() > 0.0 else math.nan
    )
    top_case_effective_share = (
        float(positive_effective_extra.max() / positive_effective_extra.sum())
        if len(positive_effective_extra) and positive_effective_extra.sum() > 0.0
        else math.nan
    )

    row = {
        "run_group": run_group,
        "n_cases": n_cases,
        "lp_success_cases": int(len(success_rows)),
        "closed_pump_m3": closed_total,
        "deadband_pump_m3": deadband_total,
        "net_lower_bound_m3": net_total,
        "closed_endpoint_net_lower_bound_m3": closed_endpoint_net_total,
        "lp_oracle_pump_m3": lp_total,
        "lp_oracle_effective_pump_m3": lp_effective_total,
        "lp_oracle_tail_to_closed_m3": lp_tail_total,
        "deadband_saving_pct_vs_closed": _pct_saving(closed_total, deadband_total),
        "oracle_saving_pct_vs_closed": _pct_saving(success_closed_total, lp_total),
        "oracle_effective_saving_pct_vs_closed": _pct_saving(success_closed_total, lp_effective_total),
        "oracle_incremental_saving_pct_vs_deadband": _pct_saving(success_deadband_total, lp_total),
        "oracle_effective_incremental_saving_pct_vs_deadband": _pct_saving(
            success_deadband_total,
            lp_effective_total,
        ),
        "oracle_incremental_saving_pp_vs_closed": (
            _pct_saving(success_closed_total, lp_total) - _pct_saving(success_closed_total, success_deadband_total)
            if np.isfinite(_pct_saving(success_closed_total, lp_total))
            and np.isfinite(_pct_saving(success_closed_total, success_deadband_total))
            else math.nan
        ),
        "deadband_gap_to_net_bound_m3": float(deadband_total - net_total),
        "deadband_gap_pct_of_deadband": float((deadband_total - net_total) / deadband_total * 100.0)
        if deadband_total > 0.0
        else math.nan,
        "deadband_gap_to_closed_endpoint_net_bound_m3": float(deadband_total - closed_endpoint_net_total)
        if np.isfinite(closed_endpoint_net_total)
        else math.nan,
        "deadband_gap_to_closed_endpoint_net_bound_pct": float(
            (deadband_total - closed_endpoint_net_total) / deadband_total * 100.0
        )
        if deadband_total > 0.0 and np.isfinite(closed_endpoint_net_total)
        else math.nan,
        "failed_lp_cases": failed_lp_cases,
        "mean_terminal_error_kg": float(success_rows["terminal_tank_error_kg"].mean()) if len(success_rows) else math.nan,
        "max_terminal_error_kg": float(success_rows["terminal_tank_error_kg"].max()) if len(success_rows) else math.nan,
        "mean_terminal_reference_error_kg": float(success_rows["terminal_reference_error_kg"].mean())
        if len(success_rows) and "terminal_reference_error_kg" in success_rows.columns
        else math.nan,
        "max_terminal_reference_error_kg": float(success_rows["terminal_reference_error_kg"].max())
        if len(success_rows) and "terminal_reference_error_kg" in success_rows.columns
        else math.nan,
        "max_deadband_terminal_tank_error_kg": float(rows["deadband_terminal_tank_error_kg"].max()) if n_cases else math.nan,
        "max_posture_violation_deg": float(success_rows["lp_posture_violation_max_deg"].max()) if len(success_rows) else math.nan,
        "max_closed_abs_axis_deg": float(
            np.nanmax(
                rows[["closed_max_abs_pitch_deg", "closed_max_abs_roll_deg"]].to_numpy(dtype=float)
            )
        )
        if n_cases
        else math.nan,
        "max_deadband_abs_axis_deg": float(
            np.nanmax(
                rows[["deadband_max_abs_pitch_deg", "deadband_max_abs_roll_deg"]].to_numpy(dtype=float)
            )
        )
        if n_cases
        else math.nan,
        "total_closed_time_gt3_s": float(rows["closed_time_gt3_s"].sum()) if n_cases else math.nan,
        "total_deadband_time_gt3_s": float(rows["deadband_time_gt3_s"].sum()) if n_cases else math.nan,
        "top_case_incremental_saving_share": top_case_share,
        "top_case_effective_incremental_saving_share": top_case_effective_share,
        "max_fit_condition": float(rows["fit_condition"].replace([np.inf, -np.inf], np.nan).max()) if n_cases else math.nan,
        "median_fit_rmse_pitch": float(rows["fit_rmse_pitch"].median()) if n_cases else math.nan,
        "median_fit_rmse_roll": float(rows["fit_rmse_roll"].median()) if n_cases else math.nan,
    }
    return pd.DataFrame([row])


def _fmt(value: object, suffix: str = "", digits: int = 3) -> str:
    value_f = _as_float(value, default=math.nan)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.{digits}f}{suffix}"


def _write_decision(out_path: Path, rows: pd.DataFrame, agg: pd.DataFrame, args: argparse.Namespace) -> None:
    a = agg.iloc[0].to_dict()
    lp_success = int(a.get("lp_success_cases", 0))
    failed = int(a.get("failed_lp_cases", 0))
    extra_pp = _as_float(a.get("oracle_incremental_saving_pp_vs_closed"), default=math.nan)
    extra_vs_deadband = _as_float(a.get("oracle_incremental_saving_pct_vs_deadband"), default=math.nan)
    effective_extra_vs_deadband = _as_float(
        a.get("oracle_effective_incremental_saving_pct_vs_deadband"),
        default=math.nan,
    )
    terminal_max = _as_float(a.get("max_terminal_error_kg"), default=math.nan)
    terminal_reference_max = _as_float(a.get("max_terminal_reference_error_kg"), default=math.nan)
    deadband_terminal_max = _as_float(a.get("max_deadband_terminal_tank_error_kg"), default=math.nan)
    violation_max = _as_float(a.get("max_posture_violation_deg"), default=math.nan)
    top_share = _as_float(a.get("top_case_incremental_saving_share"), default=math.nan)
    top_effective_share = _as_float(a.get("top_case_effective_incremental_saving_share"), default=math.nan)
    max_closed_axis = _as_float(a.get("max_closed_abs_axis_deg"), default=math.nan)
    max_deadband_axis = _as_float(a.get("max_deadband_abs_axis_deg"), default=math.nan)
    decision_extra_vs_deadband = (
        effective_extra_vs_deadband if np.isfinite(effective_extra_vs_deadband) else extra_vs_deadband
    )
    decision_top_share = top_effective_share if np.isfinite(top_effective_share) else top_share

    material = (
        lp_success > 0
        and failed == 0
        and np.isfinite(extra_pp)
        and extra_pp >= 5.0
        and np.isfinite(decision_extra_vs_deadband)
        and decision_extra_vs_deadband >= 5.0
        and np.isfinite(terminal_reference_max)
        and terminal_reference_max <= float(args.terminal_tank_tol_kg) + 1e-6
        and np.isfinite(violation_max)
        and violation_max <= 1e-5
        and (not np.isfinite(decision_top_share) or decision_top_share <= 0.50)
    )
    verdict = "GO for runtime MPC prototype" if material else "NO-GO for runtime MPC at this stage"

    top_cases = rows.copy()
    if "lp_oracle_pump_m3" in top_cases.columns:
        top_cases["incremental_m3"] = top_cases["deadband_pump_m3"] - top_cases["lp_oracle_pump_m3"]
        top_cases = top_cases.sort_values("incremental_m3", ascending=False).head(5)

    lines = [
        "# Offline Minimum-Throughput Oracle Decision",
        "",
        f"Run group: `{a.get('run_group', '')}`",
        f"Cases: `{int(a.get('n_cases', 0))}`; LP successes: `{lp_success}`; LP failures/skips: `{failed}`",
        f"Posture limit: `{_fmt(args.posture_limit_deg, ' deg')}`; terminal tank tolerance: `{_fmt(args.terminal_tank_tol_kg, ' kg', 1)}`",
        f"Posture bound mode: `{args.posture_bound_mode}`; terminal reference: `{args.terminal_reference}`; posture margin: `{_fmt(args.posture_margin_deg, ' deg')}`",
        f"Trust region mode: `{args.trust_region_mode}`; trust margin: `{_fmt(args.trust_region_margin_kg, ' kg', 1)}`",
        "",
        "## 1. Deadband distance to theoretical minimum",
        "",
        (
            f"- Closed-only pump: `{_fmt(a.get('closed_pump_m3'), ' m3')}`; "
            f"deadband pump: `{_fmt(a.get('deadband_pump_m3'), ' m3')}`; "
            f"V0 deadband-endpoint net lower bound: `{_fmt(a.get('net_lower_bound_m3'), ' m3')}`."
        ),
        (
            f"- Deadband-to-V0 gap: `{_fmt(a.get('deadband_gap_to_net_bound_m3'), ' m3')}` "
            f"(`{_fmt(a.get('deadband_gap_pct_of_deadband'), '%')}` of deadband pump)."
        ),
        (
            f"- If forced to close to the closed-only endpoint, net lower bound is "
            f"`{_fmt(a.get('closed_endpoint_net_lower_bound_m3'), ' m3')}` and the gap is "
            f"`{_fmt(a.get('deadband_gap_to_closed_endpoint_net_bound_m3'), ' m3')}` "
            f"(`{_fmt(a.get('deadband_gap_to_closed_endpoint_net_bound_pct'), '%')}`)."
        ),
        f"- Max deadband terminal tank mismatch vs closed-only endpoint: `{_fmt(deadband_terminal_max, ' kg', 1)}`.",
        "",
        "## 2. V1 LP value beyond deadband",
        "",
        (
            f"- LP oracle pump on successful cases: `{_fmt(a.get('lp_oracle_pump_m3'), ' m3')}`; "
            f"extra saving vs deadband: `{_fmt(extra_vs_deadband, '%')}` "
            f"or `{_fmt(extra_pp, ' pp')}` vs closed-only."
        ),
        (
            f"- Effective LP pump after tail-to-closed accounting: "
            f"`{_fmt(a.get('lp_oracle_effective_pump_m3'), ' m3')}`; "
            f"tail-to-closed pump: `{_fmt(a.get('lp_oracle_tail_to_closed_m3'), ' m3')}`; "
            f"effective extra saving vs deadband: `{_fmt(effective_extra_vs_deadband, '%')}`."
        ),
        f"- Top-case share of positive incremental saving: `{_fmt(top_share * 100.0 if np.isfinite(top_share) else math.nan, '%')}`.",
        f"- Top-case share after effective tail accounting: `{_fmt(top_effective_share * 100.0 if np.isfinite(top_effective_share) else math.nan, '%')}`.",
        "",
        "## 3. Posture safety and terminal closure",
        "",
        (
            f"- Max estimated LP posture violation above the limit: `{_fmt(violation_max, ' deg')}`; "
            f"max LP terminal-to-closed error: `{_fmt(terminal_max, ' kg', 1)}`; "
            f"max terminal-reference error: `{_fmt(terminal_reference_max, ' kg', 1)}`."
        ),
        (
            f"- Existing traces under the same absolute band: max closed axis `{_fmt(max_closed_axis, ' deg')}`, "
            f"max deadband axis `{_fmt(max_deadband_axis, ' deg')}`, "
            f"closed time>3 `{_fmt(a.get('total_closed_time_gt3_s'), ' s', 1)}`, "
            f"deadband time>3 `{_fmt(a.get('total_deadband_time_gt3_s'), ' s', 1)}`."
        ),
        f"- LP status: `{lp_success}` success, `{failed}` failed/skipped.",
        "",
        "## 4. Runtime MPC decision",
        "",
        f"- Verdict: **{verdict}**.",
    ]

    if material:
        lines.append(
            "- Rationale: the offline LP ceiling clears the 5-10% incremental-saving screen, "
            "keeps the linearized posture band, closes terminal tanks, and is not dominated by one case."
        )
    else:
        lines.append(
            "- Rationale: the available LP ceiling does not clear all Go gates. Treat this as a ceiling test, "
            "not a deployable controller result."
        )

    lines.extend(
        [
            "",
            "## 5. Paper framing if MPC remains no-go",
            "",
            (
                "- Suggested wording: reactive deadband already approaches the minimum-throughput ceiling "
                "under the current plant and constraints; forecast-driven MPC therefore has limited "
                "additional pump-volume space unless a stronger plant model or objective changes the ceiling."
            ),
            "- Keep the current positive claim on DC-preserving deadband economy, and frame the oracle as a negative control before online MPC complexity.",
            "",
            "## Top incremental LP cases",
            "",
            "| case | deadband m3 | LP m3 | incremental m3 | incremental vs deadband | lp_status | fit |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for r in top_cases.to_dict("records"):
        inc = _as_float(r.get("incremental_m3"), default=math.nan)
        lines.append(
            "| {case} | {deadband} | {lp} | {inc} | {pct} | {status} | {fit} |".format(
                case=r.get("case_id", ""),
                deadband=_fmt(r.get("deadband_pump_m3"), digits=2),
                lp=_fmt(r.get("lp_oracle_pump_m3"), digits=2),
                inc=_fmt(inc, digits=2),
                pct=_fmt(r.get("oracle_incremental_saving_pct_vs_deadband"), "%", digits=2),
                status=str(r.get("lp_status", ""))[:60],
                fit=f"{r.get('fit_source', '')}/rank{r.get('fit_rank', '')}",
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--posture-limit-deg", type=float, default=3.0)
    parser.add_argument("--terminal-tank-tol-kg", type=float, default=5000.0)
    parser.add_argument("--mode", type=str, default="v0_netpath")
    parser.add_argument("--resample-s", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--primary-label-substring", type=str, default="")
    parser.add_argument(
        "--posture-bound-mode",
        choices=("absolute", "closed_envelope", "deadband_envelope"),
        default="absolute",
    )
    parser.add_argument("--posture-margin-deg", type=float, default=0.0)
    parser.add_argument(
        "--terminal-reference",
        choices=("closed", "deadband"),
        default="closed",
    )
    parser.add_argument("--rho-kg-m3", type=float, default=RHO_KG_M3)
    parser.add_argument("--tank-capacity-kg", type=float, default=DEFAULT_TANK_CAPACITY_KG)
    parser.add_argument("--pump-rate-limit-m3-min", type=float, default=15.0)
    parser.add_argument(
        "--trust-region-mode",
        choices=("off", "observed_delta"),
        default="off",
    )
    parser.add_argument("--trust-region-margin-kg", type=float, default=0.0)
    parser.add_argument("--max-fit-condition", type=float, default=1e8)
    parser.add_argument("--min-fit-rank", type=int, default=2)
    parser.add_argument("--min-fit-samples", type=int, default=20)
    parser.add_argument("--precompute-family-fit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = {m.strip() for m in str(args.mode).split(",") if m.strip()}
    allowed_modes = {"v0_netpath", "v1_lp"}
    unknown = modes - allowed_modes
    if unknown:
        raise ValueError(f"unsupported mode(s): {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_case_pairs(
        run_dir=args.run_dir,
        max_cases=int(args.max_cases),
        primary_label_substring=str(args.primary_label_substring),
    )
    if not pairs:
        raise RuntimeError("no case pairs found")

    family_fit: FitResult | None = None
    if "v1_lp" in modes and bool(args.precompute_family_fit):
        family_fit = _fit_family_response_model(
            pairs=pairs,
            resample_s=float(args.resample_s),
            max_condition=float(args.max_fit_condition),
            min_rank=int(args.min_fit_rank),
            min_samples=int(args.min_fit_samples),
        )

    rows = []
    for idx, pair in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {pair.case_id}", flush=True)
        rows.append(_case_summary_row(pair, modes=modes, family_fit=family_fit, args=args))

    case_df = pd.DataFrame(rows)
    agg_df = _aggregate(case_df, run_group=args.run_dir.name)

    case_df.to_csv(args.out_dir / "oracle_case_summary.csv", index=False)
    agg_df.to_csv(args.out_dir / "oracle_aggregate_summary.csv", index=False)
    _write_decision(args.out_dir / "decision.md", case_df, agg_df, args)

    agg = agg_df.iloc[0].to_dict()
    print(
        "done run_group={run_group} cases={n_cases} lp_success={lp_success_cases} "
        "deadband_saving={deadband_saving_pct_vs_closed:.3f}% "
        "lp_extra_vs_deadband={oracle_incremental_saving_pct_vs_deadband:.3f}% "
        "lp_extra_pp={oracle_incremental_saving_pp_vs_closed:.3f}".format(**agg),
        flush=True,
    )


if __name__ == "__main__":
    main()
