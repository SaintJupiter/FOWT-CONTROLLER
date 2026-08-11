import numpy as np

TRIM_MAP_V1 = "v1"
TRIM_MAP_V2 = "v2"

DEFAULT_TRIM_REGIME_CFG = {
    "ws_rated_mps": 11.5,
    "ws_cutout_mps": 25.0,
    "post_rated_mode": "plateau",
    "post_rated_decay_ratio": 0.8,
    "protect_trim_scale": 0.0,
}


def normalize_trim_map_version(trim_map_version):
    if trim_map_version is None:
        raise ValueError("trim_map_version must be explicitly specified (v1 or v2).")
    v = str(trim_map_version).strip().lower()
    if v in ("v1", "trim_map_v1"):
        return TRIM_MAP_V1
    if v in ("v2", "trim_map_v2"):
        return TRIM_MAP_V2
    raise ValueError(f"unsupported trim_map_version: {trim_map_version}")


def build_regime_cfg(
    regime_cfg=None,
    ws_rated_mps=None,
    ws_cutout_mps=None,
    post_rated_mode=None,
    post_rated_decay_ratio=None,
    protect_trim_scale=None,
):
    cfg = dict(DEFAULT_TRIM_REGIME_CFG)
    if isinstance(regime_cfg, dict):
        cfg.update(regime_cfg)

    if ws_rated_mps is not None:
        cfg["ws_rated_mps"] = ws_rated_mps
    if ws_cutout_mps is not None:
        cfg["ws_cutout_mps"] = ws_cutout_mps
    if post_rated_mode is not None:
        cfg["post_rated_mode"] = post_rated_mode
    if post_rated_decay_ratio is not None:
        cfg["post_rated_decay_ratio"] = post_rated_decay_ratio
    if protect_trim_scale is not None:
        cfg["protect_trim_scale"] = protect_trim_scale

    cfg["ws_rated_mps"] = float(cfg["ws_rated_mps"])
    cfg["ws_cutout_mps"] = float(cfg["ws_cutout_mps"])
    cfg["post_rated_mode"] = str(cfg.get("post_rated_mode", "plateau")).strip().lower()
    cfg["post_rated_decay_ratio"] = float(cfg.get("post_rated_decay_ratio", 0.8))
    cfg["protect_trim_scale"] = float(cfg.get("protect_trim_scale", 0.0))

    if cfg["ws_cutout_mps"] <= cfg["ws_rated_mps"]:
        raise ValueError(
            "invalid regime cfg: ws_cutout_mps must be greater than ws_rated_mps"
        )
    if cfg["post_rated_mode"] not in ("plateau", "decay"):
        raise ValueError(
            f"invalid post_rated_mode={cfg['post_rated_mode']}; expected 'plateau' or 'decay'"
        )

    cfg["post_rated_decay_ratio"] = float(np.clip(cfg["post_rated_decay_ratio"], 0.0, 1.0))
    cfg["protect_trim_scale"] = float(np.clip(cfg["protect_trim_scale"], 0.0, 1.0))
    return cfg


def get_operating_regime(ws_mps, regime_cfg):
    cfg = build_regime_cfg(regime_cfg=regime_cfg)
    ws = float(ws_mps)
    if ws >= cfg["ws_cutout_mps"]:
        return "protect"
    if ws >= cfg["ws_rated_mps"]:
        return "post_rated"
    return "pre_rated"


def compute_trim_magnitude(
    ws_mps,
    k_trim_deg_per_mps,
    max_trim_deg,
    trim_map_version,
    regime_cfg=None,
):
    ws = max(float(ws_mps), 0.0)
    k = float(k_trim_deg_per_mps)
    max_trim = float(max_trim_deg)
    map_ver = normalize_trim_map_version(trim_map_version)

    if map_ver == TRIM_MAP_V1:
        trim_mag = float(np.clip(k * ws, 0.0, max_trim))
        return trim_mag, "all"

    cfg = build_regime_cfg(regime_cfg=regime_cfg)
    regime = get_operating_regime(ws, cfg)
    rated_mag = float(np.clip(k * cfg["ws_rated_mps"], 0.0, max_trim))

    if regime == "pre_rated":
        trim_mag = float(np.clip(k * ws, 0.0, max_trim))
    elif regime == "post_rated":
        if cfg["post_rated_mode"] == "plateau":
            trim_mag = rated_mag
        else:
            frac = (ws - cfg["ws_rated_mps"]) / max(
                cfg["ws_cutout_mps"] - cfg["ws_rated_mps"],
                1e-9,
            )
            frac = float(np.clip(frac, 0.0, 1.0))
            end_mag = rated_mag * cfg["post_rated_decay_ratio"]
            trim_mag = float((1.0 - frac) * rated_mag + frac * end_mag)
    else:
        trim_mag = float(rated_mag * cfg["protect_trim_scale"])

    trim_mag = float(np.clip(trim_mag, 0.0, max_trim))
    return trim_mag, regime


def compute_trim_setpoints(
    ws_mps,
    wd_deg,
    k_trim_deg_per_mps,
    max_trim_deg,
    trim_map_version,
    regime_cfg=None,
    return_meta=False,
):
    trim_mag, regime_label = compute_trim_magnitude(
        ws_mps=ws_mps,
        k_trim_deg_per_mps=k_trim_deg_per_mps,
        max_trim_deg=max_trim_deg,
        trim_map_version=trim_map_version,
        regime_cfg=regime_cfg,
    )
    wd_rad = np.radians(float(wd_deg))
    pitch_sp = -trim_mag * np.cos(wd_rad)
    roll_sp = trim_mag * np.sin(wd_rad)

    if return_meta:
        return float(pitch_sp), float(roll_sp), {
            "trim_mag_deg": float(trim_mag),
            "regime_label": str(regime_label),
            "trim_map_version": normalize_trim_map_version(trim_map_version),
        }
    return float(pitch_sp), float(roll_sp)
