# Reference engineering data

## `thrust_curve_15mw_hub_height.xlsx`

Source copy of the 15 MW hub-height wind-speed thrust table provided in:

`Thrust force at hub height 风机叶片及轮毂相关数据.xlsx`

Current use:

- `archive/legacy_fowt_control/wind_env.py` loads this file automatically when available.
- `wind_speed_to_thrust_n()` interpolates thrust force from the table instead of using a fixed-Ct formula.
- If the file is missing, the legacy fixed-Ct formula remains as fallback.

Units:

- wind speed: m/s
- thrust force: kN in Excel, converted to N in code
