# Phase 4 Historical Skill Tracking Report

## Logic & Leakage Protection
- **Window Definition**: Trailing 30 days of valid_time.
- **Verification Lag Definition**: A forecast is verified only when the real world reaches its `valid_time`. Therefore, historical features for a forecast initialized at `T` strictly use observations where `valid_time <= T`.
- **Minimum Sample Threshold**: Grid-cell historical metrics over a 30-day window possess inadequate sample sizes (max 60). Therefore, domain-level (national) fallback aggregation is utilized to generate robust historical skill metrics, aggregating ~14,000 grid cells per timestep.
- **Leakage Test Performed**: Checked that `valid_time_historical <= init_time` for all rows. Passed.

## Coverage
- **Total skill periods computed**: 3660
- **Source × Lead Coverage**: HRES and PANGU computed per lead time [24, 48, 72, 120, 240].
- **Missing/Insufficient-history cases**: 594594 (Early January forecasts do not have historical windows).

## Historical Feature Example
An example of a historical feature row:
```json
{'valid_time': Timestamp('2021-01-10 12:00:00'), 'lead_time_hours': 240, 'skill_sample_count': 849420.0, 'rolling_hres_rmse': 2.9408016426091868, 'rolling_pangu_rmse': 2.44277693781596, 'rolling_hres_mae': 1.9870274066925049, 'rolling_pangu_mae': 1.6870851238568625, 'rolling_hres_bias': -0.05674372111874012, 'rolling_pangu_bias': -0.2565443756757304, 'pangu_relative_skill': 0.5408183308567793}
```

*Skill features have been extracted to `data/skill/historical_skill_2020.parquet` for Phase 5.*
