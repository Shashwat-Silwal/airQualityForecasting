"""One-off script: slim the LAWA dataset down to what the app needs and
compute the typical ratios used to fill in features the public isn't asked for.

Feature engineering mirrors the notebook (airQualityForecasting_12.10.ipynb).
Run:  python build_data.py path/to/forecasting_dataset.xls
"""
import json
import sys

import pandas as pd

src = sys.argv[1]
df = pd.read_csv(src, parse_dates=["date"]).drop(columns=["Unnamed: 0"], errors="ignore")
df = df.sort_values(["site_name", "date"]).reset_index(drop=True)
g = df.groupby("site_name")

# Same definitions as the notebook
df["pm10_mean7"] = g["pm10_ugm3"].transform(lambda s: s.rolling(7, min_periods=4).mean())
df["pm10_max7"] = g["pm10_ugm3"].transform(lambda s: s.rolling(7, min_periods=4).max())
df["pressure_change_1d"] = df["pressure_hpa"] - g["pressure_hpa"].shift(1)
df["cold_day"] = (df["temp_max_c"] > 16).astype(int)  # kept exactly as in the notebook
df["pm25_fraction"] = df["pm25_ugm3"] / df["pm10_ugm3"]
df["region"] = df["region"].str.strip()

keep = [
    "site_name", "town", "region", "date", "season_num",
    "pm10_ugm3", "pm25_ugm3", "pm25_rollmean_7d", "pm25_rollmean_14d",
    "exceedance_count_14d", "pm25_rollmax_7d", "pm10_mean7", "pm10_max7",
    "cold_day", "pressure_change_1d", "pm25_fraction", "month",
    "temp_max_c", "temp_min_c", "wind_max_kmh", "precip_mm",
    "pressure_hpa", "humidity_pct", "target_exceeds_tomorrow",
]
out = df[keep].copy()
out.to_csv("data/history.csv.gz", index=False, float_format="%.4f")

# Typical PM10-to-PM2.5 relationships by season, used to fill in PM10 features
d = out.dropna(subset=["pm25_ugm3", "pm10_ugm3", "pm10_mean7", "pm10_max7"])
d = d[(d.pm25_ugm3 > 0.5) & (d.pm25_rollmean_7d > 0.5) & (d.pm25_rollmax_7d > 0.5)]
ratios = {}
for s, grp in d.groupby("season_num"):
    ratios[int(s)] = {
        "pm10_per_pm25": float((grp.pm10_ugm3 / grp.pm25_ugm3).median()),
        "pm10_mean7_per_pm25_mean7": float((grp.pm10_mean7 / grp.pm25_rollmean_7d).median()),
        "pm10_max7_per_pm25_max7": float((grp.pm10_max7 / grp.pm25_rollmax_7d).median()),
    }
json.dump(ratios, open("data/derivation_ratios.json", "w"), indent=2)
print(json.dumps(ratios, indent=2), len(out))
