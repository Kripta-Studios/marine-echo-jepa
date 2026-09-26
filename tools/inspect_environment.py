"""Inspect downloaded MOSAiC environment applicability; never calibrate implicitly."""

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/raw/environment/mosaic_core/MOSAiC_daily_profiles.nc"
# In-memory opening avoids the netCDF4 Windows non-ASCII filename limitation.
handle = netCDF4.Dataset("memory", memory=path.read_bytes())
ds = xr.open_dataset(xr.backends.NetCDF4DataStore(handle))
times = np.array([
    (datetime.fromordinal(int(t)) + timedelta(days=float(t) % 1) - timedelta(days=366)).replace(tzinfo=timezone.utc).timestamp()
    for t in ds.time.values
])
records = []
gps_path = ROOT / "data/raw/pangaea/949811/xeos-rover-mosaicdown-transponder-16feb-02aug2020.csv"
with gps_path.open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        if row["Device"] != "MOSAICdown":
            continue
        match = re.search(r"Timestamp: ([^,]+).*Latitude: ([\d.-]+), Longitude: ([\d.-]+)", row["Message"])
        if match:
            t, lat, lon = match.groups()
            records.append((datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp(), float(lat), float(lon)))
gps = np.array(records)
rows = []
for i, t in enumerate(times):
    if not datetime(2020, 2, 17, tzinfo=timezone.utc).timestamp() <= t < datetime(2020, 8, 3, tzinfo=timezone.utc).timestamp():
        continue
    j = int(np.abs(gps[:, 0] - t).argmin())
    lat1, lon1 = np.radians([float(ds.latitude[i]), float(ds.longitude[i])])
    lat2, lon2 = np.radians(gps[j, 1:])
    hav = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    distance = float(6371 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1))))
    depth = ds.depth[:, i].values
    selected = (depth >= 10) & (depth <= 100)
    valid = selected & np.isfinite(ds.salinity[:, i].values) & np.isfinite(ds.temperature[:, i].values)
    rows.append({"date_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(), "nearest_gps_hours": abs(float(gps[j,0]-t))/3600, "distance_km": distance if np.isfinite(distance) else None, "valid_10_100m_levels": int(valid.sum()), "requested_10_100m_levels": int(selected.sum())})
result = {
    "status": "APPLICABILITY_AUDIT_ONLY_NOT_CALIBRATION",
    "doi": "10.18739/A21J9790B", "gps_device_filter": "MOSAICdown", "gps_records": len(gps), "profile_days_in_deployment": len(rows),
    "temperature_semantics": "Conservative Temperature, not in-situ temperature",
    "salinity_semantics": "Absolute Salinity g/kg, not Practical Salinity",
    "time_semantics": "MATLAB serial day converted by ordinal minus 366 days; daily averages may include later observations",
    "limitations": ["Locations represent the compiled profile source, not proof of acoustic instrument co-location.", "No maximum spatial separation or time interpolation rule has been scientifically accepted.", "Daily retrospective environmental corrections must not become future predictor inputs.", "Do not pass Conservative Temperature or Absolute Salinity directly to formulas expecting in-situ T or Practical S."],
    "days": rows,
}
target = ROOT / "evidence/calibration/environment_applicability.json"
target.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
distances = [r["distance_km"] for r in rows if r["distance_km"] is not None]
print(json.dumps({"gps_records":len(gps),"days":len(rows),"distance_km_min_median_max":np.percentile(distances,[0,50,100]).tolist(),"complete_10_100m_days":sum(r["valid_10_100m_levels"]==r["requested_10_100m_levels"]>0 for r in rows)},indent=2))
