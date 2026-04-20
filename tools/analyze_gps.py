import sys, io, os, glob, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_tee_file = open("gps_analysis_result.txt", "w", encoding="utf-8")
_orig_write = sys.stdout.write
def _tee_write(s):
    _tee_file.write(s)
    return _orig_write(s)
sys.stdout.write = _tee_write

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)

# Auto-detect all dataset pairs: drone_synced_data.csv, drone_synced_data2.csv, ...
script_dir = os.path.dirname(os.path.abspath(__file__))
drone_files = sorted(glob.glob(os.path.join(script_dir, "drone_synced_data*.csv")))

def get_suffix(path):
    m = re.search(r"drone_synced_data(\d*)\.csv$", os.path.basename(path))
    return m.group(1) if m else ""

pairs = []
for df in drone_files:
    suffix = get_suffix(df)
    hf = os.path.join(script_dir, f"husky_synced_data{suffix}.csv")
    if os.path.exists(hf):
        pairs.append((suffix or "1", df, hf))
    else:
        print(f"[WARN] No matching husky file for {os.path.basename(df)}, skipping.")

print(f"Found {len(pairs)} dataset pair(s).\n")

for label, drone_csv, husky_csv in pairs:
    print(f"{'─'*55}")
    print(f"Set {label}: {os.path.basename(drone_csv)} + {os.path.basename(husky_csv)}")
    print(f"{'─'*55}")

    drone = pd.read_csv(drone_csv, usecols=lambda c: c.strip() in {
        "timestamp", "latitude", "longitude", "speed(mph)",
    })
    drone.columns = drone.columns.str.strip()

    husky = pd.read_csv(husky_csv, usecols=[
        "timestamp", "gps_latitude", "gps_longitude",
        "gps_time_utc", "frame_time_utc",
        "speed_mps", "cmd_vel_linear_x", "cmd_vel_angular_z",
        "gps_vel_twist_linear_x", "gps_vel_twist_linear_y",
        "track_true_deg",
        "imu_orientation_x", "imu_orientation_y",
        "imu_orientation_z", "imu_orientation_w",
    ])

    drone["timestamp"] = pd.to_datetime(drone["timestamp"], format="mixed", utc=True)
    husky["timestamp"] = pd.to_datetime(husky["timestamp"], format="mixed", utc=True)

    drone = drone.dropna(subset=["latitude", "longitude"])
    husky = husky.dropna(subset=["gps_latitude", "gps_longitude"])

    # ── 0. Outlier detection (IQR 3×) ────────────────────────────────────────
    drone_out = iqr_mask(drone["latitude"]) | iqr_mask(drone["longitude"])
    husky_out = iqr_mask(husky["gps_latitude"]) | iqr_mask(husky["gps_longitude"])

    print("=== GPS Outlier Detection (IQR 3x) ===")
    print(f"  Drone  total rows: {len(drone):>5} | outliers: {drone_out.sum()} ({drone_out.mean()*100:.2f}%)")
    if drone_out.any():
        print(drone[drone_out][["timestamp","latitude","longitude"]].to_string(index=False))
    print(f"  Husky  total rows: {len(husky):>5} | outliers: {husky_out.sum()} ({husky_out.mean()*100:.2f}%)")
    if husky_out.any():
        print(husky[husky_out][["timestamp","gps_latitude","gps_longitude"]].to_string(index=False))

    drone_clean = drone[~drone_out]
    husky_clean = husky[~husky_out]
    print()

    # ── 1. Experiment area via convex hull ────────────────────────────────────
    all_lat = np.concatenate([drone_clean["latitude"].values, husky_clean["gps_latitude"].values])
    all_lon = np.concatenate([drone_clean["longitude"].values, husky_clean["gps_longitude"].values])
    pts = np.column_stack([all_lon, all_lat])

    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        mean_lat = np.radians(np.mean(all_lat))
        lat_m = 111320.0
        lon_m = 111320.0 * np.cos(mean_lat)
        hull_m = np.column_stack([hull_pts[:, 0] * lon_m, hull_pts[:, 1] * lat_m])
        x, y = hull_m[:, 0], hull_m[:, 1]
        area_m2 = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        all_x = all_lon * lon_m
        all_y = all_lat * lat_m
        width_m  = all_x.max() - all_x.min()
        height_m = all_y.max() - all_y.min()
        print("=== Experiment Area ===")
        print(f"  Convex hull area: {area_m2:.1f} m²  ({area_m2/10000:.4f} ha)")
        print(f"  Bounding box  X (E-W width): {width_m:.1f} m  |  Y (N-S length): {height_m:.1f} m")
        print(f"  Bounding box area: {width_m * height_m:.1f} m²")
    except Exception as e:
        print(f"  ConvexHull failed: {e}")

    # ── 2. GPS time error ────────────────────────────────────────────────────
    husky["frame_time_utc_dt"] = pd.to_datetime(husky["frame_time_utc"], utc=True)
    husky["gps_time_utc_dt"]   = pd.to_datetime(husky["gps_time_utc"],   utc=True)
    husky_internal_err = (husky["frame_time_utc_dt"] - husky["gps_time_utc_dt"]).dt.total_seconds().abs()
    print(f"\nHusky internal GPS time error (frame_time_utc vs gps_time_utc):")
    print(f"  mean: {husky_internal_err.mean()*1000:.2f} ms  |  median: {husky_internal_err.median()*1000:.2f} ms  |  max: {husky_internal_err.max()*1000:.2f} ms")

    drone_ts = drone["timestamp"].values.astype("int64")
    husky_ts = husky["timestamp"].values.astype("int64")
    idx = np.searchsorted(husky_ts, drone_ts)
    idx = np.clip(idx, 0, len(husky_ts) - 1)
    idx_prev = np.clip(idx - 1, 0, len(husky_ts) - 1)
    diff_cur  = np.abs(drone_ts - husky_ts[idx])
    diff_prev = np.abs(drone_ts - husky_ts[idx_prev])
    best_idx = np.where(diff_prev < diff_cur, idx_prev, idx)
    time_diff_ms = np.abs(drone_ts - husky_ts[best_idx]) / 1e6

    print(f"\nDrone <-> Husky cross-platform time sync error (nearest-neighbor match):")
    print(f"  mean: {time_diff_ms.mean():.2f} ms  |  median: {np.median(time_diff_ms):.2f} ms  |  max: {time_diff_ms.max():.2f} ms")

    # ── 3. Cross-comparison: Husky GPS drift ──────────────────────────────────
    n = min(len(drone), len(husky))
    d_lat = drone["latitude"].values[:n]
    d_lon = drone["longitude"].values[:n]
    h_lat = husky["gps_latitude"].values[:n]
    h_lon = husky["gps_longitude"].values[:n]
    cmd_x  = husky["cmd_vel_linear_x"].fillna(0).values[:n]
    cmd_z  = husky["cmd_vel_angular_z"].fillna(0).values[:n]
    h_spd  = husky["speed_mps"].values[:n]
    drone_spd_mps = drone["speed(mph)"].values[:n] * 0.44704

    def planar_dist_m(lat1, lon1, lat2, lon2):
        mid = np.radians((lat1 + lat2) / 2)
        return np.sqrt(((lat2 - lat1) * 111320)**2 + ((lon2 - lon1) * 111320 * np.cos(mid))**2)

    # 3a. Drone–Husky GPS separation
    sep_m = planar_dist_m(d_lat, d_lon, h_lat, h_lon)
    valid_sep = np.isfinite(sep_m)
    hover  = valid_sep & (drone_spd_mps < 0.447)   # drone < 1 mph → hovering
    moving = valid_sep & (drone_spd_mps >= 0.447)

    print(f"\n=== Cross-compare A: Drone–Husky GPS Separation ===")
    print(f"  Overall    : mean {sep_m[valid_sep].mean():.2f} m  std {sep_m[valid_sep].std():.2f} m  "
          f"min {sep_m[valid_sep].min():.2f} m  max {sep_m[valid_sep].max():.2f} m")
    if hover.sum() > 5:
        print(f"  Drone hover ({hover.sum():>4} frames): mean {sep_m[hover].mean():.2f} m  std {sep_m[hover].std():.2f} m")
    if moving.sum() > 5:
        print(f"  Drone move  ({moving.sum():>4} frames): mean {sep_m[moving].mean():.2f} m  std {sep_m[moving].std():.2f} m")

    # 3b. Stationary GPS drift (cmd_vel ≈ 0 → robot not moving)
    stationary = (np.abs(cmd_x) < 0.05) & (np.abs(cmd_z) < 0.05)
    print(f"\n=== Cross-compare B: Husky Stationary GPS Drift ({stationary.sum()} frames with cmd_vel≈0) ===")
    if stationary.sum() > 5:
        s_lat = h_lat[stationary]; s_lon = h_lon[stationary]
        mid_r = np.radians(np.nanmean(s_lat))
        h_ew = (s_lon - np.nanmean(s_lon)) * 111320 * np.cos(mid_r)
        h_ns = (s_lat - np.nanmean(s_lat)) * 111320
        h_2d = np.sqrt(h_ew**2 + h_ns**2)
        print(f"  Husky GPS pos std:  E-W {np.nanstd(h_ew):.3f} m  N-S {np.nanstd(h_ns):.3f} m  2D {np.nanstd(h_2d):.3f} m")
        print(f"  Husky max excursion from centroid: {np.nanmax(h_2d):.3f} m")
        # Drone GPS drift during same frames (reference)
        ds_lat = d_lat[stationary]; ds_lon = d_lon[stationary]
        mid_dr = np.radians(np.nanmean(ds_lat))
        d_ew = (ds_lon - np.nanmean(ds_lon)) * 111320 * np.cos(mid_dr)
        d_ns = (ds_lat - np.nanmean(ds_lat)) * 111320
        d_2d = np.sqrt(d_ew**2 + d_ns**2)
        print(f"  Drone  GPS pos std (same frames): 2D {np.nanstd(d_2d):.3f} m  max {np.nanmax(d_2d):.3f} m")
        ratio = np.nanstd(h_2d) / max(np.nanstd(d_2d), 1e-4)
        print(f"  → Husky/Drone drift ratio: {ratio:.2f}×")
    else:
        print("  (too few stationary frames to measure)")

    # 3c. Husky GPS speed vs commanded velocity
    cmd_abs   = np.abs(cmd_x)
    mv_cmd    = np.isfinite(h_spd) & (cmd_abs > 0.1)
    print(f"\n=== Cross-compare C: Husky GPS Speed vs cmd_vel ({mv_cmd.sum()} moving frames) ===")
    if mv_cmd.sum() > 5:
        spd_m = h_spd[mv_cmd]; cmd_m = cmd_abs[mv_cmd]
        mae   = np.nanmean(np.abs(spd_m - cmd_m))
        corr  = np.corrcoef(spd_m[np.isfinite(spd_m)], cmd_m[np.isfinite(spd_m)])[0, 1]
        print(f"  GPS speed:   mean {np.nanmean(spd_m):.3f} m/s")
        print(f"  cmd_vel:     mean {np.nanmean(cmd_m):.3f} m/s")
        print(f"  MAE: {mae:.3f} m/s  |  Pearson r: {corr:.3f}")
        # GPS twist speed
        gvx = husky["gps_vel_twist_linear_x"].values[:n][mv_cmd]
        gvy = husky["gps_vel_twist_linear_y"].values[:n][mv_cmd]
        twist_spd = np.sqrt(gvx**2 + gvy**2)
        vld = np.isfinite(twist_spd)
        if vld.sum() > 2:
            print(f"  GPS twist speed mean: {np.nanmean(twist_spd[vld]):.3f} m/s  "
                  f"MAE vs cmd_vel: {np.nanmean(np.abs(twist_spd[vld]-cmd_m[vld])):.3f} m/s")
    else:
        print("  (too few moving frames)")

    # 3d. Husky GPS heading vs IMU yaw (when moving)
    qx = husky["imu_orientation_x"].values[:n]
    qy = husky["imu_orientation_y"].values[:n]
    qz = husky["imu_orientation_z"].values[:n]
    qw = husky["imu_orientation_w"].values[:n]
    imu_yaw = (np.degrees(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))) % 360)
    track   = husky["track_true_deg"].values[:n]
    mv_hdg  = cmd_abs > 0.2
    print(f"\n=== Cross-compare D: Husky GPS Heading vs IMU Yaw ({mv_hdg.sum()} moving frames) ===")
    if mv_hdg.sum() > 5:
        diff = ((imu_yaw[mv_hdg] - track[mv_hdg] + 180) % 360) - 180
        vld  = np.isfinite(diff)
        if vld.sum() > 2:
            print(f"  Mean |error|: {np.mean(np.abs(diff[vld])):.2f}°  "
                  f"Std: {np.std(diff[vld]):.2f}°  Max: {np.max(np.abs(diff[vld])):.2f}°")
        else:
            print("  (insufficient valid heading data)")
    else:
        print("  (too few moving frames)")
    print()

_tee_file.flush()
_tee_file.close()
sys.stdout.write = _orig_write
print(">> Results saved to gps_analysis_result.txt")
