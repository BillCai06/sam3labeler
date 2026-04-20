import sys, os, glob, re, json, math
import pandas as pd
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def get_suffix(path):
    m = re.search(r"drone_synced_data(\d*)\.csv$", os.path.basename(path))
    return m.group(1) if m else ""

def discover_pairs(script_dir):
    drone_files = sorted(glob.glob(os.path.join(script_dir, "drone_synced_data*.csv")))
    pairs = []
    for df in drone_files:
        suffix = get_suffix(df)
        hf_poses = os.path.join(script_dir, f"husky_synced_with_poses{suffix}.csv")
        hf_plain  = os.path.join(script_dir, f"husky_synced_data{suffix}.csv")
        if os.path.exists(hf_poses):
            pairs.append({"label": suffix or "1", "suffix": suffix,
                          "drone": df, "husky": hf_poses, "has_poses": True})
        elif os.path.exists(hf_plain):
            pairs.append({"label": suffix or "1", "suffix": suffix,
                          "drone": df, "husky": hf_plain, "has_poses": False})
            print(f"[WARN] No pose file for set {suffix or '1'}, falling back.")
        else:
            print(f"[WARN] No husky file for {os.path.basename(df)}, skipping.")
    return pairs

def _nan_to_none(v):
    if v is None:
        return None
    try:
        return None if np.isnan(float(v)) else v
    except (TypeError, ValueError):
        return v

def parse_kitti_pose(s):
    """Parse 12-float KITTI 3x4 row. Returns (tx,ty,tz,[m0..m11]) or None."""
    if not s or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        m = [float(x) for x in str(s).split()]
        if len(m) != 12:
            return None
        return (m[3], m[7], m[11], m)
    except (ValueError, AttributeError):
        return None

def fit_rotation_rms(valid_rows, lat0, lon0, px0, py0, use_middle_60=True):
    """Least-squares fit for theta in R(theta) * [dx_kiss, dy_kiss] ~= [de_gps, dn_gps].

    Returns (rot_rad, diagnostics_dict) or (None, diagnostics_dict) if insufficient data.
    """
    if not valid_rows:
      return None, {"reason": "no_valid_rows"}

    rows = [
        r for r in valid_rows
        if r.get("lat") is not None and r.get("lon") is not None
        and r.get("px") is not None and r.get("py") is not None
    ]
    n_all = len(rows)
    if n_all < 3:
      return None, {"reason": "too_few_samples", "n_all": n_all}

    seg = rows
    if use_middle_60 and n_all >= 10:
        lo60, hi60 = int(n_all * 0.20), int(n_all * 0.80)
        if hi60 - lo60 >= 3:
            seg = rows[lo60:hi60]

    cos_lat0 = math.cos(math.radians(lat0))

    # Objective: minimise sum ||R(theta)k_i - g_i||^2 with closed-form theta.
    num = 0.0
    den = 0.0
    for r in seg:
      kx = float(r["px"]) - px0
      ky = float(r["py"]) - py0
      gx = (float(r["lon"]) - lon0) * 111320.0 * cos_lat0
      gy = (float(r["lat"]) - lat0) * 111320.0
      num += (kx * gy - ky * gx)
      den += (kx * gx + ky * gy)

    if abs(num) < 1e-12 and abs(den) < 1e-12:
      return None, {"reason": "degenerate_geometry", "n_seg": len(seg), "n_all": n_all}

    rot_rad = math.atan2(num, den)
    _cr, _sr = math.cos(rot_rad), math.sin(rot_rad)

    # RMS diagnostics on all valid rows (not only fitting segment)
    se = 0.0
    n = 0
    for r in rows:
      kx = float(r["px"]) - px0
      ky = float(r["py"]) - py0
      gx = (float(r["lon"]) - lon0) * 111320.0 * cos_lat0
      gy = (float(r["lat"]) - lat0) * 111320.0
      rx = kx * _cr - ky * _sr
      ry = kx * _sr + ky * _cr
      ex = rx - gx
      ey = ry - gy
      se += ex * ex + ey * ey
      n += 1

    rms = math.sqrt(se / max(n, 1))
    return rot_rad, {
        "n_seg": len(seg),
        "n_all": n_all,
        "rms_m": rms,
    }

def build_aligned_vertical_profile(husky_rows, count):
  """Calibrate vertical alignment with first 30%, then follow KISS-ICP only.

  Steps:
  - Build KISS relative height z_rel = pz - pz0 for all pose frames.
  - On first 30% frames that have GPS altitude, fit linear map:
    gps_rel ~= a * z_rel + b
  - Apply this fixed map to all frames, so later section no longer tracks GPS drift.
  """
  pose_idxs = [i for i, r in enumerate(husky_rows[:count])
         if r.get("px") is not None and r.get("py") is not None and r.get("pz") is not None]
  if len(pose_idxs) < 3:
    return {"mode": "none", "reason": "too_few_pose_points"}

  pz0 = float(husky_rows[pose_idxs[0]]["pz"])
  z_rel = {i: float(husky_rows[i]["pz"]) - pz0 for i in pose_idxs}

  alt_obs = [(i, z_rel[i], float(husky_rows[i]["alt"]))
         for i in pose_idxs if husky_rows[i].get("alt") is not None]

  # Fallback: no useful GPS altitude -> pure KISS relative z.
  if len(alt_obs) < 3:
    for i in pose_idxs:
      husky_rows[i]["pzv"] = round(z_rel[i], 4)
    return {"mode": "pz_relative", "samples_pose": len(pose_idxs)}

  alt0 = alt_obs[0][2]
  rel_obs = [(i, zr, alt - alt0) for i, zr, alt in alt_obs]

  # Only first 30% GPS data is used for calibration.
  n_train = max(3, int(len(rel_obs) * 0.30))
  n_train = min(n_train, len(rel_obs))
  train = rel_obs[:n_train]

  x_train = np.array([t[1] for t in train], dtype=float)  # KISS z_rel
  y_train = np.array([t[2] for t in train], dtype=float)  # GPS alt_rel

  if np.ptp(x_train) < 1e-6:
    scale = 1.0
    offset = float(np.median(y_train) - np.median(x_train))
  else:
    scale, offset = np.polyfit(x_train, y_train, 1)

  for i in pose_idxs:
    husky_rows[i]["pzv"] = round(scale * z_rel[i] + offset, 4)

  return {
    "mode": "gps30_calib_then_kiss",
    "samples_alt": len(rel_obs),
    "samples_pose": len(pose_idxs),
    "samples_train": n_train,
    "z_scale": float(scale),
    "z_offset": float(offset),
  }

def load_drone_csv(path, suffix):
    df = pd.read_csv(path, usecols=lambda c: c.strip() in {
        "timestamp", "latitude", "longitude",
        "height_above_takeoff(feet)", "altitude_above_seaLevel(feet)",
        "speed(mph)", "compass_heading(degrees)",
        "pitch(degrees)", "roll(degrees)", "battery_percent",
    })
    df.columns = df.columns.str.strip()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ts":  str(r.get("timestamp", "")),
            "lat": _nan_to_none(r.get("latitude")),
            "lon": _nan_to_none(r.get("longitude")),
            "alt": _nan_to_none(r.get("height_above_takeoff(feet)")),
            "hdg": _nan_to_none(r.get("compass_heading(degrees)")),
            "spd": _nan_to_none(r.get("speed(mph)")),
            "bat": _nan_to_none(r.get("battery_percent")),
            "pit": _nan_to_none(r.get("pitch(degrees)")),
            "rol": _nan_to_none(r.get("roll(degrees)")),
            "img": f"drone_frames{suffix}/drone_frame_{len(rows):06d}.jpg",
        })
    return rows

def load_husky_csv(path, suffix, has_poses):
    usecols = [
        "timestamp", "frame_index", "gps_latitude", "gps_longitude", "gps_altitude",
        "speed_mps", "imu_orientation_x", "imu_orientation_y",
        "imu_orientation_z", "imu_orientation_w",
        "cmd_vel_linear_x", "cmd_vel_angular_z", "cloud_path",
    ]
    if has_poses:
        usecols += ["kitti_pose"]

    df = pd.read_csv(path, usecols=usecols)
    rows = []
    for _, r in df.iterrows():
        fi = int(r["frame_index"]) if pd.notna(r.get("frame_index")) else len(rows)
        pose_result = None
        if has_poses and pd.notna(r.get("kitti_pose")):
            pose_result = parse_kitti_pose(r["kitti_pose"])

        row = {
            "ts":  str(r.get("timestamp", "")),
            "lat": _nan_to_none(r.get("gps_latitude")),
            "lon": _nan_to_none(r.get("gps_longitude")),
            "alt": _nan_to_none(r.get("gps_altitude")),
            "spd": _nan_to_none(r.get("speed_mps")),
            "cvx": _nan_to_none(r.get("cmd_vel_linear_x")),
            "cvz": _nan_to_none(r.get("cmd_vel_angular_z")),
            "qx":  _nan_to_none(r.get("imu_orientation_x")),
            "qy":  _nan_to_none(r.get("imu_orientation_y")),
            "qz":  _nan_to_none(r.get("imu_orientation_z")),
            "qw":  _nan_to_none(r.get("imu_orientation_w")),
            "pcd": None if pd.isna(r.get("cloud_path")) else str(r["cloud_path"]),
            "img": f"husky_frames{suffix}/husky_frame_{fi:06d}.jpg",
            "px": None, "py": None, "pz": None, "pzv": None, "mat": None,
        }
        if pose_result:
            tx, ty, tz, mat = pose_result
            row["px"] = round(tx, 4)
            row["py"] = round(ty, 4)
            row["pz"] = round(tz, 4)
            row["mat"] = [round(v, 7) for v in mat]
        rows.append(row)
    return rows

def build_set_json(label, suffix, script_dir, drone_rows, husky_rows):
    count = min(len(drone_rows), len(husky_rows))
    frames = [{"frame": i, "d": drone_rows[i], "h": husky_rows[i]} for i in range(count)]

    # Drone GPS trail
    drone_trail = [[r["lat"], r["lon"], r["alt"]] for r in drone_rows[:count]
                   if r["lat"] is not None and r["lon"] is not None]

    # Husky GPS trail (raw, drifted — kept for reference)
    husky_gps_trail = [[r["lat"], r["lon"]] for r in husky_rows[:count]
                       if r["lat"] is not None and r["lon"] is not None]

    # Husky KISS-ICP trail in GPS space: anchor at first Husky GPS fix.
    # Auto-compute rotation by minimising trajectory RMS error (least squares).
    kiss_gps_trail = []
    _cr, _sr = 1.0, 0.0
    ref = next((r for r in husky_rows[:count]
                if r["lat"] is not None and r["px"] is not None), None)
    if ref:
        lat0, lon0 = ref["lat"], ref["lon"]
        cos_lat0 = math.cos(math.radians(lat0))

        valid = [r for r in husky_rows[:count]
                 if r["lat"] is not None and r["lon"] is not None
                 and r["px"] is not None and r["py"] is not None]

        rot_rad, diag = fit_rotation_rms(
            valid_rows=valid,
            lat0=lat0,
            lon0=lon0,
            px0=float(ref["px"]),
            py0=float(ref["py"]),
            use_middle_60=True,
        )

        if rot_rad is None:
            print(f"    [WARN] RMS fit failed ({diag.get('reason','unknown')}), "
                  f"falling back to 0 deg rotation.")
            rot_rad = 0.0

        rot_rad = (rot_rad + math.pi) % (2 * math.pi) - math.pi
        _cr, _sr = math.cos(rot_rad), math.sin(rot_rad)
        print(f"    Auto rotation (RMS fit): apply {math.degrees(rot_rad):.1f} deg CCW  "
              f"(fit {diag.get('n_seg', 0)}/{diag.get('n_all', 0)} samples, "
              f"RMS={diag.get('rms_m', float('nan')):.2f} m)")

        # Subtract the first pose's rotated offset so trail starts exactly at (lat0,lon0)
        rpx0 = ref["px"] * _cr - ref["py"] * _sr
        rpy0 = ref["px"] * _sr + ref["py"] * _cr

        raw_trail = []
        for r in husky_rows[:count]:
            if r["px"] is not None:
                rpx = r["px"] * _cr - r["py"] * _sr - rpx0
                rpy = r["px"] * _sr + r["py"] * _cr - rpy0
                raw_trail.append((lat0 + rpy / 111320.0,
                                   lon0 + rpx / (111320.0 * cos_lat0)))

        # Smooth with a moving-average window to remove high-frequency jitter
        WIN = 9
        half = WIN // 2
        n_raw = len(raw_trail)
        for i in range(n_raw):
            lo, hi = max(0, i-half), min(n_raw, i+half+1)
            lat_k = sum(p[0] for p in raw_trail[lo:hi]) / (hi - lo)
            lon_k = sum(p[1] for p in raw_trail[lo:hi]) / (hi - lo)
            kiss_gps_trail.append([round(lat_k, 8), round(lon_k, 8)])

    zdiag = build_aligned_vertical_profile(husky_rows, count)
    if zdiag.get("mode") != "none":
        print(f"    Z align ({zdiag.get('mode')}): "
              f"alt={zdiag.get('samples_alt', 0)} train={zdiag.get('samples_train', 0)} "
              f"pose={zdiag.get('samples_pose', 0)} "
              f"scale={zdiag.get('z_scale', float('nan')):.4f} "
              f"offset={zdiag.get('z_offset', float('nan')):.3f}")

    # KISS-ICP 3D pose trail (with aligned vertical profile for cleaner grade)
    pose_trail = [[r["px"], r["py"], (r["pzv"] if r.get("pzv") is not None else r["pz"])] for r in husky_rows[:count]
                  if r["px"] is not None]

    has_pcd = any(
        r["pcd"] and os.path.exists(os.path.join(script_dir, r["pcd"]))
        for r in husky_rows[:count]
    )
    has_poses = bool(pose_trail)

    # Reference GPS for 3-D scene (anchor of KISS-ICP ↔ GPS overlay)
    gps_ref = [ref["lat"], ref["lon"]] if ref else None

    return {
        "label":           label,
        "drone_trail":     drone_trail,
        "husky_gps_trail": husky_gps_trail,
        "kiss_gps_trail":  kiss_gps_trail,
        "pose_trail":      pose_trail,
        "gps_ref":         gps_ref,
        "kiss_rot_cos":    round(_cr, 6) if ref else -1.0,
        "kiss_rot_sin":    round(_sr, 6) if ref else  0.0,
        "kiss_px0":        round(ref["px"], 4) if ref else 0.0,
        "kiss_py0":        round(ref["py"], 4) if ref else 0.0,
        "has_pcd":         has_pcd,
        "has_poses":       has_poses,
        "frames":          frames,
    }

# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { display:flex; flex-direction:column; height:100vh; background:#1a1a1a; color:#e0e0e0; font-family:sans-serif; overflow:hidden; }

#top-bar { flex:0 0 auto; display:flex; align-items:center; gap:12px; padding:8px 14px; background:#242424; border-bottom:1px solid #3d3d3d; }
#top-bar h1 { font-size:13px; font-weight:600; color:#aaa; white-space:nowrap; }
#set-tabs { display:flex; gap:4px; }
.set-tab { padding:4px 12px; border:1px solid #3d3d3d; border-radius:4px; background:#2d2d2d; color:#aaa; cursor:pointer; font-size:12px; }
.set-tab.active { background:#4fc3f7; color:#000; border-color:#4fc3f7; font-weight:600; }
#frame-counter { font-size:12px; color:#aaa; margin-left:auto; }
#ctrl-play { padding:4px 14px; background:#2d2d2d; border:1px solid #3d3d3d; border-radius:4px; color:#e0e0e0; cursor:pointer; font-size:13px; }
#ctrl-play:hover { background:#3d3d3d; }
#speed-select { background:#2d2d2d; border:1px solid #3d3d3d; border-radius:4px; color:#e0e0e0; padding:3px 6px; font-size:12px; }

#main-panel { flex:1 1 0; display:flex; min-height:0; }
#left-panel { flex:0 0 60%; display:flex; flex-direction:column; border-right:1px solid #3d3d3d; background:#222; }

#legend { flex:0 0 auto; padding:7px 14px; border-bottom:1px solid #3d3d3d; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
#legend h2 { font-size:10px; color:#666; text-transform:uppercase; letter-spacing:.06em; white-space:nowrap; }
.leg { display:flex; align-items:center; gap:5px; font-size:11px; color:#ccc; }
.sw { width:20px; height:3px; border-radius:2px; }
.sw.drone { background:#4fc3f7; }
.sw.kiss  { background:#66bb6a; }
.sw.gref  { background:#ef5350; border-top:1px dashed #ef5350; height:0; }
.badge { padding:1px 4px; border-radius:3px; font-size:9px; font-weight:700; }
.badge.k { background:#1b5e20; color:#a5d6a7; }
.badge.g { background:#4a1010; color:#ef9a9a; }

#view-toggle { flex:0 0 auto; display:flex; gap:6px; padding:6px 10px; border-bottom:1px solid #3d3d3d; }
.view-btn { flex:1; padding:4px 0; background:#2d2d2d; border:1px solid #3d3d3d; border-radius:4px; color:#aaa; cursor:pointer; font-size:12px; }
.view-btn.active { background:#333; color:#e0e0e0; border-color:#555; }

#map-container { flex:1 1 0; position:relative; overflow:hidden; }
#map-2d, #map-3d { position:absolute; top:0; left:0; width:100%; height:100%; }
#map-3d { display:none; }
#lidar-controls {
  position:absolute; bottom:10px; left:10px; z-index:10;
  background:rgba(0,0,0,0.65); border:1px solid #444; border-radius:6px;
  padding:8px 12px; display:none; flex-direction:column; gap:6px;
  font-size:12px; color:#ccc; pointer-events:auto;
}
#lidar-controls label { display:flex; align-items:center; gap:6px; white-space:nowrap; }
#lidar-controls input[type=range] { width:90px; cursor:pointer; }
#lidar-controls select { background:#222; color:#ccc; border:1px solid #555; border-radius:3px; padding:1px 4px; cursor:pointer; }

#right-panel { flex:0 0 40%; display:flex; flex-direction:column; min-width:0; }
.view-pane { flex:1 1 0; position:relative; overflow:hidden; background:#111; border-bottom:1px solid #2a2a2a; }
.view-pane img { width:100%; height:100%; object-fit:contain; background:#0a0a0a; display:block; }
.view-pane .placeholder { width:100%; height:100%; display:flex; align-items:center; justify-content:center; color:#555; font-size:13px; }
.overlay { position:absolute; bottom:0; left:0; right:0; padding:4px 10px; background:rgba(0,0,0,.7); font-family:'Courier New',monospace; font-size:10.5px; color:#ccc; pointer-events:none; line-height:1.6; }
.overlay .label { font-weight:700; }
.overlay.drone .label { color:#4fc3f7; }
.overlay.husky .label { color:#66bb6a; }

#timeline-bar { flex:0 0 auto; padding:10px 14px; background:#1e1e1e; border-top:1px solid #3d3d3d; display:flex; align-items:center; gap:10px; }
#scrubber { flex:1; accent-color:#4fc3f7; cursor:pointer; height:4px; }
"""

_HTML = """
<div id="top-bar">
  <h1>Robotics Viz</h1>
  <div id="set-tabs"></div>
  <span id="frame-counter">Frame: 0/0</span>
  <button id="ctrl-play">&#9654; Play</button>
  <select id="speed-select"><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option><option value="0.5">0.5x</option></select>
</div>

<div id="main-panel">
  <div id="left-panel">
    <div id="legend">
      <h2>Legend</h2>
      <div class="leg"><div class="sw drone"></div>Drone GPS</div>
      <div class="leg"><div class="sw kiss"></div>Husky <span class="badge k">KISS-ICP</span></div>
      <div class="leg"><div class="sw gref"></div>Husky GPS <span class="badge g">drift ref</span></div>
    </div>
    <div id="view-toggle">
      <button class="view-btn active" id="btn-2d">2D Map</button>
      <button class="view-btn" id="btn-3d">3D + LiDAR</button>
    </div>
    <div id="map-container">
      <canvas id="map-2d"></canvas>
      <canvas id="map-3d" style="display:none"></canvas>
      <div id="lidar-controls">
        <label>Size <input id="pt-size" type="range" min="1" max="60" value="7" step="1"> <span id="pt-size-val">0.07</span></label>
        <label>Smooth
          <select id="tr-smooth">
            <option value="off">Off</option>
            <option value="light">Light</option>
            <option value="medium" selected>Medium</option>
            <option value="strong">Strong</option>
          </select>
        </label>
        <label>PCD Pose
          <select id="pcd-pose">
            <option value="stable">Stable (translation)</option>
            <option value="full" selected>Full rotation</option>
          </select>
        </label>
        <label>Clean
          <select id="pt-clean">
            <option value="off">Off (raw)</option>
            <option value="light">Light</option>
            <option value="medium" selected>Medium</option>
            <option value="strong">Strong</option>
          </select>
        </label>
        <label>Color
          <select id="pt-color">
            <option value="height">Height (Z)</option>
            <option value="forward">Forward (X)</option>
            <option value="lateral">Lateral (Y)</option>
            <option value="distance">Distance</option>
            <option value="flat">Flat white</option>
          </select>
        </label>
      </div>
    </div>
  </div>

  <div id="right-panel">
    <div class="view-pane" id="drone-pane">
      <img id="drone-img" alt="" onerror="this.style.display='none';document.getElementById('drone-ph').style.display='flex'" />
      <div id="drone-ph" class="placeholder" style="display:none">Frame not available</div>
      <div class="overlay drone"><span class="label">DRONE</span> <span id="drone-hud"></span></div>
    </div>
    <div class="view-pane" id="husky-pane">
      <img id="husky-img" alt="" onerror="this.style.display='none';document.getElementById('husky-ph').style.display='flex'" />
      <div id="husky-ph" class="placeholder" style="display:none">Frame not available</div>
      <div class="overlay husky"><span class="label">HUSKY</span> <span id="husky-hud"></span></div>
    </div>
  </div>
</div>

<div id="timeline-bar">
  <input type="range" id="scrubber" min="0" value="0" step="1" />
</div>
"""

_JS = r"""
const DATA = /*DATA_PLACEHOLDER*/null/*END_PLACEHOLDER*/;

// ── state ──────────────────────────────────────────────────────────────────
let currentSetIdx = 0;
let currentFrame  = 0;
let playTimer     = null;
let playSpeed     = 1;
const cachedProjs = [];

// ── GPS projection (covers drone + KISS-ICP trails together) ───────────────
function buildProjection(set) {
  // Combine drone GPS + KISS-ICP-in-GPS for bounds; ignore Husky raw GPS
  // (Husky raw GPS is essentially a static dot — don't let it dominate bounds)
  const allPts = [...set.drone_trail, ...set.kiss_gps_trail];
  if (!allPts.length) return null;
  const allLat = allPts.map(p => p[0]);
  const allLon = allPts.map(p => p[1]);
  const minLat = Math.min(...allLat), maxLat = Math.max(...allLat);
  const minLon = Math.min(...allLon), maxLon = Math.max(...allLon);
  const midLat = (minLat + maxLat) / 2;
  const cosLat = Math.cos(midLat * Math.PI / 180);
  const spanLon = (maxLon - minLon) * 111320 * cosLat || 1;
  const spanLat = (maxLat - minLat) * 111320 || 1;
  return {
    minLat, maxLat, minLon, maxLon, cosLat, spanLon, spanLat,
    toXY(lat, lon, W, H, pad = 20) {
      const nx = (lon - minLon) / (maxLon - minLon || 1e-9);
      const ny = 1 - (lat - minLat) / (maxLat - minLat || 1e-9);
      const scale = Math.min((W - 2*pad) / spanLon, (H - 2*pad) / spanLat);
      const ox = pad + ((W - 2*pad) - spanLon * scale) / 2;
      const oy = pad + ((H - 2*pad) - spanLat * scale) / 2;
      return [ox + nx * spanLon * scale, oy + ny * spanLat * scale];
    }
  };
}

function getProj(idx) {
  if (!cachedProjs[idx]) cachedProjs[idx] = buildProjection(DATA.sets[idx]);
  return cachedProjs[idx];
}

// ── 2D canvas drawing ──────────────────────────────────────────────────────
const c2d = document.getElementById('map-2d');
const ctx  = c2d.getContext('2d');

function resizeCanvas() {
  const cont = document.getElementById('map-container');
  c2d.width  = cont.clientWidth;
  c2d.height = cont.clientHeight;
  drawMap();
}

function drawTrail(trail, proj, color, alpha, lineW, dash) {
  if (!trail.length) return;
  ctx.save();
  ctx.globalAlpha  = alpha;
  ctx.strokeStyle  = color;
  ctx.lineWidth    = lineW;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  trail.forEach(([lat, lon], i) => {
    const [x, y] = proj.toXY(lat, lon, c2d.width, c2d.height);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawDot(x, y, color, size = 7) {
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,.85)';
  ctx.beginPath(); ctx.arc(x, y, size * .35, 0, Math.PI*2); ctx.fill();
}

function drawMap() {
  const W = c2d.width, H = c2d.height;
  ctx.clearRect(0, 0, W, H);
  const set  = DATA.sets[currentSetIdx];
  const proj = getProj(currentSetIdx);
  if (!proj) return;

  const f = set.frames[currentFrame];

  // ── faded full trails ─────────────────────────────────────────────────
  // Husky raw GPS (drifted reference) — dashed red, very faint
  drawTrail(set.husky_gps_trail, proj, '#ef5350', 0.18, 1, [5, 5]);
  // Drone GPS full trail
  drawTrail(set.drone_trail, proj, '#4fc3f7', 0.25, 1.5, []);
  // KISS-ICP trail full
  drawTrail(set.kiss_gps_trail, proj, '#66bb6a', 0.25, 1.5, []);

  // ── active trails up to current frame ────────────────────────────────
  drawTrail(set.drone_trail.slice(0, currentFrame+1), proj, '#4fc3f7', 1, 2, []);
  drawTrail(set.kiss_gps_trail.slice(0, currentFrame+1), proj, '#66bb6a', 1, 2, []);

  // ── current position dots ─────────────────────────────────────────────
  if (f.d.lat != null && f.d.lon != null) {
    const [dx, dy] = proj.toXY(f.d.lat, f.d.lon, W, H);
    drawDot(dx, dy, '#4fc3f7');
  }
  const kidx = currentFrame < set.kiss_gps_trail.length ? currentFrame : set.kiss_gps_trail.length - 1;
  if (kidx >= 0) {
    const [klat, klon] = set.kiss_gps_trail[kidx];
    const [kx, ky] = proj.toXY(klat, klon, W, H);
    drawDot(kx, ky, '#66bb6a');
  }
}

new ResizeObserver(resizeCanvas).observe(document.getElementById('map-container'));

// ── gotoFrame ──────────────────────────────────────────────────────────────
function fmt(v, d = 1) { return v == null ? 'N/A' : Number(v).toFixed(d); }

function gotoFrame(idx) {
  const set = DATA.sets[currentSetIdx];
  idx = Math.max(0, Math.min(set.frames.length - 1, idx));
  currentFrame = idx;
  const f = set.frames[idx];

  const di = document.getElementById('drone-img');
  di.style.display = ''; document.getElementById('drone-ph').style.display = 'none';
  di.src = f.d.img;
  const hi = document.getElementById('husky-img');
  hi.style.display = ''; document.getElementById('husky-ph').style.display = 'none';
  hi.src = f.h.img;

  document.getElementById('drone-hud').textContent =
    `${f.d.ts.slice(11,19)} UTC  |  alt: ${fmt(f.d.alt,0)} ft  |  hdg: ${fmt(f.d.hdg,0)}\u00b0  |  spd: ${fmt(f.d.spd,1)} mph  |  bat: ${fmt(f.d.bat,0)}%`;

  if (f.h.px != null) {
    const hz = f.h.pzv != null ? f.h.pzv : f.h.pz;
    document.getElementById('husky-hud').textContent =
      `${f.h.ts.slice(11,19)} UTC  |  pose:(${fmt(f.h.px,1)},${fmt(f.h.py,1)},${fmt(hz,2)}) m  |  v:${fmt(f.h.spd,2)} m/s  |  w:${fmt(f.h.cvz,2)} r/s`;
  } else {
    document.getElementById('husky-hud').textContent =
      `${f.h.ts.slice(11,19)} UTC  |  GPS:${fmt(f.h.lat,6)},${fmt(f.h.lon,6)}  |  v:${fmt(f.h.spd,2)} m/s`;
  }

  document.getElementById('scrubber').value = idx;
  document.getElementById('frame-counter').textContent = `Frame: ${idx+1}/${set.frames.length}`;

  drawMap();

  if (threeActive && threeLoaded) {
    updateThreeMarkers(idx);
    const fh = f.h;
    const hStrong = husky3DStrongByFrame[idx];
    const zForPCD = hStrong ? hStrong[1] : (fh.pzv != null ? fh.pzv : fh.pz);
    if (fh.pcd && fh.mat) loadPCD(fh.pcd, fh.mat, zForPCD);
  }
}

// ── controls ───────────────────────────────────────────────────────────────
function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  document.getElementById('ctrl-play').innerHTML = '&#9654; Play';
}
function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  document.getElementById('ctrl-play').innerHTML = '&#9646;&#9646; Pause';
  playTimer = setInterval(() => {
    const set = DATA.sets[currentSetIdx];
    if (currentFrame >= set.frames.length - 1) { stopPlay(); return; }
    gotoFrame(currentFrame + 1);
  }, 1000 / playSpeed);
}
document.getElementById('scrubber').addEventListener('input', e => gotoFrame(parseInt(e.target.value)));
document.getElementById('ctrl-play').addEventListener('click', togglePlay);
document.getElementById('speed-select').addEventListener('change', e => {
  playSpeed = parseFloat(e.target.value);
  if (playTimer) { stopPlay(); togglePlay(); }
});
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'ArrowRight') { e.preventDefault(); gotoFrame(currentFrame + 1); }
  if (e.code === 'ArrowLeft')  { e.preventDefault(); gotoFrame(currentFrame - 1); }
  if (e.code === 'Space')      { e.preventDefault(); togglePlay(); }
});

// ── set tabs ───────────────────────────────────────────────────────────────
function buildTabs() {
  const cont = document.getElementById('set-tabs');
  DATA.sets.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'set-tab' + (i === 0 ? ' active' : '');
    b.textContent = `Set ${s.label}`;
    b.addEventListener('click', () => switchSet(i));
    cont.appendChild(b);
  });
}
function switchSet(idx) {
  stopPlay();
  currentSetIdx = idx;
  currentFrame  = 0;
  document.querySelectorAll('.set-tab').forEach((b, i) => b.classList.toggle('active', i === idx));
  const sc = document.getElementById('scrubber');
  sc.max = DATA.sets[idx].frames.length - 1; sc.value = 0;
  if (threeScene) rebuildThreeScene();
  gotoFrame(0);
}

// ── Three.js 3D scene ───────────────────────────────────────────────────────
// Coordinate convention: KISS-ICP (px, py, pz) → Three.js (px, pz, py)
//   Three.x = KISS-ICP.x  (forward, ~0-80 m)
//   Three.y = KISS-ICP.z  (height,  ~0-5 m)   ← y-up in Three.js
//   Three.z = KISS-ICP.y  (lateral, ~0-30 m)
//
// KITTI pose matrix [R|t] applied only as TRANSLATION (Three.position.set)
// to avoid coordinate-frame rotation ambiguity when displaying the raw cloud.

let threeActive = false, threeLoaded = false;
let threeRenderer = null, threeScene = null, threeCamera = null;
let pcdCloud = null;
let huskyTrailLine = null, droneTrailLine = null, huskyMark = null, droneMark = null;
let pcdPointSize = 0.07, pcdColorScheme = 'height';
let pcdCleanLevel = 'medium';
let threeSmoothLevel = 'medium';
let pcdPoseMode = 'full';
let pcdLoadToken = 0;
let husky3DByFrame = [], drone3DByFrame = [];
let husky3DStrongByFrame = [];

// Scene centre (updated when trail is rebuilt)
let sceneCenter = { x: 40, y: 1, z: 15 };

function kissToThree(px, py, pz) {
  // Rotate KISS-ICP frame to GPS frame, then map to Three.js (y-up, east=+x, north=-z)
  const set = DATA.sets[currentSetIdx];
  const c = set.kiss_rot_cos, s = set.kiss_rot_sin;
  const rpx = px * c - py * s;  // east  → Three.x
  const rpy = px * s + py * c;  // north → Three.z (negated: north = -z)
  return [rpx, pz, -rpy];
}

function smoothWindow(level) {
  if (level === 'off') return 1;
  if (level === 'light') return 5;
  if (level === 'strong') return 17;
  return 9; // medium
}

function smoothTrackByFrame(track, win) {
  if (!track || !track.length || win <= 1) return track;
  const half = Math.floor(win / 2);
  const out = new Array(track.length).fill(null);
  for (let i = 0; i < track.length; i++) {
    let sx = 0, sy = 0, sz = 0, n = 0;
    const lo = Math.max(0, i - half);
    const hi = Math.min(track.length - 1, i + half);
    for (let j = lo; j <= hi; j++) {
      const p = track[j];
      if (!p) continue;
      sx += p[0];
      sy += p[1];
      sz += p[2];
      n += 1;
    }
    if (n > 0) out[i] = [sx / n, sy / n, sz / n];
  }
  return out;
}

function trackToVerts(track) {
  const verts = [];
  for (let i = 0; i < track.length; i++) {
    const p = track[i];
    if (!p) continue;
    verts.push(p[0], p[1], p[2]);
  }
  return verts;
}

function buildSmoothedThreeTracks(set) {
  const frames = set.frames;
  const huskyRaw = new Array(frames.length).fill(null);
  const droneRaw = new Array(frames.length).fill(null);
  const ref = set.gps_ref;

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];

    if (f.h.px != null) {
      const hz = f.h.pzv != null ? f.h.pzv : f.h.pz;
      huskyRaw[i] = kissToThree(f.h.px, f.h.py, hz);
    }

    if (ref && f.d.lat != null && f.d.lon != null) {
      const cosRef = Math.cos(ref[0] * Math.PI / 180);
      const c = set.kiss_rot_cos, s = set.kiss_rot_sin;
      const de = (f.d.lon - ref[1]) * 111320 * cosRef;
      const dn = (f.d.lat - ref[0]) * 111320;
      const px = de * c + dn * s + set.kiss_px0;
      const py = -de * s + dn * c + set.kiss_py0;
      const dpz = f.d.alt != null ? f.d.alt * 0.3048 : 20;
      droneRaw[i] = kissToThree(px, py, dpz);
    }
  }

  const win = smoothWindow(threeSmoothLevel);
  husky3DByFrame = smoothTrackByFrame(huskyRaw, win);
  drone3DByFrame = smoothTrackByFrame(droneRaw, win);
  husky3DStrongByFrame = smoothTrackByFrame(huskyRaw, smoothWindow('strong'));
}

function rebuildThreeScene() {
  if (!threeScene) return;
  [pcdCloud, huskyTrailLine, droneTrailLine, huskyMark, droneMark].forEach(o => {
    if (o) threeScene.remove(o);
  });
  pcdCloud = huskyTrailLine = droneTrailLine = huskyMark = droneMark = null;

  const set = DATA.sets[currentSetIdx];
  buildSmoothedThreeTracks(set);

  // ── Husky KISS-ICP trajectory line ─────────────────────────────────────
  const hverts = trackToVerts(husky3DByFrame);
  if (hverts.length) {
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < hverts.length; i += 3) {
      sx += hverts[i]; sy += hverts[i + 1]; sz += hverts[i + 2];
    }
    const n = hverts.length / 3;
    sceneCenter = { x: sx/n, y: sy/n, z: sz/n };

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(hverts, 3));
    huskyTrailLine = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x66bb6a }));
    threeScene.add(huskyTrailLine);
  }

  // ── Drone GPS trajectory line ───────────────────────────────────────────
  const dverts = trackToVerts(drone3DByFrame);
  if (dverts.length) {
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.Float32BufferAttribute(dverts, 3));
    droneTrailLine = new THREE.Line(dg, new THREE.LineBasicMaterial({ color: 0x4fc3f7 }));
    threeScene.add(droneTrailLine);
  }

  // ── Markers ─────────────────────────────────────────────────────────────
  huskyMark = new THREE.Mesh(
    new THREE.SphereGeometry(0.5, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0x66bb6a }));
  droneMark = new THREE.Mesh(
    new THREE.SphereGeometry(3, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0x4fc3f7 }));
  threeScene.add(huskyMark);
  threeScene.add(droneMark);

  updateThreeMarkers(currentFrame);
}

function updateThreeMarkers(idx) {
  const set = DATA.sets[currentSetIdx];
  const f   = set.frames[idx];

  // Husky marker — KISS-ICP pose
  if (huskyMark) {
    const hp = husky3DByFrame[idx];
    if (hp) {
      huskyMark.position.set(hp[0], hp[1], hp[2]);
      huskyMark.visible = true;
    } else {
      huskyMark.visible = false;
    }
  }

  // Drone marker — GPS → metres → unrotate to KISS-ICP frame → kissToThree
  if (droneMark) {
    const dp = drone3DByFrame[idx];
    if (dp) {
      droneMark.position.set(dp[0], dp[1], dp[2]);
      droneMark.visible = true;
    } else {
      droneMark.visible = false;
    }
  }
}

document.getElementById('btn-2d').addEventListener('click', () => setView('2d'));
document.getElementById('btn-3d').addEventListener('click', () => setView('3d'));

function setView(mode) {
  const is3d = mode === '3d';
  document.getElementById('map-2d').style.display = is3d ? 'none' : 'block';
  document.getElementById('map-3d').style.display = is3d ? 'block' : 'none';
  document.getElementById('lidar-controls').style.display = is3d ? 'flex' : 'none';
  document.getElementById('btn-2d').classList.toggle('active', !is3d);
  document.getElementById('btn-3d').classList.toggle('active', is3d);
  threeActive = is3d;
  if (is3d && !threeLoaded) {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
    s.onload = () => { threeLoaded = true; initThree(); gotoFrame(currentFrame); };
    document.head.appendChild(s);
  } else if (is3d) {
    gotoFrame(currentFrame);
  } else {
    drawMap();
  }
}

function initThree() {
  const canvas = document.getElementById('map-3d');
  threeRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  threeRenderer.setPixelRatio(window.devicePixelRatio);
  threeRenderer.setClearColor(0x0f0f0f);
  threeRenderer.setSize(canvas.clientWidth, canvas.clientHeight);

  threeScene  = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.1, 5000);

  // Ground grid — XZ plane at y=0
  threeScene.add(new THREE.GridHelper(250, 50, 0x444444, 0x2a2a2a));
  // Axis helper at origin
  threeScene.add(new THREE.AxesHelper(5));

  rebuildThreeScene();

  // Position camera to see both Husky (y≈0-5m) and drone (y≈78m)
  const c = sceneCenter;
  const droneAlt = 80;  // approximate drone altitude in metres
  const midY = (c.y + droneAlt) / 2;
  threeCamera.position.set(c.x, midY + 120, c.z + 180);
  threeCamera.lookAt(c.x, midY, c.z);

  initOrbit(canvas, { x: c.x, y: midY, z: c.z });

  (function animate() {
    requestAnimationFrame(animate);
    threeRenderer.render(threeScene, threeCamera);
  })();
}

function initOrbit(canvas, center) {
  // Simple orbit around a target point
  let drag = false, lx = 0, ly = 0;
  let theta = -0.3, phi = 0.55, r = 120;
  const cx = center.x, cy = center.y, cz = center.z;

  function updateCamera() {
    threeCamera.position.set(
      cx + r * Math.sin(theta) * Math.cos(phi),
      cy + r * Math.sin(phi),
      cz + r * Math.cos(theta) * Math.cos(phi));
    threeCamera.lookAt(cx, cy, cz);
  }
  updateCamera();

  canvas.addEventListener('mousedown', e => { drag = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener('mouseup', () => { drag = false; });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    theta -= (e.clientX - lx) * 0.008;
    phi = Math.max(0.05, Math.min(Math.PI / 2 - 0.05, phi - (e.clientY - ly) * 0.008));
    lx = e.clientX; ly = e.clientY;
    updateCamera();
  });
  canvas.addEventListener('wheel', e => {
    r = Math.max(5, Math.min(3000, r + e.deltaY * 0.2));
    updateCamera();
    e.preventDefault();
  }, { passive: false });
}

// ── PCD loader ─────────────────────────────────────────────────────────────
// Uses TRANSLATION ONLY from the KITTI pose (indices 3,7,11).
// The raw point cloud is displayed at the robot's position without rotation,
// which avoids coordinate-frame ambiguity while still showing the cloud
// moving with the robot along the KISS-ICP trajectory.
//
// Serve files locally to bypass fetch() restrictions:
//   python -m http.server 8080
// then open: http://localhost:8080/visualization.html

// ── LiDAR coloring ──────────────────────────────────────────────────────────
function colorPCD(pts, scheme) {
  const cols = new Float32Array(pts.length);
  // Compute range for normalisation
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < pts.length; i += 3) {
    let v;
    if      (scheme === 'height')   v = pts[i+1];
    else if (scheme === 'forward')  v = pts[i];
    else if (scheme === 'lateral')  v = Math.abs(pts[i+2]);
    else if (scheme === 'distance') v = Math.sqrt(pts[i]*pts[i] + pts[i+1]*pts[i+1] + pts[i+2]*pts[i+2]);
    else v = 0;
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  const rng = vmax - vmin || 1;
  for (let i = 0; i < pts.length; i += 3) {
    let v, r, g, b;
    if (scheme === 'flat') { cols[i]=0.9; cols[i+1]=0.9; cols[i+2]=0.9; continue; }
    if      (scheme === 'height')   v = pts[i+1];
    else if (scheme === 'forward')  v = pts[i];
    else if (scheme === 'lateral')  v = Math.abs(pts[i+2]);
    else                            v = Math.sqrt(pts[i]*pts[i] + pts[i+1]*pts[i+1] + pts[i+2]*pts[i+2]);
    const t = (v - vmin) / rng; // 0..1
    // Jet-like: blue→cyan→green→yellow→red
    if      (t < 0.25) { r=0;       g=t*4;     b=1; }
    else if (t < 0.5)  { r=0;       g=1;       b=1-(t-0.25)*4; }
    else if (t < 0.75) { r=(t-0.5)*4; g=1;     b=0; }
    else               { r=1;       g=1-(t-0.75)*4; b=0; }
    cols[i]=r; cols[i+1]=g; cols[i+2]=b;
  }
  return cols;
}

function applyPCDColor() {
  if (!pcdCloud) return;
  const pos = pcdCloud.geometry.attributes.position.array;
  pcdCloud.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorPCD(pos, pcdColorScheme), 3));
  pcdCloud.geometry.attributes.color.needsUpdate = true;
}

function getCleanConfig(level) {
  if (level === 'off') return { voxelM: 0.0, maxPoints: 120000 };
  if (level === 'light') return { voxelM: 0.08, maxPoints: 100000 };
  if (level === 'strong') return { voxelM: 0.20, maxPoints: 50000 };
  return { voxelM: 0.12, maxPoints: 80000 }; // medium
}

function voxelMergePoints(positions, voxelM) {
  if (!positions || !positions.length || voxelM <= 0) return positions;

  // One representative point per voxel; new points merge into centroid.
  const vox = new Map();
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    const ix = Math.floor(x / voxelM);
    const iy = Math.floor(y / voxelM);
    const iz = Math.floor(z / voxelM);
    const key = `${ix},${iy},${iz}`;

    const cur = vox.get(key);
    if (!cur) {
      vox.set(key, { sx: x, sy: y, sz: z, n: 1 });
    } else {
      cur.sx += x;
      cur.sy += y;
      cur.sz += z;
      cur.n += 1;
    }
  }

  const merged = new Float32Array(vox.size * 3);
  let j = 0;
  vox.forEach(v => {
    merged[j++] = v.sx / v.n;
    merged[j++] = v.sy / v.n;
    merged[j++] = v.sz / v.n;
  });
  return merged;
}

function decimatePoints(positions, maxPoints) {
  if (!positions || !positions.length) return positions;
  const n = positions.length / 3;
  if (n <= maxPoints) return positions;

  const step = Math.ceil(n / maxPoints);
  const out = new Float32Array(Math.ceil(n / step) * 3);
  let j = 0;
  for (let i = 0; i < positions.length; i += step * 3) {
    out[j++] = positions[i];
    out[j++] = positions[i + 1];
    out[j++] = positions[i + 2];
  }
  return out.subarray(0, j);
}

// ── LiDAR controls wiring ───────────────────────────────────────────────────
document.getElementById('pt-size').addEventListener('input', function() {
  pcdPointSize = this.value / 100;
  document.getElementById('pt-size-val').textContent = pcdPointSize.toFixed(2);
  if (pcdCloud) pcdCloud.material.size = pcdPointSize;
});
document.getElementById('tr-smooth').addEventListener('change', function() {
  threeSmoothLevel = this.value;
  if (threeScene) {
    rebuildThreeScene();
    gotoFrame(currentFrame);
  }
});
document.getElementById('pcd-pose').addEventListener('change', function() {
  pcdPoseMode = this.value;
  gotoFrame(currentFrame);
});
document.getElementById('pt-clean').addEventListener('change', function() {
  pcdCleanLevel = this.value;
  gotoFrame(currentFrame); // Reload current frame cloud with new clean config.
});
document.getElementById('pt-color').addEventListener('change', function() {
  pcdColorScheme = this.value;
  applyPCDColor();
});

async function loadPCD(relativePath, posemat, zAligned) {
  const myToken = ++pcdLoadToken;
  if (pcdCloud) { threeScene.remove(pcdCloud); pcdCloud = null; }
  let buf;
  try {
    const resp = await fetch(relativePath);
    if (!resp.ok) { console.warn('PCD not found:', relativePath); return; }
    buf = await resp.arrayBuffer();
  } catch (e) {
    console.warn('PCD fetch failed (file:// restriction?):', e.message);
    return;
  }

  if (myToken !== pcdLoadToken) return;

  const text = new TextDecoder().decode(buf);
  const lines = text.split('\n');
  const dataIdx = lines.findIndex(l => l.startsWith('DATA'));
  const isASCII = dataIdx >= 0 && lines[dataIdx].trim() === 'DATA ascii';

  let positions;
  if (isASCII) {
    positions = parsePCDASCII(lines, dataIdx);
  } else {
    positions = parsePCDBinary(buf, lines, dataIdx);
  }
  if (!positions || !positions.length) return;

  if (myToken !== pcdLoadToken) return;

  const cfg = getCleanConfig(pcdCleanLevel);
  const srcPts = positions instanceof Float32Array ? positions : Float32Array.from(positions);
  let pts = voxelMergePoints(srcPts, cfg.voxelM);
  pts = decimatePoints(pts, cfg.maxPoints);

  // Reorder to Three.js axes: sensor (x,y,z) → Three.js (x,z,y)
  // (same kissToThree mapping for consistency)
  const threePts = new Float32Array(pts.length);
  for (let i = 0; i < pts.length; i += 3) {
    threePts[i]   = pts[i];      // sensor x → Three.x
    threePts[i+1] = pts[i+2];    // sensor z → Three.y
    threePts[i+2] = pts[i+1];    // sensor y → Three.z
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(threePts, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(colorPCD(threePts, pcdColorScheme), 3));

  const mat = new THREE.PointsMaterial({ size: pcdPointSize, vertexColors: true, sizeAttenuation: true });
  pcdCloud = new THREE.Points(geo, mat);

  // Full pose: S * R_h * M_kitti * S  (S=axis-swap sensor->Three.js, R_h=heading correction)
  // posemat: [r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz]
  if (posemat) {
    const set = DATA.sets[currentSetIdx];
    const c = set.kiss_rot_cos, s = set.kiss_rot_sin;
    const [r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz] = posemat;
    const yVis = zAligned != null ? zAligned : tz;
    const M = new THREE.Matrix4();
    if (pcdPoseMode === 'full') {
      M.set(
        c*r00-s*r10,    c*r02-s*r12,    c*r01-s*r11,    c*tx-s*ty,   // Three.x = east
        r20,            r22,            r21,             yVis,         // Three.y = aligned altitude
        -(s*r00+c*r10), -(s*r02+c*r12), -(s*r01+c*r11), -(s*tx+c*ty), // Three.z = -north
        0,              0,              0,               1
      );
    } else {
      // Stable mode: translation only. Reduces pitch/roll ghosting on uneven terrain.
      M.set(
        1, 0, 0, c*tx - s*ty,
        0, 1, 0, yVis,
        0, 0, 1, -(s*tx + c*ty),
        0, 0, 0, 1
      );
    }
    pcdCloud.matrix.copy(M);
    pcdCloud.matrixAutoUpdate = false;
  }

  if (myToken !== pcdLoadToken) return;
  if (pcdCloud && threeScene.children.includes(pcdCloud)) threeScene.remove(pcdCloud);
  threeScene.add(pcdCloud);
}

function parsePCDASCII(lines, dataIdx) {
  const v = [];
  for (let i = dataIdx + 1; i < lines.length; i++) {
    const p = lines[i].trim().split(/\s+/);
    if (p.length >= 3) {
      const x = +p[0], y = +p[1], z = +p[2];
      if (isFinite(x) && isFinite(y) && isFinite(z)) v.push(x, y, z);
    }
  }
  return v;
}

function parsePCDBinary(buf, lines, dataIdx) {
  const getF = k => { const l = lines.find(l2 => l2.startsWith(k + ' ')); return l ? l : ''; };
  const fields = getF('FIELDS').split(/\s+/).slice(1);
  const sizes  = getF('SIZE').split(/\s+/).slice(1).map(Number);
  const full   = new TextDecoder().decode(buf);
  const headerEnd = full.indexOf('DATA binary\n') + 'DATA binary\n'.length;
  if (headerEnd < 'DATA binary\n'.length) { console.warn('binary PCD header not found'); return []; }
  const rowSize = sizes.reduce((a, b) => a + b, 0);
  const xi = fields.indexOf('x'), yi = fields.indexOf('y'), zi = fields.indexOf('z');
  if (xi < 0 || yi < 0 || zi < 0) { console.warn('x/y/z not found'); return []; }
  const xOff = sizes.slice(0, xi).reduce((a, b) => a + b, 0);
  const yOff = sizes.slice(0, yi).reduce((a, b) => a + b, 0);
  const zOff = sizes.slice(0, zi).reduce((a, b) => a + b, 0);
  const view = new DataView(buf, headerEnd);
  const n = Math.floor((buf.byteLength - headerEnd) / rowSize);
  const v = [];
  for (let i = 0; i < n; i++) {
    const x = view.getFloat32(i * rowSize + xOff, true);
    const y = view.getFloat32(i * rowSize + yOff, true);
    const z = view.getFloat32(i * rowSize + zOff, true);
    if (isFinite(x) && isFinite(y) && isFinite(z)) v.push(x, y, z);
  }
  return v;
}

// ── init ───────────────────────────────────────────────────────────────────
buildTabs();
document.getElementById('scrubber').max = DATA.sets[0].frames.length - 1;
resizeCanvas();
gotoFrame(0);
"""

def render_html(json_payload):
    js = _JS.replace("/*DATA_PLACEHOLDER*/null/*END_PLACEHOLDER*/", json_payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Robotics Dataset Visualization</title>
  <style>{_CSS}</style>
</head>
<body>
{_HTML}
<script>
{js}
</script>
</body>
</html>"""

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pairs = discover_pairs(script_dir)
    if not pairs:
        print("No dataset pairs found.")
        sys.exit(1)
    print(f"Found {len(pairs)} pair(s).")

    all_sets = []
    for p in pairs:
        print(f"  Loading Set {p['label']}...")
        d_rows = load_drone_csv(p["drone"], p["suffix"])
        h_rows = load_husky_csv(p["husky"], p["suffix"], p["has_poses"])
        s = build_set_json(p["label"], p["suffix"], script_dir, d_rows, h_rows)
        all_sets.append(s)
        print(f"    {len(s['frames'])} frames  "
              f"pose_trail={len(s['pose_trail'])}  "
              f"kiss_gps={len(s['kiss_gps_trail'])}  "
              f"has_pcd={s['has_pcd']}")

    payload = json.dumps({"sets": all_sets}, separators=(',', ':'), allow_nan=False)
    html    = render_html(payload)
    out     = os.path.join(script_dir, "visualization.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nWritten: {out}  ({len(payload)//1024} KB)")
    print("Open visualization.html in browser.")
    print("For PCD loading: python -m http.server 8080  then open http://localhost:8080/visualization.html")

if __name__ == "__main__":
    main()
