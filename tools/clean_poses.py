"""
clean_poses.py
--------------
Align KISS-ICP poses with existing .pcd files by timestamp matching.

Inputs:
  ros2_poses_tum.txt   – 2716 poses with timestamps (TUM format)
  ros2_poses_kitti.txt – 2716 poses (KITTI 3x4 matrix, same order)
  pointclouds/         – 1267 .pcd files (timestamps encoded in filename)
  husky_synced_data.csv

Outputs:
  poses_cleaned_kitti.txt  – one KITTI pose per .pcd file
  poses_cleaned_tum.txt    – one TUM pose per .pcd file
  pcd_pose_mapping.csv     – full mapping table
  husky_synced_with_poses.csv – husky_synced_data + matched pose columns
"""

import csv
import os
import bisect

BASE = os.path.dirname(os.path.abspath(__file__))
PCD_DIR = os.path.join(BASE, "pointclouds")
TUM_FILE = os.path.join(BASE, "ros2_poses_tum.txt")
KITTI_FILE = os.path.join(BASE, "ros2_poses_kitti.txt")
SYNCED_CSV = os.path.join(BASE, "husky_synced_data.csv")

OUT_KITTI = os.path.join(BASE, "poses_cleaned_kitti.txt")
OUT_TUM = os.path.join(BASE, "poses_cleaned_tum.txt")
OUT_MAPPING = os.path.join(BASE, "pcd_pose_mapping.csv")
OUT_SYNCED = os.path.join(BASE, "husky_synced_with_poses.csv")

# Maximum allowed timestamp gap (seconds) between a pcd and its matched pose
MAX_TS_GAP = 0.08  # 80 ms — one LiDAR scan interval at 10 Hz is ~100 ms


def pcd_timestamp(fname: str) -> float:
    """Extract timestamp from pcd filename: cloud_XXXXXX_SSSSSSSSSS_NNNNNNNNN.pcd"""
    parts = fname.replace(".pcd", "").split("_")
    return float(parts[2]) + float(parts[3]) / 1e9


def load_tum(path: str):
    """Return list of (timestamp, full_line) tuples."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts = float(line.split()[0])
            poses.append((ts, line))
    return poses


def load_kitti(path: str):
    """Return list of pose strings (one per line)."""
    with open(path) as f:
        return [line.rstrip() for line in f if line.strip()]


def nearest_pose_idx(sorted_ts: list, query_ts: float) -> int:
    """Binary-search for the nearest timestamp index."""
    idx = bisect.bisect_left(sorted_ts, query_ts)
    if idx == 0:
        return 0
    if idx == len(sorted_ts):
        return len(sorted_ts) - 1
    before = sorted_ts[idx - 1]
    after = sorted_ts[idx]
    return idx - 1 if abs(query_ts - before) <= abs(query_ts - after) else idx


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    tum_poses = load_tum(TUM_FILE)
    kitti_poses = load_kitti(KITTI_FILE)
    assert len(tum_poses) == len(kitti_poses), (
        f"TUM/KITTI count mismatch: {len(tum_poses)} vs {len(kitti_poses)}"
    )

    pcd_files = sorted(
        f for f in os.listdir(PCD_DIR) if f.endswith(".pcd")
    )

    tum_timestamps = [ts for ts, _ in tum_poses]

    print(f"Poses (TUM/KITTI): {len(tum_poses)}")
    print(f"PCD files:         {len(pcd_files)}")

    # ── Match each pcd → nearest pose ─────────────────────────────────────────
    mapping = []  # (pcd_file, pcd_ts, pose_idx, pose_ts, gap, kitti_row, tum_row)
    skipped = []

    for fname in pcd_files:
        pcd_ts = pcd_timestamp(fname)
        idx = nearest_pose_idx(tum_timestamps, pcd_ts)
        pose_ts = tum_timestamps[idx]
        gap = abs(pcd_ts - pose_ts)

        if gap > MAX_TS_GAP:
            skipped.append((fname, pcd_ts, gap))
            continue

        mapping.append({
            "pcd_file": fname,
            "pcd_ts": pcd_ts,
            "pose_idx": idx,
            "pose_ts": pose_ts,
            "gap_sec": gap,
            "kitti_row": kitti_poses[idx],
            "tum_row": tum_poses[idx][1],
        })

    print(f"\nMatched:  {len(mapping)}")
    print(f"Skipped (gap > {MAX_TS_GAP}s): {len(skipped)}")
    if skipped:
        for fname, ts, gap in skipped:
            print(f"  SKIP {fname}  gap={gap:.4f}s")

    gaps = [m["gap_sec"] for m in mapping]
    print(f"Timestamp gap — mean: {sum(gaps)/len(gaps)*1000:.1f} ms  "
          f"max: {max(gaps)*1000:.1f} ms")

    # Check for duplicate pose assignments (two pcds → same pose)
    pose_counts: dict = {}
    for m in mapping:
        pose_counts[m["pose_idx"]] = pose_counts.get(m["pose_idx"], 0) + 1
    dups = {k: v for k, v in pose_counts.items() if v > 1}
    if dups:
        print(f"\nWarning: {len(dups)} poses matched by >1 pcd")
        for pidx, cnt in list(dups.items())[:5]:
            print(f"  pose #{pidx} (ts={tum_timestamps[pidx]:.4f}) ← {cnt} pcds")
    else:
        print("No duplicate pose assignments.")

    # ── Write cleaned KITTI poses ─────────────────────────────────────────────
    with open(OUT_KITTI, "w") as f:
        for m in mapping:
            f.write(m["kitti_row"] + "\n")
    print(f"\nWrote {len(mapping)} poses -> {OUT_KITTI}")

    # ── Write cleaned TUM poses ───────────────────────────────────────────────
    with open(OUT_TUM, "w") as f:
        for m in mapping:
            # Replace TUM timestamp with the pcd's own timestamp for precision
            tum_parts = m["tum_row"].split()
            tum_parts[0] = f"{m['pcd_ts']:.9f}"
            f.write(" ".join(tum_parts) + "\n")
    print(f"Wrote {len(mapping)} poses -> {OUT_TUM}")

    # ── Write mapping CSV ─────────────────────────────────────────────────────
    with open(OUT_MAPPING, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pcd_file", "pcd_ts", "pose_idx", "pose_ts", "gap_sec",
                         "kitti_pose", "tum_pose"])
        for m in mapping:
            writer.writerow([
                m["pcd_file"], f"{m['pcd_ts']:.9f}",
                m["pose_idx"], f"{m['pose_ts']:.9f}",
                f"{m['gap_sec']:.6f}",
                m["kitti_row"], m["tum_row"],
            ])
    print(f"Wrote mapping -> {OUT_MAPPING}")

    # ── Enrich husky_synced_data.csv with poses ───────────────────────────────
    # Build lookup: pcd filename → mapping entry
    pcd_to_entry = {m["pcd_file"]: m for m in mapping}

    with open(SYNCED_CSV, newline="") as f:
        synced_rows = list(csv.DictReader(f))
        synced_fieldnames = list(synced_rows[0].keys()) if synced_rows else []

    new_fields = ["pose_idx", "pose_ts", "pose_gap_sec",
                  "kitti_pose", "tum_pose", "pose_matched"]
    out_fields = synced_fieldnames + new_fields

    matched_synced = 0
    for row in synced_rows:
        lf = row.get("lidar_file", "")
        entry = pcd_to_entry.get(lf)
        if entry:
            row["pose_idx"] = entry["pose_idx"]
            row["pose_ts"] = f"{entry['pose_ts']:.9f}"
            row["pose_gap_sec"] = f"{entry['gap_sec']:.6f}"
            row["kitti_pose"] = entry["kitti_row"]
            row["tum_pose"] = entry["tum_row"]
            row["pose_matched"] = "True"
            matched_synced += 1
        else:
            for nf in new_fields:
                row[nf] = ""
            row["pose_matched"] = "False"

    with open(OUT_SYNCED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(synced_rows)

    unmatched_synced = len(synced_rows) - matched_synced
    print(f"\nhusky_synced enriched: {matched_synced} matched, "
          f"{unmatched_synced} unmatched -> {OUT_SYNCED}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n== SUMMARY ==================================================")
    print(f"  Total KISS-ICP poses:        {len(tum_poses)}")
    print(f"  Total .pcd files:            {len(pcd_files)}")
    print(f"  Cleaned pairs (pcd+pose):    {len(mapping)}")
    print(f"  Skipped pcds (bad gap):      {len(skipped)}")
    print(f"  husky_synced rows matched:   {matched_synced}/{len(synced_rows)}")
    print(f"  Output KITTI:  {os.path.basename(OUT_KITTI)}")
    print(f"  Output TUM:    {os.path.basename(OUT_TUM)}")
    print(f"  Output map:    {os.path.basename(OUT_MAPPING)}")
    print(f"  Output synced: {os.path.basename(OUT_SYNCED)}")


if __name__ == "__main__":
    main()
