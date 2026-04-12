#!/bin/bash
# mount_and_run.sh — Mount USB SSD (if needed) then batch-process all datasets.
#
# Usage:
#   ./mount_and_run.sh                          # default root: /mnt/usbssd/husky_drone_dataset
#   ./mount_and_run.sh /mnt/usbssd/other_root   # custom root
#   ./mount_and_run.sh --dry-run                # list datasets without processing

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MOUNT_POINT="/mnt/usbssd"
MOUNT_DEV="/dev/sda1"
MOUNT_OPTS="uid=1000,gid=1000,fmask=0133,dmask=0022,iocharset=utf8,errors=remount-ro"
DEFAULT_ROOT="$MOUNT_POINT/husky_drone_dataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
PYTHON="${PYTHON:-python}"

# ── Args ──────────────────────────────────────────────────────────────────────
DRY_RUN=false
DATASET_ROOT="$DEFAULT_ROOT"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --*) echo "Unknown option: $arg"; exit 1 ;;
        *) DATASET_ROOT="$arg" ;;
    esac
done

# ── Mount ─────────────────────────────────────────────────────────────────────
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "[mount] $MOUNT_POINT already mounted — skipping"
else
    echo "[mount] Mounting $MOUNT_DEV → $MOUNT_POINT ..."
    sudo mkdir -p "$MOUNT_POINT"
    sudo mount -t exfat "$MOUNT_DEV" "$MOUNT_POINT" -o "$MOUNT_OPTS"
    echo "[mount] Done"
fi

# ── Find datasets ─────────────────────────────────────────────────────────────
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: dataset root not found: $DATASET_ROOT"
    exit 1
fi

mapfile -t DATASETS < <(find "$DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d -name "sensor_record_*" | sort)

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No sensor_record_* directories found in $DATASET_ROOT"
    exit 1
fi

echo ""
echo "Found ${#DATASETS[@]} dataset(s) in $DATASET_ROOT:"
for d in "${DATASETS[@]}"; do
    # Count image files (quick estimate)
    n=$(find "$d" -maxdepth 2 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo "  $(basename "$d")  (~$n images)"
done
echo ""

if $DRY_RUN; then
    echo "[dry-run] Exiting without processing."
    exit 0
fi

# ── Process ───────────────────────────────────────────────────────────────────
FAILED_DATASETS=()
START_TIME=$(date +%s)

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    name="$(basename "$dataset")"
    num=$((i + 1))
    total=${#DATASETS[@]}

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$num/$total] $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $PYTHON "$SCRIPT_DIR/run_batch.py" \
            --input  "$dataset" \
            --config "$CONFIG" \
            --auto; then
        echo "[$num/$total] ✓ $name"
    else
        echo "[$num/$total] ✗ $name — errors above (continuing)"
        FAILED_DATASETS+=("$name")
    fi
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo "════════════════════════════════════════════════════════════"
echo "Finished: $((${#DATASETS[@]} - ${#FAILED_DATASETS[@]}))/${#DATASETS[@]} datasets OK"
printf "Elapsed:  %02d:%02d:%02d\n" $HOURS $MINS $SECS
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    for name in "${FAILED_DATASETS[@]}"; do
        echo "  ✗ $name"
    done
    echo "════════════════════════════════════════════════════════════"
    exit 1
fi
echo "════════════════════════════════════════════════════════════"
