#!/bin/bash
# Pack each dataset folder into a zip containing only:
#   - husky_frames/*.jpg
#   - drone_frames/*.jpg
#   - husky_frames/<name>/annotations/*.json
#   - drone_frames/<name>/annotations/*.json
# Output: /mnt/usbssd/preprocessed_upload/<dataset_name>.zip

SRC="/mnt/usbssd/husky_drone_dataset"
DEST="/mnt/usbssd/preprocessed_upload"
ONEDRIVE_REMOTE="onedrive:Husky Snow Data/annotated_data"

mkdir -p "$DEST"

for dataset_dir in "$SRC"/sensor_record_*_dataset; do
    name=$(basename "$dataset_dir")
    out="$DEST/${name}.zip"

    if [[ -f "$out" ]]; then
        if zip -T "$out" &>/dev/null; then
            echo "[SKIP] $name (zip valid, checking OneDrive ...)"
            rclone copy "$out" "$ONEDRIVE_REMOTE/" --progress
            echo "  -> upload done: $name"
            continue
        else
            echo "[RETRY] $name (corrupt/incomplete zip, re-packing ...)"
            rm -f "$out"
        fi
    fi

    echo "[PACK] $name ..."

    # Build list of files to include (relative paths inside the dataset dir)
    file_list=$(mktemp)

    # Original images
    find "$dataset_dir/husky_frames" -maxdepth 1 -name "*.jpg" >> "$file_list"
    find "$dataset_dir/drone_frames" -maxdepth 1 -name "*.jpg" >> "$file_list"

    # Annotations JSON (only husky_frame_*.json / drone_frame_*.json, not _annotated.json)
    find "$dataset_dir/husky_frames" -path "*/annotations/husky_frame_*.json" >> "$file_list"
    find "$dataset_dir/drone_frames" -path "*/annotations/drone_frame_*.json" >> "$file_list"

    count=$(wc -l < "$file_list")
    echo "  -> $count files"

    # zip with paths relative to SRC so the zip preserves the dataset subfolder
    cd "$SRC"
    zip -q "$out" -@ < <(sed "s|$SRC/||" "$file_list")
    echo "  -> saved: $out"

    rm "$file_list"

    echo "  -> uploading to OneDrive ..."
    rclone copy "$out" "$ONEDRIVE_REMOTE/" --progress
    echo "  -> upload done: $name"
done

echo ""
echo "All done. Zips in: $DEST, uploaded to: $ONEDRIVE"
