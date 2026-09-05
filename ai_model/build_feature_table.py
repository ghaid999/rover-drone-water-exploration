import numpy as np
import glob
import os
import csv

IMAGES_DIR = "../data/swed_sample/train/images"
LABELS_DIR = "../data/swed_sample/train/labels"
OUTPUT_CSV = "feature_table.csv"

image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.npy")))

rows = []
skipped = 0

for image_path in image_files:
    filename = os.path.basename(image_path)
    # e.g. ..._image_2_34.npy -> find matching ..._chip_2_34.npy
    suffix = filename.split("_image_")[-1]  # "2_34.npy"
    prefix = filename.split("_image_")[0]   # "..._20170409T110529"
    label_filename = f"{prefix}_chip_{suffix}"
    label_path = os.path.join(LABELS_DIR, label_filename)

    if not os.path.exists(label_path):
        skipped += 1
        continue

    img = np.load(image_path).astype(float)
    lbl = np.load(label_path)[0]  # shape (256,256)

    green = img[:, :, 2]
    nir = img[:, :, 7]
    swir1 = img[:, :, 10]

    ndwi = (green - nir) / (green + nir + 1e-6)
    ndsi = (green - swir1) / (green + swir1 + 1e-6)

    mean_ndwi = float(np.mean(ndwi))
    mean_ndsi = float(np.mean(ndsi))
    water_fraction = float(np.mean(lbl))
    label = 1 if water_fraction >= 0.5 else 0

    rows.append([filename, mean_ndwi, mean_ndsi, water_fraction, label])

print(f"Processed {len(rows)} tiles, skipped {skipped} (no matching label found)")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tile_id", "mean_ndwi", "mean_ndsi", "water_fraction", "label"])
    writer.writerows(rows)

print(f"Saved feature table to {OUTPUT_CSV}")
