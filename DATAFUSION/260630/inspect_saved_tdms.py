from pathlib import Path
import json
from nptdms import TdmsFile

RAW_DIR = Path("/mnt/a/ftk3187/HAMMER/DED/RGB/raw_test")
WORK_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630")
OUT_DIR = WORK_DIR / "tdms_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SHAPES = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 1024),
    (1440, 1080),
    (1600, 1200),
    (1920, 1080),
    (2048, 1088),
    (2048, 2048),
    (2448, 2048),
    (4096, 2160),
]

tdms_files = sorted(RAW_DIR.rglob("*.tdms"))

print(f"RAW_DIR: {RAW_DIR}")
print(f"OUT_DIR: {OUT_DIR}")
print(f"Found {len(tdms_files)} TDMS files")

summary_rows = []

for tdms_path in tdms_files:
    print(f"\nInspecting: {tdms_path.name}")

    file_info = {
        "file": str(tdms_path),
        "file_size_gb": tdms_path.stat().st_size / 1024**3,
        "groups": [],
    }

    try:
        with TdmsFile.open(tdms_path) as tdms:
            for group in tdms.groups():
                print(f"  Group: {group.name}")

                group_info = {
                    "name": group.name,
                    "properties": dict(group.properties),
                    "channels": [],
                }

                for ch in group.channels():
                    n = len(ch)
                    print(f"    Channel: {ch.name}, dtype={ch.dtype}, length={n:,}")

                    possible_shapes = []
                    for width, height in IMAGE_SHAPES:
                        gray_size = width * height
                        rgb_size = width * height * 3

                        if n % gray_size == 0:
                            possible_shapes.append({
                                "type": "grayscale",
                                "width": width,
                                "height": height,
                                "frames": n // gray_size,
                            })

                        if n % rgb_size == 0:
                            possible_shapes.append({
                                "type": "rgb",
                                "width": width,
                                "height": height,
                                "frames": n // rgb_size,
                            })

                    if possible_shapes:
                        print(f"      possible_shapes: {possible_shapes}")

                    ch_info = {
                        "name": ch.name,
                        "dtype": str(ch.dtype),
                        "length": n,
                        "properties": dict(ch.properties),
                        "possible_shapes": possible_shapes,
                    }

                    group_info["channels"].append(ch_info)

                    summary_rows.append({
                        "file": tdms_path.name,
                        "size_gb": file_info["file_size_gb"],
                        "group": group.name,
                        "channel": ch.name,
                        "dtype": str(ch.dtype),
                        "length": n,
                        "possible_shapes": possible_shapes,
                    })

                file_info["groups"].append(group_info)

        out_json = OUT_DIR / f"{tdms_path.stem}_inspection.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(file_info, f, indent=2, ensure_ascii=False, default=str)

        print(f"  Saved: {out_json}")

    except Exception as e:
        print(f"  ERROR: {e}")
        summary_rows.append({
            "file": tdms_path.name,
            "error": str(e),
        })

summary_json = OUT_DIR / "summary_all_tdms.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, indent=2, ensure_ascii=False, default=str)

summary_txt = OUT_DIR / "summary_all_tdms.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    for row in summary_rows:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

print("\nSaved summary:")
print(summary_json)
print(summary_txt)
