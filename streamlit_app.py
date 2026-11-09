"""
Kaggle Dataset Analyzer & Structure Organizer (Ultra-Fast Version)
Optimized for high-speed file system traversal and minimal overhead.
"""

import os
from pathlib import Path
import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_EXTENSIONS = VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS)

def inspect_kaggle_dataset_fast(base_dir="/kaggle/input"):
    print(f"Scanning Kaggle input directory (Fast Mode): {base_dir}")
    root = Path(base_dir)
    if not root.exists():
        print(f"Directory {base_dir} does not exist. Running in local simulation mode.")
        root = Path(".")

    records = []
    
    # استخدام os.walk للبحث السريع وتفادي بطء الفحص العشوائي
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in TARGET_EXTENSIONS:
                file_type = "Video" if ext in VIDEO_EXTENSIONS else "Image"
                full_path = os.path.join(dirpath, filename)
                try:
                    # قراءة الحجم بطريقة أسرع وأكثر أماناً
                    size_bytes = os.path.getsize(full_path)
                    size_mb = round(size_bytes / (1024 * 1024), 2)
                except (PermissionError, FileNotFoundError):
                    size_mb = 0.0

                records.append({
                    "file_name": filename,
                    "file_type": file_type,
                    "relative_path": full_path,
                    "size_mb": size_mb
                })

    return pd.DataFrame(records)

def generate_markdown_report_fast(df, output_path="dataset_summary.md"):
    if df.empty:
        md_content = "# Kaggle Dataset Inspection & Summary Report\n\nNo matching video or image files found.\n"
    else:
        video_count = len(df[df['file_type'] == 'Video'])
        image_count = len(df[df['file_type'] == 'Image'])
        
        md_content = "# Kaggle Dataset Inspection & Summary Report\n\n"
        md_content += "## Overview\n"
        md_content += f"- **Total Files Discovered:** {len(df)}\n"
        md_content += f"- **Videos Count:** {video_count}\n"
        md_content += f"- **Images Count:** {image_count}\n\n"
        
        md_content += "## File Inventory Table\n"
        md_content += "| File Name | Type | Size (MB) | Path |\n"
        md_content += "| :--- | :--- | :--- | :--- |\n"
        
        rows = []
        for _, row in df.iterrows():
            rows.append(f"| {row['file_name']} | {row['file_type']} | {row['size_mb']} | `{row['relative_path']}` |")
        md_content += "\n".join(rows) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report successfully generated at: {output_path}")

if __name__ == "__main__":
    dataset_df = inspect_kaggle_dataset_fast()
    generate_markdown_report_fast(dataset_df)
