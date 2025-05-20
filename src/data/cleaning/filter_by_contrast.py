import glob
import io
import os
import random
import shutil
import sqlite3
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ===== CONFIGURATION =====
# DATASET_PATH = "/Users/ewojcik/Code/pwr/MVD/objaverse/filter_test/renders"
# REJECTED_SAMPLES_PATH = "/Users/ewojcik/Code/pwr/MVD/objaverse/filter_test/rejected"
# PROCESSING_QUEUE_PATH = "/Users/ewojcik/Code/pwr/MVD/objaverse/filter_test/queue"
DATASET_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/renders_final"
REJECTED_SAMPLES_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/rejected"
PROCESSING_QUEUE_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/renders"
os.makedirs(REJECTED_SAMPLES_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

DB_PATH = os.path.join(os.path.dirname(REJECTED_SAMPLES_PATH), "processing_status.db")

IMG_SIZE = (1024, 1024)
NUM_OBJECTS = 100000
MAX_NUM_VIEWS = 6

# ===== HEURISTIC THRESHOLDS =====
LOW_CONTRAST_THRESHOLD = 10.0
VISUALIZE_FILTERING = False


def setup_database():
    """Setup SQLite database to track processing status."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        path TEXT PRIMARY KEY,
        processed BOOLEAN,
        is_useful BOOLEAN,
        error TEXT,
        processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    return conn


def wrap_text(text, width=50):
    if not text:
        return "No text available"
    return "\n".join(textwrap.wrap(text, width))


def get_zip_files(data_path: str, limit: int = None) -> List[str]:
    zip_files = sorted(glob.glob(os.path.join(data_path, "*.zip")))

    if limit and len(zip_files) > limit:
        random.seed(42)
        zip_files = random.sample(zip_files, limit)

    return zip_files


def load_images_from_zip(
    zip_path: str, target_size: tuple = IMG_SIZE
) -> Dict[str, Any]:
    object_data = {
        "object_uid": Path(zip_path).stem,
        "images": [],
        "image_paths": [],
        "prompt": "Not provided",
    }

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()

        prompt_files = [f for f in file_list if f.endswith("prompt.txt")]
        if prompt_files:
            with zip_ref.open(prompt_files[0]) as f:
                object_data["prompt"] = f.read().decode("utf-8").strip()
        else:
            object_data["prompt"] = "Not provided"

        png_files = sorted([f for f in file_list if f.endswith(".png")])

        for png_file in png_files:
            with zip_ref.open(png_file) as f:
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data))

                if img.mode == "RGBA":
                    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(white_bg, img)

                img = img.convert("RGB")
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                object_data["images"].append(img)
                object_data["image_paths"].append(png_file)
            continue

    return object_data


def filter_sample(images: List[Image.Image], object_uid: str) -> bool:

    if not images:
        print(f"Object {object_uid}: No images provided for filtering.")
        return False

    sample_passes_all_checks = True
    image_statuses = []

    for i, pil_img in enumerate(images):
        img_description = f"Image {i + 1} (Mode: {pil_img.mode}, Size: {pil_img.size})"

        img_np_rgb = np.array(pil_img.convert("RGB"))
        img_cv_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)

        contrast_score = gray_img.std()
        is_contrast_ok = contrast_score >= LOW_CONTRAST_THRESHOLD

        status_text = f"Contrast: {contrast_score:.2f}"
        if is_contrast_ok:
            status_text += " (OK)"
            print(
                f"Object {object_uid}, {img_description}: Contrast OK ({contrast_score:.2f})"
            )
        else:
            status_text += f" (LOW < {LOW_CONTRAST_THRESHOLD})"
            print(
                f"Object {object_uid}, {img_description}: Contrast LOW ({contrast_score:.2f})"
            )
            sample_passes_all_checks = False

        image_statuses.append(
            {"pil_image": pil_img, "title": f"Img {i + 1}: {status_text}"}
        )

    if VISUALIZE_FILTERING:
        num_images = len(image_statuses)
        if num_images > 0:
            cols = 3
            rows = (num_images + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()

            for idx, status_info in enumerate(image_statuses):
                axes[idx].imshow(status_info["pil_image"])
                axes[idx].set_title(status_info["title"])
                axes[idx].axis("off")

            for idx in range(num_images, len(axes)):
                fig.delaxes(axes[idx])

            fig.suptitle(
                f"Object: {object_uid} - Overall Useful: {sample_passes_all_checks}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    return sample_passes_all_checks


def add_prompt_to_zip(zip_path: str, prompt_text: str) -> bool:
    try:
        temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_dirs = [
            d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))
        ]
        if extracted_dirs:
            content_dir = os.path.join(temp_dir, extracted_dirs[0])
            with open(os.path.join(content_dir, "prompt.txt"), "w") as f:
                f.write(prompt_text)
        else:
            with open(os.path.join(temp_dir, "prompt.txt"), "w") as f:
                f.write(prompt_text)

        temp_zip = os.path.join(temp_dir, "temp.zip")
        with zipfile.ZipFile(temp_zip, "w") as new_zip:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file == "temp.zip":
                        continue

                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_zip.write(file_path, arcname)

        shutil.move(temp_zip, zip_path)

        shutil.rmtree(temp_dir)
        return True

    except Exception:
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def main():
    conn = setup_database()
    cursor = conn.cursor()

    zip_files = get_zip_files(PROCESSING_QUEUE_PATH, limit=NUM_OBJECTS)

    for zip_path in zip_files:
        cursor.execute(
            "INSERT OR IGNORE INTO samples (path, processed) VALUES (?, ?)",
            (zip_path, False),
        )
    conn.commit()

    cursor.execute("SELECT path FROM samples WHERE processed = 0")
    unprocessed = [row[0] for row in cursor.fetchall()]

    print(f"Found {len(unprocessed)} unprocessed samples out of {len(zip_files)} total")

    for zip_path in tqdm(unprocessed, desc="Processing objects"):
        try:
            if not os.path.exists(zip_path):
                cursor.execute(
                    "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                    ("File not found", zip_path),
                )
                conn.commit()
                continue

            object_data = load_images_from_zip(zip_path)
            object_uid = object_data["object_uid"]
            print(
                f"Processing {object_uid}: Loaded {len(object_data['images'])} images."
            )

            if len(object_data["images"]) == 0:
                cursor.execute(
                    "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                    ("No images found", zip_path),
                )
                conn.commit()
                continue

            valid_images = []

            images_to_process = (
                object_data["images"][:MAX_NUM_VIEWS]
                if len(object_data["images"]) > MAX_NUM_VIEWS
                else object_data["images"]
            )

            valid_images = [
                img.convert("RGB") if img.mode != "RGB" else img
                for img in images_to_process
            ]

            if not valid_images:
                cursor.execute(
                    "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                    ("No valid images to process", zip_path),
                )
                conn.commit()
                continue

            is_useful = filter_sample(valid_images, object_uid)
            filename = os.path.basename(zip_path)

            print(f"Object {object_uid}: Filter result - Useful: {is_useful}")

            if is_useful:
                dataset_file_path = os.path.join(DATASET_PATH, filename)
                shutil.move(zip_path, dataset_file_path)

                cursor.execute(
                    "UPDATE samples SET processed = 1, is_useful = ?, path = ? WHERE path = ?",
                    (is_useful, dataset_file_path, zip_path),
                )
            else:
                filtered_path = os.path.join(REJECTED_SAMPLES_PATH, filename)

                shutil.move(zip_path, filtered_path)

                cursor.execute(
                    "UPDATE samples SET processed = 1, is_useful = ?, path = ? WHERE path = ?",
                    (is_useful, filtered_path, zip_path),
                )

            conn.commit()

        except Exception as e:
            print(f"Error processing {zip_path}: {e}")
            cursor.execute(
                "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                (str(e), zip_path),
            )
            conn.commit()

    cursor.execute("SELECT COUNT(*) FROM samples WHERE processed = 1 AND is_useful = 1")
    useful_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM samples WHERE processed = 1 AND is_useful = 0")
    not_useful_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM samples WHERE processed = 1 AND error IS NOT NULL"
    )
    error_count = cursor.fetchone()[0]

    print("Processing complete. Summary:")
    print(f"  - Useful samples: {useful_count}")
    print(f"  - Non-useful samples: {not_useful_count}")
    print(f"  - Errors: {error_count}")

    conn.close()


if __name__ == "__main__":
    main()
