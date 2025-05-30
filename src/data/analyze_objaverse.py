import glob
import io
import logging
import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# !!! IMPORTANT: Adjust this path to your actual dataset location for analysis !!!
# This should point to the directory containing your ~20,000 .zip files.
OBJAVERSE_DATA_PATH = (
    "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/renders_final"
)
MAX_SAMPLES = None  # None for all
CSV_OUTPUT_PATH = "objaverse_eda_results.csv"
MAX_IMAGES_FOR_CONTRAST = 6  # Max images to check for contrast from each zip
IMG_SIZE_FOR_CONTRAST = (
    256,
    256,
)


def get_all_zip_files(data_path: str, max_samples: int | None = None) -> list[str]:
    if not os.path.isdir(data_path):
        logging.error(f"Dataset path does not exist or is not a directory: {data_path}")
        return []

    all_zip_files = sorted(glob.glob(os.path.join(data_path, "*.zip")))

    if not all_zip_files:
        logging.warning(f"No .zip files found in {data_path}")
        return []

    if max_samples and len(all_zip_files) > max_samples:
        logging.info(f"Sampling {max_samples} files from {len(all_zip_files)} total files.")
        random.seed(42) # for reproducibility
        return random.sample(all_zip_files, max_samples)
    
    return all_zip_files


def calculate_contrast(pil_image: Image.Image) -> float:
    try:
        img_np_rgb = np.array(pil_image.convert("RGB"))
        img_cv_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
        return gray_img.std()
    except Exception as e:
        logging.error(f"Error calculating contrast: {e}")
        return 0.0  # Return a default low contrast on error


def analyze_zip_content(zip_path: str) -> dict | None:
    object_uid = Path(zip_path).stem
    logging.debug(f"--- Analyzing ZIP: {object_uid} ---")

    analysis_data = {
        "uuid": object_uid,
        "file_size_bytes": 0,
        "prompt": None,
        "prompt_length": 0,
        "render_count": 0,
        "average_contrast": 0.0,
        "error": None,
    }

    try:
        analysis_data["file_size_bytes"] = os.path.getsize(zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            file_list = zf.namelist()

            # Extract prompt
            prompt_files = [f for f in file_list if f.endswith("prompt.txt")]
            if prompt_files:
                with zf.open(prompt_files[0]) as f_prompt:
                    prompt_content = f_prompt.read().decode("utf-8").strip()
                    analysis_data["prompt"] = prompt_content
                    analysis_data["prompt_length"] = len(prompt_content)
            else:
                logging.warning(f"prompt.txt not found in {zip_path}")
                analysis_data["prompt"] = "NOT_FOUND"  # Mark as not found
                analysis_data["prompt_length"] = 0

            # Count .png files (renders)
            png_files = sorted([f for f in file_list if f.endswith(".png")])
            analysis_data["render_count"] = len(png_files)

            # Calculate average contrast from a sample of images
            if png_files:
                contrast_scores = []
                images_to_load = png_files[:MAX_IMAGES_FOR_CONTRAST]

                for png_file_name in images_to_load:
                    try:
                        with zf.open(png_file_name) as f_img:
                            img_data = f_img.read()
                            pil_img = Image.open(io.BytesIO(img_data))

                            if pil_img.mode == "RGBA":  # Composite RGBA over white
                                white_bg = Image.new(
                                    "RGBA", pil_img.size, (255, 255, 255, 255)
                                )
                                pil_img = Image.alpha_composite(white_bg, pil_img)

                            pil_img = pil_img.convert("RGB")
                            pil_img_resized = pil_img.resize(
                                IMG_SIZE_FOR_CONTRAST, Image.Resampling.LANCZOS
                            )

                            contrast_scores.append(calculate_contrast(pil_img_resized))
                    except Exception as e:
                        logging.error(
                            f"Error processing image {png_file_name} in {zip_path} for contrast: {e}"
                        )

                if contrast_scores:
                    analysis_data["average_contrast"] = np.mean(contrast_scores)
                else:  # Handles case where images exist but all fail processing
                    analysis_data["average_contrast"] = 0.0
            else:  # No png files
                analysis_data["average_contrast"] = 0.0

        return analysis_data

    except zipfile.BadZipFile:
        logging.error(f"Corrupted ZIP file: {zip_path}")
        analysis_data["error"] = "BadZipFile"
        return analysis_data
    except FileNotFoundError:
        logging.error(
            f"File not found: {zip_path} (may have been moved or deleted during processing)"
        )
        analysis_data["error"] = "FileNotFound"
        return analysis_data
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {zip_path}: {e}")
        analysis_data["error"] = str(e)
        return analysis_data


def main():
    logging.info("Starting Objaverse EDA script (Phase 1: Data Extraction)...")

    zip_files_to_process = get_all_zip_files(OBJAVERSE_DATA_PATH, MAX_SAMPLES)

    if not zip_files_to_process:
        logging.info("No ZIP files to process. Exiting.")
        return

    logging.info(
        f"Found {len(zip_files_to_process)} ZIP files to analyze from {OBJAVERSE_DATA_PATH}"
    )

    all_results = []
    for zip_path in tqdm(zip_files_to_process, desc="Analyzing ZIPs"):
        result = analyze_zip_content(zip_path)
        if result:
            all_results.append(result)

    logging.info(
        f"--- Data Extraction Complete. Processed {len(all_results)} ZIP files. ---"
    )

    if not all_results:
        logging.warning("No data was extracted. CSV file will not be generated.")
        return

    df = pd.DataFrame(all_results)

    column_order = [
        "uuid",
        "file_size_bytes",
        "prompt",
        "prompt_length",
        "render_count",
        "average_contrast",
        "error",
    ]
    df = df[column_order]

    try:
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        logging.info(f"Successfully saved EDA results to {CSV_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save CSV file: {e}")


if __name__ == "__main__":
    main()
