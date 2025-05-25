import os
import glob
import zipfile
import tempfile
import shutil
import subprocess
import json
import hashlib
import time
import random
import platform
import sys
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fire
import fsspec
import GPUtil
from icecream import ic
from loguru import logger


def get_uid_from_str(file_identifier_str: str) -> str:
    """Placeholder UID generator if objaverse is not available."""
    return hashlib.md5(file_identifier_str.encode()).hexdigest()

# Configure Loguru
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(lambda msg: ic(msg), level="DEBUG", format="{time} {level} {message}")


def calculate_sha256(filepath: str) -> str:
    """Computes SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def log_gso_processed_object(csv_filename: str, log_dir: str, *args) -> None:
    """Log when a GSO object is done being processed."""
    processed_args = ",".join([str(arg) for arg in args])
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, csv_filename)
    # Log header if file is new
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("timestamp,file_identifier,sha256,status_or_extra_info\n")
    
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{processed_args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure."""
    for root, _, files in os.walk(path):
        for file_in_zip in files:
            arcname = os.path.join(os.path.basename(root), file_in_zip)
            ziph.write(os.path.join(root, file_in_zip), arcname=arcname)


def render_single_gso_object(
    gso_obj_path: str,
    file_identifier: str,
    sha256: str,
    gso_object_metadata: Dict[str, Any],
    render_dir: str, # Base output directory for final .zip
    parsed_gpu_devices: Union[int, List[int]],
    render_timeout: int,
    only_northern_hemisphere: bool,
    blender_script_path_global: str,
    blender_module_load_command: str,
    successful_log_file: str = "gso-successful.csv",
    failed_log_file: str = "gso-failed.csv",
) -> bool:
    """
    Renders a single GSO object using Blender, mirrors main.py's handle_found_object.
    """
    save_uid = get_uid_from_str(file_identifier)
    log_base_dir = os.path.join(render_dir, "logs")

    logger.info(f"Processing GSO object: {file_identifier} (UID: {save_uid})")
    ic(gso_obj_path, save_uid)

    # Determine GPU for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(parsed_gpu_devices, int) and parsed_gpu_devices > 0:
        num_gpus = parsed_gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(parsed_gpu_devices, list) and parsed_gpu_devices:
        gpu_i = random.choice(parsed_gpu_devices)
    elif (isinstance(parsed_gpu_devices, int) and parsed_gpu_devices == 0) or \
         (isinstance(parsed_gpu_devices, list) and not parsed_gpu_devices):
        using_gpu = False
    else:
        logger.error(f"Invalid gpu_devices configuration: {parsed_gpu_devices}")
        log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "invalid_gpu_config")
        return False

    # This temporary directory is for Blender's script and its output for this object
    with tempfile.TemporaryDirectory() as processing_temp_dir:
        blender_output_dir = os.path.join(processing_temp_dir, save_uid)
        os.makedirs(blender_output_dir, exist_ok=True)
        ic(f"Blender processing temp dir: {processing_temp_dir}")
        ic(f"Blender output target dir: {blender_output_dir}")

        # Copy blender_script.py to the processing_temp_dir
        copied_blender_script_path = os.path.join(processing_temp_dir, "blender_script.py")
        try:
            shutil.copy(blender_script_path_global, copied_blender_script_path)
            ic(f"Copied blender_script.py to {copied_blender_script_path}")
        except Exception as e:
            logger.error(f"Failed to copy blender_script.py: {e}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "blender_script_copy_fail")
            return False

        # Construct Blender arguments
        blender_args_list = [
            f"--object_path '{gso_obj_path}'",
            f"--output_dir '{blender_output_dir}'",
            "--verbose"
        ]

        if platform.system() == "Linux" and using_gpu:
            blender_args_list.append("--engine BLENDER_EEVEE")
        elif platform.system() == "Darwin" or (platform.system() == "Linux" and not using_gpu):
            blender_args_list.append("--engine CYCLES")
        else:
            logger.error(f"Unsupported platform or GPU configuration for Blender engine: {platform.system()}, GPU: {using_gpu}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "unsupported_platform_engine")
            return False

        if only_northern_hemisphere:
            blender_args_list.append("--only_northern_hemisphere")
        
        blender_args_str = " ".join(blender_args_list)

        # Construct the full command
        # Base command: blender --background --python script -- args
        # Full command example: module load Blender/... && blender --background --python script.py -- --object_path ...
        # Ensure environment variables like DISPLAY are handled if necessary (usually not for --background)
        
        # If a blender_module_load_command is provided, prepend it
        full_command = f"blender --background --python {copied_blender_script_path} -- {blender_args_str}"
        if blender_module_load_command:
            full_command = f"{blender_module_load_command} && {full_command}"
        
        if using_gpu and platform.system() == "Linux":
             # Prepend CUDA_VISIBLE_DEVICES for Linux GPU rendering
            env_vars = os.environ.copy()
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_i)
            logger.info(f"Using GPU {gpu_i} for rendering. CUDA_VISIBLE_DEVICES={env_vars['CUDA_VISIBLE_DEVICES']}")
        else:
            env_vars = os.environ.copy()

        ic(f"Executing Blender command: {full_command}")

        try:
            result = subprocess.run(
                ["bash", "-c", full_command], # Use bash -c to handle module loads and &&
                timeout=render_timeout,
                check=False, # Don't raise exception for non-zero exit codes, check manually
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False, # Handle binary output
                env=env_vars
            )

            # Log Blender output
            ic("Blender Return code:", result.returncode)
            if result.stdout:
                try:
                    stdout_text = result.stdout.decode('utf-8', errors='replace')
                    ic("Blender STDOUT (first 1000 chars):", stdout_text[:1000])
                except Exception as e:
                    ic(f"Error decoding/printing Blender STDOUT: {e}")
            if result.stderr:
                try:
                    stderr_text = result.stderr.decode('utf-8', errors='replace')
                    ic("Blender STDERR (first 1000 chars):", stderr_text[:1000])
                except Exception as e:
                    ic(f"Error decoding/printing Blender STDERR: {e}")
            
            # Check for blender_debug.log
            blender_debug_log_path = os.path.join(blender_output_dir, "blender_debug.log")
            if os.path.exists(blender_debug_log_path):
                with open(blender_debug_log_path, 'r', encoding='utf-8', errors='replace') as f_debug:
                    ic("Contents of blender_debug.log:", f_debug.read())
            else:
                ic("blender_debug.log not found.")

            # Check for render_complete.txt
            render_complete_txt_path = os.path.join(blender_output_dir, "render_complete.txt")
            if os.path.exists(render_complete_txt_path):
                with open(render_complete_txt_path, 'r', encoding='utf-8', errors='replace') as f_complete:
                    ic("Contents of render_complete.txt:", f_complete.read())
            else:
                ic("render_complete.txt not found.")
            
            # List files in blender_output_dir for verification
            if os.path.exists(blender_output_dir):
                ic(f"Files in blender_output_dir ({blender_output_dir}):", os.listdir(blender_output_dir))
            else:
                ic(f"Blender output directory {blender_output_dir} does not exist after script execution.")
                log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "blender_output_dir_missing")
                return False


            if result.returncode != 0:
                logger.error(f"Blender script failed for {file_identifier} with return code {result.returncode}.")
                log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, f"blender_error_code_{result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Blender process timed out for {file_identifier} after {render_timeout} seconds.")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "timeout")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while running Blender for {file_identifier}: {e}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, f"blender_unexpected_error_{type(e).__name__}")
            return False

        # Read actual number of renders
        actual_num_renders = 0
        num_renders_txt_path = os.path.join(blender_output_dir, "num_renders.txt")
        if os.path.exists(num_renders_txt_path):
            try:
                with open(num_renders_txt_path, 'r') as f:
                    actual_num_renders = int(f.read().strip())
                ic(f"Actual number of renders from num_renders.txt: {actual_num_renders}")
            except ValueError:
                logger.error(f"Could not parse num_renders.txt for {file_identifier}")
                log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "num_renders_parse_error")
                return False
        else:
            logger.error(f"num_renders.txt not found for {file_identifier}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "num_renders_txt_missing")
            return False
        
        if actual_num_renders == 0: # Should be 6, 8, or 12
            logger.error(f"Actual number of renders is 0, which is invalid, for {file_identifier}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "invalid_actual_num_renders_0")
            return False

        # Check output files
        png_files = glob.glob(os.path.join(blender_output_dir, "*.png"))
        npy_files = glob.glob(os.path.join(blender_output_dir, "*.npy"))
        metadata_json_path = os.path.join(blender_output_dir, "metadata.json")

        ic(f"Found files: {len(png_files)} PNGs, {len(npy_files)} NPYs. Expected: {actual_num_renders} of each.")
        ic(f"Metadata JSON path: {metadata_json_path}, Exists: {os.path.exists(metadata_json_path)}")

        if not (
            len(png_files) == actual_num_renders
            and len(npy_files) == actual_num_renders
            and os.path.exists(metadata_json_path)
        ):
            logger.error(
                f"Render output validation failed for {file_identifier}. "
                f"Expected {actual_num_renders} PNGs/NPYs and metadata.json. "
                f"Got {len(png_files)} PNGs, {len(npy_files)} NPYs, metadata.json exists: {os.path.exists(metadata_json_path)}."
            )
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "output_file_validation_failed")
            return False

        # Update metadata.json
        try:
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                blender_metadata = json.load(f)
            
            # Log texture info from blender_metadata for validation
            ic(f"Blender metadata 'has_visible_textures': {blender_metadata.get('has_visible_textures')}")
            ic(f"Blender metadata 'missing_textures': {blender_metadata.get('missing_textures')}")

            blender_metadata["sha256"] = sha256
            blender_metadata["file_identifier"] = file_identifier
            blender_metadata["save_uid"] = save_uid
            blender_metadata["num_renders"] = actual_num_renders 
            # Merge GSO-specific metadata (could overwrite if keys clash, GSO takes precedence)
            blender_metadata.update(gso_object_metadata) 

            with open(metadata_json_path, "w", encoding="utf-8") as f:
                json.dump(blender_metadata, f, indent=2, sort_keys=True)
            ic("Updated metadata.json successfully.")
        except Exception as e:
            logger.error(f"Failed to update metadata.json for {file_identifier}: {e}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "metadata_update_fail")
            return False

        # Zip the blender_output_dir
        final_zip_path_temp = f"{blender_output_dir}.zip"
        try:
            with zipfile.ZipFile(final_zip_path_temp, "w", zipfile.ZIP_DEFLATED) as ziph:
                zipdir(blender_output_dir, ziph)
            ic(f"Successfully created zip: {final_zip_path_temp}")
        except Exception as e:
            logger.error(f"Failed to zip output for {file_identifier}: {e}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "zip_creation_fail")
            return False
        
        # Move the zip to the final render directory
        try:
            fs, path_prefix = fsspec.core.url_to_fs(render_dir)
            renders_subdir = os.path.join(path_prefix, "renders")
            fs.makedirs(renders_subdir, exist_ok=True)
            
            final_zip_dest = os.path.join(renders_subdir, f"{save_uid}.zip")
            fs.put(final_zip_path_temp, final_zip_dest)
            ic(f"Moved final zip to: {final_zip_dest}")
        except Exception as e:
            logger.error(f"Failed to move final zip for {file_identifier} to {render_dir}: {e}")
            log_gso_processed_object(failed_log_file, log_base_dir, file_identifier, sha256, "zip_move_fail")
            return False

        log_gso_processed_object(successful_log_file, log_base_dir, file_identifier, sha256, "success")
        logger.info(f"Successfully rendered and packaged GSO object: {file_identifier}")
        return True


def render_gso_dataset(
    gso_dataset_path: str,
    render_dir: str,
    gpu_devices: Optional[Union[int, List[int]]] = None,
    render_timeout: int = 900, # 15 minutes
    only_northern_hemisphere: bool = False,
    blender_module_load: str = "module load Blender/3.5.0-linux-x86_64-CUDA-11.7.0", # Command to load Blender
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> None:
    """
    Processes and renders 3D models from the Google Scanned Objects (GSO) dataset.

    Args:
        gso_dataset_path: Path to the root directory containing GSO .zip files.
        render_dir: Path to save rendered outputs (final .zip files) and logs.
        gpu_devices: GPU device(s) to use. Int for count (0 for CPU), list of IDs. Defaults to all available.
        render_timeout: Timeout in seconds for each Blender rendering process.
        only_northern_hemisphere: If True, renders only the northern hemisphere of objects.
        blender_module_load: The shell command to load the Blender module (e.g., "module load Blender/...").
                             Set to empty string if Blender is in PATH and no module load is needed.
        start_index: Index of the GSO zip file to start processing from (for resuming).
        end_index: Index of the GSO zip file to stop processing at (exclusive).
    """
    logger.info("Starting GSO dataset rendering.")
    ic(gso_dataset_path, render_dir, gpu_devices, render_timeout, only_northern_hemisphere, blender_module_load)

    # Locate blender_script.py (expected to be in the same directory as this script)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    blender_script_path_global = os.path.join(current_script_dir, "blender_script.py")
    if not os.path.exists(blender_script_path_global):
        logger.error(f"CRITICAL: blender_script.py not found at {blender_script_path_global}. Exiting.")
        return
    ic(f"Found blender_script.py at: {blender_script_path_global}")

    # Determine GPU devices
    parsed_gpu_devices: Union[int, List[int]]
    if gpu_devices is None:
        try:
            parsed_gpu_devices = len(GPUtil.getGPUs())
            logger.info(f"GPUtil found {parsed_gpu_devices} GPUs. Using all for rendering if > 0.")
        except Exception as e:
            logger.warning(f"GPUtil failed to get GPU count ({e}), defaulting to CPU rendering (0). Install GPUtil for GPU support.")
            parsed_gpu_devices = 0 # Fallback to CPU rendering
    elif isinstance(gpu_devices, int):
        parsed_gpu_devices = gpu_devices
    elif isinstance(gpu_devices, list):
        parsed_gpu_devices = gpu_devices
    else:
        logger.error(f"Invalid gpu_devices value: {gpu_devices}. Defaulting to CPU (0).")
        parsed_gpu_devices = 0
    
    if parsed_gpu_devices == 0 :
         logger.info("Configured for CPU rendering.")
    elif isinstance(parsed_gpu_devices, int) and parsed_gpu_devices > 0:
         logger.info(f"Configured to use up to {parsed_gpu_devices} GPU(s) by index.")
    elif isinstance(parsed_gpu_devices, list):
         logger.info(f"Configured to use specific GPU IDs: {parsed_gpu_devices}.")


    # Prepare output directories
    renders_output_dir = os.path.join(render_dir, "renders")
    logs_output_dir = os.path.join(render_dir, "logs")
    os.makedirs(renders_output_dir, exist_ok=True)
    os.makedirs(logs_output_dir, exist_ok=True)
    ic(f"Final renders will be saved in: {renders_output_dir}")
    ic(f"Logs will be saved in: {logs_output_dir}")

    gso_zip_files = sorted(glob.glob(os.path.join(gso_dataset_path, "*.zip")))
    if not gso_zip_files:
        logger.error(f"No .zip files found in {gso_dataset_path}. Exiting.")
        return
    
    logger.info(f"Found {len(gso_zip_files)} GSO .zip files.")

    # Apply slicing for start/end index
    effective_end_index = end_index if end_index is not None else len(gso_zip_files)
    files_to_process = gso_zip_files[start_index:effective_end_index]
    logger.info(f"Processing files from index {start_index} to {effective_end_index-1} ({len(files_to_process)} files).")


    for i, zip_file_path in enumerate(files_to_process):
        current_file_index_abs = start_index + i
        logger.info(f"--- Processing file {current_file_index_abs + 1}/{len(gso_zip_files)}: {os.path.basename(zip_file_path)} ---")

        file_identifier = os.path.splitext(os.path.basename(zip_file_path))[0]
        save_uid = get_uid_from_str(file_identifier)

        # Check if already rendered
        final_zip_location = os.path.join(renders_output_dir, f"{save_uid}.zip")
        if os.path.exists(final_zip_location):
            logger.info(f"Object {file_identifier} (UID: {save_uid}) already rendered. Skipping.")
            log_gso_processed_object("gso-skipped.csv", logs_output_dir, file_identifier, "N/A", "already_rendered")
            continue

        # Temporary directory for unzipping this specific GSO archive
        with tempfile.TemporaryDirectory() as gso_unzip_temp_dir:
            ic(f"Temporary directory for unzipping {os.path.basename(zip_file_path)}: {gso_unzip_temp_dir}")
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(gso_unzip_temp_dir)
                ic(f"Successfully unzipped to {gso_unzip_temp_dir}")
            except Exception as e:
                logger.error(f"Failed to unzip {zip_file_path}: {e}")
                log_gso_processed_object("gso-failed.csv", logs_output_dir, file_identifier, "N/A", f"unzip_error_{type(e).__name__}")
                continue

            # Define expected relative paths within the unzipped GSO structure
            obj_rel_path = "meshes/model.obj"
            texture_original_rel_path = "materials/textures/texture.png" # User-provided original path
            texture_target_dir_rel_path = "meshes" # User wants texture here
            texture_filename = "texture.png"
            
            # Construct absolute paths within gso_unzip_temp_dir
            abs_obj_path = os.path.join(gso_unzip_temp_dir, obj_rel_path)
            abs_texture_original_path = os.path.join(gso_unzip_temp_dir, texture_original_rel_path)
            abs_texture_target_path = os.path.join(gso_unzip_temp_dir, texture_target_dir_rel_path, texture_filename)

            ic(f"Expected OBJ path: {abs_obj_path}")
            ic(f"Expected original Texture path: {abs_texture_original_path}")
            ic(f"Target Texture path (in meshes dir): {abs_texture_target_path}")

            # Validate file existence
            if not os.path.exists(abs_obj_path):
                logger.error(f"model.obj not found at {abs_obj_path} for {zip_file_path}")
                log_gso_processed_object("gso-failed.csv", logs_output_dir, file_identifier, "N/A", "obj_not_found")
                continue
            
            if not os.path.exists(abs_texture_original_path):
                logger.warning(f"Original texture.png not found at {abs_texture_original_path} for {zip_file_path}. Proceeding without texture move.")
                # Blender script might use a default color or fail if texture is critical and not found by MTL.
            else:
                # Move texture.png to meshes/ folder
                try:
                    os.makedirs(os.path.dirname(abs_texture_target_path), exist_ok=True)
                    shutil.move(abs_texture_original_path, abs_texture_target_path)
                    ic(f"Moved texture from {abs_texture_original_path} to {abs_texture_target_path}")
                except Exception as e:
                    logger.error(f"Failed to move texture for {zip_file_path}: {e}")
                    log_gso_processed_object("gso-failed.csv", logs_output_dir, file_identifier, "N/A", f"texture_move_error_{type(e).__name__}")
                    # Decide if to continue or skip. For now, try rendering anyway.
            
            # Calculate SHA256 of the model.obj
            try:
                obj_sha256 = calculate_sha256(abs_obj_path)
                ic(f"SHA256 for {os.path.basename(abs_obj_path)}: {obj_sha256}")
            except Exception as e:
                logger.error(f"Failed to calculate SHA256 for {abs_obj_path}: {e}")
                log_gso_processed_object("gso-failed.csv", logs_output_dir, file_identifier, "N/A", f"sha256_error_{type(e).__name__}")
                continue
                
            gso_specific_metadata = {
                "source": "GSO",
                "original_zip_filename": os.path.basename(zip_file_path),
                "gso_dataset_path": gso_dataset_path,
            }

            # Call the rendering function for this object
            render_success = render_single_gso_object(
                gso_obj_path=abs_obj_path,
                file_identifier=file_identifier,
                sha256=obj_sha256,
                gso_object_metadata=gso_specific_metadata,
                render_dir=render_dir, # Pass base render_dir
                parsed_gpu_devices=parsed_gpu_devices,
                render_timeout=render_timeout,
                only_northern_hemisphere=only_northern_hemisphere,
                blender_script_path_global=blender_script_path_global,
                blender_module_load_command=blender_module_load
            )

            if render_success:
                logger.info(f"Successfully processed {file_identifier}")
            else:
                logger.error(f"Failed to process {file_identifier}")
            
            # Optional: small delay
            time.sleep(1) 

    logger.info("GSO dataset rendering script finished.")


if __name__ == "__main__":
    fire.Fire(render_gso_dataset)
