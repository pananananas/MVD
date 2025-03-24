from typing import Any, Dict, List, Literal, Optional, Union
from functools import partial
import multiprocessing
import subprocess
import platform
import tempfile
import zipfile
import random
import glob
import json
import time
import os
import shutil
import http.client
import urllib.error
import socket
import sys

import fire
import fsspec
import GPUtil
import tenacity
import pandas as pd
from icecream import ic
from loguru import logger
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type
)

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str


def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.

    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.

    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    dirname = os.path.expanduser(f"/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure.

    Args:
        path (str): Path to the directory to zip.
        ziph (zipfile.ZipFile): ZipFile handler object to write to.

    Returns:
        None
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # this ensures the structure inside the zip starts at folder/
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)


def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    save_uid = get_uid_from_str(file_identifier)
    args = f"--object_path '{local_path}' --output_dir"
    ic(local_path, save_uid)
    
    # Log the file type for debugging
    file_extension = os.path.splitext(local_path)[1].lower().lstrip('.')
    ic(f"File extension: {file_extension}")
    
    # Store the repository path for cleanup
    repo_path = None
    if 'github.com' in file_identifier:
        # For GitHub, get the repo directory (parent directory of the file)
        repo_path = os.path.dirname(os.path.dirname(local_path))
        ic(f"GitHub repository path for cleanup: {repo_path}")
    elif 'thingiverse.com' in file_identifier:
        # For Thingiverse, the directory structure might be different
        repo_path = os.path.dirname(local_path)
        ic(f"Thingiverse directory path for cleanup: {repo_path}")
    else:
        # For other sources, try to get the parent directory
        repo_path = os.path.dirname(local_path)
        ic(f"Source directory path for cleanup: {repo_path}")

    # get the GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" {target_directory}"

        # Check if blender_script.py exists and copy it to the temp directory
        if not os.path.exists("blender_script.py"):
            logger.error("blender_script.py not found")
            ic("blender_script.py not found in", os.getcwd())
            if failed_log_file is not None:
                log_processed_object(
                    failed_log_file,
                    file_identifier,
                    sha256,
                    "missing_script"
                )
            return False
        
        # Copy the script to ensure it's accessible
        shutil.copy("blender_script.py", temp_dir)
        ic(f"Copied blender_script.py to {temp_dir}")
        
        # Update args to use the copied script
        script_path = os.path.join(temp_dir, "blender_script.py")

        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        # First try without DISPLAY to see if it helps
        command = f"module load Blender/3.5.0-linux-x86_64-CUDA-11.7.0 && blender --background --python {script_path} -- {args} --verbose"
        ic(command)

        # render the object and capture ALL output
        try:
            # Run with text=False to handle binary output
            result = subprocess.run(
                ["bash", "-c", command],
                timeout=render_timeout,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Change to False to handle binary data
                bufsize=1    # Line buffered
            )
            
            # Always log output for debugging
            ic("Return code:", result.returncode)
            
            # Process stderr with better handling for binary data
            if result.stderr:
                try:
                    stderr_text = result.stderr.decode('utf-8', errors='replace')
                    stderr_lines = stderr_text.splitlines()
                    ic("STDERR first 50 lines:")
                    for line in stderr_lines[:50]:
                        ic(line)
                except Exception as e:
                    ic(f"Error decoding stderr: {str(e)}")
                    ic(f"Raw stderr length: {len(result.stderr)} bytes")
            
            # Process stdout with better handling for binary data
            if result.stdout:
                try:
                    stdout_text = result.stdout.decode('utf-8', errors='replace')
                    stdout_lines = stdout_text.splitlines()
                    ic("STDOUT first 50 lines:")
                    for line in stdout_lines[:50]:
                        ic(line)
                except Exception as e:
                    ic(f"Error decoding stdout: {str(e)}")
                    ic(f"Raw stdout length: {len(result.stdout)} bytes")
                    
            # Look for specific debug log file
            debug_log_path = os.path.join(target_directory, "blender_debug.log")
            if os.path.exists(debug_log_path):
                with open(debug_log_path, 'r', encoding='utf-8', errors='replace') as f:
                    debug_log = f.read()
                    ic("BLENDER DEBUG LOG:")
                    ic(debug_log)
            
            # Check for the num_renders.txt file that contains the actual number of renders
            actual_num_renders = num_renders  # Default to the original value
            num_renders_path = os.path.join(target_directory, "num_renders.txt")
            if os.path.exists(num_renders_path):
                try:
                    with open(num_renders_path, 'r') as f:
                        actual_num_renders = int(f.read().strip())
                        ic(f"Found actual num_renders: {actual_num_renders}")
                except Exception as e:
                    ic(f"Error reading num_renders.txt: {str(e)}")
            
            # List files in target directory
            if os.path.exists(target_directory) and os.path.isdir(target_directory):
                files = os.listdir(target_directory)
                ic(f"Files in target directory ({target_directory}):", files)
            else:
                ic(f"Target directory issue: {target_directory}")
                ic(f"Exists: {os.path.exists(target_directory)}")
                ic(f"Is directory: {os.path.isdir(target_directory)}")
                # If it exists but isn't a directory, it means Blender saved a file at that path
                if os.path.exists(target_directory) and not os.path.isdir(target_directory):
                    # Create the directory and move the file
                    os.rename(target_directory, f"{target_directory}.blend")
                    os.makedirs(target_directory, exist_ok=True)
                    ic("Created directory and renamed the file")
        except subprocess.TimeoutExpired:
            ic("Blender process timed out after", render_timeout, "seconds")
            if failed_log_file is not None:
                log_processed_object(
                    failed_log_file,
                    file_identifier,
                    sha256,
                    "timeout"
                )
            return False

        # check that the renders were saved successfully
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        ic(f"Found files: {len(png_files)} PNGs, {len(metadata_files)} JSONs, {len(npy_files)} NPYs")
        ic(f"Expected files: {actual_num_renders} PNGs, 1 JSON, {actual_num_renders} NPYs")
        
        if (
            (len(png_files) != actual_num_renders)
            or (len(npy_files) != actual_num_renders)
            or (len(metadata_files) != 1)
        ):
            logger.error(
                f"Found object {file_identifier} was not rendered successfully! "
                f"Expected {actual_num_renders} renders, got {len(png_files)} PNGs and {len(npy_files)} NPYs."
            )
            if failed_log_file is not None:
                log_processed_object(
                    failed_log_file,
                    file_identifier,
                    sha256,
                )
            return False

        # update the metadata
        metadata_path = os.path.join(target_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)
        metadata_file["sha256"] = sha256
        metadata_file["file_identifier"] = file_identifier
        metadata_file["save_uid"] = save_uid
        metadata_file["metadata"] = metadata
        metadata_file["num_renders"] = actual_num_renders  # Add the actual number to metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        # Make a zip of the target_directory.
        # Keeps the {save_uid} directory structure when unzipped
        with zipfile.ZipFile(
            f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        # move the zip to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)

        # move the zip to the render_dir
        fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(path, "renders", f"{save_uid}.zip"),
        )

        # log that this object was rendered successfully
        if successful_log_file is not None:
            log_processed_object(successful_log_file, file_identifier, sha256)
        
        # CLEANUP: Delete the repository directory after successful rendering and transfer
        if repo_path and os.path.exists(repo_path):
            try:
                # For safety, verify this is a download directory before removing
                if any(marker in repo_path for marker in ['objaverse', 'download', 'tmp', 'temp']):
                    ic(f"Cleaning up repository directory: {repo_path}")
                    if os.path.isdir(repo_path):
                        shutil.rmtree(repo_path)
                    else:
                        os.remove(repo_path)
                    logger.info(f"Successfully removed repository files at {repo_path}")
                else:
                    # Safety check failed, log but don't delete
                    logger.warning(f"Skipped cleanup of {repo_path} - path doesn't appear to be a temp directory")
            except Exception as e:
                logger.error(f"Error during repository cleanup: {str(e)}")
                # Continue despite cleanup errors - they shouldn't affect the main process
        
        ic("Successfully rendered", file_identifier)
        return True


def handle_new_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    log_file: str = "handle-new-object.csv",
) -> None:
    """Called when a new object is found.

    Here, the object is not used in Objaverse-XL, but is still downloaded with the
    repository. The object may have not been used because it does not successfully
    import into Blender. If None, the object will be downloaded, but nothing will be
    done with it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): The file identifier of the new 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, including the GitHub
            organization and repo names.
        log_file (str): Name of the log file to save the handle_new_object logs to.

    Returns:
        None
    """
    # log the new object
    log_processed_object(log_file, file_identifier, sha256)


def handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
) -> None:
    """Called when a modified object is found and downloaded.

    Here, the object is successfully downloaded, but it has a different sha256 than the
    one that was downloaded with Objaverse-XL. This is not expected to happen very
    often, because the same commit hash is used for each repo. If None, the object will
    be downloaded, but nothing will be done with it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        new_sha256 (str): SHA256 of the contents of the newly downloaded 3D object.
        old_sha256 (str): Expected SHA256 of the contents of the 3D object as it was
            when it was downloaded with Objaverse-XL.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.

    Returns:
        None
    """
    success = handle_found_object(
        local_path=local_path,
        file_identifier=file_identifier,
        sha256=new_sha256,
        metadata=metadata,
        num_renders=num_renders,
        render_dir=render_dir,
        only_northern_hemisphere=only_northern_hemisphere,
        gpu_devices=gpu_devices,
        render_timeout=render_timeout,
        successful_log_file=None,
        failed_log_file=None,
    )

    if success:
        log_processed_object(
            "handle-modified-object-successful.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )
    else:
        log_processed_object(
            "handle-modified-object-failed.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )


def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    log_file: str = "handle-missing-object.csv",
) -> None:
    """Called when an object that is in Objaverse-XL is not found.

    Here, it is likely that the repository was deleted or renamed. If None, nothing
    will be done with the missing object.

    Args:
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the original 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        log_file (str): Name of the log file to save missing renders to.

    Returns:
        None
    """
    # log the missing object
    log_processed_object(log_file, file_identifier, sha256)


def get_sample_objects(sample_size: int = 50) -> pd.DataFrame:
    """Returns a DataFrame of sample objects from Objaverse-XL.
    
    Args:
        sample_size (int): Number of objects to sample from the filtered dataset.
        
    Returns:
        pd.DataFrame: DataFrame containing the sampled objects.
    """
    logger.info("Fetching annotations from Objaverse-XL...")
    annotations = oxl.get_annotations(download_dir='/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse')
    logger.info(f"Retrieved {len(annotations)} total annotations")
    
    # Filter to preferred formats - prioritizing formats less likely to use Git LFS
    preferred_formats = ['glb', 'gltf', 'obj', 'fbx', 'stl', 'ply']
    filtered_annotations = annotations[
        annotations['fileType'].isin(preferred_formats)
    ]
    logger.info(f"Filtered to {len(filtered_annotations)} objects with preferred formats")

    ic(filtered_annotations.head())
    ic(filtered_annotations.shape)

    # Aggressively filter out repos likely to use LFS
    if 'source' in filtered_annotations.columns:
        # Non-GitHub sources are less likely to use LFS
        non_github = filtered_annotations[filtered_annotations['source'] != 'github']
        if len(non_github) >= sample_size:
            filtered_annotations = non_github
            logger.info(f"Using only non-GitHub sources: {len(filtered_annotations)} objects")
    
    ic(filtered_annotations.head())
    ic(filtered_annotations.shape)

    # Filter out large repositories (most likely to use Git LFS)
    if 'repoSize' in filtered_annotations.columns:
        # Only keep the smallest repos (under 100MB to be safe)
        size_filtered = filtered_annotations[filtered_annotations['repoSize'] < 100000000]
        if len(size_filtered) >= sample_size:
            filtered_annotations = size_filtered
            logger.info(f"Further filtered to {len(filtered_annotations)} objects with repo size < 10MB")
    
    ic(filtered_annotations.head())
    ic(filtered_annotations.shape)

    # Sample a subset
    if len(filtered_annotations) > sample_size:
        sampled_objects = filtered_annotations.sample(sample_size, random_state=42)
        logger.info(f"Sampled {len(sampled_objects)} objects for rendering")
    else:
        sampled_objects = filtered_annotations
        logger.warning(f"Only {len(filtered_annotations)} objects available after filtering")
    
    ic(sampled_objects.head())
    ic(sampled_objects.shape)

    # Ensure DataFrame has expected format
    required_columns = ["fileIdentifier", "sha256", "source"]
    for col in required_columns:
        if col not in sampled_objects.columns:
            logger.error(f"Required column '{col}' not found in annotations")
            raise ValueError(f"Required column '{col}' not found in annotations")
    
    ic(sampled_objects.head())
    ic(sampled_objects.shape)

    return sampled_objects


def get_example_objects() -> pd.DataFrame:
    """Returns a DataFrame of example objects to use for debugging."""
    # For compatibility, keeping this function but making it call the sample function
    logger.info("Using sample objects from Objaverse-XL dataset instead of examples")
    return get_sample_objects(sample_size=50)


def render_objects(
    render_dir: str = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse",
    download_dir: Optional[str] = None,
    num_renders: int = 12,
    processes: Optional[int] = None,
    save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
    only_northern_hemisphere: bool = False,
    render_timeout: int = 900,
    gpu_devices: Optional[Union[int, List[int]]] = None,
    use_example_objects: bool = False,
    sample_size: int = 1_000_000,
    batch_size: int = 50,
    max_download_retries: int = 5,
    max_concurrent_downloads: int = 8,
) -> None:
    """Renders objects in the Objaverse-XL dataset with Blender"""
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )
    if download_dir is None and save_repo_format is not None:
        raise ValueError(
            f"If {save_repo_format=} is not None, {download_dir=} must be specified."
        )
    if download_dir is not None and save_repo_format is None:
        logger.warning(
            f"GitHub repos will not save. While {download_dir=} is specified, {save_repo_format=} None."
        )

    # Try to install backoff if not already installed
    try:
        import tenacity
    except ImportError:
        logger.info("Installing tenacity package for retry functionality...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tenacity"])
        import tenacity
        logger.info("Successfully installed tenacity package")

    # get the gpu devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    # Limit the number of processes to avoid overwhelming the network
    if processes is None:
        processes = min(multiprocessing.cpu_count(), max_concurrent_downloads)
    else:
        processes = min(processes, max_concurrent_downloads)
    
    logger.info(f"Using {processes} processes for downloads (limited to {max_concurrent_downloads})")

    # get the objects to render
    if use_example_objects:
        objects = pd.read_json("example-objects.json", orient="records")
        logger.info(f"Using {len(objects)} example objects from JSON file.")
    else:
        objects = get_sample_objects(sample_size=sample_size)
        logger.info(f"Using {len(objects)} sample objects from Objaverse-XL.")
    
    # Ensure file identifiers are accessible
    if len(objects) > 0:
        first_id = objects.iloc[0]["fileIdentifier"]
        logger.info(f"First file identifier: {first_id}")
    objects = objects.copy()
    logger.info(f"Provided {len(objects)} objects to render.")

    # get the already rendered objects
    fs, path = fsspec.core.url_to_fs(render_dir)
    try:
        zip_files = fs.glob(os.path.join(path, "renders", "*.zip"), refresh=True)
    except TypeError:
        # s3fs may not support refresh depending on the version
        zip_files = fs.glob(os.path.join(path, "renders", "*.zip"))

    saved_ids = set(zip_file.split("/")[-1].split(".")[0] for zip_file in zip_files)
    logger.info(f"Found {len(saved_ids)} objects already rendered.")

    # filter out the already rendered objects
    objects["saveUid"] = objects["fileIdentifier"].apply(get_uid_from_str)
    objects = objects[~objects["saveUid"].isin(saved_ids)]
    objects = objects.reset_index(drop=True)
    logger.info(f"Rendering {len(objects)} new objects.")

    # shuffle the objects
    objects = objects.sample(frac=1).reset_index(drop=True)
    
    # Group objects by source for more efficient batch processing
    objects_by_source = {source: group for source, group in objects.groupby("source")}
    
    # Patch the objaverse download method to include retries
    original_download_objects = oxl.download_objects
    
    @retry(
        stop=stop_after_attempt(max_download_retries),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((http.client.IncompleteRead, urllib.error.URLError, socket.timeout, ConnectionError)),
        before_sleep=lambda retry_state: logger.info(f"Download attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def download_objects_with_retry(*args, **kwargs):
        # Add a timeout parameter if not already provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300  # 5 minutes timeout
        
        try:
            return original_download_objects(*args, **kwargs)
        except (http.client.IncompleteRead, urllib.error.URLError, socket.timeout, ConnectionError) as e:
            logger.warning(f"Download error: {str(e)}. Retrying...")
            raise  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unhandled exception during download: {str(e)}")
            raise  # Re-raise any other exceptions
    
    # Temporarily replace the download_objects function
    oxl.download_objects = download_objects_with_retry
    
    try:
        for source, source_objects in objects_by_source.items():
            logger.info(f"Processing {len(source_objects)} objects from source: {source}")
            
            # Process the source's objects in batches
            for batch_start in range(0, len(source_objects), batch_size):
                batch_end = min(batch_start + batch_size, len(source_objects))
                batch_objects = source_objects.iloc[batch_start:batch_end].reset_index(drop=True)
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(source_objects) + batch_size - 1)//batch_size} "
                           f"({batch_start}-{batch_end-1}) from {source}")
                
                try:
                    # Download and process just this batch with retry wrapper
                    oxl.download_objects(
                        objects=batch_objects,
                        processes=processes,
                        save_repo_format=save_repo_format,
                        download_dir=download_dir,
                        handle_found_object=partial(
                            handle_found_object,
                            render_dir=render_dir,
                            num_renders=num_renders,
                            only_northern_hemisphere=only_northern_hemisphere,
                            gpu_devices=parsed_gpu_devices,
                            render_timeout=render_timeout,
                        ),
                        handle_new_object=handle_new_object,
                        handle_modified_object=partial(
                            handle_modified_object,
                            render_dir=render_dir,
                            num_renders=num_renders,
                            only_northern_hemisphere=only_northern_hemisphere,
                            gpu_devices=parsed_gpu_devices,
                            render_timeout=render_timeout,
                        ),
                        handle_missing_object=handle_missing_object,
                        timeout=300,  # Add timeout parameter to avoid hanging downloads
                    )
                    
                    # Force garbage collection after each batch
                    import gc
                    gc.collect()
                    
                    logger.info(f"Successfully processed batch {batch_start//batch_size + 1} from {source}")
                    
                    # Brief pause between batches to let the system stabilize
                    time.sleep(5)  # Increased pause between batches
                    
                except subprocess.CalledProcessError as e:
                    if "git lfs" in str(e):
                        logger.error(f"Git LFS error in batch {batch_start//batch_size + 1} from {source}: {e}")
                        logger.info("Continuing with next batch")
                    else:
                        logger.error(f"Error during batch {batch_start//batch_size + 1} download: {e}")
                        # Don't raise - continue with next batch
                        logger.info("Continuing with next batch despite error")
                except Exception as e:
                    logger.error(f"Unexpected error during batch {batch_start//batch_size + 1} download: {e}")
                    logger.info("Continuing with next batch despite error")
                
                # Log progress after each batch
                logger.info(f"Completed {batch_end}/{len(source_objects)} objects from {source} "
                           f"({batch_end/len(source_objects)*100:.1f}%)")
    finally:
        # Restore the original download_objects function
        oxl.download_objects = original_download_objects
    
    logger.info("All objects from all sources have been processed")


if __name__ == "__main__":
    fire.Fire(render_objects)