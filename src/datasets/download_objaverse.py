# # Objaverse-XL API Tutorial
# 
# Objaverse-XL is a Universe of 10M+ 3D Objects.
# 
# It is hosted on 🤗[Hugging Face](https://huggingface.co/datasets/allenai/objaverse-xl) and includes a [Python API on GitHub](https://github.com/allenai/objaverse-xl). This notebook provides a tutorial on downloading objects and annotations!

from icecream import ic
from datetime import datetime
from typing import Any, Dict, Hashable
import objaverse.xl as oxl
import time
import json
import os

os.environ['TMPDIR'] = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse_tmp"
os.makedirs(os.environ['TMPDIR'], exist_ok=True)
print(f"Using temporary directory: {os.environ['TMPDIR']}")

# Keep your original download path
DOWNLOAD_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse"
TRACKING_FILE = os.path.join(DOWNLOAD_PATH, "download_progress.json")

# Create download directory
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# Get annotations and show basic info
annotations = oxl.get_annotations(download_dir=DOWNLOAD_PATH)
ic(annotations.shape)  # Show how many objects total

# Basic data exploration
ic(annotations["source"].value_counts())
ic(annotations["fileType"].value_counts())

# Filter to a manageable subset based on preferred formats
preferred_formats = ['glb', 'gltf', 'obj', 'fbx', 'stl', 'usdz', 'blend']
filtered_annotations = annotations[
    (annotations['fileType'].isin(preferred_formats))
]

# Don't limit by random sampling upfront
MAX_FILES = 500000  # Target number of successful downloads
print(f"Will download until reaching {MAX_FILES} successful files")
print(f"Using filtered set of {len(filtered_annotations)} files")

# Show distribution of the filtered dataset
ic(filtered_annotations["source"].value_counts())
ic(filtered_annotations["fileType"].value_counts())
ic(f"Selected {len(filtered_annotations)} files out of {len(annotations)} total")


def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    source = metadata.get('source', 'Unknown')
    file_type = local_path.split('.')[-1] if '.' in local_path else 'Unknown'
    
    with open(os.path.join(DOWNLOAD_PATH, "success_log.txt"), "a") as f:
        f.write(f"{file_identifier}|{source}|{file_type}|{sha256}\n")
    
    if hash(file_identifier) % 100 == 0:  # Only print status for ~1% of downloads
        print(f"\nDownloaded ({source}): {os.path.basename(local_path)}")


def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    with open(os.path.join(DOWNLOAD_PATH, "failed_log.txt"), "a") as f:
        f.write(f"{file_identifier}|{metadata.get('source', 'Unknown')}|{sha256}\n")


def download_with_chunking_and_resume():
    """Download the filtered dataset in chunks with resume capability"""
    
    if not os.path.exists(os.path.join(DOWNLOAD_PATH, "success_log.txt")):
        with open(os.path.join(DOWNLOAD_PATH, "success_log.txt"), "w") as f:
            f.write("file_identifier|source|file_type|sha256\n")
    
    if not os.path.exists(os.path.join(DOWNLOAD_PATH, "failed_log.txt")):
        with open(os.path.join(DOWNLOAD_PATH, "failed_log.txt"), "w") as f:
            f.write("file_identifier|source|sha256\n")
    
    downloaded_ids = set()
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            progress = json.load(f)
            downloaded_ids = set(progress.get('downloaded_ids', []))
            print(f"Resuming download - already downloaded {len(downloaded_ids)} files")
    
    # check success log for downloaded files
    if os.path.exists(os.path.join(DOWNLOAD_PATH, "success_log.txt")):
        with open(os.path.join(DOWNLOAD_PATH, "success_log.txt"), "r") as f:
            for line in f:
                if "|" in line:
                    file_id = line.split("|")[0]
                    if file_id and file_id != "file_identifier":
                        downloaded_ids.add(file_id)
    
    # track the number of successfully downloaded files
    success_count = len(downloaded_ids)
    print(f"Current successful download count: {success_count}")
    
    # check if we've already reached our target
    if success_count >= MAX_FILES:
        print(f"Already reached target of {MAX_FILES} successfully downloaded files!")
        return
    
    # process by source to better manage rate limits
    for source_name, source_group in filtered_annotations.groupby('source'):
        # stop if we've reached the max files
        if success_count >= MAX_FILES:
            print(f"Reached target of {MAX_FILES} successfully downloaded files!")
            break
            
        to_download = source_group[~source_group['fileIdentifier'].isin(downloaded_ids)]
        
        if len(to_download) == 0:
            print(f"Skipping {source_name} - all {len(source_group)} files already downloaded")
            continue
        
        print(f"\n\n======= Downloading files from {source_name} =======")
        print(f"Current progress: {success_count}/{MAX_FILES} files")
        
        # process in smaller chunks to avoid overwhelming the API
        chunk_size = 500  # smaller chunk size for better reliability
        
        for i in range(0, len(to_download), chunk_size):
            # stop if we've reached the max files
            if success_count >= MAX_FILES:
                print(f"Reached target of {MAX_FILES} successfully downloaded files!")
                break
                
            chunk = to_download.iloc[i:i+chunk_size]
            chunk_start_time = datetime.now()
            
            print(f"\nProcessing chunk {i//chunk_size + 1}/{(len(to_download) + chunk_size - 1)//chunk_size} "
                  f"({i}-{min(i+chunk_size-1, len(to_download)-1)}) from {source_name}")
            
            # apply rate limiting for all sources (more conservative)
            rate_limit_delay = 1.0 if source_name == "Thingiverse" else 0.2
            
            # track successful downloads in this chunk
            chunk_success_count = 0
            initial_success_count = success_count
            
            # try with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # custom success handler that counts successful downloads
                    def count_success_handler(local_path, file_identifier, sha256, metadata):
                        nonlocal chunk_success_count, success_count
                        handle_found_object(local_path, file_identifier, sha256, metadata)
                        chunk_success_count += 1
                        success_count += 1
                    
                    results = oxl.download_objects(
                        objects=chunk,
                        handle_found_object=count_success_handler,
                        handle_missing_object=handle_missing_object,
                        download_dir=DOWNLOAD_PATH
                    )
                    
                    # record successful downloads
                    for file_id in chunk['fileIdentifier'].tolist():
                        downloaded_ids.add(file_id)
                    
                    # update progress file after each chunk for better resumability
                    with open(TRACKING_FILE, 'w') as f:
                        json.dump({
                            'downloaded_ids': list(downloaded_ids),
                            'last_updated': datetime.now().isoformat(),
                            'total_files': len(filtered_annotations),
                            'downloaded_count': len(downloaded_ids),
                            'success_count': success_count
                        }, f)
                    
                    # calculate and show progress
                    elapsed = (datetime.now() - chunk_start_time).total_seconds()
                    rate = chunk_size / elapsed if elapsed > 0 else 0
                    print(f'# {"*" * 100}')
                    print(f"Chunk completed in {elapsed:.1f}s ({rate:.1f} files/sec)")
                    print(f"Successful downloads this chunk: {chunk_success_count}/{len(chunk)}")
                    print(f"Overall progress: {success_count}/{MAX_FILES} "
                          f"({100*success_count/MAX_FILES:.2f}%)")
                    
                    # longer break between chunks to avoid API limits
                    time.sleep(rate_limit_delay * 15)
                    break
                
                except Exception as e:
                    print(f"Error during chunk download: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 60  # longer exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to download chunk after {max_retries} attempts")
                        # mark these as failed but continue with next chunk
                        with open(os.path.join(DOWNLOAD_PATH, "failed_chunks.txt"), "a") as f:
                            f.write(f"{source_name}|{i}|{i+chunk_size-1}|{datetime.now().isoformat()}\n")
                        # give the API a break before continuing
                        time.sleep(30)

# run the download with chunking and resume capability
print("\nStarting filtered dataset download with chunking and resume capability...")
download_with_chunking_and_resume()

# final verification
print("\nFinal verification of downloads...")
success_count = 0
if os.path.exists(os.path.join(DOWNLOAD_PATH, "success_log.txt")):
    with open(os.path.join(DOWNLOAD_PATH, "success_log.txt"), "r") as f:
        success_count = sum(1 for line in f if "|" in line) - 1  # subtract header

failed_count = 0
if os.path.exists(os.path.join(DOWNLOAD_PATH, "failed_log.txt")):
    with open(os.path.join(DOWNLOAD_PATH, "failed_log.txt"), "r") as f:
        failed_count = sum(1 for line in f if "|" in line) - 1  # subtract header

print(f"Successfully downloaded: {success_count} files")
print(f"Failed downloads: {failed_count} files")
print(f"Total processed: {success_count + failed_count} of {len(filtered_annotations)}")

# calculate storage usage
print("\nStorage usage:")
total_size = 0
for dirpath, dirnames, filenames in os.walk(DOWNLOAD_PATH):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        if os.path.exists(fp):
            total_size += os.path.getsize(fp)

print(f"Total storage used: {total_size / (1024**3):.2f} GB")
print("Download process completed :>")


# ## Next Steps
# 
# Take a look at the [Blender rendering code](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering) for rendering Objaverse-XL objects in Blender and extracting metadata from the objects!
# 