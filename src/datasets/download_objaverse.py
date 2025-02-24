# # Objaverse-XL API Tutorial
# 
# Objaverse-XL is a Universe of 10M+ 3D Objects.
# 
# It is hosted on 🤗[Hugging Face](https://huggingface.co/datasets/allenai/objaverse-xl) and includes a [Python API on GitHub](https://github.com/allenai/objaverse-xl). This notebook provides a tutorial on downloading objects and annotations!

import objaverse
import objaverse.xl as oxl
from icecream import ic


# ## Annotations
# 
# The objects that appear in the dataset can be obtained with the `get_annotations` function:
# 
# ```python
# oxl.get_annotations(
#     download_dir: str = "~/Code/datasets/objaverse_tutorial",
# ) -> pd.DataFrame
# ```
# 
# The function takes in a parameter for `download_dir: str = "~/Code/datasets/objaverse_tutorialets/objaverse_tutorial"`, which is the directory to cache the downloaded annotations. After the annotations are downloaded for the first time, they do not need to be downloaded again, as they are cached.
# 
# For example:
# 

DOWNLOAD_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse"


annotations = oxl.get_annotations(
    download_dir=DOWNLOAD_PATH # default download directory
)
ic(annotations)


# > Note: Some objects are still under review for being publicly released.
# 
# Here, `annotations` is a pandas DataFrame. These annotations are meant to provide a minimal amount of information about the objects that are standarized across each source and allow it to be downloaded. Each object is a row, and has attributes for:
# 
# - `fileIdentifier` - A unique identifier of the 3D object. Typically the URL that links back to the object.
# - `source` - The website where the 3D object comes from.
# - `license` - The license that the original object was distributed under.
# - `fileType` - The 3D file type of the object (e.g., `fbx`, `obj`, `glb`).
# - `sha256` - The cryptographic hash of the contents of the object (used for deduplication and to check if the object has not been modified since the dataset was originally released).
# - `metadata` - Additional metadata of the object that might be site specific (e.g., the file name). To keep the `annotations` DataFrame lightweight, more detailed annotations may be available as standalone functions (e.g., `objaverse.load_lvis_annotations()`). See the Objaverse 1.0 documentation for more specific annotations.
# 
# Since the annotations is a pandas DataFrame object, we can do standard operations on it, such as getting the value counts of different attributes:
# 

ic(annotations["source"].value_counts())

ic(annotations["fileType"].value_counts())


# And randomly sample objects:
# 

ic(annotations.sample(5))


# ## Alignment Fine-tuning Annotations
# 
# For training Zero123-XL, we first started by training on the entire dataset, and then performed fine-tuning on a smaller, more high-quality 3D dataset. To load in the dataset that was used for fine-tuning, we can run:
# 

alignment_annotations = oxl.get_alignment_annotations(
    download_dir=DOWNLOAD_PATH
)
ic(alignment_annotations)


# ## Download Objects
# 
# Downloading objects can be done through the `download_objects` function:
# 
# ```python
# oxl.download_objects(
#     # Base parameters:
#     objects: pd.DataFrame,
#     download_dir: str = "~/Code/datasets/objaverse_tutorial",
#     processes: Optional[int] = None,  # None => multiprocessing.cpu_count()
# 
#     # optional callback functions:
#     handle_found_object: Optional[Callable] = None,
#     handle_modified_object: Optional[Callable] = None,
#     handle_missing_object: Optional[Callable] = None,
# 
#     # GitHub specific:
#     save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
#     handle_new_object: Optional[Callable] = None,
# )
# ```
# 
# The function supports several different types of parameters, which we've broken down into base parameters, callback functions, and GitHub specific parameters.
# 
# **Base parameters.**
# 
# - `objects: pd.DataFrame` a pandas DataFrame the objects to download. Must have columns for the object "fileIdentifier", "source", and "sha256". Use the `oxl.get_annotations` function to get all objects as a DataFrame.
# - `download_dir: Optional[str] = "~/Code/datasets/objaverse_tutorial"` specifies where to download the objects.
# 
#   Thanks to fsspec, we support writing files to many file systems. To use it, simply use the prefix of your filesystem before the path. For example hdfs://, s3://, http://, gcs://, or ssh://. Some of these file systems require installing an additional package (for example s3fs for s3, gcsfs for gcs, fsspec/sshfs for ssh). Start [here](https://github.com/rom1504/img2dataset#file-system-support) for more details on fsspec.
# 
#   If None, the objects will be deleted after they are downloaded. Defaults to "~/Code/datasets/objaverse_tutorial".
# 
# - `processes: Optional[int] = None` number of processes to use when downloading the objects. If None, it will use the number of CPUs on the machine (which comes from `multiprocessing.cpu_count()`). Defaults to None.
# 
# **Callback function parameters.**
# The function also supports several callback functions, which are called right after an object is locally downloaded. Common use cases for these callback functions may include downloading objects on the fly and processing them with Blender, rendering them, then discarding them. The specific callback functions include:
# 
# - `handle_found_object: Optional[Callable] = None` is called when an object is successfully found and downloaded. Here, the object has the same sha256 as the one that was downloaded with Objaverse-XL. If None, the object will be downloaded, but nothing will be done with it.
# 
#   Parameters for the function must include:
# 
#   - `local_path: str` Local path to the downloaded 3D object.
#   - `file_identifier: str` File identifier of the 3D object.
#   - `sha256: str` sha256 of the contents of the 3D object.
#   - `metadata: Dict[Hashable, Any]` Metadata about the 3D object, such as the GitHub organization and repo name.
# 
#   The return of the function is not used.
# 
# - `handle_modified_object: Optional[Callable] = None` is called when a modified object is found and downloaded. Here, the object is successfully downloaded, but it has a different sha256 than the one that was downloaded with Objaverse-XL. This is not expected to happen very often, because the same commit hash is used for each repo. If None, the object will be downloaded, but nothing will be done with it.
# 
#   Parameters for the function must include:
# 
#   - `local_path: str` Local path to the downloaded 3D object.
#   - `file_identifier: str` File identifier of the 3D object.
#   - `new_sha256: str` sha256 of the contents of the newly downloaded 3D object.
#   - `old_sha256: str` Provided sha256 representing the contents of the 3D object as it was originally intended to be downloaded (coming from the `objects` argument).
#   - `metadata: Dict[Hashable, Any]` Metadata about the 3D object, such as the GitHub organization and repo name.
# 
#   The return of the function is not used.
# 
# - `handle_missing_object: Optional[Callable] = None` is called when a specified object cannot be found. Here, it is likely that the object was deleted or the repository was deleted or renamed. If None, nothing will be done with the missing object.
# 
#   Parameters for the function must include:
# 
#   - `file_identifier: str` File identifier of the 3D object.
#   - `sha256: str` Provided sha256 representing the contents of the 3D object as it was originally intended to be downloaded (coming from the `objects` argument).
#   - `metadata: Dict[Hashable, Any]` Metadata about the 3D object, which is particular to the source.
# 
#   The return of the function is not used.
# 
# **GitHub specific parameters.** There are several parameters that are only used when downloading objects from GitHub. These parameters can still be passed in when downloading objects from other sources, but they will not be used. These parameters include:
# 
# - `save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None` specifies the format to save the GitHub repository. Unlike other sources, GitHub objects are not standalone 3D files, and may link to other assets, such as textures. If None, the repository will not be saved. If "files" is specified, each file will be saved individually in a standard folder structure. Otherwise, the repository can be saved as a "zip", "tar", or "tar.gz" file. Defaults to None.
# 
# - `handle_new_object: Optional[Callable]` is called when a new object is found. Here, the object is not used in Objaverse-XL, but is still downloaded as part of the repository. Note that the object may have not been used because it does not successfully import into Blender. If None, the object will be downloaded, but nothing will be done with it.
# 
#   Parameters for the function must include:
# 
#   - `local_path: str` Local path to the downloaded 3D object.
#   - `file_identifier: str` GitHub URL of the 3D object.
#   - `sha256: str` sha256 of the contents of the 3D object.
#   - `metadata: Dict[str, Any]` Metadata about the 3D object, such as the GitHub organization and repo names.
# 
#   The return of the function is not used.
# 
# The following is a minimal example of using `oxl.download_objects`:
# 

# sample a single object from each source
sampled_df = annotations.groupby('source').apply(lambda x: x.sample(1)).reset_index(drop=True)
ic(sampled_df)

oxl.download_objects(objects=sampled_df, download_dir=DOWNLOAD_PATH)


# Great! As we can see, the objects were successfully downloaded. Note that the GitHub objects were not saved, since `save_repo_format` defaults to None, so they are not included in the output return.
# 
# Next, we'll show an example using callback functions, which work well when downloading and processing GitHub objects.
# 
# We'll start by removing the `~/Code/datasets/objaverse_tutorialets/objaverse_tutorial` directory to clear the cache of the objects that we just downloaded, so they'll be downloaded again from scratch. Otherwise, the objects will be cached and not downloaded for a 2nd time:
# 

import shutil
import os

shutil.rmtree(os.path.expanduser(DOWNLOAD_PATH), ignore_errors=True)


# And we'll define our `handle_found_object` function, which is called after an object is downloaded and has a sha256 that matches the one that we supplied:
# 

from typing import Any, Dict, Hashable

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    print("\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")


# Now, after running the same function with the `handle_found_object` callback, we have:
# 

oxl.download_objects(
    objects=sampled_df,
    handle_found_object=handle_found_object,
    download_dir=DOWNLOAD_PATH
)


# Notice that our custom `handle_found_object` function is called right after each object is locally downloaded!
# 
# Next, for the `handle_modified_object` callback, let's change the sha256 of one of the objects and then try to download it:
# 

modified_df = sampled_df.copy()
modified_df.iloc[0]["sha256"] = "modified-sha256"
ic(modified_df)

def handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[Hashable, Any],
) -> None:
    print("\n\n\n---HANDLE_MODIFIED_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {old_sha256=}\n  {new_sha256}\n  {metadata=}\n\n\n")

# remove previously downloaded objects
shutil.rmtree(os.path.expanduser(DOWNLOAD_PATH), ignore_errors=True)

# redownload
oxl.download_objects(
    objects=modified_df,
    handle_found_object=handle_found_object,
    handle_modified_object=handle_modified_object,
    download_dir=DOWNLOAD_PATH
)


# Notice that `handle_found_object` was called 3 times and `handle_modified_object` was called once, for the object that has its sha256 modified!
# 
# We'll do something similar to experiment with `handle_missing_object`, where we'll add modify the path of one of the objects to something that doesn't exist:
# 

missing_df = sampled_df.copy()
missing_df.iloc[1]["fileIdentifier"] += "-i-do-not-exist"

print(missing_df.iloc[1]["fileIdentifier"])
ic(missing_df)

def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    print("\n\n\n---HANDLE_MISSING_OBJECT CALLED---\n",
          f"  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")

# remove previously downloaded objects
shutil.rmtree(os.path.expanduser(DOWNLOAD_PATH), ignore_errors=True)

# redownload
oxl.download_objects(
    objects=missing_df,
    handle_found_object=handle_found_object,
    handle_modified_object=handle_modified_object,
    handle_missing_object=handle_missing_object,
    download_dir=DOWNLOAD_PATH
)


# Great! Notice how we get an error that the object could not be found and that our `handle_missing_object` callback is called!
# 
# Finally, we'll also add a callback for `handle_new_object`, which will be called for every object that is in the repository, but not in the objects that we supplied for it to expect to download:
# 

def handle_new_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    print("\n\n\n---HANDLE_NEW_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")

# remove previously downloaded objects
shutil.rmtree(os.path.expanduser(DOWNLOAD_PATH), ignore_errors=True)

# redownload
oxl.download_objects(
    objects=sampled_df,
    handle_found_object=handle_found_object,
    handle_modified_object=handle_modified_object,
    handle_missing_object=handle_missing_object,
        handle_new_object=handle_new_object,
    download_dir=DOWNLOAD_PATH
)


# Notice that `handle_new_object` gets called a bunch of times!
# 
# For even more objects, one may want to experiment with using the latest Git commits, instead of the ones used with Objaverse-XL, as it'll likely lead to more objects being available. Here, `handle_new_object` would be quite a useful callback!
# 

ic("Done")


# ## Next Steps
# 
# Take a look at the [Blender rendering code](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering) for rendering Objaverse-XL objects in Blender and extracting metadata from the objects!
# 