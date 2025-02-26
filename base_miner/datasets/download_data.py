from typing import Optional
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import datasets
import argparse
import time
import sys
import os
import subprocess
import glob
import requests

from base_miner.config import IMAGE_DATASETS, HUGGINGFACE_CACHE_DIR

datasets.logging.set_verbosity_warning()
datasets.disable_progress_bar()

from datasets import load_dataset, load_from_disk
from typing import Optional
import os


from datasets import load_dataset, load_from_disk
from typing import Optional, Union, Dict, Any
import os

def load_huggingface_dataset(
    path: str,
    split: Optional[str] = 'train',
    name: Optional[str] = None,
    download_mode: str = 'reuse_cache_if_exists',
    cache_dir: Optional[str] = HUGGINGFACE_CACHE_DIR
) -> Union[datasets.Dataset, Dict[str, datasets.Dataset]]:
    """Load a dataset from Hugging Face or a local directory.
    
    Args:
        path (str): Path to dataset. Can be:
            - A Hugging Face dataset path (<organization>/<dataset-name>)
            - An image folder path (imagefolder:<path/to/directory>)
            - A local path to a saved dataset (for load_from_disk)
        split (str, optional): Dataset split to load. If None, returns all splits. Default: 'train'
        name (str, optional): Dataset configuration name. Default: None
        download_mode (str, optional): Download mode for Hugging Face datasets. 
            Default: 'reuse_cache_if_exists'
        cache_dir (str, optional): Cache directory for downloaded datasets.
            Default: None (uses default HF cache)
            
    Returns:
        Union[datasets.Dataset, Dict[str, datasets.Dataset]]: 
            The loaded dataset split or all splits if split=None
    """
    # Check if it's a local path with a saved dataset
    if os.path.isdir(os.path.expanduser(path)):
        # Dataset artifacts that indicate this is a saved dataset
        path = os.path.expanduser(path)        
        dataset_artifacts = [
            "dataset_info.json", 
            "state.json", 
            # Check for .arrow files or data directory
            lambda files: any(f.endswith('.arrow') for f in files)
        ]
        
        path_contents = os.listdir(path)
        is_saved_dataset = any(
            artifact in path_contents if isinstance(artifact, str) 
            else artifact(path_contents) 
            for artifact in dataset_artifacts
        )
        
        if is_saved_dataset:
            try:
                dataset = load_from_disk(path)
                if split is not None and split in dataset:
                    return dataset[split]
                return dataset
            except Exception as e:
                print(f"Failed to load dataset from disk: {e}")
                # Continue to other loading methods

    #  Hugging Face dataset, not local
    try:
        dataset_kwargs = {
            "path": path,
            "download_mode": download_mode
        }
        if name:
            dataset_kwargs["name"] = name
        if cache_dir:
            dataset_kwargs["cache_dir"] = cache_dir
            
        dataset = load_dataset(**dataset_kwargs)
        if split is not None and split in dataset:
            return dataset[split]
        return dataset
    except Exception as e:
        print(f"Failed to load Hugging Face dataset: {e}")
        raise


def download_image(url: str) -> Image.Image:
    """Download an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image or None: The downloaded image as a PIL Image object if
            successful, otherwise None.
    """
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    else:
        #print(f"Failed to download image: {response.status_code}")
        return None


def download_dataset(
    dataset_path: str,
    dataset_name: str,
    download_mode: str,
    cache_dir: str,
    max_wait: int = 300
):
    """Downloads the datasets present in datasets.json with exponential backoff.

    Args:
        dataset_path (str): Path to the dataset on Hugging Face
        dataset_name (str): Name/config of the dataset subset
        download_mode (str): Either 'force_redownload' or 'use_cache_if_exists'
        cache_dir (str): Huggingface cache directory. ~/.cache/huggingface by default
        max_wait (int, optional): Maximum wait time between retries in seconds. Defaults to 300.

    Returns:
        Dataset: The downloaded Hugging Face dataset
    """
    retry_wait = 10  # initial wait time in seconds
    attempts = 0
    print(f"Downloading {dataset_path} (subset={dataset_name}) dataset...")
    while True:
        try:
            dataset = load_dataset(
                dataset_path,
                name=dataset_name,  # config/subset name
                cache_dir=cache_dir,
                download_mode=download_mode,
                trust_remote_code=True)
            break
        except Exception as e:
            print(e)
            if '429' in str(e) or 'ReadTimeoutError' in str(e):
                print(f"Rate limit hit or timeout, retrying in {retry_wait}s...")
            elif isinstance(e, PermissionError):
                file_path = str(e).split(": '")[1].rstrip("'")
                print(f"Permission error at {file_path}, attempting to fix...")
                fix_permissions(file_path)  # Attempt to fix permissions directly
                clean_cache(cache_dir)      # Clear cache to remove any incomplete or locked files
            else:
                print(f"Unexpected error, stopping retries for {dataset_path}")
                raise e

            if retry_wait > max_wait:
                print(f"Download failed for {dataset_path} after {attempts} attempts. Try again later")
                sys.exit(1)

            time.sleep(retry_wait)
            retry_wait *= 2  # exponential backoff
            attempts += 1

    print(f"Downloaded {dataset_path} dataset to {cache_dir}")
    return dataset


def clean_cache(cache_dir):
    """Clears lock files and incomplete downloads from the cache directory.

    Args:
        cache_dir (str): Path to the Hugging Face cache directory
    """
    lock_files = glob.glob(os.path.join(cache_dir, "*lock"))
    incomplete_files = glob.glob(os.path.join(cache_dir, "downloads", "**", "*.incomplete"), recursive=True)
    try:
        if lock_files:
            subprocess.run(["rm", *lock_files], check=True)
        if incomplete_files:
            for file in incomplete_files:
                os.remove(file)
        print("Hugging Face cache lock files cleared successfully.")
    except Exception as e:
        print(f"Failed to clear Hugging Face cache lock files: {e}")


def fix_permissions(path):
    """Attempts to fix permission issues on a given path.

    Args:
        path (str): Path to fix permissions for
    """
    try:
        subprocess.run(["chmod", "-R", "775", path], check=True)
        print(f"Fixed permissions for {path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to fix permissions for {path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Hugging Face datasets for validator challenge generation and miner training.')
    parser.add_argument('--force_redownload', action='store_true', help='force redownload of datasets')
    parser.add_argument('--modality', default='image', choices=['video', 'image'], help='download image or video datasets')
    parser.add_argument('--cache_dir', type=str, default=HUGGINGFACE_CACHE_DIR, help='huggingface cache directory')
    args = parser.parse_args()

    download_mode = "reuse_cache_if_exists"
    if args.force_redownload:
        download_mode = "force_redownload"

    os.makedirs(args.cache_dir, exist_ok=True)
    clean_cache(args.cache_dir)  # Clear the cache of lock and incomplete files.

    if args.modality == 'image':
        dataset_meta = IMAGE_DATASETS
    #elif args.modality == 'video':
    #    dataset_meta = VIDEO_DATASET_META

    for dataset_type in dataset_meta:
        for dataset in dataset_meta[dataset_type]:
            download_dataset(
                dataset_path=dataset['path'],
                dataset_name=dataset.get('name', None),
                download_mode=download_mode,
                cache_dir=args.cache_dir)
