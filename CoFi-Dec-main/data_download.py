import os
import requests
import json
import zipfile
from tqdm import tqdm
from pathlib import Path

class DatasetDownloader:
    def __init__(self, base_dir="/data/ce/data"):
        self.base_dir = Path(base_dir)
        self.pope_dir = self.base_dir / "POPE"
        self.coco_dir = self.base_dir / "coco"

    def download_file(self, url, filepath, chunk_size=8192):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)

    def download_coco_val2014(self):
        """Download COCO val2014 dataset"""
        print("\nDownloading COCO val2014 dataset...")
        url = "http://images.cocodataset.org/zips/val2014.zip"
        zip_path = self.coco_dir / "val2014.zip"

        # Download
        self.download_file(url, zip_path)

        # Extract
        print("\nExtracting val2014.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.coco_dir)

        # Clean up
        os.remove(zip_path)
        print("✓ COCO val2014 downloaded and extracted")

    def setup_pope(self):
        """Setup POPE directory structure"""
        print("\nSetting up POPE directory structure...")

        # Create directories
        self.pope_dir.mkdir(parents=True, exist_ok=True)
        (self.pope_dir / "coco").mkdir(exist_ok=True)

        # Download POPE files from the GitHub repository
        base_url = "https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/Evaluation/evaluation/POPE/output"
        types = ["random", "popular", "adversarial"]

        for type_ in types:
            filename = f"coco_pope_{type_}.json"
            url = f"{base_url}/{filename}"
            self.download_file(url, self.pope_dir / "coco" / filename)

        print("✓ POPE files downloaded")

    def setup_all(self):
        """Download and setup all required datasets"""
        print("🚀 Starting dataset setup...")

        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Download and setup datasets
        self.setup_pope()
        self.download_coco_val2014()

        print("\n✨ All datasets prepared successfully!")
        print(f"Datasets location: {self.base_dir}")

def main():
    # Ask for base directory
    default_dir = "/data/ce/data"
    base_dir = input(f"Enter base directory for datasets (default: {default_dir}): ").strip()
    if not base_dir:
        base_dir = default_dir

    # Initialize and run downloader
    downloader = DatasetDownloader(base_dir)
    downloader.setup_all()

if __name__ == "__main__":
    main()