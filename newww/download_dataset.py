import os
import sys
import shutil
import zipfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from a URL with a progress bar
    
    Parameters:
    url (str): URL to download
    filename (str): Destination filename
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        print(f"Downloading {filename}...")
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Download complete: {filename}")

def setup_dataset():
    """
    Download and set up the Alzheimer's dataset
    """
    # Create main dataset directory
    dataset_dir = os.path.join(os.getcwd(), "Alzheimer_s Dataset")
    train_dir = os.path.join(dataset_dir, "train")
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    
    # Dataset classes
    classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    
    # Instructions for manual download
    print("\n" + "="*80)
    print("Alzheimer's Dataset Setup")
    print("="*80)
    print("\nThe script will guide you through setting up the Alzheimer's dataset.")
    print("\nOptions:")
    print("1. Download dataset from Kaggle API (requires Kaggle API credentials)")
    print("2. Manual download instructions")
    print("3. Specify path to already downloaded dataset")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        try:
            import kaggle
            print("\nDownloading dataset using Kaggle API...")
            
            # Check if ~/.kaggle/kaggle.json exists
            kaggle_dir = os.path.expanduser("~/.kaggle")
            if not os.path.exists(os.path.join(kaggle_dir, "kaggle.json")):
                print("\nKaggle API credentials not found.")
                print("Please follow these steps to set up your Kaggle API credentials:")
                print("1. Go to https://www.kaggle.com/account")
                print("2. Click 'Create New API Token'")
                print("3. Save the kaggle.json file to ~/.kaggle/")
                print("4. Run this script again")
                return
            
            # Download the dataset
            print("\nDownloading Alzheimer's dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                "tourist55/alzheimers-dataset-4-class-of-images",
                path=dataset_dir,
                unzip=True
            )
            print("Dataset downloaded successfully!")
            
            # Move files to the right structure if needed
            src_folder = os.path.join(dataset_dir, "Alzheimer_s Dataset", "train")
            if os.path.exists(src_folder):
                for cls in classes:
                    src_path = os.path.join(src_folder, cls)
                    dst_path = os.path.join(train_dir, cls)
                    if os.path.exists(src_path):
                        print(f"Moving {cls} images to the correct location...")
                        for file in os.listdir(src_path):
                            shutil.copy(
                                os.path.join(src_path, file),
                                os.path.join(dst_path, file)
                            )
                print("Dataset has been set up successfully!")
            
        except ImportError:
            print("\nKaggle API package not found. Installing...")
            print("Run: pip install kaggle")
            print("\nAfter installation, run this script again.")
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print("\nPlease try manual download (option 2).")
    
    elif choice == "2":
        print("\nManual Download Instructions:")
        print("1. Go to https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images")
        print("2. Click 'Download' button (requires Kaggle account)")
        print("3. Extract the downloaded zip file")
        print(f"4. Move the contents of the extracted 'Alzheimer_s Dataset/train' folder to:")
        print(f"   {train_dir}")
        print("5. Ensure the following folders exist with MRI images inside:")
        for cls in classes:
            print(f"   - {os.path.join(train_dir, cls)}")
    
    elif choice == "3":
        print("\nEnter the path to your downloaded dataset folder:")
        print("(This should be the folder containing MildDemented, ModerateDemented, etc. subfolders)")
        src_path = input()
        
        if not os.path.exists(src_path):
            print(f"Error: The path {src_path} does not exist.")
            return
        
        # Check if source folder has the expected structure
        missing_folders = []
        for cls in classes:
            if not os.path.exists(os.path.join(src_path, cls)):
                missing_folders.append(cls)
        
        if missing_folders:
            print("\nWarning: The following required folders are missing in the source path:")
            for folder in missing_folders:
                print(f"- {folder}")
            print("\nPlease ensure your dataset has the correct structure.")
            return
        
        # Copy dataset to the expected location
        print("\nCopying dataset to the expected location...")
        for cls in classes:
            src_cls_path = os.path.join(src_path, cls)
            dst_cls_path = os.path.join(train_dir, cls)
            
            # Create destination folder if it doesn't exist
            os.makedirs(dst_cls_path, exist_ok=True)
            
            # Copy all files
            files = os.listdir(src_cls_path)
            for i, file in enumerate(files):
                # Show progress every 10 files
                if i % 10 == 0:
                    print(f"Copying {cls}: {i+1}/{len(files)}", end="\r")
                    
                src_file = os.path.join(src_cls_path, file)
                dst_file = os.path.join(dst_cls_path, file)
                shutil.copy2(src_file, dst_file)
            
            print(f"Copied {len(files)} files for {cls}" + " " * 20)
            
        print("\nDataset has been set up successfully!")
    
    else:
        print("\nInvalid choice. Please run the script again.")
        return
    
    # Check if setup was successful
    success = True
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        if not os.path.exists(cls_path) or len(os.listdir(cls_path)) == 0:
            success = False
            break
    
    if success:
        print("\nDataset is ready to use. Run 'python new.py' to train the model.")
    else:
        print("\nDataset setup appears incomplete. Please check the folder structure:")
        print(f"Expected structure: {train_dir}/[class_name]/[images]")
        print("Where class_name is one of:", ", ".join(classes))

if __name__ == "__main__":
    setup_dataset() 