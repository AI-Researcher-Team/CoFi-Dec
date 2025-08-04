import os
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_model(model_name, save_dir=None):
    """
    Download a model from Hugging Face Hub with progress tracking
    """
    try:
        print(f"\nDownloading {model_name}...")
        save_path = os.path.join(save_dir, model_name.split('/')[-1]) if save_dir else None
        
        # Create directory if it doesn't exist
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ Successfully downloaded {model_name}")
        if save_path:
            print(f"  Saved to: {save_path}")
            
    except Exception as e:
        print(f"× Error downloading {model_name}: {str(e)}")
        return False
    
    return True

def main():
    # Define models to download
    models = [
        "liuhaotian/llava-v1.5-7b"
        # "Salesforce/instructblip-vicuna-7b",
        # 'lmsys/vicuna-7b-v1.1'
    ]
    
    # Ask for download directory
    save_dir = input("Enter directory to save models (press Enter for current directory): ").strip()
    if not save_dir:
        save_dir = "models"
    
    print("\n📥 Starting downloads...")
    
    # Download each model
    for model in models:
        success = download_model(model, save_dir)
        if not success:
            print(f"\nWarning: Failed to download {model}")
    
    print("\n✨ Download process completed!")

if __name__ == "__main__":
    main()