import os
import requests
import pandas as pd
from dotenv import load_dotenv
from time import sleep
from tqdm import tqdm
import sys

# Load environment variables
load_dotenv()
MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')

if not MAPBOX_TOKEN:
    print("❌ ERROR: MAPBOX_ACCESS_TOKEN not found in .env file!")
    sys.exit(1)

class SatelliteImageFetcher:
    def __init__(self, zoom=18, size="400x400"):
        """
        Initialize the fetcher
        zoom: 17 (neighborhood), 18 (property+neighbors), 19 (building detail)
        size: Image dimensions (e.g., "400x400", "640x640")
        """
        self.token = MAPBOX_TOKEN
        self.zoom = zoom
        self.size = size
        self.base_url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
        
    def fetch_image(self, lat, lon, save_path):
        """Fetch a single satellite image"""
        # Construct URL: lon,lat,zoom/widthxheight
        url = f"{self.base_url}/{lon},{lat},{self.zoom}/{self.size}?access_token={self.token}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"❌ Failed: Status {response.status_code} for {save_path}")
                return False
        except Exception as e:
            print(f"❌ Error downloading {save_path}: {str(e)}")
            return False
    
    def fetch_dataset_images(self, df, output_folder, dataset_name="train"):
        """Fetch images for entire dataset"""
        os.makedirs(output_folder, exist_ok=True)
        
        successful = 0
        failed = 0
        
        print(f"\n📥 Downloading {len(df)} {dataset_name} images...")
        print(f"📁 Saving to: {output_folder}")
        print(f"🔧 Settings: Zoom={self.zoom}, Size={self.size}\n")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Fetching {dataset_name}"):
            property_id = row['id']
            lat = row['lat']
            lon = row['long']
            
            save_path = os.path.join(output_folder, f"{property_id}.jpg")
            
            # Skip if already downloaded
            if os.path.exists(save_path):
                successful += 1
                continue
            
            if self.fetch_image(lat, lon, save_path):
                successful += 1
            else:
                failed += 1
            
            # Rate limiting: Be nice to the API
            sleep(0.1)  # 10 images per second max
        
        print(f"\n✅ Download Complete!")
        print(f"   Success: {successful}/{len(df)}")
        print(f"   Failed: {failed}/{len(df)}")
        return successful, failed

def main():
    # Initialize fetcher
    fetcher = SatelliteImageFetcher(zoom=18, size="400x400")
    
    # Load datasets
    print("📂 Loading datasets...")
    train_df = pd.read_excel('../data/raw/train.xlsx')
    test_df = pd.read_excel('../data/raw/test.xlsx')
    
    print(f"   Train: {len(train_df)} properties")
    print(f"   Test: {len(test_df)} properties")
    
    # Download train images
    fetcher.fetch_dataset_images(
        train_df, 
        'data/images/train', 
        dataset_name="train"
    )
    
    # Download test images
    fetcher.fetch_dataset_images(
        test_df, 
        'data/images/test', 
        dataset_name="test"
    )
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
