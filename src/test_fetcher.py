from data_fetcher import SatelliteImageFetcher
import pandas as pd

# Load just 5 rows for testing
train_df = pd.read_excel('../data/raw/train.xlsx').head(5)

print("Testing with 5 sample images...")
fetcher = SatelliteImageFetcher(zoom=18, size="400x400")
fetcher.fetch_dataset_images(train_df, 'data/images/test_sample', dataset_name="test")

print("\n✅ Check data/images/test_sample/ folder to see the images!")
