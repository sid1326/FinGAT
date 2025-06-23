import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
DATA_FOLDER = "nifty_stocks"  # Folder containing individual stock CSV files
WEEK = 4  # Length of temporal input sequence
OUTPUT_PKL = f"dataset_week{WEEK}.pkl"

# === LOAD DATA FROM MULTIPLE FILES ===
def load_stock_data(folder_path):
    """Load data from all CSV files in the specified folder"""
    stock_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_data = []
    
    for file in stock_files:
        file_path = os.path.join(folder_path, file)
        stock_name = file.split('.')[0]  # Remove file extension
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Add stock identifier
            df['index_id'] = stock_name
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df.columns = df.columns.str.strip()  # Clean whitespace
            col_mapping = {}
            
            for req_col in required_cols:
                for actual_col in df.columns:
                    if actual_col.lower() == req_col.lower():
                        col_mapping[actual_col] = req_col.lower()
                        break
            
            df = df.rename(columns=col_mapping)

            # Rename to standard format
            df = df.rename(columns={
                'date': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            # Convert timestamp with support for both date-only and datetime formats
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

            all_data.append(df)
            print(f"✅ Loaded {stock_name}: {len(df)} records")
            
        except Exception as e:
            print(f"❌ Error loading {file}: {str(e)}")
            continue
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# === LOAD AND PREPROCESS DATA ===
print("Loading stock data files...")
df = load_stock_data(DATA_FOLDER)

# Convert timestamp and sort
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values(by=["index_id", "timestamp"], inplace=True)

print(f"Total records loaded: {len(df)}")
print(f"Unique stocks: {df['index_id'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# === PREPROCESS ===
index_ids = df["index_id"].unique()
stock_data = {}

print("\nProcessing individual stocks...")
for idx in index_ids:
    data = df[df["index_id"] == idx].copy().reset_index(drop=True)
    
    # Skip if insufficient data
    if len(data) <50:  # Need at least 50 days for features
        print(f"⚠️  Skipping {idx}: insufficient data ({len(data)} records)")
        continue

    # Normalize close price
    scaler = StandardScaler()
    data["nor_close"] = scaler.fit_transform(data[["close"]])

    # Return ratio
    data["return ratio"] = data["close"].pct_change().fillna(0)

    # Feature engineering - relative prices
    data["c_open"] = (data["open"] / data["close"]) - 1
    data["c_high"] = (data["high"] / data["close"]) - 1
    data["c_low"] = (data["low"] / data["close"]) - 1

    # Moving averages relative to current price
    for i in [5, 10, 15, 20, 25, 30]:
        data[f"{i}-days"] = (data["close"].rolling(window=i).mean() / data["close"]) - 1
        data[f"{i}-days"] = data[f"{i}-days"].fillna(0)

    stock_data[idx] = {
        "stock_price": data,
        "scaler": scaler
    }
    
    print(f"✅ Processed {idx}: {len(data)} records")

# Filter out stocks with insufficient data
valid_index_ids = list(stock_data.keys())
print(f"\nValid stocks for training: {len(valid_index_ids)}")

# === BUILD FEATURES AND LABELS ===
features = {}
Y_buy_or_not = {}

for idx in valid_index_ids:
    data = stock_data[idx]["stock_price"]
    # Use data from day 30 onwards (after moving averages are calculated)
    features[idx] = data.iloc[30:, 7:].reset_index(drop=True)  # Skip first 7 cols (basic OHLC data)
    Y_buy_or_not[idx] = (features[idx]['return ratio'] >= 0).astype(int)

# === SPLIT INTO TRAIN AND TEST ===
train_data, test_data = {}, {}
train_Y_buy_or_not, test_Y_buy_or_not = {}, {}

# Use 80% for training, 20% for testing
split_ratio = 0.8

for idx in valid_index_ids:
    split_index = int(len(features[idx]) * split_ratio)
    
    train_data[idx] = features[idx].iloc[:split_index]
    test_data[idx] = features[idx].iloc[split_index:]
    train_Y_buy_or_not[idx] = Y_buy_or_not[idx].iloc[:split_index]
    test_Y_buy_or_not[idx] = Y_buy_or_not[idx].iloc[split_index:]

print(f"Train/Test split completed. Train samples: {split_index}, Test samples: {len(features[idx]) - split_index}")

# === CREATE SEQUENCE WINDOWS ===
def create_sequences(week):
    """Create sequential data for time series prediction"""
    
    # Training sequences
    train = {}
    sequence_length = 7  # Use 7 days of data for each sequence
    
    # Create input sequences for each week
    for w in range(week):
        train_x = []
        max_sequences = len(train_data[valid_index_ids[0]]) - sequence_length - (week - 1)
        
        for seq_idx in range(max_sequences):
            sequence_batch = []
            for stock_idx in valid_index_ids:
                start_idx = seq_idx + w
                end_idx = start_idx + sequence_length
                sequence_batch.append(train_data[stock_idx].iloc[start_idx:end_idx].values)
            train_x.append(sequence_batch)
        train[f"x{w+1}"] = np.array(train_x)

    # Create target variables
    train_y1, train_y2 = [], []
    for seq_idx in range(max_sequences):
        target_idx = seq_idx + (week - 1) + sequence_length
        batch_y1, batch_y2 = [], []
        
        for stock_idx in valid_index_ids:
            batch_y1.append(train_data[stock_idx]["return ratio"].iloc[target_idx])
            batch_y2.append(train_Y_buy_or_not[stock_idx].iloc[target_idx])
        
        train_y1.append(batch_y1)
        train_y2.append(batch_y2)
    
    train["y_return ratio"] = np.array(train_y1)
    train["y_up_or_down"] = np.array(train_y2)

    # Testing sequences
    test = {}
    for w in range(week):
        test_x = []
        max_test_sequences = len(test_data[valid_index_ids[0]]) - sequence_length - (week - 1)
        
        for seq_idx in range(max_test_sequences):
            sequence_batch = []
            for stock_idx in valid_index_ids:
                start_idx = seq_idx + w
                end_idx = start_idx + sequence_length
                sequence_batch.append(test_data[stock_idx].iloc[start_idx:end_idx].values)
            test_x.append(sequence_batch)
        test[f"x{w+1}"] = np.array(test_x)

    # Create test targets
    test_y1, test_y2 = [], []
    for seq_idx in range(max_test_sequences):
        target_idx = seq_idx + (week - 1) + sequence_length
        batch_y1, batch_y2 = [], []
        
        for stock_idx in valid_index_ids:
            batch_y1.append(test_data[stock_idx]["return ratio"].iloc[target_idx])
            batch_y2.append(test_Y_buy_or_not[stock_idx].iloc[target_idx])
        
        test_y1.append(batch_y1)
        test_y2.append(batch_y2)
    
    test["y_return ratio"] = np.array(test_y1)
    test["y_up_or_down"] = np.array(test_y2)

    return {"train": train, "test": test}

# === RUN AND SAVE OUTPUT ===
print(f"\nCreating sequences with WEEK={WEEK}...")
processed = create_sequences(WEEK)

# Add metadata
processed["metadata"] = {
    "stocks": valid_index_ids,
    "total_stocks": len(valid_index_ids),
    "week_length": WEEK,
    "sequence_length": 7,
    "train_sequences": len(processed["train"]["y_return ratio"]),
    "test_sequences": len(processed["test"]["y_return ratio"]),
    "features_per_stock": processed["train"]["x1"].shape[-1],
    "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}"
}

# Save processed data
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(processed, f)

print(f"Saved processed dataset to: {OUTPUT_PKL}")
print(f"Dataset summary:")
print(f"   - Stocks: {len(valid_index_ids)}")
print(f"   - Training sequences: {len(processed['train']['y_return ratio'])}")
print(f"   - Testing sequences: {len(processed['test']['y_return ratio'])}")
print(f"   - Features per stock: {processed['train']['x1'].shape[-1]}")
print(f"   - Input shape: {processed['train']['x1'].shape}")
