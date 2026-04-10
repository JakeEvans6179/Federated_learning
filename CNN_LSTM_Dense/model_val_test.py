import tensorflow as tf
from pathlib import Path
import pandas as pd

from tensorflow.keras.models import load_model

import Helper_functions

from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
CNN-LSTM_Dense federated training - validation screening
'''




data_path = Path("selected_100_normalised_ph.parquet")

max_min_path = Path("global_weather_scaler.csv")

local_kwh_scaling = Path("local_kwh_scaler.csv")


HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"

num_chunks = 40


#model input features
feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]













df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)   #load global weather scalers and local kwh scalers
#house_ids = sorted(df["LCLid"].unique())[:15]
house_ids = sorted(df["LCLid"].unique())
#results = []


x_val = []
y_val = []


val_data = {}
    


for i, house_id in enumerate(house_ids, start = 1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, feature_cols)

    #house_x_train, house_y_train = Helper_functions.make_xy(train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    house_x_val, house_y_val = Helper_functions.make_xy(val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    

    if len(house_x_val) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    #x_train.append(house_x_train)
    #y_train.append(house_y_train)

    x_val.append(house_x_val)
    y_val.append(house_y_val)

    #store x_val, y_val and min max kwh values in dictionary sorted by house_id


    val_data[house_id] = {
        "x_val": house_x_val,
        "y_val": house_y_val,
        "kwh_min": kwh_min,
        "kwh_max": kwh_max
    }



print("Number of houses in val_data:", len(val_data))

chunk_val_metrics = {}



#First loop through all models to determine the performance of each, plot the validation loss against communication chunks
for i in range(1, num_chunks + 1):
    tf.keras.backend.clear_session()  # Clear the Keras session to free up resources
    model_path = f"chunk_checkpoints/global_chunk_{i:03d}_CNN_LSTM_Dense.keras"
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)

    val_metrics = []

    #now evaluate the model on the global validation set and plot the validation loss against communication chunks
    for house_id in val_data.keys():
        

        pred_scaled = model.predict(val_data[house_id]["x_val"], verbose=0)
        kwh_min = float(val_data[house_id]["kwh_min"])
        kwh_max = float(val_data[house_id]["kwh_max"])

        y_scaled = val_data[house_id]["y_val"]

        metrics, y_raw, pred_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=y_scaled,
        pred_scaled=pred_scaled,
        min_val=kwh_min,
        max_val=kwh_max
    )
        
        val_metrics.append(metrics["mean_rmse_across_horizons"])

    mean_val_rmse = np.mean(val_metrics)
    chunk_val_metrics[i] = mean_val_rmse
    print(f"Chunk {i}: Mean RMSE across horizons on global validation set: {mean_val_rmse:.4f}")


#plot the validation loss against communication chunks
plt.figure(figsize=(10, 6))
plt.plot(list(chunk_val_metrics.keys()), list(chunk_val_metrics.values()), marker='o')
plt.title("Mean RMSE across horizons on global validation set vs Communication Chunks")
plt.xlabel("Communication Chunk")
plt.ylabel("Mean RMSE across horizons")
plt.savefig("validation_loss_vs_chunks.png")
plt.close()

        
    

summary_df = pd.DataFrame({
    "chunk": list(chunk_val_metrics.keys()),
    "mean_rmse_kwh": list(chunk_val_metrics.values())
})
summary_df["model"] = "CNN_LSTM_Dense_federated"
summary_df.to_csv("chunk_validation_results.csv", index=False)








