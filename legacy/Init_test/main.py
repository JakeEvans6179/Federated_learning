import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
import random 
from pathlib import Path
import legacy.Init_test.Helper_functions as Helper_functions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 0. LOCK DOWN REPRODUCIBILITY (THE SEED)
# ==========================================
SEED = 42

# 1. Lock Python's random (Locks Flower's client selection)
random.seed(SEED)

# 2. Lock NumPy's random
np.random.seed(SEED)

# 3. Lock TensorFlow's random (Locks weight initialization)
tf.random.set_seed(SEED)


print(f"Global seed set to {SEED}. Deterministic mode enabled.")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================
HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"
feature_cols = [
    "kwh", "hour_sin", "hour_cos", "year_sin", "year_cos",
    "dow_sin", "dow_cos", "weekend", "temperature", "humidity"
]

data_path = Path("selected_100_normalised_ph.parquet")
max_min_path = Path("global_weather_scaler.csv")
local_kwh_scaling = Path("local_kwh_scaler.csv")

df, local_kwh_scaler_df, *_ = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)
house_ids = sorted(df["LCLid"].unique())

print("Pre-processing all 100 households into memory...")
client_data = {}

for house_id in house_ids:
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, feature_cols)
    
    X_train, y_train = Helper_functions.make_xy(train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon=HORIZON)
    X_val, y_val = Helper_functions.make_xy(val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon=HORIZON)
    
    # Only keep houses with enough data
    if len(X_train) > 0 and len(X_val) > 0:
        client_data[house_id] = {
            "x_train": X_train, "y_train": y_train,
            "x_val": X_val, "y_val": y_val
        }

# Update house_ids list to only include valid houses
valid_house_ids = list(client_data.keys())
print(f"Successfully loaded {len(valid_house_ids)} clients into memory.")

# ==========================================
# 2. THE FLOWER CLIENT CLASS
# ==========================================
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.2),
        Dense(HORIZON, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

class HouseClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # 1. The Server gives this client the global weights
        self.model.set_weights(parameters)
        
        # 2. Train LOCALLY on this specific house's data for 1 epoch
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=256, verbose=0)
        
        # 3. Return the new weights, number of samples (for weighted averaging), and a metrics dict
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, rmse = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return loss, len(self.x_val), {"rmse": rmse}

# ==========================================
# 3. THE CLIENT FACTORY
# ==========================================
# Flower calls this function whenever it needs to wake up a specific house
def client_fn(cid: str) -> fl.client.Client:
    # Flower passes 'cid' as a string ("0", "1", "2"...)
    house_id = valid_house_ids[int(cid)]
    
    # Grab the pre-processed data for this specific house
    data = client_data[house_id]
    
    # Build a fresh model skeleton
    input_shape = data["x_train"].shape[1:]
    model = build_model(input_shape)
    
    return HouseClient(model, data["x_train"], data["y_train"], data["x_val"], data["y_val"])

# ==========================================
# 4. THE FEDERATED SERVER STRATEGY
# ==========================================
num_clients = len(valid_house_ids)

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,            # Train on 30% of houses per round
    fraction_evaluate=0.2,       # Evaluate on 20% of houses per round
    min_fit_clients=int(num_clients * 0.3),
    min_evaluate_clients=int(num_clients * 0.2),
    min_available_clients=num_clients,
)

# ==========================================
# 5. START THE SIMULATION
# ==========================================
print("Starting Federated Learning Simulation")
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=50), # Number of global aggregation rounds
    strategy=strategy,
)