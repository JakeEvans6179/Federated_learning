from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import flwr as fl

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from flwr.common import parameters_to_ndarrays

import Helper_functions

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ==========================================
# 0. REPRODUCIBILITY
# ==========================================
SEED = 42
tf.keras.utils.set_random_seed(SEED)
#tf.config.experimental.enable_op_determinism()

print(f"Global seed set to {SEED}.")
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


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

out_dir = Path("fl_outputs")
out_dir.mkdir(exist_ok=True)

df, local_kwh_scaler_df, *_ = Helper_functions.load_data(
    data_path, max_min_path, local_kwh_scaling
)

house_ids = sorted(df["LCLid"].unique())

print("Pre-processing households into memory...")
client_data = {}

for house_id in house_ids:
    train_df, val_df, _ = Helper_functions.get_house_split(df, house_id, feature_cols)

    X_train, y_train = Helper_functions.make_xy(
        train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon=HORIZON
    )
    X_val, y_val = Helper_functions.make_xy(
        val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon=HORIZON
    )

    if len(X_train) > 0 and len(X_val) > 0:
        client_data[house_id] = {
            "x_train": X_train,
            "y_train": y_train,
            "x_val": X_val,
            "y_val": y_val,
        }
        #client_data[house_id]["x_val"] = X_val  # keep explicit

valid_house_ids = list(client_data.keys())
num_clients = len(valid_house_ids)

print(f"Successfully loaded {num_clients} valid clients into memory.")


# ==========================================
# 2. MODEL
# ==========================================
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.2),
        Dense(HORIZON, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


# Need one dummy model for saving final aggregated weights later
dummy_input_shape = next(iter(client_data.values()))["x_train"].shape[1:]


# ==========================================
# 3. FLOWER CLIENT
# ==========================================
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
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=256,
            verbose=0,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, rmse = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return loss, len(self.x_val), {"rmse": float(rmse)}


def client_fn(cid: str) -> fl.client.Client:
    tf.keras.backend.clear_session() # <--- Clears the GPU RAM before building
    house_id = valid_house_ids[int(cid)]
    data = client_data[house_id]
    model = build_model(data["x_train"].shape[1:])
    return HouseClient(
        model,
        data["x_train"], data["y_train"],
        data["x_val"], data["y_val"],
    )


# ==========================================
# 4. TRACKING STRATEGY
# ==========================================
class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, house_id_lookup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.house_id_lookup = house_id_lookup
        self.fit_selection_log = []
        self.eval_selection_log = []
        self.round_eval_log = []
        self.final_parameters = None

    def configure_fit(self, server_round, parameters, client_manager):
        fit_cfg = super().configure_fit(server_round, parameters, client_manager)

        selected = []
        for client_proxy, _ in fit_cfg:
            cid = client_proxy.cid
            house_id = self.house_id_lookup[int(cid)]
            selected.append(house_id)

        self.fit_selection_log.append({
            "round": server_round,
            "n_fit_clients": len(selected),
            "fit_house_ids": selected,
        })
        print(f"[Round {server_round}] Fit clients: {selected}")
        return fit_cfg

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_cfg = super().configure_evaluate(server_round, parameters, client_manager)

        if eval_cfg is None:
            return None

        selected = []
        for client_proxy, _ in eval_cfg:
            cid = client_proxy.cid
            house_id = self.house_id_lookup[int(cid)]
            selected.append(house_id)

        self.eval_selection_log.append({
            "round": server_round,
            "n_eval_clients": len(selected),
            "eval_house_ids": selected,
        })
        print(f"[Round {server_round}] Eval clients: {selected}")
        return eval_cfg

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        self.round_eval_log.append({
            "round": server_round,
            "aggregated_val_loss": aggregated_loss,
        })
        print(f"[Round {server_round}] Aggregated val loss: {aggregated_loss}")
        return aggregated_loss, aggregated_metrics


strategy = TrackingFedAvg(
    house_id_lookup=valid_house_ids,
    fraction_fit=0.3,
    fraction_evaluate=0.2,
    min_fit_clients=int(num_clients * 0.3),
    min_evaluate_clients=max(1, int(num_clients * 0.2)),
    min_available_clients=num_clients,
)


# ==========================================
# 5. RUN SIMULATION
# ==========================================
print("Starting Federated Learning Simulation...")
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)


# ==========================================
# 6. SAVE LOGS
# ==========================================
fit_log_df = pd.DataFrame(strategy.fit_selection_log)
eval_log_df = pd.DataFrame(strategy.eval_selection_log)
round_eval_df = pd.DataFrame(strategy.round_eval_log)

fit_log_df.to_csv(out_dir / "fl_fit_client_selection.csv", index=False)
eval_log_df.to_csv(out_dir / "fl_eval_client_selection.csv", index=False)
round_eval_df.to_csv(out_dir / "fl_round_validation_loss.csv", index=False)

print("Saved round/client logs.")


# ==========================================
# 7. QUICK CHECK FOR SPECIFIC HOUSES
# ==========================================
watch_houses = {"MAC000012", "MAC001522"}

fit_hits = []
for row in strategy.fit_selection_log:
    chosen = set(row["fit_house_ids"])
    hit = sorted(list(chosen.intersection(watch_houses)))
    if hit:
        fit_hits.append({"round": row["round"], "houses": hit})

eval_hits = []
for row in strategy.eval_selection_log:
    chosen = set(row["eval_house_ids"])
    hit = sorted(list(chosen.intersection(watch_houses)))
    if hit:
        eval_hits.append({"round": row["round"], "houses": hit})

pd.DataFrame(fit_hits).to_csv(out_dir / "fl_watch_houses_fit_hits.csv", index=False)
pd.DataFrame(eval_hits).to_csv(out_dir / "fl_watch_houses_eval_hits.csv", index=False)

print("Saved watch-house hit tables.")


# ==========================================
# 8. SAVE FINAL GLOBAL MODEL
# ==========================================
if strategy.final_parameters is None:
    raise RuntimeError("No final aggregated parameters were captured.")

final_weights = parameters_to_ndarrays(strategy.final_parameters)

final_model = build_model(dummy_input_shape)
final_model.set_weights(final_weights)
final_model.save(out_dir / "fl_lstm64_global_round50.keras")

print("Saved final FL global model.")


# ==========================================
# 9. PLOT VALIDATION LOSS VS ROUND
# ==========================================
if not round_eval_df.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(
        round_eval_df["round"],
        round_eval_df["aggregated_val_loss"],
        marker="o",
        label="Aggregated validation loss",
    )
    plt.xlabel("Communication round")
    plt.ylabel("Validation loss (MSE)")
    plt.title("FL validation loss vs communication round")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fl_validation_loss_vs_round.png", dpi=200)
    plt.close()

    print("Saved FL validation-loss plot.")