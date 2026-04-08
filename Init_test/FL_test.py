import os
import gc
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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


# ==========================================
# 0. GLOBAL CONFIG
# ==========================================
SEED = 42
HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"

FEATURE_COLS = [
    "kwh",
    "hour_sin",
    "hour_cos",
    "year_sin",
    "year_cos",
    "dow_sin",
    "dow_cos",
    "weekend",
    "temperature",
    "humidity",
]

DATA_PATH = Path("selected_100_normalised_ph.parquet")
MAX_MIN_PATH = Path("global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("local_kwh_scaler.csv")

CLIENT_DATA_DIR = Path("client_npz")
CLIENT_DATA_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = CLIENT_DATA_DIR / "manifest.csv"

OUT_DIR = Path("fl_outputs_gpu")
OUT_DIR.mkdir(exist_ok=True)

# Safer defaults for stabilizing the run first.
# Once this is stable, increase them.
NUM_ROUNDS = 5
FRACTION_FIT = 0.3
FRACTION_EVALUATE = 0.2

CLIENT_NUM_CPUS = 2
CLIENT_NUM_GPUS = 1.0
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


# ==========================================
# 1. UTILS
# ==========================================
def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.2),
        Dense(HORIZON, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    all_keys = set()
    for _, metric_dict in metrics:
        all_keys.update(metric_dict.keys())

    for key in all_keys:
        weighted_sum = 0.0
        used_examples = 0
        for num_examples, metric_dict in metrics:
            if key in metric_dict:
                weighted_sum += num_examples * metric_dict[key]
                used_examples += num_examples
        aggregated[key] = weighted_sum / used_examples if used_examples > 0 else None

    return aggregated


# ==========================================
# 2. PRECOMPUTE PER-HOUSE NPZ FILES
# ==========================================
def precompute_client_npz():
    print("Creating per-house NPZ files...")

    df, *_ = Helper_functions.load_data(
        DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING
    )

    house_ids = sorted(df["LCLid"].unique())
    records = []

    for house_id in house_ids:
        train_df, val_df, _ = Helper_functions.get_house_split(
            df, house_id, FEATURE_COLS
        )

        X_train, y_train = Helper_functions.make_xy(
            train_df,
            window_size=WINDOW_SIZE,
            target_col=TARGET_COL,
            horizon=HORIZON,
        )
        X_val, y_val = Helper_functions.make_xy(
            val_df,
            window_size=WINDOW_SIZE,
            target_col=TARGET_COL,
            horizon=HORIZON,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        X_train = X_train.astype(np.float32, copy=False)
        y_train = y_train.astype(np.float32, copy=False)
        X_val = X_val.astype(np.float32, copy=False)
        y_val = y_val.astype(np.float32, copy=False)

        out_path = CLIENT_DATA_DIR / f"{house_id}.npz"
        np.savez_compressed(
            out_path,
            x_train=X_train,
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
        )

        records.append({
            "house_id": house_id,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "timesteps": int(X_train.shape[1]),
            "n_features": int(X_train.shape[2]),
            "horizon": int(y_train.shape[1]),
            "file": out_path.name,
        })

        del X_train, y_train, X_val, y_val
        gc.collect()

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    del df
    gc.collect()

    print(f"Saved {len(records)} client NPZ files to {CLIENT_DATA_DIR}")


def ensure_precomputed():
    if not MANIFEST_PATH.exists():
        precompute_client_npz()
        return

    manifest_df = pd.read_csv(MANIFEST_PATH)
    if manifest_df.empty:
        precompute_client_npz()
        return

    first_house = manifest_df.iloc[0]["house_id"]
    first_path = CLIENT_DATA_DIR / f"{first_house}.npz"
    if not first_path.exists():
        precompute_client_npz()
        return

    print(f"Found existing precomputed client data in {CLIENT_DATA_DIR}")


def load_manifest():
    manifest_df = pd.read_csv(MANIFEST_PATH)
    if manifest_df.empty:
        raise RuntimeError("Manifest is empty. Precompute step did not produce valid clients.")

    valid_house_ids = manifest_df["house_id"].tolist()

    first_house = valid_house_ids[0]
    with np.load(CLIENT_DATA_DIR / f"{first_house}.npz") as data:
        dummy_input_shape = data["x_train"].shape[1:]

    return valid_house_ids, dummy_input_shape, manifest_df


# ==========================================
# 3. REPRODUCIBILITY + DRIVER GPU CHECK
# ==========================================
tf.keras.utils.set_random_seed(SEED)
enable_gpu_memory_growth()

print(f"Global seed set to {SEED}.")
print("Driver TF GPUs:", tf.config.list_physical_devices("GPU"))


# ==========================================
# 4. PRECOMPUTE IF NEEDED, THEN LOAD MANIFEST
# ==========================================
ensure_precomputed()
valid_house_ids, dummy_input_shape, manifest_df = load_manifest()
num_clients = len(valid_house_ids)

print(f"Valid clients available for FL: {num_clients}")

if num_clients == 0:
    raise RuntimeError("No valid clients available.")


# ==========================================
# 5. FLOWER CLIENT
#    Load one house from disk on demand.
# ==========================================
class HouseClient(fl.client.NumPyClient):
    def __init__(self, model, house_id):
        self.model = model
        self.house_id = house_id

    def _load_train_arrays(self):
        path = CLIENT_DATA_DIR / f"{self.house_id}.npz"
        with np.load(path) as data:
            x_train = data["x_train"]
            y_train = data["y_train"]
        return x_train, y_train

    def _load_val_arrays(self):
        path = CLIENT_DATA_DIR / f"{self.house_id}.npz"
        with np.load(path) as data:
            x_val = data["x_val"]
            y_val = data["y_val"]
        return x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        x_train, y_train = self._load_train_arrays()

        history = self.model.fit(
            x_train,
            y_train,
            epochs=1,
            batch_size=min(BATCH_SIZE, len(x_train)),
            verbose=0,
        )

        train_loss = float(history.history["loss"][-1])
        updated_weights = self.model.get_weights()
        num_examples = int(len(x_train))

        del x_train, y_train, history
        tf.keras.backend.clear_session()
        gc.collect()

        return updated_weights, num_examples, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        x_val, y_val = self._load_val_arrays()

        loss, rmse = self.model.evaluate(
            x_val,
            y_val,
            batch_size=min(BATCH_SIZE, len(x_val)),
            verbose=0,
        )

        num_examples = int(len(x_val))

        del x_val, y_val
        tf.keras.backend.clear_session()
        gc.collect()

        return float(loss), num_examples, {"rmse": float(rmse)}


def client_fn(cid: str) -> fl.client.Client:
    tf.keras.backend.clear_session()
    gc.collect()

    house_id = valid_house_ids[int(cid)]
    model = build_model(dummy_input_shape)

    return HouseClient(model, house_id).to_client()


# ==========================================
# 6. TRACKING STRATEGY
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

        if aggregated_metrics:
            print(f"[Round {server_round}] Aggregated fit metrics: {aggregated_metrics}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        self.round_eval_log.append({
            "round": server_round,
            "aggregated_val_loss": aggregated_loss,
            "aggregated_val_rmse": (
                aggregated_metrics.get("rmse") if aggregated_metrics else None
            ),
        })

        print(f"[Round {server_round}] Aggregated val loss: {aggregated_loss}")
        if aggregated_metrics:
            print(f"[Round {server_round}] Aggregated eval metrics: {aggregated_metrics}")

        return aggregated_loss, aggregated_metrics


strategy = TrackingFedAvg(
    house_id_lookup=valid_house_ids,
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,
    min_fit_clients=max(1, int(np.ceil(num_clients * FRACTION_FIT))),
    min_evaluate_clients=(
        0 if FRACTION_EVALUATE == 0 else max(1, int(np.ceil(num_clients * FRACTION_EVALUATE)))
    ),
    min_available_clients=num_clients,
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
)


# ==========================================
# 7. MAIN
# ==========================================
def main():
    print("Starting Federated Learning Simulation...")

    _history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={
            "num_cpus": CLIENT_NUM_CPUS,
            "num_gpus": CLIENT_NUM_GPUS,
        },
        ray_init_args={
            "ignore_reinit_error": True,
            "num_gpus": 1,
        },
    )

    # ==========================================
    # 8. SAVE LOGS
    # ==========================================
    fit_log_df = pd.DataFrame(strategy.fit_selection_log)
    eval_log_df = pd.DataFrame(strategy.eval_selection_log)
    round_eval_df = pd.DataFrame(strategy.round_eval_log)

    fit_log_df.to_csv(OUT_DIR / "fl_fit_client_selection.csv", index=False)
    eval_log_df.to_csv(OUT_DIR / "fl_eval_client_selection.csv", index=False)
    round_eval_df.to_csv(OUT_DIR / "fl_round_validation_metrics.csv", index=False)

    print("Saved round/client logs.")

    # ==========================================
    # 9. QUICK CHECK FOR SPECIFIC HOUSES
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

    pd.DataFrame(fit_hits).to_csv(OUT_DIR / "fl_watch_houses_fit_hits.csv", index=False)
    pd.DataFrame(eval_hits).to_csv(OUT_DIR / "fl_watch_houses_eval_hits.csv", index=False)

    print("Saved watch-house hit tables.")

    # ==========================================
    # 10. SAVE FINAL GLOBAL MODEL
    # ==========================================
    if strategy.final_parameters is None:
        raise RuntimeError("No final aggregated parameters were captured.")

    final_weights = parameters_to_ndarrays(strategy.final_parameters)

    final_model = build_model(dummy_input_shape)
    final_model.set_weights(final_weights)
    final_model.save(OUT_DIR / f"fl_lstm64_global_round{NUM_ROUNDS}.keras")

    print("Saved final FL global model.")

    # ==========================================
    # 11. PLOT VALIDATION METRICS
    # ==========================================
    if not round_eval_df.empty and "aggregated_val_loss" in round_eval_df.columns:
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
        plt.savefig(OUT_DIR / "fl_validation_loss_vs_round.png", dpi=200)
        plt.close()

        if "aggregated_val_rmse" in round_eval_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(
                round_eval_df["round"],
                round_eval_df["aggregated_val_rmse"],
                marker="o",
                label="Aggregated validation RMSE",
            )
            plt.xlabel("Communication round")
            plt.ylabel("Validation RMSE")
            plt.title("FL validation RMSE vs communication round")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT_DIR / "fl_validation_rmse_vs_round.png", dpi=200)
            plt.close()

        print("Saved FL validation plots.")

    print("Done.")


if __name__ == "__main__":
    main()