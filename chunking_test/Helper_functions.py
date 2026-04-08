from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""
Helper functions for:
- loading dataset and scaler files
- extracting per-house scaling values
- splitting one house into train / val / test
- creating windowed X, y datasets for single-step or multi-step forecasting
- unscaling predictions back to raw kWh
- evaluating both single-step and multi-step forecasts
"""


def load_data(data_path, max_min_path, local_kwh_scaling):
    """
    Load the main dataset and scaling metadata.

    Returns:
        df: full parquet dataset
        local_kwh_scaler_df: per-house min/max kwh values
        global_temp_min, global_temp_max: global weather scaling values
        global_hum_min, global_hum_max: global weather scaling values
    """
    df = pd.read_parquet(data_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

    weather_scaler_df = pd.read_csv(max_min_path)
    local_kwh_scaler_df = pd.read_csv(local_kwh_scaling)

    global_temp_min = float(weather_scaler_df["global_temp_min"].iloc[0])
    global_temp_max = float(weather_scaler_df["global_temp_max"].iloc[0])

    global_hum_min = float(weather_scaler_df["global_hum_min"].iloc[0])
    global_hum_max = float(weather_scaler_df["global_hum_max"].iloc[0])

    return df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max


def extract_kwh(local_kwh_scaler_df, house_id):
    """
    Extract per-house min/max kwh values from the local scaling CSV.
    """
    kwh_min = float(
        local_kwh_scaler_df[local_kwh_scaler_df["house_id"] == house_id]["kwh_min"].item()
    )
    kwh_max = float(
        local_kwh_scaler_df[local_kwh_scaler_df["house_id"] == house_id]["kwh_max"].item()
    )
    return kwh_min, kwh_max


def get_house_split(df: pd.DataFrame, house_id: str, feature_cols):
    """
    Return train / val / test dataframes for one house,
    keeping only the requested feature columns.
    """
    house_df = df[df["LCLid"] == house_id].copy().sort_values("DateTime")

    train_df = house_df[house_df["split"] == "train"].copy()
    val_df = house_df[house_df["split"] == "val"].copy()
    test_df = house_df[house_df["split"] == "test"].copy()

    train_df = train_df[feature_cols].copy()
    val_df = val_df[feature_cols].copy()
    test_df = test_df[feature_cols].copy()

    return train_df, val_df, test_df


def make_xy(df_house: pd.DataFrame, window_size: int = 24, target_col: str = "kwh", horizon: int = 1):
    """
    Convert one split dataframe into windowed X and y arrays.

    For each sample:
    - X = the previous `window_size` hours of all input features
    - y = the next `horizon` values of the target column

    If horizon = 1:
        y shape will be (n_samples, 1)
    If horizon = 3:
        y shape will be (n_samples, 3)

    Example:
        window_size = 24
        horizon = 3

        X sample = hours [t-23 ... t]
        y sample = [t+1, t+2, t+3]
    """
    values = df_house.to_numpy(dtype=np.float32)
    target_idx = df_house.columns.get_loc(target_col)

    X = []
    y = []

    # Need enough room for both the input window and the forecast horizon
    for i in range(len(values) - window_size - horizon + 1):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size:i + window_size + horizon, target_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def unscale(arr_scaled, min_val, max_val):
    """
    Convert scaled values back to raw kWh.
    Works for both 1D and 2D arrays.
    """
    return arr_scaled * (max_val - min_val) + min_val


def evaluate_predictions(y_scaled, pred_scaled, min_val, max_val):
    """
    Single-step evaluation function.

    Inputs:
        y_scaled: shape (n_samples,) or (n_samples, 1)
        pred_scaled: same shape

    Returns:
        metrics dict, y_raw, pred_raw
    """
    y_raw = unscale(y_scaled, min_val, max_val)
    pred_raw = unscale(pred_scaled, min_val, max_val)

    y_raw = np.asarray(y_raw).reshape(-1)
    pred_raw = np.asarray(pred_raw).reshape(-1)

    pred_raw = np.clip(pred_raw, a_min = 0, a_max = None)

    rmse = np.sqrt(mean_squared_error(y_raw, pred_raw))
    mae = mean_absolute_error(y_raw, pred_raw)

    y_std = np.std(y_raw)
    y_mean = np.mean(y_raw)

    nrmse_std = rmse / y_std if y_std != 0 else np.nan
    nrmse_mean = rmse / y_mean if y_mean != 0 else np.nan

    return {
        "rmse_kwh": rmse,
        "mae_kwh": mae,
        "nrmse_std": nrmse_std,
        "nrmse_mean": nrmse_mean,
    }, y_raw, pred_raw


def evaluate_predictions_multistep(y_scaled, pred_scaled, min_val, max_val):
    """
    Multi-step evaluation function.

    Inputs:
        y_scaled: shape (n_samples, horizon)
        pred_scaled: shape (n_samples, horizon)

    Returns:
        metrics dict containing:
            rmse_t+1, rmse_t+2, ...
            mae_t+1, mae_t+2, ...
            mean_rmse_across_horizons
            mean_mae_across_horizons
        and the unscaled arrays y_raw, pred_raw
    """
    y_raw = unscale(y_scaled, min_val, max_val)
    pred_raw = unscale(pred_scaled, min_val, max_val)

    pred_raw = np.clip(pred_raw, a_min = 0, a_max = None)

    n_horizons = y_raw.shape[1]
    metrics = {}

    rmse_list = []
    mae_list = []

    for h in range(n_horizons):
        y_h = y_raw[:, h]
        pred_h = pred_raw[:, h]

        rmse_h = np.sqrt(mean_squared_error(y_h, pred_h))
        mae_h = mean_absolute_error(y_h, pred_h)

        metrics[f"rmse_t+{h+1}"] = rmse_h
        metrics[f"mae_t+{h+1}"] = mae_h

        rmse_list.append(rmse_h)
        mae_list.append(mae_h)

    metrics["mean_rmse_across_horizons"] = float(np.mean(rmse_list))
    metrics["mean_mae_across_horizons"] = float(np.mean(mae_list))

    return metrics, y_raw, pred_raw