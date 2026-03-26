import json
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy.orm import Session
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

from app.models.fish_training_price import FishTrainingPrice

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = BASE_DIR / "ml_models"
DEPLOYED_DIR = ML_MODELS_DIR / "deployed"

ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)

ANN_BATCH_SIZE = 256
ANN_EPOCHS = 80
ANN_ALPHA = 0.6
VAL_WEEKS = 12
TEST_WEEKS = 12


def normalize_month_to_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.6:
        return numeric.round().astype("Int64")

    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    s2 = s.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return s2.map(month_map).astype("Int64")


def normalize_week_to_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    nums = s.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(nums, errors="coerce").astype("Int64")


def build_week_start(year: int, month: int, week: int) -> pd.Timestamp:
    month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
    return month_start + pd.Timedelta(days=(int(week) - 1) * 7)


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(
        np.abs((np.array(y_true) - np.array(y_pred)) / np.clip(np.abs(y_true), 1e-6, None))
    ) * 100
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(mse), float(mape), float(r2)


def get_deployed_model_info():
    ann_model_path = DEPLOYED_DIR / "ann_xgb_hybrid_ann_model.keras"
    xgb_model_path = DEPLOYED_DIR / "ann_xgb_hybrid_xgb_model.pkl"
    meta_path = DEPLOYED_DIR / "ann_xgb_hybrid_metadata.json"

    missing = []

    if not ann_model_path.exists():
        missing.append("ann_xgb_hybrid_ann_model.keras")
    if not xgb_model_path.exists():
        missing.append("ann_xgb_hybrid_xgb_model.pkl")
    if not meta_path.exists():
        missing.append("ann_xgb_hybrid_metadata.json")

    if missing:
        raise FileNotFoundError(f"Missing deployed model files: {missing}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "modelName": metadata.get("model_name", "Hybrid_ANN_XGBoost"),
        "trainedAt": metadata.get("trained_at"),
        "bestWeights": {
            "ANN": metadata.get("best_w_ann"),
            "XGBoost": metadata.get("best_w_xgb"),
        },
        "annAlpha": metadata.get("ann_alpha"),
        "metrics": metadata.get("metrics", {}),
        "files": {
            "annModel": str(ann_model_path),
            "xgbModel": str(xgb_model_path),
            "metadata": str(meta_path),
        },
    }


def load_training_dataframe_from_db(db: Session) -> pd.DataFrame:
    rows = db.query(FishTrainingPrice).all()

    if not rows:
        raise ValueError("No rows found in fish_training_prices. Sync long format data first.")

    data = [
        {
            "Sinhala Name": row.sinhala_name,
            "Common Name": row.common_name,
            "Year": row.year,
            "Month": row.month,
            "Week": row.week,
            "Price": float(row.price) if row.price is not None else None,
        }
        for row in rows
    ]

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("fish_training_prices returned no usable data.")

    return df


def add_tabular_lags(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.sort_values(["fish_id", "week_start"]).copy()

    for lag_no in [1, 2, 3, 4]:
        out[f"lag_{lag_no}"] = out.groupby("fish_id")["price"].shift(lag_no)

    out["roll4_mean"] = (
        out.groupby("fish_id")["price"]
        .shift(1)
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["diff_1"] = out["lag_1"] - out["lag_2"]
    out["diff_2"] = out["lag_2"] - out["lag_3"]
    out["pct_change_1"] = (
        (out["lag_1"] - out["lag_2"]) /
        np.clip(np.abs(out["lag_2"]), 1e-6, None)
    )

    return out


def _prepare_training_artifacts(db: Session):
    df = load_training_dataframe_from_db(db)

    required = ["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    df = df.rename(columns={
        "Sinhala Name": "sinhala_name",
        "Common Name": "common_name",
        "Year": "year",
        "Month": "month",
        "Week": "week_in_month",
        "Price": "price",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = normalize_month_to_int(df["month"])
    df["week_in_month"] = normalize_week_to_int(df["week_in_month"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["sinhala_name", "common_name", "year", "month", "week_in_month", "price"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["week_in_month"] = df["week_in_month"].astype(int)
    df["price"] = df["price"].astype(np.float32)

    df = df[df["week_in_month"].between(1, 4)].copy()

    df["fish_id"] = (
        df["sinhala_name"].astype(str).str.strip()
        + " | "
        + df["common_name"].astype(str).str.strip()
    )

    df["week_start"] = df.apply(
        lambda r: build_week_start(r["year"], r["month"], r["week_in_month"]),
        axis=1,
    )
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    df = (
        df.groupby(["fish_id", "week_start", "week_end"], as_index=False)
        .agg(
            sinhala_name=("sinhala_name", "first"),
            common_name=("common_name", "first"),
            year=("year", "first"),
            month=("month", "first"),
            week_in_month=("week_in_month", "first"),
            price=("price", "mean"),
        )
    )

    df = df.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows found after preprocessing fish_training_prices.")

    unique_fish = sorted(df["fish_id"].dropna().unique().tolist())
    fish_token_map = {fish: f"fish_{i}" for i, fish in enumerate(unique_fish)}
    df["fish_token"] = df["fish_id"].map(fish_token_map).fillna("UNKNOWN")

    df["holiday_count"] = 0.0
    df["is_holiday_week"] = 0.0
    df["poya_count"] = 0.0
    df["temp_mean"] = 0.0
    df["precip_sum"] = 0.0
    df["wind_max"] = 0.0
    df["humidity_mean"] = 0.0
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["year_trend"] = df["year"] - df["year"].min()

    tab_df = add_tabular_lags(df)

    num_cols = [
        "week_in_month",
        "holiday_count", "is_holiday_week", "poya_count",
        "temp_mean", "precip_sum", "wind_max", "humidity_mean",
        "month_sin", "month_cos",
        "year_trend",
        "lag_1", "lag_2", "lag_3", "lag_4",
        "roll4_mean", "diff_1", "diff_2", "pct_change_1",
    ]

    required_lags = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "roll4_mean", "diff_1", "diff_2", "pct_change_1",
    ]

    tab_df = tab_df.dropna(subset=required_lags).copy()

    if len(tab_df) < 50:
        raise ValueError("Not enough processed rows after lag creation to train candidate model.")

    unique_weeks = sorted(tab_df["week_start"].drop_duplicates())
    if len(unique_weeks) < (VAL_WEEKS + TEST_WEEKS + 5):
        raise ValueError("Not enough week history to create train/validation/test splits.")

    test_start = unique_weeks[-TEST_WEEKS]
    val_start = unique_weeks[-(VAL_WEEKS + TEST_WEEKS)]

    train_df = tab_df[tab_df["week_start"] < val_start].copy()
    val_df = tab_df[(tab_df["week_start"] >= val_start) & (tab_df["week_start"] < test_start)].copy()
    test_df = tab_df[tab_df["week_start"] >= test_start].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/validation/test split failed.")

    for col in num_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    train_df[num_cols] = train_df[num_cols].fillna(0.0).astype(np.float32)
    val_df[num_cols] = val_df[num_cols].fillna(0.0).astype(np.float32)
    test_df[num_cols] = test_df[num_cols].fillna(0.0).astype(np.float32)

    return {
        "df": df,
        "tab_df": tab_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "num_cols": num_cols,
        "fish_token_map": fish_token_map,
    }


def _train_hybrid_core(db: Session):
    prepared = _prepare_training_artifacts(db)

    df = prepared["df"]
    tab_df = prepared["tab_df"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    num_cols = prepared["num_cols"]
    fish_token_map = prepared["fish_token_map"]

    # ANN
    ann_train = train_df.copy()
    ann_val = val_df.copy()
    ann_test = test_df.copy()

    ann_train["delta"] = ann_train["price"] - ann_train["lag_1"]
    ann_val["delta"] = ann_val["price"] - ann_val["lag_1"]
    ann_test["delta"] = ann_test["price"] - ann_test["lag_1"]

    def make_ann_ds(df_in, y_col, shuffle=True):
        fish_tensor = tf.constant(
            df_in["fish_token"].fillna("UNKNOWN").astype(str).tolist(),
            dtype=tf.string,
            shape=(len(df_in), 1),
        )
        num_tensor = df_in[num_cols].to_numpy(dtype=np.float32)
        y_tensor = df_in[y_col].to_numpy(dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((
            {
                "fish_id": fish_tensor,
                "num_features": num_tensor,
            },
            y_tensor,
        ))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(df_in), 20000), seed=SEED)
        return ds.batch(ANN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ann_train_ds = make_ann_ds(ann_train, "delta", shuffle=True)
    ann_val_ds = make_ann_ds(ann_val, "delta", shuffle=False)

    fish_lookup = layers.StringLookup(output_mode="int")
    fish_lookup.adapt(ann_train["fish_token"].fillna("UNKNOWN").astype(str).values)

    vocab_size = fish_lookup.vocabulary_size()
    embed_dim = int(min(32, max(8, vocab_size // 2)))

    normalizer = layers.Normalization()
    normalizer.adapt(ann_train[num_cols].astype(np.float32).values)

    fish_in = keras.Input(shape=(1,), dtype=tf.string, name="fish_id")
    num_in = keras.Input(shape=(len(num_cols),), dtype=tf.float32, name="num_features")

    xf = fish_lookup(fish_in)
    xf = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(xf)
    xf = layers.Flatten()(xf)

    xn = normalizer(num_in)

    x = layers.Concatenate()([xf, xn])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    ann_model = keras.Model(inputs={"fish_id": fish_in, "num_features": num_in}, outputs=out)
    ann_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    ann_model.fit(
        ann_train_ds,
        validation_data=ann_val_ds,
        epochs=ANN_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    X_ann_test = {
        "fish_id": tf.constant(
            ann_test["fish_token"].fillna("UNKNOWN").astype(str).tolist(),
            dtype=tf.string,
            shape=(len(ann_test), 1),
        ),
        "num_features": ann_test[num_cols].to_numpy(dtype=np.float32),
    }

    ann_pred_delta = ann_model.predict(X_ann_test, verbose=0).reshape(-1)
    ann_baseline = ann_test["lag_1"].to_numpy(dtype=np.float32)
    ann_pred_test = ann_baseline + (ANN_ALPHA * ann_pred_delta)

    ann_mae, ann_rmse, ann_mse, ann_mape, ann_r2 = eval_metrics(
        ann_test["price"].values,
        ann_pred_test,
    )

    # XGBoost
    xgb_feature_train = pd.get_dummies(
        train_df[["fish_id"] + num_cols].copy(),
        columns=["fish_id"],
        drop_first=False,
    )
    xgb_feature_val = pd.get_dummies(
        val_df[["fish_id"] + num_cols].copy(),
        columns=["fish_id"],
        drop_first=False,
    )
    xgb_feature_test = pd.get_dummies(
        test_df[["fish_id"] + num_cols].copy(),
        columns=["fish_id"],
        drop_first=False,
    )

    feature_cols = sorted(
        set(xgb_feature_train.columns)
        | set(xgb_feature_val.columns)
        | set(xgb_feature_test.columns)
    )

    xgb_feature_train = xgb_feature_train.reindex(columns=feature_cols, fill_value=0).astype(np.float32)
    xgb_feature_val = xgb_feature_val.reindex(columns=feature_cols, fill_value=0).astype(np.float32)
    xgb_feature_test = xgb_feature_test.reindex(columns=feature_cols, fill_value=0).astype(np.float32)

    y_train = train_df["price"].to_numpy(dtype=np.float32)
    y_val = val_df["price"].to_numpy(dtype=np.float32)
    y_test = test_df["price"].to_numpy(dtype=np.float32)

    xgb_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=SEED,
        tree_method="hist",
    )

    xgb_model.fit(
        xgb_feature_train,
        y_train,
        eval_set=[(xgb_feature_val, y_val)],
        verbose=False,
    )

    xgb_pred_test = xgb_model.predict(xgb_feature_test)

    xgb_mae, xgb_rmse, xgb_mse, xgb_mape, xgb_r2 = eval_metrics(
        y_test,
        xgb_pred_test,
    )

    # Hybrid search
    weight_sets = [
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.4, 0.6),
    ]

    results = []
    for w_ann, w_xgb in weight_sets:
        pred = (w_ann * ann_pred_test) + (w_xgb * xgb_pred_test)
        mae, rmse, mse, mape, r2 = eval_metrics(y_test, pred)
        results.append({
            "w_ann": float(w_ann),
            "w_xgb": float(w_xgb),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MSE": float(mse),
            "MAPE": float(mape),
            "R2": float(r2),
        })

    results_df = pd.DataFrame(results).sort_values("MAPE").reset_index(drop=True)
    best = results_df.iloc[0]

    best_w_ann = float(best["w_ann"])
    best_w_xgb = float(best["w_xgb"])

    hybrid_pred = (best_w_ann * ann_pred_test) + (best_w_xgb * xgb_pred_test)
    hyb_mae, hyb_rmse, hyb_mse, hyb_mape, hyb_r2 = eval_metrics(y_test, hybrid_pred)

    # Full models for deploy only
    known_full_ann = tab_df.copy()
    known_full_ann["delta"] = known_full_ann["price"] - known_full_ann["lag_1"]

    fish_lookup_full = layers.StringLookup(output_mode="int")
    fish_lookup_full.adapt(known_full_ann["fish_token"].fillna("UNKNOWN").astype(str).values)

    vocab_full = fish_lookup_full.vocabulary_size()
    embed_full = int(min(32, max(8, vocab_full // 2)))

    normalizer_full = layers.Normalization()
    normalizer_full.adapt(known_full_ann[num_cols].astype(np.float32).values)

    fish_in2 = keras.Input(shape=(1,), dtype=tf.string, name="fish_id")
    num_in2 = keras.Input(shape=(len(num_cols),), dtype=tf.float32, name="num_features")

    xf2 = fish_lookup_full(fish_in2)
    xf2 = layers.Embedding(input_dim=vocab_full, output_dim=embed_full)(xf2)
    xf2 = layers.Flatten()(xf2)
    xn2 = normalizer_full(num_in2)

    z = layers.Concatenate()([xf2, xn2])
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(32, activation="relu")(z)
    out2 = layers.Dense(1)(z)

    ann_model_full = keras.Model(inputs={"fish_id": fish_in2, "num_features": num_in2}, outputs=out2)
    ann_model_full.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(),
    )

    full_ds = tf.data.Dataset.from_tensor_slices((
        {
            "fish_id": tf.constant(
                known_full_ann["fish_token"].fillna("UNKNOWN").astype(str).tolist(),
                dtype=tf.string,
                shape=(len(known_full_ann), 1),
            ),
            "num_features": known_full_ann[num_cols].to_numpy(dtype=np.float32),
        },
        known_full_ann["delta"].to_numpy(dtype=np.float32),
    )).shuffle(buffer_size=min(len(known_full_ann), 20000), seed=SEED).batch(ANN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ann_model_full.fit(full_ds, epochs=40, verbose=0)

    xgb_all_known = pd.get_dummies(
        tab_df[["fish_id"] + num_cols].copy(),
        columns=["fish_id"],
        drop_first=False,
    ).reindex(columns=feature_cols, fill_value=0).astype(np.float32)

    xgb_model_full = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=SEED,
        tree_method="hist",
    )
    xgb_model_full.fit(xgb_all_known, tab_df["price"].to_numpy(dtype=np.float32))

    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    version_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    metadata = {
        "model_name": "Hybrid_ANN_XGBoost",
        "trained_at": trained_at,
        "best_w_ann": best_w_ann,
        "best_w_xgb": best_w_xgb,
        "ann_alpha": float(ANN_ALPHA),
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "fish_token_map": fish_token_map,
        "metrics": {
            "ANN": {
                "MAE": ann_mae,
                "RMSE": ann_rmse,
                "MAPE": ann_mape,
                "R2": ann_r2,
            },
            "XGBoost": {
                "MAE": xgb_mae,
                "RMSE": xgb_rmse,
                "MAPE": xgb_mape,
                "R2": xgb_r2,
            },
            "Hybrid_ANN_XGBoost": {
                "MAE": hyb_mae,
                "RMSE": hyb_rmse,
                "MAPE": hyb_mape,
                "R2": hyb_r2,
            },
        },
    }

    return {
        "version": version_name,
        "trainedAt": trained_at,
        "metrics": {
            "MAE": hyb_mae,
            "RMSE": hyb_rmse,
            "MAPE": hyb_mape,
            "R2": hyb_r2,
            "best_w_ann": best_w_ann,
            "best_w_xgb": best_w_xgb,
        },
        "allMetrics": metadata["metrics"],
        "fishCount": int(df["fish_id"].nunique()),
        "trainRows": int(len(train_df)),
        "valRows": int(len(val_df)),
        "testRows": int(len(test_df)),
        "ann_model_full": ann_model_full,
        "xgb_model_full": xgb_model_full,
        "metadata": metadata,
    }


def train_full_hybrid_ann_xgb_pipeline(db: Session):
    result = _train_hybrid_core(db)

    return {
        "message": "Candidate model trained successfully from fish_training_prices",
        "version": result["version"],
        "trainedAt": result["trainedAt"],
        "metrics": result["metrics"],
        "allMetrics": result["allMetrics"],
        "fishCount": result["fishCount"],
        "trainRows": result["trainRows"],
        "valRows": result["valRows"],
        "testRows": result["testRows"],
    }


def train_and_save_deployed_hybrid_model(db: Session, version_name: str):
    result = _train_hybrid_core(db)

    ann_path = DEPLOYED_DIR / "ann_xgb_hybrid_ann_model.keras"
    xgb_path = DEPLOYED_DIR / "ann_xgb_hybrid_xgb_model.pkl"
    meta_path = DEPLOYED_DIR / "ann_xgb_hybrid_metadata.json"

    result["ann_model_full"].save(ann_path)
    joblib.dump(result["xgb_model_full"], xgb_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result["metadata"], f, indent=2, ensure_ascii=False)

    return {
        "message": "Deployed model trained and saved successfully",
        "version": version_name,
        "trainedAt": result["trainedAt"],
        "metrics": result["metrics"],
        "files": {
            "annModel": str(ann_path),
            "xgbModel": str(xgb_path),
            "metadata": str(meta_path),
        },
    }