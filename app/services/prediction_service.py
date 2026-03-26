import os
import numpy as np
import pandas as pd
import tensorflow as tf

from app.services.model_loader import load_ann_xgb_hybrid

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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


def build_week_start(year, month, week):
    month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
    return month_start + pd.Timedelta(days=(int(week) - 1) * 7)


def get_next_week_label(last_year, last_month, last_week):
    if last_week < 4:
        return last_year, last_month, last_week + 1
    if last_month < 12:
        return last_year, last_month + 1, 1
    return last_year + 1, 1, 1


def week_suffix(week: int) -> str:
    if week == 1:
        return "st"
    if week == 2:
        return "nd"
    if week == 3:
        return "rd"
    return "th"


def month_int_to_name(month: int) -> str:
    month_map = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    return month_map.get(int(month), str(month))


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


def _safe_string_tensor(values):
    cleaned = pd.Series(values).fillna("UNKNOWN").astype(str).tolist()
    return tf.constant(cleaned, dtype=tf.string, shape=(len(cleaned), 1))


def generate_next_week_predictions_with_saved_hybrid(long_csv_path: str):
    ann_model, xgb_model, metadata = load_ann_xgb_hybrid()


    fish_token_map = metadata.get("fish_token_map", {})
    num_cols = metadata.get("num_cols", [])
    feature_cols = metadata.get("feature_cols", [])
    best_w_ann = float(metadata.get("best_w_ann", 0.5))
    best_w_xgb = float(metadata.get("best_w_xgb", 0.5))
    ann_alpha = float(metadata.get("ann_alpha", 1.0))

    if not num_cols:
        raise ValueError("Metadata missing 'num_cols'.")
    if not feature_cols:
        raise ValueError("Metadata missing 'feature_cols'.")
    if not fish_token_map:
        raise ValueError("Metadata missing 'fish_token_map'.")

    if not os.path.exists(long_csv_path):
        raise FileNotFoundError(f"Long format dataset not found: {long_csv_path}")

    df = pd.read_csv(long_csv_path)

    required = ["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in long format file: {missing}")

    df = df.rename(columns={
        "Sinhala Name": "sinhala_name",
        "Common Name": "common_name",
        "Year": "year",
        "Month": "month",
        "Week": "week_in_month",
        "Price": "price",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = normalize_month_to_int(df["month"])
    df["week_in_month"] = normalize_week_to_int(df["week_in_month"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(
        subset=["sinhala_name", "common_name", "year", "month", "week_in_month", "price"]
    ).copy()

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

    df["fish_token"] = df["fish_id"].map(fish_token_map)
    df["fish_token"] = df["fish_token"].fillna("UNKNOWN").astype(str)

    df["week_start"] = df.apply(
        lambda r: build_week_start(r["year"], r["month"], r["week_in_month"]),
        axis=1,
    )
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    # Merge duplicate weekly rows if any
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

    # restore fish_token after groupby
    df["fish_token"] = df["fish_id"].map(fish_token_map)
    df["fish_token"] = df["fish_token"].fillna("UNKNOWN").astype(str)

    df = df.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows found after preprocessing.")

    last_row = df.sort_values(["year", "month", "week_in_month"]).tail(1).iloc[0]

    pred_year, pred_month, pred_week = get_next_week_label(
        int(last_row["year"]),
        int(last_row["month"]),
        int(last_row["week_in_month"]),
    )

    pred_week_start = build_week_start(pred_year, pred_month, pred_week)

    future = df[["fish_id", "fish_token", "sinhala_name", "common_name"]].drop_duplicates().copy()
    future["fish_token"] = future["fish_token"].fillna("UNKNOWN").astype(str)

    future["week_start"] = pred_week_start
    future["week_end"] = pred_week_start + pd.Timedelta(days=6)
    future["price"] = np.nan
    future["year"] = pred_year
    future["month"] = pred_month
    future["week_in_month"] = pred_week

    # Placeholder external features
    future["holiday_count"] = 0.0
    future["is_holiday_week"] = 0.0
    future["poya_count"] = 0.0
    future["temp_mean"] = 0.0
    future["precip_sum"] = 0.0
    future["wind_max"] = 0.0
    future["humidity_mean"] = 0.0
    future["month_sin"] = np.sin(2 * np.pi * future["month"] / 12.0).astype(np.float32)
    future["month_cos"] = np.cos(2 * np.pi * future["month"] / 12.0).astype(np.float32)
    future["year_trend"] = (future["year"] - df["year"].min()).astype(np.float32)

    known_df = df.copy()
    known_df["holiday_count"] = 0.0
    known_df["is_holiday_week"] = 0.0
    known_df["poya_count"] = 0.0
    known_df["temp_mean"] = 0.0
    known_df["precip_sum"] = 0.0
    known_df["wind_max"] = 0.0
    known_df["humidity_mean"] = 0.0
    known_df["month_sin"] = np.sin(2 * np.pi * known_df["month"] / 12.0).astype(np.float32)
    known_df["month_cos"] = np.cos(2 * np.pi * known_df["month"] / 12.0).astype(np.float32)
    known_df["year_trend"] = (known_df["year"] - known_df["year"].min()).astype(np.float32)

    df_all = pd.concat([known_df, future], ignore_index=True)
    df_all = df_all.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

    tab_df = add_tabular_lags(df_all)

    required_lags = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "roll4_mean", "diff_1", "diff_2", "pct_change_1",
    ]

    future_df = tab_df[tab_df["week_start"] == pred_week_start].copy()
    future_df = future_df.dropna(subset=required_lags).copy()

    if future_df.empty:
        raise ValueError("No future rows available after lag creation.")

    # Ensure all expected numeric feature columns exist
    for col in num_cols:
        if col not in future_df.columns:
            future_df[col] = 0.0

    # Force exact types
    for col in num_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")

    future_df[num_cols] = future_df[num_cols].replace([np.inf, -np.inf], np.nan)
    future_df[num_cols] = future_df[num_cols].fillna(0.0)

    for col in num_cols:
        future_df[col] = future_df[col].astype(np.float32)

    future_df["fish_token"] = future_df["fish_token"].fillna("UNKNOWN").astype(str)
    future_df["fish_id"] = future_df["fish_id"].fillna("UNKNOWN").astype(str)

    # ANN input
    fish_id_input = _safe_string_tensor(future_df["fish_token"])
    num_features_input = tf.constant(
        np.asarray(future_df[num_cols].values, dtype=np.float32),
        dtype=tf.float32,
    )

    x_ann_future = {
        "fish_id": fish_id_input,
        "num_features": num_features_input,
    }

    ann_future_delta = ann_model.predict(x_ann_future, verbose=0).reshape(-1)
    ann_pred = future_df["lag_1"].to_numpy(dtype=np.float32) + (ann_alpha * ann_future_delta)

    # XGBoost input
    xgb_feature_future = pd.get_dummies(
        future_df[["fish_id"] + num_cols].copy(),
        columns=["fish_id"],
        drop_first=False,
    )

    xgb_feature_future = xgb_feature_future.reindex(columns=feature_cols, fill_value=0)
    xgb_feature_future = xgb_feature_future.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    xgb_feature_future = xgb_feature_future.astype(np.float32)

    xgb_pred = xgb_model.predict(xgb_feature_future)

    final_pred = (best_w_ann * ann_pred) + (best_w_xgb * xgb_pred)

    out = future_df[["sinhala_name", "common_name", "year", "month", "week_in_month"]].copy()
    out["Predicted_Price"] = np.round(final_pred, 2)

    out = out.rename(columns={
        "sinhala_name": "Sinhala Name",
        "common_name": "Common Name",
        "year": "Year",
        "month": "Month",
        "week_in_month": "Week",
    })

    # convert month number to month name
    out["Month"] = out["Month"].apply(month_int_to_name)

    display_week_label = (
        f"{pred_week}{week_suffix(pred_week)} week of {month_int_to_name(pred_month)} {pred_year}"
    )
    out["Week_Label"] = display_week_label

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hybrid_predictions_{timestamp}.csv"
    save_path = os.path.join(UPLOAD_DIR, filename)

    out.to_csv(save_path, index=False, encoding="utf-8-sig")

    return {
        "filename": filename,
        "date": display_week_label,
        "rowCount": int(len(out)),
        "modelName": "Hybrid_ANN_XGBoost",
        "preview": out.to_dict(orient="records"),
    }


# def generate_next_week_predictions_with_saved_hybrid(long_csv_path: str):
#     import traceback

#     print("🚀 START prediction function")

#     try:
#         print("📦 Loading models...")
#         ann_model, xgb_model, metadata = load_ann_xgb_hybrid()
#         print("✅ Models loaded")

#         fish_token_map = metadata.get("fish_token_map", {})
#         num_cols = metadata.get("num_cols", [])
#         feature_cols = metadata.get("feature_cols", [])
#         best_w_ann = float(metadata.get("best_w_ann", 0.5))
#         best_w_xgb = float(metadata.get("best_w_xgb", 0.5))
#         ann_alpha = float(metadata.get("ann_alpha", 1.0))

#         print("📊 Metadata loaded")

#         if not num_cols:
#             raise ValueError("Metadata missing 'num_cols'.")
#         if not feature_cols:
#             raise ValueError("Metadata missing 'feature_cols'.")
#         if not fish_token_map:
#             raise ValueError("Metadata missing 'fish_token_map'.")

#         print("📂 Loading CSV...")
#         df = pd.read_csv(long_csv_path)
#         print("✅ CSV shape:", df.shape)

#         required = ["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]
#         missing = [c for c in required if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing columns: {missing}")

#         print("🧹 Cleaning data...")
#         df = df.rename(columns={
#             "Sinhala Name": "sinhala_name",
#             "Common Name": "common_name",
#             "Year": "year",
#             "Month": "month",
#             "Week": "week_in_month",
#             "Price": "price",
#         })

#         df["year"] = pd.to_numeric(df["year"], errors="coerce")
#         df["month"] = normalize_month_to_int(df["month"])
#         df["week_in_month"] = normalize_week_to_int(df["week_in_month"])
#         df["price"] = pd.to_numeric(df["price"], errors="coerce")

#         df = df.dropna(subset=["sinhala_name", "common_name", "year", "month", "week_in_month", "price"]).copy()

#         df["year"] = df["year"].astype(int)
#         df["month"] = df["month"].astype(int)
#         df["week_in_month"] = df["week_in_month"].astype(int)
#         df["price"] = df["price"].astype(np.float32)

#         df["fish_id"] = df["sinhala_name"] + " | " + df["common_name"]
#         df["fish_token"] = df["fish_id"].map(fish_token_map).fillna("UNKNOWN").astype(str)

#         df["week_start"] = df.apply(lambda r: build_week_start(r["year"], r["month"], r["week_in_month"]), axis=1)

#         print("📅 Building future week...")
#         last_row = df.sort_values(["year", "month", "week_in_month"]).tail(1).iloc[0]

#         pred_year, pred_month, pred_week = get_next_week_label(
#             int(last_row["year"]),
#             int(last_row["month"]),
#             int(last_row["week_in_month"]),
#         )

#         pred_week_start = build_week_start(pred_year, pred_month, pred_week)

#         future = df[["fish_id", "fish_token", "sinhala_name", "common_name"]].drop_duplicates().copy()
#         future["week_start"] = pred_week_start
#         future["price"] = np.nan
#         future["year"] = pred_year
#         future["month"] = pred_month
#         future["week_in_month"] = pred_week

#         print("⚙️ Creating features...")
#         df_all = pd.concat([df, future], ignore_index=True)
#         df_all = df_all.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

#         tab_df = add_tabular_lags(df_all)

#         future_df = tab_df[tab_df["week_start"] == pred_week_start].copy()

#         print("DEBUG future_df shape:", future_df.shape)

#         if future_df.empty:
#             raise ValueError("❌ future_df is EMPTY")

#         for col in num_cols:
#             if col not in future_df:
#                 future_df[col] = 0.0

#         future_df[num_cols] = future_df[num_cols].fillna(0).astype(np.float32)

#         print("🧠 Preparing ANN input...")

#         fish_id_input = future_df["fish_token"].astype(str).values.reshape(-1, 1)
#         num_features_input = future_df[num_cols].values.astype(np.float32)

#         print("DEBUG fish shape:", fish_id_input.shape)
#         print("DEBUG num shape:", num_features_input.shape)

#         print("🔮 ANN predicting...")
#         ann_pred_delta = ann_model.predict({
#             "fish_id": fish_id_input,
#             "num_features": num_features_input
#         }, verbose=0).reshape(-1)

#         ann_pred = future_df["lag_1"].values + (ann_alpha * ann_pred_delta)

#         print("🔮 XGB predicting...")
#         xgb_feature_future = pd.get_dummies(
#             future_df[["fish_id"] + num_cols],
#             columns=["fish_id"],
#             drop_first=False,
#         )

#         xgb_feature_future = xgb_feature_future.reindex(columns=feature_cols, fill_value=0)

#         xgb_pred = xgb_model.predict(xgb_feature_future)

#         print("🎯 Combining predictions...")
#         final_pred = best_w_ann * ann_pred + best_w_xgb * xgb_pred

#         print("✅ Prediction SUCCESS")

#         return {
#             "rowCount": len(final_pred),
#             "preview": []
#         }

#     except Exception as e:
#         print("❌ ERROR INSIDE FUNCTION")
#         traceback.print_exc()
#         raise e