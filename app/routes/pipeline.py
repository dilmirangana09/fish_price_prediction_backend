from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List
import math
import os
import re
import time
import traceback
import pandas as pd
import json

from sqlalchemy import desc
from uuid import uuid4
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.fish_training_price import FishTrainingPrice
from app.models.upload_log import UploadLog
from app.models.prediction_result import PredictionResult
from app.models.pipeline_activity_log import PipelineActivityLog
from app.services.system_status import (
    read_status,
    update_fish_count,
    update_last_upload,
    write_status,
)
from app.services.training_service import get_deployed_model_info
from app.services.prediction_service import generate_next_week_predictions_with_saved_hybrid
from app.models.model_version import ModelVersion
from app.services.training_service import (
    train_and_save_deployed_hybrid_model,
    train_full_hybrid_ann_xgb_pipeline
)
from app.models.candidate_result import CandidateResult

router = APIRouter(prefix="/admin/pipeline", tags=["admin-pipeline"])

UPLOAD_DIR = "uploads"
MODEL_DIR = "models_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MASTER_PATH = os.path.join(UPLOAD_DIR, "master_merged.csv")
MERGED_EDITABLE_TEMP_PATH = os.path.join(UPLOAD_DIR, "merged_editable_temp.csv")


class ValidateRequest(BaseModel):
    storedFilename: str


class FinalizeUploadIn(BaseModel):
    filename: str
    rows: List[List[Any]]


class MergeRequest(BaseModel):
    storedFilename: str


class SyncLongToDbRequest(BaseModel):
    filename: str


def read_csv_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def safe_write_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig", retries: int = 5):
    temp_path = f"{path}.tmp"

    for i in range(retries):
        try:
            df.to_csv(temp_path, index=False, encoding=encoding)
            os.replace(temp_path, path)
            return
        except PermissionError:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            if i == retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not save file because it is open or locked: {os.path.basename(path)}. Please close it and try again."
                )
            time.sleep(0.5)


def derive_week_label(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace("_", " ").strip()
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    return stem


def split_week_label(label: str):
    s = str(label).strip()
    m = re.match(r"(\d+)(st|nd|rd|th)\s+week\s+of\s+([A-Za-z]+)\s+(\d{4})", s)
    if not m:
        return None, None, None
    return int(m.group(4)), m.group(3), int(m.group(1))


def read_uploaded_week_file(file_path: str, stored_filename: str) -> pd.DataFrame:
    lower_name = stored_filename.lower()

    if lower_name.endswith(".xlsx"):
        df = pd.read_excel(
            file_path,
            sheet_name="Retail",
            skiprows=2,
            engine="openpyxl",
        )
        if df.shape[1] < 6:
            raise HTTPException(status_code=400, detail="XLSX file does not contain expected columns")

        data = df.iloc[:, [1, 2, 5]].copy()
        data.columns = ["Sinhala Name", "Common Name", "Price"]
        return data

    if lower_name.endswith(".csv"):
        df = read_csv_flexible(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        expected_cols = ["Sinhala Name", "Common Name", "Price"]
        if all(c in df.columns for c in expected_cols):
            return df[expected_cols].copy()

        if df.shape[1] < 3:
            raise HTTPException(status_code=400, detail="CSV file must contain at least 3 columns")

        data = df.iloc[:, :3].copy()
        data.columns = expected_cols
        return data

    raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are supported")


def log_upload_action(
    db: Session,
    filename: str | None,
    stored_filename: str | None,
    week_label: str | None,
    fish_count: int | None,
    action: str,
    status: str,
):
    log = UploadLog(
        filename=filename,
        stored_filename=stored_filename,
        week_label=week_label,
        fish_count=fish_count,
        action=action,
        status=status,
    )
    db.add(log)
    db.commit()


def add_pipeline_log(db: Session, action: str, status: str, notes: str | None = None):
    log = PipelineActivityLog(
        action=action,
        status=status,
        notes=notes,
    )
    db.add(log)
    db.commit()


def month_number_to_name(month_value):
    if month_value is None:
        return None

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

    try:
        month_int = int(month_value)
        return month_map.get(month_int, str(month_value))
    except Exception:
        return str(month_value)


def upsert_weekly_prices_to_db(
    weekly_df: pd.DataFrame,
    year: int,
    month: str,
    week: int,
    db: Session,
):
    value_cols = [c for c in weekly_df.columns if c not in ["Sinhala Name", "Common Name"]]
    if not value_cols:
        raise ValueError("Weekly dataframe has no price column")

    value_col = value_cols[0]

    for _, row in weekly_df.iterrows():
        sinhala_name = str(row.get("Sinhala Name", "")).strip()
        common_name = str(row.get("Common Name", "")).strip()
        raw_price = row.get(value_col)

        if not sinhala_name and not common_name:
            continue

        price = None
        if pd.notna(raw_price):
            try:
                price = float(raw_price)
            except Exception:
                price = None

        existing = (
            db.query(FishWeeklyPrice)
            .filter(
                FishWeeklyPrice.sinhala_name == sinhala_name,
                FishWeeklyPrice.common_name == common_name,
                FishWeeklyPrice.year == year,
                FishWeeklyPrice.month == month,
                FishWeeklyPrice.week == week,
            )
            .first()
        )

        if existing:
            existing.price = price
            existing.updated_at = datetime.now()
        else:
            db.add(
                FishWeeklyPrice(
                    sinhala_name=sinhala_name,
                    common_name=common_name,
                    year=year,
                    month=month,
                    week=week,
                    price=price,
                )
            )

    db.commit()


def upsert_long_format_to_training_db(long_df: pd.DataFrame, db: Session):
    for _, row in long_df.iterrows():
        sinhala_name = str(row.get("Sinhala Name", "")).strip()
        common_name = str(row.get("Common Name", "")).strip()
        year = row.get("Year")
        month = str(row.get("Month", "")).strip()
        week = row.get("Week")
        price = row.get("Price")

        if not sinhala_name and not common_name:
            continue
        if pd.isna(year) or pd.isna(week):
            continue

        try:
            year = int(year)
            week = int(week)
        except Exception:
            continue

        db_price = None
        if pd.notna(price):
            try:
                db_price = float(price)
            except Exception:
                db_price = None

        existing = (
            db.query(FishTrainingPrice)
            .filter(
                FishTrainingPrice.sinhala_name == sinhala_name,
                FishTrainingPrice.common_name == common_name,
                FishTrainingPrice.year == year,
                FishTrainingPrice.month == month,
                FishTrainingPrice.week == week,
            )
            .first()
        )

        if existing:
            existing.price = db_price
            existing.updated_at = datetime.now()
        else:
            db.add(
                FishTrainingPrice(
                    sinhala_name=sinhala_name,
                    common_name=common_name,
                    year=year,
                    month=month,
                    week=week,
                    price=db_price,
                )
            )

    db.commit()


@router.post("/upload-csv")
async def upload_weekly_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are allowed")

    orig = (file.filename or "uploaded_file").replace(" ", "_")
    safe_name = os.path.basename(orig)
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    base, ext = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(save_path):
        safe_name = f"{base}({counter}){ext}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        counter += 1

    contents = await file.read()
    if not contents or len(contents) < 10:
        raise HTTPException(status_code=400, detail="File is empty or invalid")

    with open(save_path, "wb") as f:
        f.write(contents)

    update_last_upload(safe_name)

    log_upload_action(
        db=db,
        filename=file.filename,
        stored_filename=safe_name,
        week_label=None,
        fish_count=None,
        action="upload",
        status="success",
    )

    return {
        "message": "Uploaded successfully",
        "uploadedBy": admin.get("email"),
        "storedFilename": safe_name,
        "originalFilename": file.filename,
        "uploadedDate": datetime.now().strftime("%Y-%m-%d"),
    }


@router.post("/preview-file")
async def preview_file(
    file: UploadFile = File(...),
    admin=Depends(get_current_admin),
):
    name = (file.filename or "").lower()

    if not (name.endswith(".csv") or name.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        if name.endswith(".xlsx"):
            data = pd.read_excel(
                BytesIO(contents),
                sheet_name="Retail",
                skiprows=2,
                engine="openpyxl",
            )
            if data.shape[1] < 6:
                raise HTTPException(status_code=400, detail="XLSX file does not contain expected columns")

            data = data.iloc[:, [1, 2, 5]].copy()
            data.columns = ["Sinhala Name", "Common Name", "Price"]
        else:
            try:
                data = pd.read_csv(BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                data = pd.read_csv(BytesIO(contents), encoding="latin1")

            expected = ["Sinhala Name", "Common Name", "Price"]
            if all(c in data.columns for c in expected):
                data = data[expected].copy()
            else:
                data = data.iloc[:, :3].copy()
                data.columns = expected

        data["Sinhala Name"] = data["Sinhala Name"].astype(str).str.strip()
        data["Common Name"] = data["Common Name"].astype(str).str.strip()
        data["Sinhala Name"] = data["Sinhala Name"].replace(["", "nan", "None"], pd.NA)
        data["Common Name"] = data["Common Name"].replace(["", "nan", "None"], pd.NA)

        data = data[
            data["Sinhala Name"].notna() | data["Common Name"].notna()
        ].reset_index(drop=True)

        data["Price"] = data["Price"].replace(["nan", "None"], "")
        rows = data.fillna("").values.tolist()

        return {
            "columns": ["Sinhala Name", "Common Name", "Price"],
            "rows": rows,
            "rowCount": int(len(rows)),
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preview failed: {str(e)}")


@router.post("/finalize-upload")
def finalize_upload(
    payload: FinalizeUploadIn,
    admin=Depends(get_current_admin),
):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows to upload")

    df = pd.DataFrame(payload.rows, columns=["Sinhala Name", "Common Name", "Price"])
    df = df.replace("", pd.NA).dropna(how="all").reset_index(drop=True)

    if df.empty:
        raise HTTPException(status_code=400, detail="No rows available to save")

    safe_name = os.path.basename(payload.filename).replace(" ", "_")
    if not safe_name.lower().endswith(".csv"):
        safe_name = os.path.splitext(safe_name)[0] + ".csv"

    save_path = os.path.join(UPLOAD_DIR, safe_name)
    base, ext = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(save_path):
        safe_name = f"{base}({counter}){ext}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        counter += 1

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    update_last_upload(safe_name)

    return {
        "message": "Uploaded successfully",
        "storedFilename": safe_name,
        "uploadedDate": datetime.now().strftime("%Y-%m-%d"),
        "rowCount": int(len(df)),
    }


@router.post("/validate-csv")
def validate_weekly_csv(
    payload: ValidateRequest,
    admin=Depends(get_current_admin),
):
    safe_name = os.path.basename(payload.storedFilename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    try:
        df = read_uploaded_week_file(file_path, safe_name)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File read failed: {str(e)}")

    rows, cols = df.shape
    if rows == 0 or cols == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    fish_count = int(df["Common Name"].nunique()) if "Common Name" in df.columns else 0
    update_fish_count(fish_count)

    missing_total = int(df.isna().sum().sum())
    missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns}
    duplicate_rows = int(df.duplicated().sum())
    duplicate_columns = [c for c in df.columns[df.columns.duplicated()].tolist()]

    errors = []
    warnings = []

    expected = {"Sinhala Name", "Common Name", "Price"}
    if set(df.columns) != expected:
        warnings.append(f"Unexpected columns after extraction: {list(df.columns)}")

    if duplicate_columns:
        errors.append(f"Duplicate column names found: {duplicate_columns[:10]}")

    summary = "Valid ✅" if len(errors) == 0 else "Has issues ⚠️"

    return {
        "summary": summary,
        "rows": int(rows),
        "columns": int(cols),
        "fileType": "XLSX" if safe_name.lower().endswith(".xlsx") else "CSV",
        "fishCount": fish_count,
        "missingTotal": missing_total,
        "missingByColumn": missing_by_col,
        "duplicateRows": duplicate_rows,
        "duplicateColumns": duplicate_columns,
        "errors": errors,
        "warnings": warnings,
        "validatedBy": admin.get("email"),
        "storedFilename": safe_name,
    }


@router.post("/preprocess-merge")
def preprocess_merge(
    payload: MergeRequest,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    stored_filename = payload.storedFilename
    input_path = os.path.join(UPLOAD_DIR, stored_filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    uploaded_df = read_uploaded_week_file(input_path, stored_filename)
    uploaded_df.columns = [str(c).strip() for c in uploaded_df.columns]
    uploaded_df["Sinhala Name"] = uploaded_df["Sinhala Name"].astype(str).str.strip()
    uploaded_df["Common Name"] = uploaded_df["Common Name"].astype(str).str.strip()

    week_label = derive_week_label(stored_filename)
    weekly_df = uploaded_df.rename(columns={"Price": week_label})

    year, month, week = split_week_label(week_label)
    if year is None or month is None or week is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid weekly filename format: {week_label}. Expected format like '1st week of June 2025'"
        )

    upsert_weekly_prices_to_db(weekly_df, year, month, week, db)

    if os.path.exists(MASTER_PATH):
        master_df = read_csv_flexible(MASTER_PATH)
        master_df.columns = [str(c).strip() for c in master_df.columns]
        master_df["Sinhala Name"] = master_df["Sinhala Name"].astype(str).str.strip()
        master_df["Common Name"] = master_df["Common Name"].astype(str).str.strip()

        master_idx = master_df.set_index(["Sinhala Name", "Common Name"])
        weekly_idx = weekly_df.set_index(["Sinhala Name", "Common Name"])

        if week_label in master_idx.columns:
            master_idx[week_label] = weekly_idx[week_label].combine_first(master_idx[week_label])
        else:
            master_idx = master_idx.join(weekly_idx[[week_label]], how="outer")

        merged_df = master_idx.reset_index()
    else:
        merged_df = weekly_df.copy()

    safe_write_csv(merged_df, MERGED_EDITABLE_TEMP_PATH)
    safe_write_csv(merged_df, MASTER_PATH)

    fish_count = int(merged_df["Common Name"].nunique()) if "Common Name" in merged_df.columns else 0

    status = read_status()
    status["lastMergedEditableFilename"] = "merged_editable_temp.csv"
    status["lastMergedFilename"] = "master_merged.csv"
    status["lastWeekLabel"] = week_label
    write_status(status)

    log_upload_action(
        db=db,
        filename=stored_filename,
        stored_filename="merged_editable_temp.csv",
        week_label=week_label,
        fish_count=fish_count,
        action="merge",
        status="success",
    )

    return {
        "message": "Merged dataset generated and weekly rows saved to database",
        "weekLabel": week_label,
        "filename": "merged_editable_temp.csv",
        "columns": list(merged_df.columns),
        "rows": merged_df.fillna("").values.tolist(),
        "rowCount": int(len(merged_df)),
        "dbSaved": True,
    }


@router.post("/preprocess-filter")
def preprocess_filter(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    if not os.path.exists(MERGED_EDITABLE_TEMP_PATH):
        raise HTTPException(status_code=404, detail="Merged editable dataset not found")

    merged_df = read_csv_flexible(MERGED_EDITABLE_TEMP_PATH)
    merged_df.columns = [str(c).strip() for c in merged_df.columns]

    week_cols = [c for c in merged_df.columns if c not in ["Sinhala Name", "Common Name"]]
    if not week_cols:
        raise HTTPException(status_code=400, detail="No week columns found")

    merged_num = merged_df.copy()
    merged_num[week_cols] = merged_num[week_cols].apply(pd.to_numeric, errors="coerce")

    available_counts = merged_num[week_cols].notna().sum(axis=1)
    total_weeks = len(week_cols)
    threshold = math.ceil(total_weeks * 0.5)

    summary_df = merged_df[["Sinhala Name", "Common Name"]].copy()
    summary_df["available_weeks"] = available_counts
    summary_df["total_weeks"] = total_weeks
    summary_df["availability_pct"] = ((available_counts / total_weeks) * 100).round(2)
    summary_df["kept"] = available_counts >= threshold

    filtered_df = merged_num[available_counts >= threshold].copy().reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered_filename = f"filtered_50pct_{ts}.csv"
    summary_filename = f"filtered_summary_{ts}.csv"

    filtered_path = os.path.join(UPLOAD_DIR, filtered_filename)
    summary_path = os.path.join(UPLOAD_DIR, summary_filename)

    filtered_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    status = read_status()
    status["lastFilteredFilename"] = filtered_filename
    status["lastFilteredSummaryFilename"] = summary_filename
    write_status(status)

    kept_fish_count = int(filtered_df["Common Name"].nunique()) if "Common Name" in filtered_df.columns else 0

    log_upload_action(
        db=db,
        filename="merged_editable_temp.csv",
        stored_filename=filtered_filename,
        week_label=status.get("lastWeekLabel"),
        fish_count=kept_fish_count,
        action="filter_50pct",
        status="success",
    )

    return {
        "message": "50% filter applied",
        "filteredFilename": filtered_filename,
        "summaryFilename": summary_filename,
        "stats": {
            "totalWeeks": total_weeks,
            "thresholdWeeks": threshold,
            "keptFishCount": kept_fish_count,
        },
        "summaryPreview": summary_df.fillna("").head(50).to_dict(orient="records"),
    }


@router.post("/preprocess-interpolate")
def preprocess_interpolate(admin=Depends(get_current_admin)):
    status = read_status()
    filtered_filename = status.get("lastFilteredFilename")

    if not filtered_filename:
        raise HTTPException(status_code=404, detail="Filtered dataset not found. Apply 50% filter first.")

    filtered_path = os.path.join(UPLOAD_DIR, filtered_filename)
    if not os.path.exists(filtered_path):
        raise HTTPException(status_code=404, detail="Filtered dataset file not found")

    filtered_df = read_csv_flexible(filtered_path)
    filtered_df.columns = [str(c).strip() for c in filtered_df.columns]

    week_cols = [c for c in filtered_df.columns if c not in ["Sinhala Name", "Common Name"]]
    if not week_cols:
        raise HTTPException(status_code=400, detail="No week columns found in filtered dataset")

    interpolated_df = filtered_df.copy()
    interpolated_df[week_cols] = interpolated_df[week_cols].apply(pd.to_numeric, errors="coerce")
    interpolated_df[week_cols] = interpolated_df[week_cols].transpose().interpolate(method="linear").transpose()
    interpolated_df[week_cols] = interpolated_df[week_cols].bfill(axis=1).ffill(axis=1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    interpolated_filename = f"interpolated_{ts}.csv"
    interpolated_path = os.path.join(UPLOAD_DIR, interpolated_filename)

    interpolated_df.to_csv(interpolated_path, index=False, encoding="utf-8-sig", float_format="%.2f")

    fish_count = int(interpolated_df["Common Name"].nunique()) if "Common Name" in interpolated_df.columns else 0
    update_fish_count(fish_count)

    status["lastInterpolatedFilename"] = interpolated_filename
    write_status(status)

    return {
        "message": "Interpolation completed",
        "filename": interpolated_filename,
        "stats": {
            "fishCount": fish_count,
            "weekCount": len(week_cols),
            "rowCount": int(len(interpolated_df)),
        },
        "preview": interpolated_df.head(20).fillna("").to_dict(orient="records"),
    }


@router.post("/preprocess-long-format")
def preprocess_long_format(admin=Depends(get_current_admin)):
    status = read_status()
    interpolated_filename = status.get("lastInterpolatedFilename")

    if not interpolated_filename:
        raise HTTPException(status_code=404, detail="Interpolated dataset not found. Run interpolation first.")

    interpolated_path = os.path.join(UPLOAD_DIR, interpolated_filename)
    if not os.path.exists(interpolated_path):
        raise HTTPException(status_code=404, detail="Interpolated dataset file not found")

    interpolated_df = read_csv_flexible(interpolated_path)
    interpolated_df.columns = [str(c).strip() for c in interpolated_df.columns]

    week_cols = [c for c in interpolated_df.columns if c not in ["Sinhala Name", "Common Name"]]
    if not week_cols:
        raise HTTPException(status_code=400, detail="No week columns found in interpolated dataset")

    long_df = interpolated_df.melt(
        id_vars=["Sinhala Name", "Common Name"],
        value_vars=week_cols,
        var_name="Week_Label",
        value_name="Price",
    )

    long_df[["Year", "Month", "Week"]] = long_df["Week_Label"].apply(
        lambda x: pd.Series(split_week_label(x))
    )

    long_df = long_df[["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]].copy()
    long_df = long_df.dropna(subset=["Year", "Month", "Week", "Price"]).reset_index(drop=True)

    long_df["Year"] = long_df["Year"].astype(int)
    long_df["Week"] = long_df["Week"].astype(int)
    long_df["Price"] = pd.to_numeric(long_df["Price"], errors="coerce")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_filename = f"fish_prices_long_{ts}.csv"
    long_path = os.path.join(UPLOAD_DIR, long_filename)

    long_df.to_csv(long_path, index=False, encoding="utf-8-sig", float_format="%.2f")

    status["lastLongFilename"] = long_filename
    status["lastPreprocessDate"] = datetime.now().strftime("%Y-%m-%d")
    write_status(status)

    return {
        "message": "Long format dataset generated",
        "filename": long_filename,
        "stats": {
            "rowCount": int(len(long_df)),
            "fishCount": int(long_df["Common Name"].nunique()) if "Common Name" in long_df.columns else 0,
        },
        "preview": long_df.head(50).fillna("").to_dict(orient="records"),
    }


@router.post("/sync-long-to-db")
def sync_long_to_db(
    payload: SyncLongToDbRequest,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    safe_name = os.path.basename(payload.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Long format file not found: {safe_name}")

    df = read_csv_flexible(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df = df[required_cols].copy()
    df["Sinhala Name"] = df["Sinhala Name"].astype(str).str.strip()
    df["Common Name"] = df["Common Name"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Sinhala Name"] = df["Sinhala Name"].replace(["", "nan", "None"], pd.NA)
    df["Common Name"] = df["Common Name"].replace(["", "nan", "None"], pd.NA)
    df["Month"] = df["Month"].replace(["", "nan", "None"], pd.NA)
    df = df.dropna(subset=["Sinhala Name", "Common Name", "Year", "Month", "Week"]).reset_index(drop=True)

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid rows found in long format dataset")

    upsert_long_format_to_training_db(df, db)

    add_pipeline_log(
        db,
        action="Sync Long Format to Training DB",
        status="Completed",
        notes=f"{len(df)} rows synced from {safe_name}",
    )

    return {
        "message": "Long format training dataset stored in training database successfully",
        "filename": safe_name,
        "fishCount": int(df["Common Name"].nunique()),
        "rowCount": int(len(df)),
    }


@router.post("/train-hybrid-model")
def train_hybrid_model_route(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    try:
        result = get_deployed_model_info()

        status = read_status()
        status["lastModelStatus"] = "Ready"
        status["lastModelName"] = result["modelName"]
        status["lastModelTrainedAt"] = result["trainedAt"]
        status["lastModelSource"] = "Colab"
        status["lastModelWeights"] = result["bestWeights"]
        status["lastModelAnnAlpha"] = result["annAlpha"]
        status["lastModelMetrics"] = result.get("metrics", {})
        write_status(status)

        add_pipeline_log(
            db,
            action="Load Deployed Model",
            status="Completed",
            notes=f"Model ready: {result['modelName']}",
        )

        return {
            "message": "Deployed hybrid ANN + XGBoost model is ready",
            "finalModel": result["modelName"],
            "trainedAt": result["trainedAt"],
            "bestWeights": result["bestWeights"],
            "annAlpha": result["annAlpha"],
            "source": "Colab",
            "metrics": result.get("metrics", {}),
            "testSamples": "—",
            "files": result["files"],
        }

    except Exception as e:
        add_pipeline_log(
            db,
            action="Load Deployed Model",
            status="Failed",
            notes=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-next-week")
def predict_next_week(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()
    long_filename = status.get("lastLongFilename")
    print(f"Predicting next week using long format file: {long_filename}")

    if not long_filename:
        raise HTTPException(
            status_code=404,
            detail="Long format dataset not found. Generate long format dataset first."
        )

    long_path = os.path.join(UPLOAD_DIR, long_filename)
    if not os.path.exists(long_path):
        raise HTTPException(
            status_code=404,
            detail="Long format dataset file not found."
        )

    try:
        print("Calling prediction generator with long format dataset...")
        result = generate_next_week_predictions_with_saved_hybrid(long_path)
        
        preview = result.get("preview", [])
        filename = result.get("filename")
        output_week = result.get("date")
        row_count = int(result.get("rowCount", 0))
        final_model = result.get("modelName", "Hybrid_ANN_XGBoost")

        if not preview:
            raise HTTPException(
                status_code=500,
                detail="No prediction rows returned from prediction generator."
            )

        batch_id = uuid4().hex

        db.query(PredictionResult).filter(
            PredictionResult.is_published == False
        ).delete(synchronize_session=False)
        db.commit()

        for row in preview:
            row_year = row.get("Year")
            row_month = row.get("Month")
            row_week = row.get("Week")

            row_year = int(row_year) if row_year not in [None, ""] else None
            row_week = int(row_week) if row_week not in [None, ""] else None

            month_name = month_number_to_name(row_month) if row_month not in [None, ""] else None

            week_label = None
            if row_week and month_name and row_year:
                suffix = "th"
                if row_week == 1:
                    suffix = "st"
                elif row_week == 2:
                    suffix = "nd"
                elif row_week == 3:
                    suffix = "rd"

                week_label = f"{row_week}{suffix} week of {month_name} {row_year}"

            predicted_price = row.get("Predicted_Price")

            db_row = PredictionResult(
                batch_id=batch_id,
                model_name=final_model,
                sinhala_name=row.get("Sinhala Name"),
                common_name=row.get("Common Name"),
                year=row_year,
                month=month_name,
                week=row_week,
                week_label=week_label,
                predicted_price=float(predicted_price) if predicted_price not in [None, ""] else None,
                # source_long_file=long_filename,
                # source_prediction_file=filename,
                is_published=False,
            )
            db.add(db_row)

        db.commit()

        add_pipeline_log(
            db,
            action="Generate Predictions",
            status="Completed",
            notes=f"{row_count} predictions saved to DB. Batch: {batch_id}",
        )

        status["lastPredictionFilename"] = filename
        status["lastPredictionBatchId"] = batch_id
        status["lastPredictionDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status["lastModelName"] = final_model
        write_status(status)

        return {
            "message": "Next week predictions generated and saved to DB successfully",
            "date": output_week,
            "rowCount": row_count,
            "filename": filename,
            "batchId": batch_id,
            "savedToDb": True,
            "preview": preview[:50],
            "modelName": final_model,
        }

    except Exception as e:
        db.rollback()
        add_pipeline_log(
            db,
            action="Generate Predictions",
            status="Failed",
            notes=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/publish-predictions")
def publish_predictions(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()
    batch_id = status.get("lastPredictionBatchId")

    if not batch_id:
        raise HTTPException(status_code=404, detail="No prediction batch found to publish.")

    rows = db.query(PredictionResult).filter(PredictionResult.batch_id == batch_id).all()

    if not rows:
        raise HTTPException(status_code=404, detail="Prediction rows not found for latest batch.")

    db.query(PredictionResult).filter(PredictionResult.is_published == True).update(
        {"is_published": False, "published_at": None},
        synchronize_session=False
    )

    published_time = datetime.utcnow()
    for row in rows:
        row.is_published = True
        row.published_at = published_time

    db.commit()

    add_pipeline_log(
        db,
        action="Publish to User Pages",
        status="Completed",
        notes=f"Published batch: {batch_id} ({len(rows)} predictions)",
    )

    status["lastPublishedBatchId"] = batch_id
    status["lastPublishedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_status(status)

    return {
        "message": "Predictions published successfully",
        "batchId": batch_id,
        "publishedCount": len(rows),
        "publishedAt": published_time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/download")
def download_file(filename: str, admin=Depends(get_current_admin)):
    safe_name = os.path.basename(filename)

    upload_path = os.path.join(UPLOAD_DIR, safe_name)
    model_path = os.path.join(MODEL_DIR, safe_name)

    if os.path.exists(upload_path):
        return FileResponse(upload_path, filename=safe_name)

    if os.path.exists(model_path):
        return FileResponse(model_path, filename=safe_name)

    raise HTTPException(status_code=404, detail="File not found")


@router.get("/activity-logs")
def get_activity_logs(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    logs = (
        db.query(PipelineActivityLog)
        .order_by(desc(PipelineActivityLog.created_at))
        .limit(5)
        .all()
    )

    return {
        "rows": [
            {
                "id": row.id,
                "date": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else "—",
                "action": row.action,
                "status": row.status,
                "notes": row.notes or "—",
            }
            for row in logs
        ]
    }


@router.post("/train-candidate-model")
def train_candidate_model(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    try:
        result = train_full_hybrid_ann_xgb_pipeline(db)

        new_candidate = CandidateResult(
            model_name="Hybrid_ANN_XGBoost",
            version_name=result["version"],
            mae=result["metrics"]["MAE"],
            rmse=result["metrics"]["RMSE"],
            mape=result["metrics"]["MAPE"],
            r2=result["metrics"]["R2"],
            ann_weight=result["metrics"]["best_w_ann"],
            xgb_weight=result["metrics"]["best_w_xgb"],
            fish_count=result["fishCount"],
            train_rows=result["trainRows"],
            val_rows=result["valRows"],
            test_rows=result["testRows"],
        )

        db.add(new_candidate)
        db.commit()
        db.refresh(new_candidate)

        add_pipeline_log(
            db,
            action="Train Candidate Model",
            status="Completed",
            notes=f"Candidate result saved: {result['version']}",
        )

        return {
            "message": "Candidate model trained and candidate results saved to DB",
            "version": result["version"],
            "metrics": result["metrics"],
            "candidateResultId": new_candidate.id,
        }

    except Exception as e:
        db.rollback()
        add_pipeline_log(
            db,
            action="Train Candidate Model",
            status="Failed",
            notes=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Candidate training failed: {str(e)}")

@router.get("/model-comparison")
def compare_models(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    current = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_deployed == True)
        .order_by(desc(ModelVersion.created_at))
        .first()
    )

    candidate = (
        db.query(CandidateResult)
        .order_by(desc(CandidateResult.created_at))
        .first()
    )

    if not current or not candidate:
        return {"message": "Not enough models to compare"}

    is_better = (
        candidate.mape is not None
        and current.mape is not None
        and candidate.rmse is not None
        and current.rmse is not None
        and candidate.mape < current.mape
        and candidate.rmse <= current.rmse
    )

    return {
        "current": {
            "MAPE": current.mape,
            "RMSE": current.rmse,
            "MAE": current.mae,
            "R2": current.r2,
        },
        "candidate": {
            "MAPE": candidate.mape,
            "RMSE": candidate.rmse,
            "MAE": candidate.mae,
            "R2": candidate.r2,
        },
        "isBetter": is_better,
        "candidateVersion": candidate.version_name,
    }

@router.post("/deploy-model/{version_id}")
def deploy_model(
    version_id: int,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()

    if not model:
        raise HTTPException(404, "Model not found")

    # remove old deployed
    db.query(ModelVersion).update({"is_deployed": False})

    model.is_deployed = True
    db.commit()

    return {"message": "Model deployed successfully"}

@router.get("/current-deployed-model")
def get_current_deployed_model(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    row = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_deployed == True)
        .order_by(desc(ModelVersion.created_at))
        .first()
    )

    if not row:
        return {
            "row": None,
            "message": "No deployed model found"
        }

    return {
        "row": {
            "id": row.id,
            "model_name": row.model_name,
            "version_name": row.version_name,
            "mae": row.mae,
            "rmse": row.rmse,
            "mape": row.mape,
            "r2": row.r2,
            "ann_weight": row.ann_weight,
            "xgb_weight": row.xgb_weight,
            "is_deployed": row.is_deployed,
            "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None,
        }
    }
    
@router.post("/sync-deployed-model-to-db")
def sync_deployed_model_to_db(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    deployed_dir = Path("app") / "ml_models" / "deployed"
    meta_path = deployed_dir / "ann_xgb_hybrid_metadata.json"

    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Deployed metadata file not found: {meta_path}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata.get("model_name", "Hybrid_ANN_XGBoost")
    trained_at = metadata.get("trained_at")
    best_w_ann = metadata.get("best_w_ann")
    best_w_xgb = metadata.get("best_w_xgb")

    metrics = metadata.get("metrics", {})
    hybrid_metrics = metrics.get("Hybrid_ANN_XGBoost", {})

    version_name = (
        trained_at.replace(":", "").replace("-", "").replace(" ", "_")
        if trained_at else "deployed_model"
    )

    existing = (
        db.query(ModelVersion)
        .filter(ModelVersion.version_name == version_name)
        .first()
    )

    # only one deployed model in DB
    db.query(ModelVersion).update({"is_deployed": False}, synchronize_session=False)

    if existing:
        existing.model_name = model_name
        existing.mae = hybrid_metrics.get("MAE")
        existing.rmse = hybrid_metrics.get("RMSE")
        existing.mape = hybrid_metrics.get("MAPE")
        existing.r2 = hybrid_metrics.get("R2")
        existing.ann_weight = best_w_ann
        existing.xgb_weight = best_w_xgb
        existing.is_deployed = True
        db.commit()
        db.refresh(existing)

        return {
            "message": "Deployed model already existed and was synced successfully",
            "row": {
                "id": existing.id,
                "model_name": existing.model_name,
                "version_name": existing.version_name,
                "mae": existing.mae,
                "rmse": existing.rmse,
                "mape": existing.mape,
                "r2": existing.r2,
                "ann_weight": existing.ann_weight,
                "xgb_weight": existing.xgb_weight,
                "is_deployed": existing.is_deployed,
            },
        }

    new_row = ModelVersion(
        model_name=model_name,
        version_name=version_name,
        mae=hybrid_metrics.get("MAE"),
        rmse=hybrid_metrics.get("RMSE"),
        mape=hybrid_metrics.get("MAPE"),
        r2=hybrid_metrics.get("R2"),
        ann_weight=best_w_ann,
        xgb_weight=best_w_xgb,
        is_deployed=True,
    )

    db.add(new_row)
    db.commit()
    db.refresh(new_row)

    return {
        "message": "Current deployed model synced to DB successfully",
        "row": {
            "id": new_row.id,
            "model_name": new_row.model_name,
            "version_name": new_row.version_name,
            "mae": new_row.mae,
            "rmse": new_row.rmse,
            "mape": new_row.mape,
            "r2": new_row.r2,
            "ann_weight": new_row.ann_weight,
            "xgb_weight": new_row.xgb_weight,
            "is_deployed": new_row.is_deployed,
        },
    }

@router.post("/deploy-latest-candidate-model")
def deploy_latest_candidate_model(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    candidate = (
        db.query(CandidateResult)
        .order_by(desc(CandidateResult.created_at))
        .first()
    )

    if not candidate:
        raise HTTPException(status_code=404, detail="No candidate result found to deploy.")

    current = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_deployed == True)
        .order_by(desc(ModelVersion.created_at))
        .first()
    )

    if current and current.mape is not None and current.rmse is not None:
        is_better = (
            candidate.mape is not None
            and candidate.rmse is not None
            and candidate.mape < current.mape
            and candidate.rmse <= current.rmse
        )
        if not is_better:
            raise HTTPException(
                status_code=400,
                detail="Latest candidate result is not better than the current deployed model."
            )

    try:
        deployed_result = train_and_save_deployed_hybrid_model(db, candidate.version_name)

        db.query(ModelVersion).filter(ModelVersion.is_deployed == True).update(
            {"is_deployed": False},
            synchronize_session=False,
        )

        deployed_row = ModelVersion(
            model_name="Hybrid_ANN_XGBoost",
            version_name=candidate.version_name,
            mae=candidate.mae,
            rmse=candidate.rmse,
            mape=candidate.mape,
            r2=candidate.r2,
            ann_weight=candidate.ann_weight,
            xgb_weight=candidate.xgb_weight,
            is_deployed=True,
        )

        db.add(deployed_row)
        db.commit()
        db.refresh(deployed_row)

        add_pipeline_log(
            db,
            action="Deploy Candidate Model",
            status="Completed",
            notes=f"Deployed version: {candidate.version_name}",
        )

        status = read_status()
        status["lastModelStatus"] = "Deployed"
        status["lastModelName"] = deployed_row.model_name
        status["lastModelTrainedAt"] = deployed_result["trainedAt"]
        status["lastModelMetrics"] = deployed_result["metrics"]
        write_status(status)

        return {
            "message": "Latest candidate deployed successfully",
            "version": candidate.version_name,
            "trainedAt": deployed_result["trainedAt"],
            "metrics": deployed_result["metrics"],
        }

    except Exception as e:
        db.rollback()
        add_pipeline_log(
            db,
            action="Deploy Candidate Model",
            status="Failed",
            notes=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Deploy failed: {str(e)}")