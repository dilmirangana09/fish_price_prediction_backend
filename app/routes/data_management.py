from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.services.system_status import read_status
import os
import pandas as pd
from fastapi.responses import FileResponse

router = APIRouter(prefix="/admin/data", tags=["admin-data"])


@router.get("/db-stats")
def get_db_stats(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    row_count = db.query(func.count(FishWeeklyPrice.id)).scalar() or 0
    fish_count = db.query(
        func.count(
            func.distinct(
                func.concat(
                    FishWeeklyPrice.sinhala_name,
                    " | ",
                    FishWeeklyPrice.common_name,
                )
            )
        )
    ).scalar() or 0

    latest = db.query(func.max(FishWeeklyPrice.updated_at)).scalar()

    status = read_status()
    source_file = status.get("lastLongFilename") or "—"

    return {
        "fishCount": int(fish_count),
        "rowCount": int(row_count),
        "lastUpdated": latest.strftime("%Y-%m-%d %H:%M:%S") if latest else "—",
        "sourceFile": source_file,
    }


@router.get("/list")
def list_db_rows(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    rows = (
        db.query(FishWeeklyPrice)
        .order_by(
            FishWeeklyPrice.year.desc(),
            FishWeeklyPrice.month.desc(),
            FishWeeklyPrice.week.desc(),
            FishWeeklyPrice.common_name.asc(),
        )
        .all()
    )

    return {
        "rows": [
            {
                "id": r.id,
                "sinhala_name": r.sinhala_name,
                "common_name": r.common_name,
                "year": r.year,
                "month": r.month,
                "week": r.week,
                "price": float(r.price) if r.price is not None else None,
            }
            for r in rows
        ]
    }


@router.get("/export")
def export_dataset(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    rows = (
        db.query(FishWeeklyPrice)
        .order_by(
            FishWeeklyPrice.year.asc(),
            FishWeeklyPrice.week.asc(),
            FishWeeklyPrice.common_name.asc(),
        )
        .all()
    )

    data = [
        {
            "Sinhala Name": r.sinhala_name,
            "Common Name": r.common_name,
            "Year": r.year,
            "Month": r.month,
            "Week": r.week,
            "Price": float(r.price) if r.price is not None else None,
        }
        for r in rows
    ]

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    file_path = os.path.join(export_dir, "actual_fish_prices_export.csv")
    pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8-sig")

    return FileResponse(file_path, filename="actual_fish_prices_export.csv")