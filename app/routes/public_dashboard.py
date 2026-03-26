from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case

from app.db.session import get_db
from app.models.prediction_result import PredictionResult
from app.models.fish_weekly_price import FishWeeklyPrice

router = APIRouter(prefix="/public", tags=["Public Dashboard"])


def get_week_suffix(week: int):
    if week == 1:
        return "st"
    elif week == 2:
        return "nd"
    elif week == 3:
        return "rd"
    return "th"


@router.get("/dashboard-overview")
def get_dashboard_overview(db: Session = Depends(get_db)):
    month_order = case(
        (FishWeeklyPrice.month == "January", 1),
        (FishWeeklyPrice.month == "February", 2),
        (FishWeeklyPrice.month == "March", 3),
        (FishWeeklyPrice.month == "April", 4),
        (FishWeeklyPrice.month == "May", 5),
        (FishWeeklyPrice.month == "June", 6),
        (FishWeeklyPrice.month == "July", 7),
        (FishWeeklyPrice.month == "August", 8),
        (FishWeeklyPrice.month == "September", 9),
        (FishWeeklyPrice.month == "October", 10),
        (FishWeeklyPrice.month == "November", 11),
        (FishWeeklyPrice.month == "December", 12),
        else_=0,
    )

    # Latest actual week
    latest_actual_row = (
        db.query(FishWeeklyPrice)
        .order_by(
            desc(FishWeeklyPrice.year),
            desc(month_order),
            desc(FishWeeklyPrice.week),
            desc(FishWeeklyPrice.id),
        )
        .first()
    )

    actual_rows = []
    actual_week_label = None
    actual_count = 0

    if latest_actual_row:
        actual_year = latest_actual_row.year
        actual_month = latest_actual_row.month
        actual_week = latest_actual_row.week
        suffix = get_week_suffix(actual_week)
        actual_week_label = f"{actual_week}{suffix} week of {actual_month} {actual_year}"

        actual_db_rows = (
            db.query(FishWeeklyPrice)
            .filter(
                FishWeeklyPrice.year == actual_year,
                FishWeeklyPrice.month == actual_month,
                FishWeeklyPrice.week == actual_week,
            )
            .order_by(FishWeeklyPrice.common_name.asc())
            .all()
        )

        actual_count = len(actual_db_rows)

        actual_rows = [
            {
                "id": row.id,
                "sinhalaName": row.sinhala_name,
                "commonName": row.common_name,
                "year": row.year,
                "month": row.month,
                "week": row.week,
                "actualPrice": float(row.price) if row.price is not None else None,
            }
            for row in actual_db_rows[:5]
        ]

    # Latest published predictions
    published_rows = (
        db.query(PredictionResult)
        .filter(PredictionResult.is_published == True)
        .order_by(PredictionResult.common_name.asc())
        .all()
    )

    predicted_rows = [
        {
            "id": row.id,
            "sinhalaName": row.sinhala_name,
            "commonName": row.common_name,
            "year": row.year,
            "month": row.month,
            "week": row.week,
            "predictedPrice": float(row.predicted_price) if row.predicted_price is not None else None,
        }
        for row in published_rows[:5]
    ]

    predicted_week_label = published_rows[0].week_label if published_rows else None
    published_at = (
        published_rows[0].created_at.strftime("%Y-%m-%d %H:%M:%S")
        if published_rows and published_rows[0].created_at
        else None
    )

    return {
        "actualWeekLabel": actual_week_label,
        "predictedWeekLabel": predicted_week_label,
        "actualCount": actual_count,
        "predictedCount": len(published_rows),
        "publishedAt": published_at,
        "actualRows": actual_rows,
        "predictedRows": predicted_rows,
    }