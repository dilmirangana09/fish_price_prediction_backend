
from fastapi import APIRouter, Depends,HTTPException, Query
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import get_db
from app.models.prediction_result import PredictionResult
from app.models.fish_weekly_price import FishWeeklyPrice
from app.services.prediction_service import generate_next_week_predictions_with_saved_hybrid
import os
from sqlalchemy import or_

UPLOAD_DIR = "uploads"

router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post("/generate-hybrid-predictions")
def generate_hybrid_predictions(admin=Depends(get_current_admin)):
    status = read_status()
    long_filename = status.get("lastLongFilename")

    if not long_filename:
        raise HTTPException(status_code=404, detail="Long format dataset not found. Generate long format dataset first.")

    try:
        pred_df = generate_next_week_predictions_with_saved_hybrid(long_filename)

        pred_filename = f"hybrid_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pred_path = os.path.join(UPLOAD_DIR, pred_filename)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        status["lastPredictionFilename"] = pred_filename
        status["lastPredictionDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_status(status)

        return {
            "message": "Hybrid predictions generated successfully",
            "filename": pred_filename,
            "rowCount": int(len(pred_df)),
            "preview": pred_df.head(30).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")






@router.get("/latest-published")
def get_latest_published_predictions(
    db: Session = Depends(get_db),
    search: str | None = Query(default=None),
    year: int | None = Query(default=None),
    month: str | None = Query(default=None),
    week: int | None = Query(default=None),
    limit: int | None = Query(default=None),
):
    query = db.query(PredictionResult).filter(PredictionResult.is_published == True)

    # If no filters, get latest published batch first
    if year is None and month is None and week is None and not search:
        latest_batch = (
            db.query(PredictionResult.batch_id)
            .filter(PredictionResult.is_published == True)
            .order_by(PredictionResult.created_at.desc())
            .first()
        )

        if not latest_batch:
            return {
                "rowCount": 0,
                "rows": [],
            }

        batch_id = latest_batch[0]
        query = query.filter(PredictionResult.batch_id == batch_id)

    # Apply filters
    if year is not None:
        query = query.filter(PredictionResult.year == year)

    if month is not None and month != "":
        query = query.filter(PredictionResult.month == month)

    if week is not None:
        query = query.filter(PredictionResult.week == week)

    if search:
        search_text = f"%{search.strip()}%"
        query = query.filter(
            or_(
                PredictionResult.sinhala_name.ilike(search_text),
                PredictionResult.common_name.ilike(search_text),
            )
        )

    query = query.order_by(
        PredictionResult.year.desc(),
        PredictionResult.week.desc(),
        PredictionResult.common_name.asc(),
    )

    # Apply limit only when no filters
    if (
        limit is not None
        and limit > 0
        and not search
        and year is None
        and month is None
        and week is None
    ):
        query = query.limit(limit)

    rows = query.all()

    return {
        "rowCount": len(rows),
        "rows": [
            {
                "id": row.id,
                "Sinhala Name": row.sinhala_name,
                "Common Name": row.common_name,
                "Year": row.year,
                "Month": row.month,
                "Week": row.week,
                "Week_Label": row.week_label,
                "Predicted_Price": float(row.predicted_price) if row.predicted_price is not None else None,
                "Published_At": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None,
            }
            for row in rows
        ],
    }

@router.get("/filter-options")
def get_prediction_filter_options(db: Session = Depends(get_db)):
    years = [
        row[0]
        for row in db.query(PredictionResult.year)
        .filter(PredictionResult.is_published == True)
        .distinct()
        .order_by(PredictionResult.year.desc())
        .all()
    ]

    months = [
        row[0]
        for row in db.query(PredictionResult.month)
        .filter(PredictionResult.is_published == True)
        .distinct()
        .order_by(PredictionResult.month.asc())
        .all()
    ]

    weeks = [
        row[0]
        for row in db.query(PredictionResult.week)
        .filter(PredictionResult.is_published == True)
        .distinct()
        .order_by(PredictionResult.week.asc())
        .all()
    ]

    return {
        "years": years,
        "months": months,
        "weeks": weeks,
    }