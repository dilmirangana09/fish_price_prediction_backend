from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.core.security import get_current_admin
from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.prediction_result import PredictionResult
from app.models.feedback import Feedback

from app.services.system_status import read_status

router = APIRouter(prefix="/admin/dashboard", tags=["admin-dashboard"])

@router.get("/stats")
def get_dashboard_stats(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()

    fish_count = db.query(func.count(func.distinct(FishWeeklyPrice.common_name))).scalar() or 0
    total_feedback = db.query(func.count(Feedback.id)).scalar() or 0
    published_prediction_count = (
        db.query(func.count(PredictionResult.id))
        .filter(PredictionResult.is_published == True)
        .scalar()
        or 0
    )

    latest_actual = (
        db.query(FishWeeklyPrice)
        .order_by(
            desc(FishWeeklyPrice.year),
            desc(FishWeeklyPrice.month),
            desc(FishWeeklyPrice.week),
        )
        .first()
    )

    latest_prediction = (
        db.query(PredictionResult)
        .filter(PredictionResult.is_published == True)
        .order_by(desc(PredictionResult.published_at))
        .first()
    )

    latest_actual_week = "—"
    if latest_actual:
        latest_actual_week = f"{latest_actual.month} {latest_actual.year} - Week {latest_actual.week}"

    latest_prediction_week = "—"
    if latest_prediction:
        latest_prediction_week = latest_prediction.week_label or "—"

    return {
        "fishCount": fish_count,
        "totalFeedback": total_feedback,
        "publishedPredictionCount": published_prediction_count,
        "latestActualWeek": latest_actual_week,
        "latestPredictionWeek": latest_prediction_week,
        "lastUploadDate": status.get("lastUploadDate", "—"),
        "lastModelName": status.get("lastModelName", "—"),
        "lastModelTrainedAt": status.get("lastModelTrainedAt", "—"),
    }