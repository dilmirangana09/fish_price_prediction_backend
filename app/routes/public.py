from fastapi import APIRouter
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.prediction_result import PredictionResult
from fastapi import Depends

router = APIRouter(prefix="/public", tags=["Public"])

@router.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    results = db.query(PredictionResult).order_by(PredictionResult.week.desc()).all()

    return [
        {
            "fish": r.fish_name,
            "predicted_price": r.predicted_price,
            "week": r.week
        }
        for r in results
    ]