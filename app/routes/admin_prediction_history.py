from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.core.security import get_current_admin
from app.models.prediction_result import PredictionResult

router = APIRouter(prefix="/admin/prediction-history", tags=["admin-prediction-history"])


@router.get("/")
def get_prediction_history(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    rows = (
        db.query(
            PredictionResult.batch_id.label("batch_id"),
            func.max(PredictionResult.model_name).label("model_name"),
            func.max(PredictionResult.week_label).label("week_label"),
            func.max(PredictionResult.year).label("year"),
            func.max(PredictionResult.month).label("month"),
            func.max(PredictionResult.week).label("week"),
            func.count(PredictionResult.id).label("row_count"),
            func.max(PredictionResult.is_published).label("is_published"),
            func.max(PredictionResult.source_prediction_file).label("source_prediction_file"),
            func.max(PredictionResult.created_at).label("created_at"),
        )
        .filter(PredictionResult.batch_id.isnot(None))
        .group_by(PredictionResult.batch_id)
        .order_by(func.max(PredictionResult.created_at).desc())
        .all()
    )

    return {
        "rowCount": len(rows),
        "rows": [
            {
                "batchId": row.batch_id,
                "modelName": row.model_name,
                "weekLabel": row.week_label,
                "year": row.year,
                "month": row.month,
                "week": row.week,
                "rowCount": int(row.row_count or 0),
                "isPublished": bool(row.is_published),
                "sourcePredictionFile": row.source_prediction_file,
                "createdAt": row.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if row.created_at
                else None,
            }
            for row in rows
        ],
    }


@router.get("/{batch_id}")
def get_prediction_history_details(
    batch_id: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    rows = (
        db.query(PredictionResult)
        .filter(PredictionResult.batch_id == batch_id)
        .order_by(PredictionResult.common_name.asc())
        .all()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Prediction batch not found")

    first = rows[0]

    return {
        "batchId": batch_id,
        "modelName": first.model_name,
        "weekLabel": first.week_label,
        "isPublished": first.is_published,
        "createdAt": first.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if first.created_at
        else None,
        "rows": [
            {
                "id": row.id,
                "sinhalaName": row.sinhala_name,
                "commonName": row.common_name,
                "year": row.year,
                "month": row.month,
                "week": row.week,
                "weekLabel": row.week_label,
                "predictedPrice": float(row.predicted_price) if row.predicted_price is not None else None,
                "sourceLongFile": row.source_long_file,
                "sourcePredictionFile": row.source_prediction_file,
            }
            for row in rows
        ],
    }


@router.post("/{batch_id}/publish")
def publish_prediction_batch(
    batch_id: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    exists = (
        db.query(PredictionResult)
        .filter(PredictionResult.batch_id == batch_id)
        .first()
    )

    if not exists:
        raise HTTPException(status_code=404, detail="Prediction batch not found")

    db.query(PredictionResult).update(
        {PredictionResult.is_published: False},
        synchronize_session=False,
    )

    db.query(PredictionResult).filter(
        PredictionResult.batch_id == batch_id
    ).update(
        {PredictionResult.is_published: True},
        synchronize_session=False,
    )

    db.commit()

    return {
        "message": "Prediction batch published successfully",
        "batchId": batch_id,
    }


@router.delete("/{batch_id}")
def delete_prediction_batch(
    batch_id: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    q = db.query(PredictionResult).filter(PredictionResult.batch_id == batch_id)
    count = q.count()

    if count == 0:
        raise HTTPException(status_code=404, detail="Prediction batch not found")

    q.delete(synchronize_session=False)
    db.commit()

    return {
        "message": "Prediction batch deleted successfully",
        "batchId": batch_id,
        "deletedRows": count,
    }