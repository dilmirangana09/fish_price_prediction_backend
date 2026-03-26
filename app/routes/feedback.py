from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.feedback import Feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    message: str = Field(..., min_length=5, max_length=2000)


def delete_old_feedback(db: Session):
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)

    db.query(Feedback).filter(
        Feedback.created_at < cutoff_date
    ).delete(synchronize_session=False)

    db.commit()


@router.post("/")
def create_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    try:
        # auto delete feedback older than 3 months
        delete_old_feedback(db)

        feedback = Feedback(
            name=payload.name.strip(),
            email=payload.email.strip(),
            message=payload.message.strip(),
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        return {
            "message": "Feedback submitted successfully",
            "id": feedback.id,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")