from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.db.session import get_db
from app.core.security import get_current_admin
from app.models.feedback import Feedback

router = APIRouter(prefix="/admin/feedback", tags=["admin-feedback"])

@router.get("/")
def get_all_feedback(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
    search: str | None = Query(default=None),
    date_filter: str | None = Query(default="all"),
    sort: str | None = Query(default="newest"),
    limit: int | None = Query(default=None),
):
    delete_old_feedback(db)

    query = db.query(Feedback)

    has_filters = bool(search) or (date_filter is not None and date_filter != "all") or (sort == "oldest")

    if search:
        search_text = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Feedback.name.ilike(search_text),
                Feedback.email.ilike(search_text),
                Feedback.message.ilike(search_text),
            )
        )

    now = datetime.now(timezone.utc)

    if date_filter == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(Feedback.created_at >= start)
    elif date_filter == "7days":
        start = now - timedelta(days=7)
        query = query.filter(Feedback.created_at >= start)
    elif date_filter == "30days":
        start = now - timedelta(days=30)
        query = query.filter(Feedback.created_at >= start)

    if sort == "oldest":
        query = query.order_by(Feedback.created_at.asc(), Feedback.id.asc())
    else:
        query = query.order_by(Feedback.created_at.desc(), Feedback.id.desc())

    if not has_filters and limit is not None and limit > 0:
        query = query.limit(limit)

    rows = query.all()

    return {
        "rowCount": len(rows),
        "rows": [
            {
                "id": row.id,
                "name": row.name,
                "email": row.email,
                "message": row.message,
                "createdAt": row.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if row.created_at
                else None,
            }
            for row in rows
        ],
    }


def delete_old_feedback(db: Session):
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
    db.query(Feedback).filter(
        Feedback.created_at < cutoff_date
    ).delete(synchronize_session=False)
    db.commit()

@router.get("/stats")
def get_feedback_stats(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    delete_old_feedback(db)

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    last_7_days = now - timedelta(days=7)
    last_30_days = now - timedelta(days=30)

    total_count = db.query(Feedback).count()
    today_count = db.query(Feedback).filter(Feedback.created_at >= today_start).count()
    last_7_days_count = db.query(Feedback).filter(Feedback.created_at >= last_7_days).count()
    last_30_days_count = db.query(Feedback).filter(Feedback.created_at >= last_30_days).count()

    return {
        "totalCount": total_count,
        "todayCount": today_count,
        "last7DaysCount": last_7_days_count,
        "last30DaysCount": last_30_days_count,
    }