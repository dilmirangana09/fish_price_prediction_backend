from fastapi import APIRouter, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from app.db.session import SessionLocal
from app.models.fish_weekly_price import FishWeeklyPrice

router = APIRouter(prefix="/actual-prices", tags=["actual-prices"])


@router.get("/filter-options")
def get_filter_options():
    db: Session = SessionLocal()
    try:
        rows = db.query(FishWeeklyPrice).all()

        years = sorted({row.year for row in rows if row.year is not None})
        months = sorted({row.month for row in rows if row.month is not None})
        weeks = sorted({row.week for row in rows if row.week is not None})

        return {
            "years": years,
            "months": months,
            "weeks": weeks,
        }
    finally:
        db.close()


@router.get("/latest")
def get_latest_actual_prices(
    search: str | None = Query(None),
    year: str | None = Query(None),
    month: str | None = Query(None),
    week: str | None = Query(None),
    limit: int | None = Query(None),
):
    db: Session = SessionLocal()
    try:
        q = db.query(FishWeeklyPrice)

        if search:
            search_text = f"%{search.strip()}%"
            q = q.filter(
                or_(
                    FishWeeklyPrice.sinhala_name.ilike(search_text),
                    FishWeeklyPrice.common_name.ilike(search_text),
                )
            )

        if year:
            q = q.filter(FishWeeklyPrice.year == int(year))

        if month:
            q = q.filter(FishWeeklyPrice.month == month)

        if week:
            q = q.filter(FishWeeklyPrice.week == int(week))

        q = q.order_by(
            desc(FishWeeklyPrice.year),
            desc(FishWeeklyPrice.month),
            desc(FishWeeklyPrice.week),
            desc(FishWeeklyPrice.id),
        )

        if limit:
            q = q.limit(limit)

        rows = q.all()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row.id,
                    "Sinhala Name": row.sinhala_name,
                    "Common Name": row.common_name,
                    "Year": row.year,
                    "Month": row.month,
                    "Week": row.week,
                    "Week_Label": f"{row.week} week of {row.month} {row.year}" if row.year and row.month and row.week else None,
                    "Actual_Price": float(row.price) if row.price is not None else None,
                    "Recorded_At": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if getattr(row, "created_at", None) else None,
                }
            )

        return {"rows": result}
    finally:
        db.close()