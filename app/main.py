from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import engine
from app.db.models import Base
from app.routes.auth import router as auth_router
from app.routes.pipeline import router as pipeline_router
from app.routes.admin_dashboard import router as admin_dashboard_router
from app.routes.feedback import router as feedback_router
from app.routes.public import router as public_router
from app.routes.data_management import router as data_management_router
from app.routes.predictions import router as predictions_router
from app.db.base import Base
from app.routes.admin_prediction_history import router as admin_prediction_history_router
from app.routes.public_dashboard import router as public_dashboard_router
from app.routes.actual_prices import router as actual_prices_router
from app.routes.admin_feedback import router as admin_feedback_router

from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from app.db.session import SessionLocal
from app.models.feedback import Feedback
from app.routes.admin_dashboard import router as admin_dashboard_router



app = FastAPI(title="Fish Price Prediction API")
app.include_router(pipeline_router)
app.include_router(admin_dashboard_router)
app.include_router(feedback_router)
app.include_router(public_router)
app.include_router(data_management_router)
app.include_router(predictions_router)
app.include_router(admin_prediction_history_router)
app.include_router(public_dashboard_router)
app.include_router(actual_prices_router)
app.include_router(admin_feedback_router)
app.include_router(admin_dashboard_router)

scheduler = BackgroundScheduler()

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables (simple for MVP)
Base.metadata.create_all(bind=engine)

app.include_router(auth_router)

@app.get("/health")
def health():
    return {"status": "ok"}



def delete_old_feedback_job():
    db = SessionLocal()
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        db.query(Feedback).filter(
            Feedback.created_at < cutoff_date
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


scheduler.add_job(delete_old_feedback_job, "interval", days=1)
scheduler.start()
