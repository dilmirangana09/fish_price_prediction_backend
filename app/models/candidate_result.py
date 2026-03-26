from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from app.db.base import Base


class CandidateResult(Base):
    __tablename__ = "candidate_results"

    id = Column(Integer, primary_key=True, index=True)

    model_name = Column(String(100), nullable=False)
    version_name = Column(String(100), nullable=False, unique=True)

    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)

    ann_weight = Column(Float, nullable=True)
    xgb_weight = Column(Float, nullable=True)

    fish_count = Column(Integer, nullable=True)
    train_rows = Column(Integer, nullable=True)
    val_rows = Column(Integer, nullable=True)
    test_rows = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)