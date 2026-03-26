from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
from app.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)

    model_name = Column(String, nullable=False)
    version_name = Column(String, nullable=False)

    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    r2 = Column(Float)

    ann_weight = Column(Float)
    xgb_weight = Column(Float)

    is_deployed = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)