from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime

from app.db.base_class import Base


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(100), index=True, nullable=False)
    model_name = Column(String(20), nullable=False, default="Hybrid3_ANN_LSTM_XGB")

    sinhala_name = Column(String(50), nullable=True)
    common_name = Column(String(50), nullable=True)

    year = Column(Integer, nullable=True)
    month = Column(String(50), nullable=True)
    week = Column(Integer, nullable=True)
    week_label = Column(String(50), nullable=True)

    predicted_price = Column(Float, nullable=True)

    # source_long_file = Column(String(255), nullable=True)
    # source_prediction_file = Column(String(255), nullable=True)

    is_published = Column(Boolean, default=False)
    published_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)