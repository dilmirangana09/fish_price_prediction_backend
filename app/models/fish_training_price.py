from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, Numeric, String, UniqueConstraint
from app.db.base_class import Base


class FishTrainingPrice(Base):
    __tablename__ = "fish_training_prices"

    id = Column(Integer, primary_key=True, index=True)
    sinhala_name = Column(String(255), nullable=False)
    common_name = Column(String(255), nullable=False)
    year = Column(Integer, nullable=False)
    month = Column(String(50), nullable=False)
    week = Column(Integer, nullable=False)
    price = Column(Numeric(10, 2), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint(
            "sinhala_name",
            "common_name",
            "year",
            "month",
            "week",
            name="uq_training_fish_week",
        ),
    )