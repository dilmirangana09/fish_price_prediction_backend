from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from app.db.base_class import Base


class UploadLog(Base):
    __tablename__ = "upload_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=True)
    stored_filename = Column(String(255), nullable=True)
    week_label = Column(String(100), nullable=True)
    fish_count = Column(Integer, nullable=True)
    action = Column(String(100), nullable=True)
    status = Column(String(50), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))