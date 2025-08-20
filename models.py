# models.py
from sqlalchemy import Column, Integer, String, DateTime, Float
import datetime as dt
from database import Base

class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False)
    question = Column(String, nullable=False)
    video_path = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    transcript = Column(String, nullable=False)
    eye_contact_score = Column(Float, default=0.0)
    posture_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    answer_quality_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    response_speed_score = Column(Float, default=0.0)
    final_engagement_score = Column(Float, default=0.0)
    smile_score = Column(Float, default=0.0)
    blink_rate_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
