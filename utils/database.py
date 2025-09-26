import os
import datetime as dt
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine.url import URL
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# Engine & session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model
Base = declarative_base()

# Transcript model
class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
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


# Final interview result model
class InterviewResult(Base):
    __tablename__ = "interview_results"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False, unique=True)
    final_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)
