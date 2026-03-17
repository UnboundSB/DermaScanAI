import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

# --- 1. Database Connection ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./derma_history.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. Database Schemas ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    
    # Relationship mapping
    records = relationship("PredictionRecord", back_populates="owner")


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    scan_type = Column(String) # "normal" or "10_day"
    report_data = Column(Text) # The JSON blob from your PyTorch engine
    
    # THE BLOB VAULT: Stores the raw image bytes natively inside SQLite
    image_data = Column(LargeBinary, nullable=True) 
    
    # Automatically stamps the exact date and time the scan was processed
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="records")

# --- 3. Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()