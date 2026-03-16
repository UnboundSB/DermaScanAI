from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

# 1. Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./derma_history.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. Database Models (Schemas)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # Relationship to link users to their scan history
    scans = relationship("ScanHistory", back_populates="owner")

class ScanHistory(Base):
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    scan_type = Column(String) # "baseline" or "10_day"
    image_path = Column(String) # E.g., "/uploads/user1_scan5.jpg"
    
    # Store the PyTorch/Replyer output
    primary_diagnosis = Column(String)
    severity = Column(String)
    full_report = Column(Text) 
    
    # Automatically stamps the exact date and time the row is created
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="scans")

# 3. Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()