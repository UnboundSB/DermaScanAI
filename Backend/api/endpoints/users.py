import json
import hashlib
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

# --- The Ghost-Proof Import ---
from core_db.db import get_db, User, PredictionRecord

router = APIRouter()

def hash_password(password: str):
    """Hashes the password using SHA-256 for database security."""
    return hashlib.sha256(password.encode()).hexdigest()

@router.post("/register")
def register_user(username: str, password: str, db: Session = Depends(get_db)):
    """Creates a new user account."""
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = User(username=username, password_hash=hash_password(password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

@router.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    """Validates user credentials and returns the user ID."""
    user = db.query(User).filter(User.username == username, User.password_hash == hash_password(password)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": user.id}

@router.get("/{user_id}/history")
def get_user_history(user_id: int, db: Session = Depends(get_db)):
    """
    Fetches the chronological timeline of a user's scans.
    Deliberately excludes the 'image_data' BLOB so the response is lightning fast.
    """
    # Order by descending timestamp (latest first)
    records = db.query(PredictionRecord).filter(
        PredictionRecord.user_id == user_id
    ).order_by(desc(PredictionRecord.timestamp)).all()
    
    history = []
    for r in records:
        history.append({
            "id": r.id,
            "scan_type": r.scan_type,
            "timestamp": r.timestamp.isoformat(),
            "report": json.loads(r.report_data)
        })
        
    return {"user_id": user_id, "history": history}