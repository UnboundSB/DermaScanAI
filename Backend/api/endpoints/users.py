from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json
from Backend.Database.db import get_db, User, PredictionRecord
import hashlib

router = APIRouter()

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

@router.post("/register")
def register_user(username: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = User(username=username, password_hash=hash_password(password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

@router.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username, User.password_hash == hash_password(password)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": user.id}

@router.get("/{user_id}/history")
def get_user_history(user_id: int, db: Session = Depends(get_db)):
    records = db.query(PredictionRecord).filter(PredictionRecord.user_id == user_id).order_by(PredictionRecord.timestamp.desc()).all()
    history = []
    for r in records:
        history.append({
            "id": r.id,
            "scan_type": r.scan_type,
            "timestamp": r.timestamp.isoformat(),
            "report": json.loads(r.report_data)
        })
    return {"user_id": user_id, "history": history}